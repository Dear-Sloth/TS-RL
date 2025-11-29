import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

import instructions
from data_factory import data_provider
from m4 import M4Meta
from utils.m4_summary import M4Summary
from utils.tools import visual


class test_generation:
    """vLLM based inference pipeline for M4-style forecasting datasets."""

    FUTURE_PATTERN = re.compile(r"<future>(.*?)</future>", flags=re.IGNORECASE | re.DOTALL)

    def __init__(self, args):
        self.args = args

        if self.args.data == "m4":
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            self.args.seq_len = 2 * self.args.pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if getattr(self.args, "tensor_parallel_size", None) is None or self.args.tensor_parallel_size <= 0:
            self.args.tensor_parallel_size = max(1, torch.cuda.device_count())

        self.llm = LLM(
            model=self.args.model_path,
            tensor_parallel_size=self.args.tensor_parallel_size,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
        )

    def _get_data(self, flag):
        dataset, dataloader = data_provider(self.args, flag)
        return dataset, dataloader

    def _format_system_prompt(self) -> str:
        try:
            return instructions.system_train.format(future_len=self.args.pred_len)
        except Exception:
            return instructions.system_train

    def _format_user_prompt(self, history_values: np.ndarray) -> str:
        template = instructions.use_norm if self.args.norm else instructions.nonorm_noy_train
        history_str = ",".join(map(str, history_values))
        return template.format(history_len=self.args.seq_len, future_len=self.args.pred_len, history=history_str)

    def _build_prompts(self, history: np.ndarray) -> tuple[list[str], list[str]]:
        """
        仅修复问题点：使用 Llama 3.1 Instruct 模板构造对话。
        返回：(chat_prompts, user_prompts)
        """
        chat_prompts: list[str] = []
        user_prompts: list[str] = []
        system_prompt = self._format_system_prompt()
        for series in history:
            user_prompt = self._format_user_prompt(series)
            # Llama 3.1 Instruct 风格模板
            chat_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}\n<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
            chat_prompts.append(chat_prompt)
            user_prompts.append(user_prompt)
        return chat_prompts, user_prompts

    def test(self):
        _, train_loader = self._get_data(flag="train")
        _, test_loader = self._get_data(flag="test")

        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        ids = test_loader.dataset.ids

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(-1)

        if self.args.norm:
            means = x_tensor.mean(dim=1, keepdim=True)
            stdev = torch.sqrt(torch.var(x_tensor, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_normed = (x_tensor - means) / stdev
        else:
            x_normed = x_tensor

        history_np = x_normed.detach().cpu().numpy()[..., 0]
        per_device_batch = self.args.batch_size if self.args.batch_size > 0 else 16
        effective_batch = per_device_batch * max(1, self.args.tensor_parallel_size)
        chat_prompts, user_prompts = self._build_prompts(history_np)
        total_samples = len(chat_prompts)

        predictions = np.zeros((total_samples, self.args.pred_len), dtype=np.float32)
        final_responses: list[str | None] = [None] * total_samples
        fail_list: list[int] = []
        saved_indices: set[int] = set()

        output_root = Path("m4_results") / self.args.model_path.replace("/", "_")
        output_root.mkdir(parents=True, exist_ok=True)
        debug_conversation_file = output_root / f"debug_conversation_{self.args.seasonal_patterns}.json"
        test_results_dir = Path("test_results")
        test_results_dir.mkdir(parents=True, exist_ok=True)

        def process_batch(indices: list[int]):
            prompt_list = [chat_prompts[idx] for idx in indices]
            outputs = self.llm.generate(prompt_list, self.sampling_params)

            for local_idx, output in enumerate(outputs):
                real_idx = indices[local_idx]
                response = output.outputs[0].text
                future_values = self._extract_future_values(response)

                if len(future_values) == self.args.pred_len:
                    predictions[real_idx, :] = future_values
                    final_responses[real_idx] = response
                    if real_idx < 10 and real_idx not in saved_indices:
                        self._save_debug_and_figure(
                            debug_conversation_file,
                            real_idx,
                            user_prompts[real_idx],
                            response,
                            x_tensor,
                            y,
                            predictions,
                            test_results_dir,
                            chat_prompt=chat_prompts[real_idx],
                        )
                        saved_indices.add(real_idx)
                else:
                    fail_list.append(real_idx)

        # first pass
        idx = 0
        while idx < total_samples:
            batch_indices = list(range(idx, min(idx + effective_batch, total_samples)))
            process_batch(batch_indices)
            idx += effective_batch

        print(f"[1st stage done]: total fails = {len(fail_list)} / {total_samples}")

        # retry loop
        max_rounds = 20
        round_id = 0
        while fail_list and round_id < max_rounds:
            round_id += 1
            print(f"==== Retry Round {round_id}, fail_list size = {len(fail_list)} ====")
            pending = fail_list
            fail_list = []
            idx = 0
            while idx < len(pending):
                batch_indices = pending[idx : min(idx + effective_batch, len(pending))]
                process_batch(batch_indices)
                idx += effective_batch
            print(f"Round {round_id} done, leftover fails = {len(fail_list)}")

        # fallback
        force_fixed_count = 0
        if fail_list:
            print(f"[!] Fallback forced for {len(fail_list)} samples after {round_id} rounds")
            for index in fail_list:
                force_fixed_count += 1
                predictions[index, :] = 0.0
                final_responses[index] = "Fallback forced"
                if index < 10 and index not in saved_indices:
                    self._save_debug_and_figure(
                        debug_conversation_file,
                        index,
                        user_prompts[index],
                        final_responses[index],
                        x_tensor,
                        y,
                        predictions,
                        test_results_dir,
                        chat_prompt=chat_prompts[index],
                    )
                    saved_indices.add(index)
        print(f"[!][!][!] Force fix the number of sequences: {force_fixed_count}")

        if self.args.norm:
            means_np = means.detach().cpu().numpy()
            stdev_np = stdev.detach().cpu().numpy()
            for i in range(total_samples):
                predictions[i, :] = predictions[i, :] * stdev_np[i, 0, 0] + means_np[i, 0, 0]

        df = pd.DataFrame(predictions, columns=[f"V{i+1}" for i in range(self.args.pred_len)])
        df.insert(0, "id", ids)
        out_file = output_root / f"{self.args.seasonal_patterns}_forecast.csv"
        df.to_csv(out_file, index=False)
        print("Forecast CSV saved:", out_file)

        required_files = {
            "Weekly_forecast.csv",
            "Monthly_forecast.csv",
            "Yearly_forecast.csv",
            "Daily_forecast.csv",
            "Hourly_forecast.csv",
            "Quarterly_forecast.csv",
        }
        existing = set(os.listdir(output_root))
        if required_files.issubset(existing):
            m4_summary = M4Summary(str(output_root), self.args.root_path)
            smape_results, owa_results, mape_results, mase_results = m4_summary.evaluate()
            print("smape:", smape_results)
            print("mape:", mape_results)
            print("mase:", mase_results)
            print("owa:", owa_results)
        else:
            print("After all 6 tasks are finished, you can calculate the averaged index")

    def _save_debug_and_figure(
        self,
        debug_file: Path,
        idx: int,
        user_prompt: str,
        assistant_response: str,
        x_tensor: torch.Tensor,
        ground_truth: np.ndarray,
        preds: np.ndarray,
        test_results_dir: Path,
        *,
        chat_prompt: str | None = None,
    ):
        debug_info = {
            "index": idx,
            "system_prompt": self._format_system_prompt(),
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
            "ground_truth": ground_truth[idx].tolist(),
            "predicted": preds[idx].tolist(),
            "parsed_future": self._extract_future_values(assistant_response),
        }
        with open(debug_file, "a", encoding="utf-8") as fp:
            json.dump(debug_info, fp, ensure_ascii=False)
            fp.write("\n")

        x_cpu = x_tensor.detach().cpu().numpy()
        gt_series = np.concatenate((x_cpu[idx, :, 0], ground_truth[idx]), axis=0)
        pred_series = np.concatenate((x_cpu[idx, :, 0], preds[idx, :]), axis=0)
        visual(gt_series, pred_series, str(test_results_dir / f"{idx}.pdf"))

    def _extract_future_values(self, response: str) -> list[float]:
        match = self.FUTURE_PATTERN.search(response)
        if not match:
            return []
        payload = match.group(1)
        tokens = [token.strip() for token in payload.split(",") if token.strip()]
        try:
            return [float(token) for token in tokens]
        except ValueError:
            return []
