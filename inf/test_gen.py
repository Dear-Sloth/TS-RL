import os
import torch
import numpy as np
import pandas as pd
import json
from data_factory import data_provider
from utils.tools import visual
from utils.m4_summary import M4Summary
from m4 import M4Meta

from vllm import LLM, SamplingParams
import instructions

class test_generation:
    def __init__(self, args):
        self.args = args

        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            self.args.seq_len = 2 * self.args.pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        
        self.llm = LLM(
            model=self.args.model_path,
            tensor_parallel_size=self.args.tensor_parallel_size,
            gpu_memory_utilization=self.args.gpu_memory_utilization
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=15000
        )

    def _get_data(self, flag):

        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def test(self):
        """
        实现：
        1) 第一次遍历所有样本分批推理
        2) 将失败的收集到 fail_list
        3) 对 fail_list 做多轮批量重试（可设置max_rounds，避免无限循环）
        4) 在成功/修复时若序号<10则立刻存 debug+图
        5) 全部完成后输出 CSV
        """
        # ========== 1. 获取数据 ==========
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
    
        x, _ = train_loader.dataset.last_insample_window()  # shape (B, seq_len)
        y = test_loader.dataset.timeseries  # shape (B, pred_len)
        ids = test_loader.dataset.ids       # (B,)
        x = torch.tensor(x, dtype=torch.float32).to('cuda').unsqueeze(-1)  # (B, seq_len, 1)
    
        if self.args.norm:
            means = x.mean(dim=1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_normed = (x - means) / stdev
        else:
            x_normed = x
    
        x_normed = x_normed.cpu().numpy()
        B = x_normed.shape[0]
    
        # ========== 2. 构造全部prompts ==========
        all_prompts = []
        for i in range(B):
            sample_x = x_normed[i, :, 0]
            sys_prompt = instructions.system
            if self.args.norm:
                input_query = instructions.use_norm.format(
                    history_len=self.args.seq_len,
                    future_len=self.args.pred_len,
                    history=",".join(map(str, sample_x))
                )
            else:
                input_query = instructions.basic_inp.format(
                    history_len=self.args.seq_len,
                    future_len=self.args.pred_len,
                    history=",".join(map(str, sample_x))
                )
            chatml_prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{input_query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            all_prompts.append(chatml_prompt)
    
        # ========== 3. 数据结构准备 ==========
        preds = np.zeros((B, self.args.pred_len), dtype=np.float32)
        final_responses = [None]*B
        saved_indices = set()  # 已经保存过前10条的序列索引
    
        folder_path = './test_results/'
        os.makedirs(folder_path, exist_ok=True)
        folder_path_m4 = './m4_results/' + self.args.model_path + '/'
        os.makedirs(folder_path_m4, exist_ok=True)
        debug_conversation_file = os.path.join(folder_path_m4, f"debug_conversation_{self.args.seasonal_patterns}.json")
    
        force_fixed_count = 0
    
        # ========== 4. 第一次遍历: 常规 batch 生成 + 推理 ==========
        batch_size = self.args.batch_size if self.args.batch_size>0 else 16
    
        fail_list = []  # 用来收集所有没成功解析的
        i = 0
        while i < B:
            batch_end = min(i+batch_size, B)
            batch_indices = list(range(i, batch_end))
    
            # 一次性并行推理
            prompt_list = [all_prompts[idx] for idx in batch_indices]
            batch_outputs = self.llm.generate(prompt_list, self.sampling_params)
    
            for b_i, out in enumerate(batch_outputs):
                real_idx = batch_indices[b_i]
                response = out.outputs[0].text
                splitted = response.split("[future]")
                pred_str = splitted[1].strip() if len(splitted)>1 else ""
    
                try:
                    local_vals = [float(x_) for x_ in pred_str.split(",")]
                except:
                    local_vals = []
    
                if len(local_vals)==self.args.pred_len:
                    # 成功
                    final_responses[real_idx] = response
                    preds[real_idx,:] = local_vals
                    # 若是前10条,立即存
                    if real_idx<10 and real_idx not in saved_indices:
                        self._save_debug_and_figure(
                            debug_conversation_file,
                            real_idx, 
                            all_prompts[real_idx],
                            response,
                            x,
                            y,
                            preds
                        )
                        saved_indices.add(real_idx)
                else:
                    # 失败 => 加入 fail_list
                    fail_list.append(real_idx)
    
            i = batch_end
    
        print(f"[1st stage done]: total fails = {len(fail_list)} / {B}")
    
        # ========== 5. 第二阶段: 多轮批量重试 ==========
    
        # 你可以不设置无限循环，以免死循环
        # max_rounds 仅做兜底，防止一直无法成功
        max_rounds = 20
        round_cnt = 0
    
        while len(fail_list)>0 and round_cnt<max_rounds:
            round_cnt += 1
            print(f"==== Retry Round {round_cnt}, fail_list size = {len(fail_list)} ====")
    
            new_fail_list = []
    
            # 分批推理 fail_list
            i2 = 0
            while i2 < len(fail_list):
                batch_end = min(i2+batch_size, len(fail_list))
                batch_indices = fail_list[i2:batch_end]
    
                prompt_list = [all_prompts[idx] for idx in batch_indices]
                batch_outputs = self.llm.generate(prompt_list, self.sampling_params)
    
                for b_i, out in enumerate(batch_outputs):
                    real_idx = batch_indices[b_i]
                    response = out.outputs[0].text
                    splitted = response.split("[future]")
                    pred_str = splitted[1].strip() if len(splitted)>1 else ""
                    try:
                        local_vals = [float(x_) for x_ in pred_str.split(",")]
                    except:
                        local_vals = []
    
                    if len(local_vals)==self.args.pred_len:
                        # 成功
                        final_responses[real_idx] = response
                        preds[real_idx,:] = local_vals
                        if real_idx<10 and real_idx not in saved_indices:
                            self._save_debug_and_figure(
                                debug_conversation_file,
                                real_idx,
                                all_prompts[real_idx],
                                response,
                                x,
                                y,
                                preds
                            )
                            saved_indices.add(real_idx)
                    else:
                        # 还失败 => 继续留在 new_fail_list
                        new_fail_list.append(real_idx)
    
                i2 = batch_end
    
            fail_list = new_fail_list
            print(f"Round {round_cnt} done, leftover fails = {len(fail_list)}")
    
        # 结束后，如果还剩 fail_list，就 fallback
        if len(fail_list)>0:
            print(f"[!] Fallback forced for {len(fail_list)} samples after {round_cnt} rounds")
            for idx in fail_list:
                force_fixed_count += 1
                # 强制修复
                final_responses[idx] = "Fallback forced"
                preds[idx, :] = [0.0]*self.args.pred_len
                if idx<10 and idx not in saved_indices:
                    self._save_debug_and_figure(
                        debug_conversation_file,
                        idx,
                        all_prompts[idx],
                        final_responses[idx],
                        x,
                        y,
                        preds
                    )
                    saved_indices.add(idx)
    
        print(f"[!][!][!] Force fix the number of sequences: {force_fixed_count}")
    
        # ========== 6. 全部完成 -> 反归一化, 再输出 CSV ==========
        if self.args.norm:
            m_ = means.cpu().numpy()
            s_ = stdev.cpu().numpy()
            for i in range(B):
                m = m_[i, 0, 0]
                s = s_[i, 0, 0]
                preds[i, :] = [(val*s + m) for val in preds[i,:]]
    
        # 最后写CSV
        df = pd.DataFrame(preds, columns=[f"V{i+1}" for i in range(self.args.pred_len)])
        df.index = ids
        df.index.name = 'id'
        out_file = os.path.join(folder_path_m4, f"{self.args.seasonal_patterns}_forecast.csv")
        df.to_csv(out_file)
        print("Forecast CSV saved:", out_file)


            # M4Summary评估
        if 'Weekly_forecast.csv' in os.listdir(folder_path_m4) \
                    and 'Monthly_forecast.csv' in os.listdir(folder_path_m4) \
                    and 'Yearly_forecast.csv' in os.listdir(folder_path_m4) \
                    and 'Daily_forecast.csv' in os.listdir(folder_path_m4) \
                    and 'Hourly_forecast.csv' in os.listdir(folder_path_m4) \
                    and 'Quarterly_forecast.csv' in os.listdir(folder_path_m4):
                m4_summary = M4Summary(folder_path_m4, self.args.root_path)
                smape_results, owa_results, mape_results, mase_results = m4_summary.evaluate()
                print('smape:', smape_results)
                print('mape:', mape_results)
                print('mase:', mase_results)
                print('owa:', owa_results)
        else:
                print('After all 6 tasks are finished, you can calculate the averaged index')

        return

    def _save_debug_and_figure(
        self,
        debug_conversation_file: str,
        idx: int,
        user_prompt: str,
        assistant_response: str,
        x_tensor: torch.Tensor,
        y_np: np.ndarray,
        preds_np: np.ndarray,
    ):
        """
        辅助方法：针对单条序列 idx，
        1) 记录到 debug json
        2) 保存可视化对比图
        """
        debug_info = {
            "index": idx,
            "system_prompt": instructions.system,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
            "ground_truth": y_np[idx].tolist(),
            # 你也可加上 "predicted": preds_np[idx].tolist() 方便对比
        }
        with open(debug_conversation_file, "a", encoding="utf-8") as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=4)
            f.write("\n")
    
        # 画图
        # x_tensor: shape (B, seq_len, 1)
        x_cpu = x_tensor.cpu().numpy()
        folder_path = './test_results/'
        os.makedirs(folder_path, exist_ok=True)
    
        gt = np.concatenate((x_cpu[idx, :, 0], y_np[idx]), axis=0)
        pd_ = np.concatenate((x_cpu[idx, :, 0], preds_np[idx, :]), axis=0)
        from utils.tools import visual
        visual(gt, pd_, os.path.join(folder_path, f"{idx}.pdf"))

