import argparse
import json
import os
from typing import Dict, List

import numpy as np

from data_loader import Dataset_M4
from m4 import M4Meta
import instructions


def generate_prompt_content(history_values: List[float], seq_len: int, pred_len: int) -> str:
    """
    构造不归一化、无 Y 的用户提示（与 tsrl/instructions.nonorm_noy_train 一致）。
    """
    history_str = ",".join(str(v) for v in history_values)
    return instructions.nonorm_noy_train.format(
        history_len=seq_len,
        future_len=pred_len,
        history=history_str,
    )


def build_record(
    data_source: str,
    ability: str,
    content: str,
    ground_truth: List[float],
    extra_info: Dict,
    system_prompt: str | None,
) -> Dict:
    """
    生成一条 VERL 风格样本记录。
    prompt 仅包含 user 消息；奖励信息通过 reward_model.ground_truth 传入，供 RM/环境计算。
    """
    prompt_msgs = []
    if system_prompt:
        prompt_msgs.append({
            "content": system_prompt,
            "role": "system",
        })
    prompt_msgs.append({
        "content": content,
        "role": "user",
    })

    return {
        "data_source": data_source,
        "prompt": prompt_msgs,
        "ability": ability,
        "reward_model": {
            "ground_truth": ground_truth,
            "style": "ts_m4",
        },
        "extra_info": extra_info,
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_for_pattern(
    root_path: str,
    seasonal_pattern: str,
    out_dir: str,
    data_source: str,
    ability: str,
    include_extra_qa: bool,
    use_test_horizon: bool,
    include_system: bool,
) -> str:
    """
    基于 M4 指定季节频率，构造 JSONL 数据集（不归一化、无 Y）。
    只依赖训练集窗口；ground_truth 仅用于 reward_model，不出现在 prompt。
    返回输出文件路径。
    """
    pred_len = M4Meta.horizons_map[seasonal_pattern]
    seq_len = 2 * pred_len
    label_len = 0

    # 训练集：用于构造历史窗口
    train_dataset = Dataset_M4(
        args=None,
        root_path=root_path,
        flag="train",
        size=[seq_len, label_len, pred_len],
        features="S",
        data_path="",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="",
        seasonal_patterns=seasonal_pattern,
    )
    # 测试集：可选，用于严格提供 pred_len 长度的 horizon
    test_dataset = None
    if use_test_horizon:
        test_dataset = Dataset_M4(
            args=None,
            root_path=root_path,
            flag="test",
            size=[seq_len, label_len, pred_len],
            features="S",
            data_path="",
            target="OT",
            scale=False,
            inverse=False,
            timeenc=0,
            freq="",
            seasonal_patterns=seasonal_pattern,
        )

    pattern_dir = os.path.join(out_dir, f"_{seasonal_pattern}")
    ensure_dir(pattern_dir)
    out_file = os.path.join(pattern_dir, "train.jsonl")

    with open(out_file, "w", encoding="utf-8") as f:
        total = len(train_dataset)
        for idx in range(total):
            if use_test_horizon:
                # 使用训练集最后窗口 + 测试集 horizon
                ins_all, _ = train_dataset.last_insample_window()
                history_values = [float(v) for v in ins_all[idx].tolist()]
                # 测试集 timeseries 作为地真 horizon，确保长度==pred_len
                raw_future = test_dataset.timeseries[idx]
                future_values = [float(v) for v in raw_future.tolist()]
                if len(future_values) != pred_len:
                    # 兜底：截断或补零
                    if len(future_values) > pred_len:
                        future_values = future_values[:pred_len]
                    else:
                        future_values = future_values + [0.0] * (pred_len - len(future_values))
            else:
                # 从 __getitem__ 随机切片，未来窗口可能短于 pred_len
                insample, outsample, _, _ = train_dataset.__getitem__(idx)
                history_values = [float(v) for v in insample[:, 0].tolist()]
                future_values = [float(v) for v in outsample[:, 0].tolist()]
                if len(future_values) != pred_len:
                    # 兜底：补齐长度
                    if len(future_values) > pred_len:
                        future_values = future_values[:pred_len]
                    else:
                        future_values = future_values + [0.0] * (pred_len - len(future_values))

            content = generate_prompt_content(history_values, seq_len, pred_len)

            extra_info = {
                "series_id": str(train_dataset.ids[idx]) if hasattr(train_dataset, "ids") else str(idx),
                "seasonal_pattern": seasonal_pattern,
                "seq_len": seq_len,
                "pred_len": pred_len,
            }

            if include_extra_qa:
                # 可选：与 GSM8K 风格对齐；对本任务并非必需
                extra_info["question"] = content
                extra_info["answer"] = None

            record = build_record(
                data_source=data_source,
                ability=ability,
                content=content,
                ground_truth=future_values,
                extra_info=extra_info,
                system_prompt=instructions.system_train if include_system else None,
            )

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Build VERL-style RL dataset for M4")
    parser.add_argument("--root_path", type=str, default="tsrl/m4", help="M4 数据路径（包含 training.npz/test.npz/M4-info.csv）")
    parser.add_argument("--output_dir", type=str, default="tsrl/rl_data", help="输出目录")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="Yearly,Quarterly,Monthly,Weekly,Daily,Hourly 或 all")
    parser.add_argument("--data_source", type=str, default="m4", help="data_source 字段")
    parser.add_argument("--ability", type=str, default="time_series_forecast", help="ability 字段")
    parser.add_argument("--include_extra_qa", action="store_true", help="extra_info 中附带 question/answer（可选）")
    parser.add_argument("--use_test_horizon", action="store_true", help="使用测试集 horizon 作为 ground_truth（严格等长）")
    parser.add_argument("--no_system", action="store_true", help="不写入 system prompt（默认写入 instructions.system_train）")

    args = parser.parse_args()

    if args.seasonal_patterns.lower() == "all":
        patterns = M4Meta.seasonal_patterns
    else:
        patterns = [p.strip() for p in args.seasonal_patterns.split(",") if p.strip()]

    ensure_dir(args.output_dir)

    outputs = []
    for sp in patterns:
        out_file = build_for_pattern(
            root_path=args.root_path,
            seasonal_pattern=sp,
            out_dir=args.output_dir,
            data_source=args.data_source,
            ability=args.ability,
            include_extra_qa=args.include_extra_qa,
            use_test_horizon=args.use_test_horizon,
            include_system=not args.no_system,
        )
        outputs.append(out_file)

    print("Built files:")
    for p in outputs:
        print("  ", p)


if __name__ == "__main__":
    main()


