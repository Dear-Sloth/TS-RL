#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert VERL-style JSON/JSONL datasets into train/test Parquet files.

Example:
    python tsrl/jsonl_to_parquet.py ^
        --input tsrl/rl_data/_Weekly/train.jsonl ^
        --train-output tsrl/rl_data/_Weekly/train.parquet ^
        --test-output tsrl/rl_data/_Weekly/test.parquet ^
        --test-ratio 0.1
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, load_dataset


def _load_dataset(path: Path) -> Dataset:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path))["train"]
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path))["train"]
    raise ValueError(f"Unsupported input format: {path}")


def _export(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    print(f"Saved {len(dataset)} records -> {output_path}")


def _ground_truth_lengths(dataset: Dataset) -> list[int]:
    reward_column = dataset["reward_model"]
    lengths: list[int] = []
    for item in reward_column:
        gt = []
        if isinstance(item, dict):
            gt = item.get("ground_truth", [])
        if not isinstance(gt, list):
            gt = []
        lengths.append(len(gt))
    return lengths


def _filter_by_gt_length(dataset: Dataset, expected_length: Optional[int]) -> Tuple[Dataset, Optional[int], int]:
    lengths = _ground_truth_lengths(dataset)
    if not lengths:
        return dataset, expected_length, 0

    inferred_length = expected_length
    if inferred_length is None:
        counter = Counter(lengths)
        inferred_length, _ = counter.most_common(1)[0]
        print(f"[jsonl_to_parquet] Auto-detected ground-truth length {inferred_length} (counts: {dict(counter)})")

    keep_indices = [idx for idx, length in enumerate(lengths) if length == inferred_length]
    removed = len(dataset) - len(keep_indices)
    if removed:
        dataset = dataset.select(keep_indices)
        print(f"[jsonl_to_parquet] Removed {removed} samples with ground-truth length != {inferred_length}")

    return dataset, inferred_length, removed


def convert(
    input_path: Path,
    train_output: Path,
    test_output: Path,
    test_ratio: float,
    seed: int,
    shuffle: bool,
    test_input: Optional[Path] = None,
    expected_gt_len: Optional[int] = None,
    disable_length_filter: bool = False,
) -> None:
    base_dataset = _load_dataset(input_path)
    if shuffle:
        base_dataset = base_dataset.shuffle(seed=seed)

    gt_len = expected_gt_len
    total_removed = 0
    if not disable_length_filter:
        base_dataset, gt_len, removed = _filter_by_gt_length(base_dataset, expected_gt_len)
        total_removed += removed

    if test_input is not None:
        test_dataset = _load_dataset(test_input)
        train_dataset = base_dataset
        if shuffle:
            test_dataset = test_dataset.shuffle(seed=seed)
        if not disable_length_filter:
            test_dataset, _, removed = _filter_by_gt_length(test_dataset, gt_len)
            total_removed += removed
    else:
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("--test-ratio must be between 0 and 1 when no --test-input is provided.")
        split = base_dataset.train_test_split(test_size=test_ratio, seed=seed)
        train_dataset = split["train"]
        test_dataset = split["test"]
        if not disable_length_filter:
            train_dataset, gt_len, removed_train = _filter_by_gt_length(train_dataset, gt_len)
            test_dataset, _, removed_test = _filter_by_gt_length(test_dataset, gt_len)
            total_removed += removed_train + removed_test

    if not disable_length_filter:
        print(f"[jsonl_to_parquet] Total removed samples: {total_removed}")

    _export(train_dataset, train_output)
    _export(test_dataset, test_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON/JSONL data into train/test Parquet files.")
    parser.add_argument("--input", type=Path, required=True, help="Source JSON/JSONL dataset.")
    parser.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Optional separate dataset for the test split.",
    )
    parser.add_argument("--train-output", type=Path, required=True, help="Output Parquet file for training split.")
    parser.add_argument("--test-output", type=Path, required=True, help="Output Parquet file for test split.")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction reserved for the test split if --test-input is not supplied. Default: 0.1",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/splitting.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling prior to splitting (shuffle is enabled by default).",
    )
    parser.add_argument(
        "--expected-gt-len",
        type=int,
        default=None,
        help="Expected length of reward_model.ground_truth. Defaults to the most common length.",
    )
    parser.add_argument(
        "--disable-length-filter",
        action="store_true",
        help="Skip removal of samples whose ground-truth length differs from the majority.",
    )

    args = parser.parse_args()

    convert(
        input_path=args.input,
        test_input=args.test_input,
        train_output=args.train_output,
        test_output=args.test_output,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        expected_gt_len=args.expected_gt_len,
        disable_length_filter=args.disable_length_filter,
    )


if __name__ == "__main__":
    main()
