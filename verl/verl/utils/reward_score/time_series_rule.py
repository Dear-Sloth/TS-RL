# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rule-based reward helpers for time-series forecasting tasks.

The default implementation extracts predictions from the last `<future>...</future>`
block and scores them against the provided ground truth using a SMAPE-style metric:

    score = 1 - mean(|pred - gt| / (|pred| + |gt|))

This file is intended to be customized. Feel free to modify `compute_score` to plug
in your own evaluation logic or add additional helper utilities.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence


FUTURE_PATTERN = re.compile(r"<future>(.*?)</future>", flags=re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

# Default n-gram repetition penalty settings (always enabled by default).
# Set REPETITION_PENALTY_COEF to 0.0 to disable.
REPETITION_NGRAM_N: int = 3
REPETITION_PENALTY_COEF: float = 1.0


def _ensure_numeric_sequence(values: Iterable) -> list[float]:
    """Convert raw ground-truth values to a flat list of floats."""
    if isinstance(values, (str, bytes)):
        extracted = NUMBER_PATTERN.findall(values.decode() if isinstance(values, bytes) else values)
        return [float(v) for v in extracted]

    normalized: list[float] = []
    for item in values:
        if isinstance(item, (int, float)):
            normalized.append(float(item))
        elif isinstance(item, str):
            extracted = NUMBER_PATTERN.findall(item)
            normalized.extend(float(v) for v in extracted)
    return normalized


def _extract_future_block(solution_str: str) -> list[float]:
    """Extract numeric predictions from the `<future>` block in the solution."""
    match = FUTURE_PATTERN.search(solution_str)
    if not match:
        return []

    payload = match.group(1)
    numbers = NUMBER_PATTERN.findall(payload)
    return [float(num) for num in numbers]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_score(
    solution_str: str,
    ground_truth: Sequence[float] | dict,
    extra_info: dict | None = None,
) -> float:
    """Compute a rule-based reward for time-series predictions.

    Returns a single float. Applies an n-gram repetition penalty over the
    entire decoded response text using module-level defaults
    `REPETITION_NGRAM_N` and `REPETITION_PENALTY_COEF`.

    Set `REPETITION_PENALTY_COEF = 0.0` to disable the penalty.
    """
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("values", [])

    target = _ensure_numeric_sequence(ground_truth)
    prediction = _extract_future_block(solution_str)

    # Compute base SMAPE score; if prediction is invalid, return 0.0
    if not prediction:
        return 0.0
    series_length = min(len(prediction), len(target))
    if series_length == 0:
        return 0.0
    if len(prediction) != len(target):
        return 0.0

    smape_terms: list[float] = []
    for pred, tgt in zip(prediction[:series_length], target[:series_length]):
        denom = abs(pred) + abs(tgt)
        if math.isclose(denom, 0.0):
            continue
        smape_terms.append(abs(pred - tgt) / denom)
    smape = _mean(smape_terms)
    base_score = float(1.0 - smape)

    # n-gram repetition penalty over the entire response text
    def _tokenize_for_repetition(s: str) -> list[str]:
        # Simple version: split by whitespace, keep all non-whitespace tokens (includes special markers)
        return re.findall(r"\S+", s.lower())

    def _ngram_repetition_rate_tokens(tokens: Sequence[str], n: int) -> float:
        if n <= 0 or len(tokens) < n:
            return 0.0
        total = len(tokens) - n + 1
        if total <= 0:
            return 0.0
        counts: dict[tuple[str, ...], int] = {}
        for i in range(total):
            gram = tuple(tokens[i : i + n])
            counts[gram] = counts.get(gram, 0) + 1
        repeats = sum(c - 1 for c in counts.values() if c > 1)
        return repeats / total if total > 0 else 0.0

    # Use module-level defaults
    rep_n = int(REPETITION_NGRAM_N)
    rep_coef = float(REPETITION_PENALTY_COEF)

    repetition_rate = 0.0
    repetition_penalty = 0.0
    if rep_n > 0 and rep_coef > 0:
        tokens = _tokenize_for_repetition(solution_str)
        if tokens:
            repetition_rate = _ngram_repetition_rate_tokens(tokens, rep_n)
            repetition_penalty = rep_coef * repetition_rate

    final_score = max(0.0, min(1.0, base_score - repetition_penalty))

    return final_score


__all__ = ["compute_score"]

