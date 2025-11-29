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

from __future__ import annotations

from verl.utils.reward_score.time_series_rule import compute_score as time_series_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager


@register("time_series_rule")
class TimeSeriesRuleRewardManager(DAPORewardManager):
    """DAPO-style reward manager with a default time-series rule reward."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key: str = "data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score or time_series_compute_score,
            reward_fn_key=reward_fn_key,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=overlong_buffer_cfg,
        )


__all__ = ["TimeSeriesRuleRewardManager"]

