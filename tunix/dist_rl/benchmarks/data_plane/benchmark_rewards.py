# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrator-side RSS / HWM telemetry reward hook for data_plane_benchmark."""

from __future__ import annotations

from collections.abc import Sequence
import logging

from tunix.dist_rl.benchmarks.data_plane import data_plane_benchmark


def rss_telemetry_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    **kwargs: object,
) -> list[float]:
  """Zero-bonus reward hook logging live CPU Orchestrator RSS / HWM."""
  del prompts, kwargs
  rss_mib, hwm_mib = data_plane_benchmark.read_process_memory_mib()
  logging.info(
      "[DataPlaneBench][Orchestrator] VmRSS=%.1f MiB VmHWM=%.1f MiB"
      " batch_items=%d",
      rss_mib,
      hwm_mib,
      len(completions),
  )
  return [0.0] * len(completions)
