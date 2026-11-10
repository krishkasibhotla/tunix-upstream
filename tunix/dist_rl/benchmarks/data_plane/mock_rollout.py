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

"""Mock RolloutWorker builder and node entrypoint for data_plane_benchmark."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import importlib
import logging
from typing import cast

import jax
from tunix.dist_rl.benchmarks.data_plane import data_plane_benchmark
from tunix.dist_rl.common import test_utils
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import rollout as rollout_node
from tunix.dist_rl.worker import rollout_worker
from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib

LOCAL_ROLLOUT_WORKER_ID = "rollout-0"


def _resolve_response_tokens_per_turn(
    model_kwargs: Mapping[str, object], max_resp: int
) -> int:
  """Resolves per-turn mock generation length.

  Defaults to `max_resp - 1` so single-turn runs (`max_turns == 1`) leave
  budget for `env.step()` instead of clipping at `MAX_CONTEXT_LIMIT_REACHED`.
  Multi-turn runs (`max_turns >= 2`) must set
  `model_config.kwargs.mock_response_tokens_per_turn` explicitly so the
  cumulative assistant + env tokens across turns stay below `max_resp`.

  Args:
    model_kwargs: `rollout_model_config.kwargs` mapping.
    max_resp: `rollout_config.total_generation_steps` response budget.

  Returns:
    Number of tokens `MockBaseSamplerImpl` emits per turn.
  """
  if "mock_response_tokens_per_turn" in model_kwargs:
    return data_plane_benchmark.kwarg_int(
        model_kwargs, "mock_response_tokens_per_turn", max_resp
    )
  if max_resp < 2:
    raise ValueError(
        "rollout_config.total_generation_steps must be >= 2 when"
        " mock_response_tokens_per_turn is unset so trajectories leave"
        f" budget for env.step(), got {max_resp}."
    )
  return max_resp - 1


def build_mock_rollout_worker(
    hp: resolve_config.DistHyperParameters,
    worker_id_str: str,
    tokenizer: node_utils.HfTokenizer,
) -> rollout_worker.RolloutWorker:
  """Constructs a `RolloutWorker` backed by `MockBaseSamplerImpl`."""
  rollout_node.validate_rollout_config(hp)
  grpo_cfg = hp.config_mapping("agentic_grpo_config")
  model_kwargs = cast(
      Mapping[str, object], hp.config_mapping("rollout_model_config")["kwargs"]
  )
  worker_rollout_cfg = rollout_node.build_rollout_config(
      hp, tokenizer, sampler_type="vanilla"
  )
  tok_adapter = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  shape_spec, num_experts, _ = data_plane_benchmark.resolve_routed_experts_spec(
      hp.config_mapping("vllm_config"), model_kwargs
  )
  tokens_per_turn = _resolve_response_tokens_per_turn(
      model_kwargs, worker_rollout_cfg.max_tokens_to_generate
  )
  decode_sleep_s = data_plane_benchmark.kwarg_float(
      model_kwargs, "mock_decode_sleep_s", 0.05
  )
  eos_id = (
      worker_rollout_cfg.eos_tokens[0]
      if worker_rollout_cfg.eos_tokens
      else None
  )

  sampler = test_utils.MockBaseSamplerImpl(
      sampler_name=worker_id_str,
      default_delay=decode_sleep_s,
      server_id=worker_id_str,
      config=worker_rollout_cfg,
      response_tokens_per_turn=tokens_per_turn,
      routed_experts_shape_spec=shape_spec,
      num_experts=num_experts,
      eos_token_id=eos_id,
      tokenizer=tok_adapter,
  )
  return rollout_worker.RolloutWorker(
      worker_id=worker_id_str,
      config=worker_rollout_cfg,
      sampler=sampler,
      tokenizer=tok_adapter,
      chat_parser=rollout_node.create_chat_parser(hp, tokenizer),
      env_name=str(hp.config["env_class_path"]),
      agent_name=str(hp.config["agent_class_path"]),
      env_config=hp.config_mapping("env_kwargs"),
      agent_config=hp.config_mapping("agent_kwargs"),
      max_concurrency=int(grpo_cfg["max_concurrency"]),
  )


def rollout_main(
    argv: list[str], context: context_lib.ProcessContext | None = None
) -> None:
  """Serves a mock `RolloutWorker` over gRPC + Bulk TCP until stopped."""
  jax.config.update("jax_platforms", "cpu")
  process_context = node_utils.require_discovery_context(context)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [MockRolloutNode] %(message)s",
      force=True,
  )
  hp = resolve_config.initialize(argv)
  node_utils.apply_library_env(hp, "rollout")
  worker_id_str = node_utils.worker_id(hp, "rollout", LOCAL_ROLLOUT_WORKER_ID)
  port = node_utils.service_port(hp, "rollout")
  node_utils.setup_otel_metrics(hp, "rollout", worker_id_str)
  node_utils.ensure_repo_root_on_path()
  registry_mod = str(hp.config_mapping("cluster_config")["registry_module"])
  if registry_mod:
    importlib.import_module(registry_mod)
  config_sha256 = node_utils.log_resolved_config(hp, "rollout")
  tokenizer = node_utils.load_hf_tokenizer(hp)

  async def _serve() -> None:
    worker_service = build_mock_rollout_worker(hp, worker_id_str, tokenizer)
    rss_mib, hwm_mib = data_plane_benchmark.read_process_memory_mib()
    logging.info(
        "[DataPlaneBench][Rollout %s] Initialized VmRSS=%.1f MiB"
        " VmHWM=%.1f MiB",
        worker_id_str,
        rss_mib,
        hwm_mib,
    )

    async def _drain() -> None:
      end_rss_mib, end_hwm_mib = data_plane_benchmark.read_process_memory_mib()
      logging.info(
          "[DataPlaneBench][Rollout %s] Draining VmRSS=%.1f MiB VmHWM=%.1f MiB",
          worker_id_str,
          end_rss_mib,
          end_hwm_mib,
      )
      try:
        worker_service.stop()
      finally:
        await worker_service.sampler.stop()

    await node_utils.serve_worker(
        worker_service,
        process_context=process_context,
        role="rollout",
        worker_id_str=worker_id_str,
        port=port,
        config_sha256=config_sha256,
        extra_metadata={
            "trajectory_store_config": (
                worker_service.trajectory_store.to_config()
                if worker_service.trajectory_store is not None
                else None
            ),
        },
        drain_fn=_drain,
    )

  asyncio.run(_serve())
