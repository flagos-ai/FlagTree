#!/usr/bin/env python3
# Copyright 2026, The FlagOS Contributors.
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

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from triton.experimental.tle.raw.nvshmem.utils import init_torch_distributed

CODE_DIR = Path(__file__).parent.resolve()
DEFAULT_TOKENS = [1, 16, 64, 128, 256, 1024, 4096]
WARMUP = 200
MEASURE_ITERS = 1000
FRESH_REPS = 5
ABS_TOL = 1.25e-1
REL_TOL = 4.0


def _load_module():
    path = CODE_DIR / "allreduce.py"
    spec = importlib.util.spec_from_file_location("_tle_ar_rmsnorm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
Operator = MODULE.FusedAllReduceResidualRMSNorm
HIDDEN_SIZE = MODULE.HIDDEN_SIZE

try:
    import flashinfer.comm as flashinfer_comm  # type: ignore

    if not hasattr(flashinfer_comm, "trtllm_allreduce_fusion"):
        flashinfer_comm = None
except Exception:
    flashinfer_comm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NVSHMEM fused AllReduce + Residual + RMSNorm")
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def _timed_loop(fn, iters: int, device: torch.device) -> list[float]:
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _gather_samples(local: list[float], group) -> list[list[float]]:
    gathered: list[list[float] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    return [entry or [] for entry in gathered]


def _max_across_ranks(per_rank: list[list[float]]) -> list[float]:
    if not per_rank:
        return []
    length = len(per_rank[0])
    return [max(rank_values[i] for rank_values in per_rank) for i in range(length)]


def _summarize_repetitions(repetition_samples: list[list[float]]) -> dict[str, float]:
    pooled = [sample for rep in repetition_samples for sample in rep]
    rep_p50s = [_percentile(rep, 0.5) for rep in repetition_samples if rep]
    return {
        "p50": _percentile(pooled, 0.5),
        "p90": _percentile(pooled, 0.9),
        "p99": _percentile(pooled, 0.99),
        "repeat_p50_mean": statistics.mean(rep_p50s) if rep_p50s else float("nan"),
        "repeat_p50_std": statistics.pstdev(rep_p50s) if len(rep_p50s) > 1 else 0.0,
    }


def _seeded_batch(device: torch.device, tokens: int, rank: int, rep: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260904 + tokens * 1009 + rank * 10007 + rep * 97)
    x = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=generator)
    residual = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=generator)
    weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=generator)
    return x, residual, weight


class FlashInferBaseline:
    def __init__(self, op: Operator, group) -> None:
        self.available = False
        self.reason = "flashinfer.comm is unavailable"
        self.group = group
        self.rank = op.rank
        self.world_size = op.world_size
        self.device = op.device
        self._ipc_handles: Any = None
        self.workspace = None
        self.input = None
        self.residual_out = None
        self.norm_out = None

        if flashinfer_comm is None:
            return

        create_fn = getattr(
            flashinfer_comm,
            "trtllm_create_ipc_workspace_for_all_reduce_fusion",
            None,
        )
        if create_fn is None:
            self.reason = "flashinfer.comm missing trtllm_create_ipc_workspace_for_all_reduce_fusion"
            return

        kwargs = dict(
            tp_rank=self.rank,
            tp_size=self.world_size,
            max_token_num=op.max_tokens,
            hidden_dim=HIDDEN_SIZE,
            group=group,
            use_fp32_lamport=False,
        )
        try:
            self._ipc_handles, self.workspace = create_fn(**kwargs)
        except TypeError:
            kwargs.pop("group")
            self._ipc_handles, self.workspace = create_fn(**kwargs)
        except Exception as exc:
            self.reason = f"flashinfer workspace init failed: {exc}"
            return

        self.input = torch.empty((op.max_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=self.device)
        self.residual_out = torch.empty_like(self.input)
        self.norm_out = torch.empty_like(self.input)
        self.available = True
        self.reason = ""

    def close(self) -> None:
        if not self.available or self._ipc_handles is None:
            return
        destroy_fn = getattr(
            flashinfer_comm,
            "trtllm_destroy_ipc_workspace_for_all_reduce",
            None,
        )
        if destroy_fn is None:
            return
        destroy_fn(self._ipc_handles, self.group)

    def prepare_input(self, x: torch.Tensor) -> None:
        if not self.available:
            raise RuntimeError(self.reason)
        self.input[: x.shape[0]].copy_(x, non_blocking=True)

    def run_prepared(self, tokens: int, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        if not self.available:
            raise RuntimeError(self.reason)
        kwargs = dict(
            allreduce_in=self.input[:tokens],
            token_num=tokens,
            residual_in=residual,
            residual_out=self.residual_out[:tokens],
            norm_out=self.norm_out[:tokens],
            rms_gamma=weight,
            rms_eps=float(eps),
            hidden_dim=HIDDEN_SIZE,
            workspace_ptrs=self.workspace,
            pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
            allreduce_out=None,
            quant_out=None,
            scale_out=None,
            layout_code=None,
            scale_factor=None,
            use_oneshot=True,
            world_rank=self.rank,
            world_size=self.world_size,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            fp32_acc=True,
        )
        flashinfer_comm.trtllm_allreduce_fusion(**kwargs)

    def stage_inclusive_step(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        self.input[: x.shape[0]].copy_(x, non_blocking=True)
        self.run_prepared(x.shape[0], residual, weight, eps)


def _check_correctness(
    op: Operator,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[float, float]:
    reduced, norm = op.forward(x, residual, weight, eps=eps)
    dist.barrier(group=op.group)
    ref_reduced, ref_norm = op.reference(x, residual, weight, eps=eps)
    abs_reduced = (reduced - ref_reduced).abs().max().item()
    abs_norm = (norm - ref_norm).abs().max().item()
    rel_reduced = (
        (reduced - ref_reduced).abs()
        / torch.maximum(
            ref_reduced.abs(),
            torch.tensor(1e-2, device=op.device, dtype=ref_reduced.dtype),
        )
    ).max().item()
    rel_norm = (
        (norm - ref_norm).abs()
        / torch.maximum(
            ref_norm.abs(),
            torch.tensor(1e-2, device=op.device, dtype=ref_norm.dtype),
        )
    ).max().item()
    return max(abs_reduced, abs_norm), max(rel_reduced, rel_norm)


def _benchmark_variant(
    op: Operator,
    batch_factory,
    warmup_fn,
    timed_fn,
    tokens: int,
    eps: float,
    warmup: int,
    iters: int,
    reps: int,
) -> dict[str, float]:
    repetition_samples: list[list[float]] = []
    for rep in range(reps):
        x, residual, weight = batch_factory(rep)
        for _ in range(warmup):
            warmup_fn(x, residual, weight, eps)
        torch.cuda.synchronize(op.device)
        dist.barrier(group=op.group)
        local = _timed_loop(lambda: timed_fn(x, residual, weight, eps), iters, op.device)
        max_series = _max_across_ranks(_gather_samples(local, op.group))
        repetition_samples.append(max_series)
        dist.barrier(group=op.group)
    return _summarize_repetitions(repetition_samples)


def _benchmark_tokens(op: Operator, tokens: int, baseline: FlashInferBaseline, rank: int) -> dict[str, Any]:
    candidate = {
        "prepared": {"status": "skipped"},
        "stage_inclusive": {"status": "skipped"},
    }
    flashinfer = {
        "prepared": {"status": "unavailable", "reason": baseline.reason},
        "stage_inclusive": {"status": "unavailable", "reason": baseline.reason},
    }

    def batch_factory(rep: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _seeded_batch(op.device, tokens, rank, rep)

    for rep in range(FRESH_REPS):
        x, residual, weight = batch_factory(rep)
        max_abs, max_rel = _check_correctness(op, x, residual, weight, 1.0e-6)
        if rank == 0:
            status = "PASS" if max_abs <= ABS_TOL and max_rel <= REL_TOL else "FAIL"
            print(
                f"tokens={tokens} max_abs={max_abs:.6f} max_rel={max_rel:.6f} status={status}",
                flush=True,
            )
        if max_abs > ABS_TOL:
            raise RuntimeError(f"correctness failed for tokens={tokens}")

    def candidate_prepare(x, residual, weight, eps):
        op.prepare_input(x)

    def candidate_prepared(_, residual, weight, eps):
        op.run_prepared(tokens, residual, weight, eps=eps)

    def candidate_stage(x, residual, weight, eps):
        op.forward(x, residual, weight, eps=eps)

    candidate["prepared"] = _benchmark_variant(
        op,
        batch_factory,
        candidate_prepare,
        candidate_prepared,
        tokens,
        1.0e-6,
        WARMUP,
        MEASURE_ITERS,
        FRESH_REPS,
    )
    candidate["stage_inclusive"] = _benchmark_variant(
        op,
        batch_factory,
        candidate_stage,
        candidate_stage,
        tokens,
        1.0e-6,
        WARMUP,
        MEASURE_ITERS,
        FRESH_REPS,
    )

    if baseline.available:
        def baseline_prepare(x, residual, weight, eps):
            baseline.prepare_input(x)

        def baseline_prepared(_, residual, weight, eps):
            baseline.run_prepared(tokens, residual, weight, eps)

        def baseline_stage(x, residual, weight, eps):
            baseline.stage_inclusive_step(x, residual, weight, eps)

        flashinfer["prepared"] = _benchmark_variant(
            op,
            batch_factory,
            baseline_prepare,
            baseline_prepared,
            tokens,
            1.0e-6,
            WARMUP,
            MEASURE_ITERS,
            FRESH_REPS,
        )
        flashinfer["stage_inclusive"] = _benchmark_variant(
            op,
            batch_factory,
            baseline_stage,
            baseline_stage,
            tokens,
            1.0e-6,
            WARMUP,
            MEASURE_ITERS,
            FRESH_REPS,
        )

    return {"candidate": candidate, "flashinfer": flashinfer}


def main() -> None:
    args = parse_args()
    group = init_torch_distributed()
    rank = dist.get_rank(group)

    op = Operator(max(args.tokens), group=group)
    baseline = FlashInferBaseline(op, group)
    try:
        results: dict[str, Any] = {
            "world_size": op.world_size,
            "hidden_size": HIDDEN_SIZE,
            "warmup": WARMUP,
            "measurement_iterations": MEASURE_ITERS,
            "fresh_repetitions": FRESH_REPS,
            "timing_definition": "max latency across ranks, wall-clock milliseconds per iteration",
            "results": {},
        }
        for tokens in args.tokens:
            if rank == 0:
                print(f"benchmarking tokens={tokens}", flush=True)
            results["results"][str(tokens)] = _benchmark_tokens(op, tokens, baseline, rank)
        if rank == 0:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Results saved to {args.output_json}", flush=True)
    finally:
        baseline.close()
        op.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
