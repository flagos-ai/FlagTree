"""Reproducible AMD benchmark for the TLE TopK tutorial."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch

import triton

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPK_TUTORIAL_PATH = REPO_ROOT / "python" / "tutorials" / "tle" / "03-topk.py"

# --- Self-contained benchmark helpers (timing quantiles, round summaries,
# bootstrap CI, and environment capture; previously imported from benchmark_amd). ---
QUANTILES = (0.5, 0.2, 0.8)
BENCH_WARMUP_MS = 25
BENCH_REP_MS = 100
RANDOM_SEED = 0
BOOTSTRAP_SAMPLES = 10_000


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(REPO_ROOT), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _environment() -> dict[str, object]:
    target = triton.runtime.driver.active.get_current_target()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "triton_version": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "backend": target.backend,
        "arch": target.arch,
        "warp_size": target.warp_size,
    }


def _summarize_rounds(rounds: list[dict[str, float]]) -> dict[str, object]:
    p50_values = [round_result["p50_ms"] for round_result in rounds]
    p50_median = statistics.median(p50_values)
    p50_mean = statistics.mean(p50_values)
    coefficient_of_variation = (statistics.pstdev(p50_values) / p50_mean if len(p50_values) > 1 else 0.0)
    return {
        "rounds": rounds,
        "p50_ms_median": p50_median,
        "p50_us_median": p50_median * 1_000.0,
        "p50_coefficient_of_variation": coefficient_of_variation,
    }


def _bootstrap_speedup_ci(
    triton_rounds: list[dict[str, float]],
    tle_rounds: list[dict[str, float]],
) -> list[float]:
    rng = random.Random(RANDOM_SEED)
    sample_count = len(triton_rounds)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(sample_count) for _ in range(sample_count)]
        triton_median = statistics.median(triton_rounds[index]["p50_ms"] for index in indices)
        tle_median = statistics.median(tle_rounds[index]["p50_ms"] for index in indices)
        estimates.append(triton_median / tle_median)

    estimates.sort()
    lower_index = math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1))
    upper_index = math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1))
    return [estimates[lower_index], estimates[upper_index]]


BALANCED_PROVIDER_ORDERS = (
    ("radix", "triton", "torch"),
    ("triton", "torch", "radix"),
    ("torch", "radix", "triton"),
    ("torch", "triton", "radix"),
    ("triton", "radix", "torch"),
    ("radix", "torch", "triton"),
)
SHAPES = {
    "short-small-k": {"m": 64, "n": 128, "k": 8, "row_class": "short", "k_class": "small"},
    "short-medium-k": {"m": 64, "n": 1024, "k": 32, "row_class": "short", "k_class": "medium"},
    "long-medium-k": {"m": 64, "n": 8192, "k": 128, "row_class": "long", "k_class": "medium"},
    "long-large-k": {"m": 128, "n": 32768, "k": 256, "row_class": "long", "k_class": "large"},
}


def _load_topk_tutorial() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tle_topk_tutorial", TOPK_TUTORIAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TopK tutorial from {TOPK_TUTORIAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOPK = _load_topk_tutorial()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _provider_configs(m: int, n: int, k: int) -> dict[str, dict[str, object]]:
    radix_block_n, radix_bits, radix_num_warps = TOPK._radix_launch_config(n)
    triton_block_n, triton_num_warps = TOPK._triton_launch_config(m, n, k)
    return {
        "radix": {
            "algorithm": "tle_shared_memory_radix_select",
            "block_n": radix_block_n,
            "radix_bits": radix_bits,
            "num_warps": radix_num_warps,
            "num_stages": 1,
        },
        "triton": {
            "algorithm": "triton_streaming_topk",
            "block_n": triton_block_n,
            "num_warps": triton_num_warps,
            "num_stages": 1,
        },
        "torch": {"algorithm": "torch.topk", "sorted": False},
    }


def _comparison(
    baseline_rounds: list[dict[str, float]],
    candidate_rounds: list[dict[str, float]],
    baseline_summary: dict[str, object],
    candidate_summary: dict[str, object],
) -> dict[str, object]:
    speedup = baseline_summary["p50_ms_median"] / candidate_summary["p50_ms_median"]
    return {
        "speedup": speedup,
        "speedup_95_ci": _bootstrap_speedup_ci(baseline_rounds, candidate_rounds),
    }


def _measure_providers(
    launches: dict[str, Callable[[], None]],
    rounds: int,
    stabilization_rounds: int,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    samples: dict[str, list[dict[str, float]]] = {provider: [] for provider in launches}
    stabilization_provider_order: list[list[str]] = []
    provider_order_by_round: list[list[str]] = []

    for launch in launches.values():
        launch()
    torch.cuda.synchronize()

    for round_index in range(stabilization_rounds):
        order = list(BALANCED_PROVIDER_ORDERS[round_index % len(BALANCED_PROVIDER_ORDERS)])
        stabilization_provider_order.append(order)
        for provider in order:
            triton.testing.do_bench(
                launches[provider],
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=QUANTILES,
            )

    for round_index in range(rounds):
        order_index = stabilization_rounds + round_index
        order = list(BALANCED_PROVIDER_ORDERS[order_index % len(BALANCED_PROVIDER_ORDERS)])
        provider_order_by_round.append(order)
        for provider in order:
            p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
                launches[provider],
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=QUANTILES,
            )
            samples[provider].append({
                "p20_ms": float(p20_ms),
                "p50_ms": float(p50_ms),
                "p80_ms": float(p80_ms),
            })

    summaries = {provider: _summarize_rounds(provider_rounds) for provider, provider_rounds in samples.items()}
    return {
        "providers": summaries,
        "stabilization_provider_order": stabilization_provider_order,
        "provider_order_by_round": provider_order_by_round,
        "radix_vs_triton": _comparison(
            samples["triton"],
            samples["radix"],
            summaries["triton"],
            summaries["radix"],
        ),
        "radix_vs_torch": _comparison(
            samples["torch"],
            samples["radix"],
            summaries["torch"],
            summaries["radix"],
        ),
        "triton_vs_torch": _comparison(
            samples["torch"],
            samples["triton"],
            summaries["torch"],
            summaries["triton"],
        ),
    }


def _run_shape(
    shape_name: str,
    shape: dict[str, object],
    dtype_name: str,
    rounds: int,
    stabilization_rounds: int,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    m = int(shape["m"])
    n = int(shape["n"])
    k = int(shape["k"])
    dtype = _torch_dtype(dtype_name)
    device = triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand((m, n), device=device, dtype=dtype)

    radix_values = torch.empty((m, k), device=device, dtype=dtype)
    radix_indices = torch.empty((m, k), device=device, dtype=torch.int32)
    triton_values = torch.empty_like(radix_values)
    triton_indices = torch.empty_like(radix_indices)
    torch_values = torch.empty_like(radix_values)
    torch_indices = torch.empty((m, k), device=device, dtype=torch.int64)

    launches = {
        "radix": lambda: TOPK.triton_radix_topk(
            x,
            k,
            out_vals=radix_values,
            out_idx=radix_indices,
        ),
        "triton": lambda: TOPK.triton_topk(
            x,
            k,
            out_vals=triton_values,
            out_idx=triton_indices,
        ),
        "torch": lambda: torch.topk(
            x,
            k,
            dim=1,
            sorted=False,
            out=(torch_values, torch_indices),
        ),
    }

    for launch in launches.values():
        launch()
    torch.cuda.synchronize()
    expected_values = torch.sort(torch_values, dim=1, descending=True).values
    for values, indices in (
        (radix_values, radix_indices),
        (triton_values, triton_indices),
    ):
        actual_values = torch.sort(values, dim=1, descending=True).values
        torch.testing.assert_close(actual_values, expected_values, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(x.gather(1, indices.to(torch.int64)), values, rtol=1e-3, atol=1e-3)

    selected_provider = TOPK._topk_provider(n, k)
    measurements = _measure_providers(
        launches,
        rounds,
        stabilization_rounds,
        warmup_ms,
        rep_ms,
    )
    if selected_provider == "radix":
        selected_vs_triton = dict(measurements["radix_vs_triton"])
        selected_vs_torch = dict(measurements["radix_vs_torch"])
    else:
        selected_vs_triton = {"speedup": 1.0, "speedup_95_ci": [1.0, 1.0]}
        selected_vs_torch = dict(measurements["triton_vs_torch"])
    measurements["selected_vs_triton"] = selected_vs_triton
    measurements["selected_vs_torch"] = selected_vs_torch

    return {
        "case": "topk",
        "shape_name": shape_name,
        "selected_provider": selected_provider,
        "parameters": {
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype_name,
            "row_class": shape["row_class"],
            "k_class": shape["k_class"],
            "provider_configs": _provider_configs(m, n, k),
        },
        "correct": True,
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("all", *SHAPES), default="all")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--stabilization-rounds", type=int, default=1)
    parser.add_argument("--warmup-ms", type=int, default=BENCH_WARMUP_MS)
    parser.add_argument("--rep-ms", type=int, default=BENCH_REP_MS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.stabilization_rounds < 0:
        parser.error("--stabilization-rounds cannot be negative")
    if args.warmup_ms < 1 or args.rep_ms < 1:
        parser.error("--warmup-ms and --rep-ms must be at least 1")

    torch.manual_seed(RANDOM_SEED)
    selected_shapes = SHAPES.items() if args.shape == "all" else ((args.shape, SHAPES[args.shape]), )
    results = [
        _run_shape(
            shape_name,
            shape,
            args.dtype,
            args.rounds,
            args.stabilization_rounds,
            args.warmup_ms,
            args.rep_ms,
        ) for shape_name, shape in selected_shapes
    ]
    report = {
        "schema_version": 2,
        "environment": _environment(),
        "measurement_config": {
            "rounds": args.rounds,
            "stabilization_rounds_discarded": args.stabilization_rounds,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "quantiles": list(QUANTILES),
            "provider_order": "balanced cycle over all six provider permutations",
            "random_seed": RANDOM_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "confidence_level": 0.95,
            "confidence_method": "paired bootstrap of ratio of p50 medians",
            "output_allocation": "preallocated for all providers",
            "cache_policy": "triton.testing.do_bench clears L2 before every timed sample",
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
