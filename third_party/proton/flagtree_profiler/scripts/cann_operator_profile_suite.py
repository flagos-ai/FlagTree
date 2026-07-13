#!/usr/bin/env python3
"""Profile Triton-developed operators with Proton CANN.

Default mode is a driver:

    python third_party/proton/flagtree_profiler/scripts/cann_operator_profile_suite.py \
      --out /tmp/proton_cann_triton_operator_suite --clean

It launches this file again under external msprof, runs a fixed set of
@triton.jit kernels on Ascend NPU tensors, imports exported CANN CSV files into
Proton artifacts, and writes summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import triton
import triton.language as tl
import triton.profiler as proton
from triton._C.libproton import proton as libproton


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def _vector_fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y + z, mask=mask)


@triton.jit
def _relu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    y = x + bias
    tl.store(out_ptr + offsets, tl.where(y > 0, y, 0.0), mask=mask)


@triton.jit
def _exp_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.exp(x), mask=mask)


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _cast_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float16), mask=mask)


@triton.jit
def _row_sum_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    y = tl.sum(x, axis=0)
    tl.store(out_ptr + row, y)


@triton.jit
def _row_max_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    y = tl.max(x, axis=0)
    tl.store(out_ptr + row, y)


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


@triton.jit
def _transpose_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    tl.store(out_ptr + offs_n[None, :] * M + offs_m[:, None], x, mask=mask)


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + k[:, None] * N + offs_n[None, :], mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=mask)


@triton.jit
def _masked_tail_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x - y, mask=mask)


@dataclass(frozen=True)
class TritonCase:
    name: str
    category: str
    description: str
    make: Callable


def _load_torch_npu():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for the Triton operator suite.") from exc

    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch_npu is required for NPU tensor allocation.") from exc

    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is unavailable after importing torch_npu.")
    if not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed, but torch.npu.is_available() is false.")
    return torch


def _grid_1d(n_elements: int, block_size: int):
    return (triton.cdiv(n_elements, block_size), )


def _triton_cases(torch, device) -> list[TritonCase]:

    def vector_add_fp32():
        n = 1_048_576
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float32)
        y = torch.randn((n, ), device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        return lambda: (_vector_add_kernel[_grid_1d(n, block)](x, y, out, n, BLOCK_SIZE=block), out)[1]

    def vector_add_fp16():
        n = 1_048_576
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float16)
        y = torch.randn((n, ), device=device, dtype=torch.float16)
        out = torch.empty_like(x)
        return lambda: (_vector_add_kernel[_grid_1d(n, block)](x, y, out, n, BLOCK_SIZE=block), out)[1]

    def vector_fma_fp32():
        n = 1_048_576
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float32)
        y = torch.randn((n, ), device=device, dtype=torch.float32)
        z = torch.randn((n, ), device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        return lambda: (_vector_fma_kernel[_grid_1d(n, block)](x, y, z, out, n, BLOCK_SIZE=block), out)[1]

    def relu_fp16():
        n = 1_048_576
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float16)
        bias = torch.randn((n, ), device=device, dtype=torch.float16)
        out = torch.empty_like(x)
        return lambda: (_relu_kernel[_grid_1d(n, block)](x, bias, out, n, BLOCK_SIZE=block), out)[1]

    def exp_fp32():
        n = 262_144
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        return lambda: (_exp_kernel[_grid_1d(n, block)](x, out, n, BLOCK_SIZE=block), out)[1]

    def copy_fp16():
        n = 2_097_152
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float16)
        out = torch.empty_like(x)
        return lambda: (_copy_kernel[_grid_1d(n, block)](x, out, n, BLOCK_SIZE=block), out)[1]

    def cast_fp32_to_fp16():
        n = 1_048_576
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float32)
        out = torch.empty((n, ), device=device, dtype=torch.float16)
        return lambda: (_cast_kernel[_grid_1d(n, block)](x, out, n, BLOCK_SIZE=block), out)[1]

    def row_sum_fp32():
        rows = 1024
        cols = 512
        block = triton.next_power_of_2(cols)
        x = torch.randn((rows, cols), device=device, dtype=torch.float32)
        out = torch.empty((rows, ), device=device, dtype=torch.float32)
        return lambda: (_row_sum_kernel[(rows, )](x, out, cols, BLOCK_SIZE=block), out)[1]

    def row_max_fp32():
        rows = 1024
        cols = 512
        block = triton.next_power_of_2(cols)
        x = torch.randn((rows, cols), device=device, dtype=torch.float32)
        out = torch.empty((rows, ), device=device, dtype=torch.float32)
        return lambda: (_row_max_kernel[(rows, )](x, out, cols, BLOCK_SIZE=block), out)[1]

    def softmax_fp32():
        rows = 1024
        cols = 512
        block = triton.next_power_of_2(cols)
        x = torch.randn((rows, cols), device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        return lambda: (_softmax_kernel[(rows, )](x, out, cols, BLOCK_SIZE=block), out)[1]

    def transpose_fp16():
        m = 512
        n = 512
        bm = 16
        bn = 16
        x = torch.randn((m, n), device=device, dtype=torch.float16)
        out = torch.empty((n, m), device=device, dtype=torch.float16)
        grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
        return lambda: (_transpose_kernel[grid](x, out, m, n, BLOCK_M=bm, BLOCK_N=bn), out)[1]

    def matmul_fp16():
        m = 256
        n = 256
        k = 256
        bm = 64
        bn = 64
        bk = 64
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        c = torch.empty((m, n), device=device, dtype=torch.float16)
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), )
        return lambda: (_matmul_kernel[grid](a, b, c, m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk), c)[1]

    def masked_tail_fp32():
        n = 1_000_003
        block = 1024
        x = torch.randn((n, ), device=device, dtype=torch.float32)
        y = torch.randn((n, ), device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        return lambda: (_masked_tail_kernel[_grid_1d(n, block)](x, y, out, n, BLOCK_SIZE=block), out)[1]

    return [
        TritonCase("triton_vector_add_fp32", "elementwise", "tl.load + add + tl.store, fp32", vector_add_fp32),
        TritonCase("triton_vector_fma_fp32", "elementwise", "tl.load + multiply-add + tl.store", vector_fma_fp32),
        TritonCase("triton_relu_fp16", "activation", "tl.where relu-style activation", relu_fp16),
        TritonCase("triton_exp_fp32", "math", "tl.exp vector kernel", exp_fp32),
        TritonCase("triton_copy_fp16", "memory", "contiguous load/store copy", copy_fp16),
        TritonCase("triton_cast_fp32_to_fp16", "cast", "fp32 load converted to fp16 store", cast_fp32_to_fp16),
        TritonCase("triton_row_sum_fp32", "reduction", "row-wise tl.sum", row_sum_fp32),
        TritonCase("triton_row_max_fp32", "reduction", "row-wise tl.max", row_max_fp32),
        TritonCase("triton_softmax_fp32", "normalization", "fused row-wise softmax", softmax_fp32),
        TritonCase("triton_transpose_fp16", "layout", "2D tiled transpose", transpose_fp16),
        TritonCase("triton_matmul_fp16", "dense_compute", "tl.dot matrix multiplication", matmul_fp16),
        TritonCase("triton_masked_tail_fp32", "masking", "non-power-of-two masked tail kernel", masked_tail_fp32),
    ]


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/proton_cann_triton_operator_suite")
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--operator",
        action="append",
        dest="operators",
        help="Triton case to run. May be repeated. Default: all cases.",
    )
    parser.add_argument("--aic-metrics", default="MemoryAccess")
    parser.add_argument("--sys-hardware-mem-freq", type=int, default=100)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--allow-op-failures",
        action="store_true",
        help="Continue and return success even if a Triton case fails.",
    )
    parser.add_argument("--workload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    parser.add_argument("--vendor-output", help=argparse.SUPPRESS)
    parser.add_argument("--external-msprof", action="store_true", help=argparse.SUPPRESS)
    return parser


def _checksum(out) -> float:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not hasattr(out, "float"):
        return 0.0
    return float(out.float().mean().cpu().item())


def _measure_op(torch, op: Callable, iters: int) -> tuple[float, object]:
    out = None
    start = time.perf_counter()
    for _ in range(iters):
        out = op()
    torch.npu.synchronize()
    return time.perf_counter() - start, out


def _timing_fields(baseline_elapsed_s: float, profiled_elapsed_s: float, iters: int) -> dict:
    overhead_s = profiled_elapsed_s - baseline_elapsed_s
    overhead_ratio = None
    overhead_percent = None
    if baseline_elapsed_s > 0:
        overhead_ratio = overhead_s / baseline_elapsed_s
        overhead_percent = overhead_ratio * 100.0
    return {
        "baseline_elapsed_s": baseline_elapsed_s,
        "profiled_elapsed_s": profiled_elapsed_s,
        "elapsed_s": profiled_elapsed_s,
        "overhead_s": overhead_s,
        "overhead_ratio": overhead_ratio,
        "overhead_percent": overhead_percent,
        "baseline_per_iter_s": baseline_elapsed_s / iters if iters else None,
        "profiled_per_iter_s": profiled_elapsed_s / iters if iters else None,
        "overhead_per_iter_s": overhead_s / iters if iters else None,
    }


def _summarize_timing(results: list[dict]) -> dict:
    timed = [
        result for result in results if result.get("status") == "ok" and result.get("baseline_elapsed_s") is not None
        and result.get("profiled_elapsed_s") is not None
    ]
    ratios = [result["overhead_ratio"] for result in timed if result.get("overhead_ratio") is not None]
    baseline_total = sum(result["baseline_elapsed_s"] for result in timed)
    profiled_total = sum(result["profiled_elapsed_s"] for result in timed)
    weighted_ratio = None
    if baseline_total > 0:
        weighted_ratio = (profiled_total - baseline_total) / baseline_total
    return {
        "timed_case_count": len(timed),
        "baseline_total_s": baseline_total,
        "profiled_total_s": profiled_total,
        "overhead_total_s": profiled_total - baseline_total,
        "average_overhead_ratio": sum(ratios) / len(ratios) if ratios else None,
        "average_overhead_percent": (sum(ratios) / len(ratios) * 100.0) if ratios else None,
        "weighted_overhead_ratio": weighted_ratio,
        "weighted_overhead_percent": weighted_ratio * 100.0 if weighted_ratio is not None else None,
    }


def _run_workload(args: argparse.Namespace) -> int:
    if not args.name or not args.vendor_output:
        raise RuntimeError("--workload requires --name and --vendor-output")

    base = pathlib.Path(args.name)
    vendor_output = pathlib.Path(args.vendor_output)
    base.parent.mkdir(parents=True, exist_ok=True)
    vendor_output.mkdir(parents=True, exist_ok=True)
    os.chmod(vendor_output, 0o700)

    torch = _load_torch_npu()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    cases = _triton_cases(torch, device)
    case_by_name = {case.name: case for case in cases}
    selected_names = args.operators or [case.name for case in cases]
    unknown = sorted(set(selected_names) - set(case_by_name))
    if unknown:
        raise RuntimeError(f"Unknown Triton case(s): {', '.join(unknown)}")

    prepared = []
    results = []
    failures = []
    for name in selected_names:
        case = case_by_name[name]
        scope_name = f"proton_cann_triton::{case.name}"
        try:
            op = case.make()
            for _ in range(args.warmup):
                op()
            torch.npu.synchronize()
            baseline_elapsed_s, _ = _measure_op(torch, op, args.iters)
            prepared.append((case, scope_name, op, baseline_elapsed_s))
        except Exception as exc:
            failures.append({"name": case.name, "phase": "baseline", "error": repr(exc)})
            results.append({
                "name": case.name,
                "category": case.category,
                "description": case.description,
                "status": "failed",
                "phase": "baseline",
                "error": repr(exc),
            })
            if not args.allow_op_failures:
                break

    mode = ("runtime_base:"
            "vendor_metrics=aicore,bandwidth:"
            f"device_id={args.device}:"
            f"aclprof_output_path={vendor_output}:"
            "runtime_host_timing_fallback=true:"
            f"aclprof_runtime_enabled={'false' if args.external_msprof else 'true'}:"
            f"aclprof_auto_export={'false' if args.external_msprof else 'true'}:"
            "mstx_enabled=true:"
            "mstx_domain=proton")
    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=mode,
    )

    try:
        for case, scope_name, op, baseline_elapsed_s in prepared:
            try:
                scope_id = libproton.record_scope()
                out = None
                libproton.enter_op(scope_id, scope_name)
                try:
                    profiled_elapsed_s, out = _measure_op(torch, op, args.iters)
                finally:
                    libproton.exit_op(scope_id, scope_name)

                result = {
                    "name": case.name,
                    "scope": scope_name,
                    "category": case.category,
                    "description": case.description,
                    "iters": args.iters,
                    "checksum": _checksum(out),
                    "status": "ok",
                }
                result.update(_timing_fields(baseline_elapsed_s, profiled_elapsed_s, args.iters))
                results.append(result)
            except Exception as exc:
                failures.append({"name": case.name, "phase": "profiled", "error": repr(exc)})
                results.append({
                    "name": case.name,
                    "category": case.category,
                    "description": case.description,
                    "status": "failed",
                    "phase": "profiled",
                    "error": repr(exc),
                })
                if not args.allow_op_failures:
                    raise
    finally:
        proton.finalize(session_id)

    summary = {
        "ran_triton_kernel": bool(results) and any(r["status"] == "ok" for r in results),
        "device": str(device),
        "operator_count": len(selected_names),
        "ok_count": sum(1 for r in results if r["status"] == "ok"),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
        "timing": _summarize_timing(results),
        "base": str(base),
        "vendor_json": str(base.with_suffix(".vendor.json")),
        "workload_summary_json": str(base.with_suffix(".workload_summary.json")),
    }
    base.with_suffix(".workload_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures and not args.allow_op_failures:
        return 4
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    if not args.name:
        raise RuntimeError("--baseline-only requires --name")

    base = pathlib.Path(args.name)
    base.parent.mkdir(parents=True, exist_ok=True)

    torch = _load_torch_npu()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    cases = _triton_cases(torch, device)
    case_by_name = {case.name: case for case in cases}
    selected_names = args.operators or [case.name for case in cases]
    unknown = sorted(set(selected_names) - set(case_by_name))
    if unknown:
        raise RuntimeError(f"Unknown Triton case(s): {', '.join(unknown)}")

    results = []
    failures = []
    for name in selected_names:
        case = case_by_name[name]
        try:
            op = case.make()
            for _ in range(args.warmup):
                op()
            torch.npu.synchronize()
            baseline_elapsed_s, out = _measure_op(torch, op, args.iters)
            results.append({
                "name": case.name,
                "category": case.category,
                "description": case.description,
                "iters": args.iters,
                "checksum": _checksum(out),
                "status": "ok",
                "baseline_elapsed_s": baseline_elapsed_s,
                "baseline_per_iter_s": baseline_elapsed_s / args.iters if args.iters else None,
            })
        except Exception as exc:
            failures.append({"name": case.name, "phase": "baseline", "error": repr(exc)})
            results.append({
                "name": case.name,
                "category": case.category,
                "description": case.description,
                "status": "failed",
                "phase": "baseline",
                "error": repr(exc),
            })
            if not args.allow_op_failures:
                break

    summary = {
        "device": str(device),
        "operator_count": len(selected_names),
        "ok_count": sum(1 for r in results if r["status"] == "ok"),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
        "overhead_method": "separate_process_no_profiler_baseline",
        "baseline_summary_json": str(base.with_suffix(".baseline_summary.json")),
    }
    base.with_suffix(".baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures and not args.allow_op_failures:
        return 4
    return 0


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _summarize_post_import(out: pathlib.Path, post_import_base: pathlib.Path) -> dict:
    vendor_path = post_import_base.with_suffix(".vendor.json")
    meta_path = post_import_base.with_suffix(".meta.json")
    vendor = json.loads(vendor_path.read_text())
    meta = json.loads(meta_path.read_text())

    source_counts = Counter(assoc.get("source", "") for assoc in vendor.get("associations", []))
    bandwidth_count = sum(1 for assoc in vendor.get("associations", []) if "bandwidth_gb_s" in assoc.get("metrics", {}))
    mstx_messages = sorted({
        assoc.get("metrics", {}).get("message")
        for assoc in vendor.get("associations", [])
        if assoc.get("source") == "msprof_mstx" and assoc.get("metrics", {}).get("message")
    })
    op_types = Counter(
        assoc.get("metrics", {}).get("op_type")
        for assoc in vendor.get("associations", [])
        if assoc.get("source") == "aclprof_op_summary" and assoc.get("metrics", {}).get("op_type"))
    triton_ranges = [msg for msg in mstx_messages if msg.startswith("proton_cann_triton::")]

    return {
        "post_import_vendor_json": str(vendor_path),
        "post_import_meta_json": str(meta_path),
        "raw_input_count": len(vendor.get("raw_inputs", [])),
        "association_sources": dict(source_counts),
        "bandwidth_association_count": bandwidth_count,
        "mstx_triton_operator_ranges": triton_ranges,
        "triton_range_count": len(triton_ranges),
        "top_op_types": dict(op_types.most_common(30)),
        "degrade_reasons": meta.get("degrade_reasons", []),
        "summary_json": str(out / "summary.json"),
    }


def _run_driver(args: argparse.Namespace) -> int:
    out = pathlib.Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    msprof_out = out / "msprof"
    msprof_out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    os.chmod(msprof_out, 0o700)

    script = pathlib.Path(__file__).resolve()
    post_import = script.parent / "cann_post_import_msprof.py"
    profile_base = out / "triton_operator_suite"
    baseline_base = out / "baseline_operator_suite"
    post_import_base = out / "post_import"

    baseline_cmd = [
        sys.executable,
        str(script),
        "--baseline-only",
        "--name",
        str(baseline_base),
        "--device",
        str(args.device),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]
    for op_name in args.operators or []:
        baseline_cmd.extend(["--operator", op_name])
    if args.allow_op_failures:
        baseline_cmd.append("--allow-op-failures")

    _run(baseline_cmd)

    msprof_cmd = [
        args.msprof,
        "--msproftx=on",
        "--ai-core=on",
        f"--aic-metrics={args.aic_metrics}",
        "--task-memory=on",
        "--sys-hardware-mem=on",
        f"--sys-hardware-mem-freq={args.sys_hardware_mem_freq}",
        f"--output={msprof_out}",
        sys.executable,
        str(script),
        "--workload",
        "--name",
        str(profile_base),
        "--vendor-output",
        str(msprof_out),
        "--device",
        str(args.device),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--external-msprof",
    ]
    for op_name in args.operators or []:
        msprof_cmd.extend(["--operator", op_name])
    if args.allow_op_failures:
        msprof_cmd.append("--allow-op-failures")

    _run(msprof_cmd)
    csv_files = sorted(msprof_out.rglob("*.csv"))
    print("exported_csv_count", len(csv_files))
    for path in csv_files[:40]:
        print("exported_csv", path)

    _run([
        sys.executable,
        str(post_import),
        "--base",
        str(post_import_base),
        "--msprof-output",
        str(msprof_out),
    ])

    summary = _summarize_post_import(out, post_import_base)
    baseline_summary_path = baseline_base.with_suffix(".baseline_summary.json")
    baseline_summary = (json.loads(baseline_summary_path.read_text()) if baseline_summary_path.exists() else {})
    baseline_by_name = {
        result.get("name"): result
        for result in baseline_summary.get("results", [])
        if result.get("status") == "ok"
    }
    workload_summary_path = profile_base.with_suffix(".workload_summary.json")
    if workload_summary_path.exists():
        workload_summary = json.loads(workload_summary_path.read_text())
        summary["workload_summary_json"] = str(workload_summary_path)
        summary["baseline_summary_json"] = str(baseline_summary_path)
        summary["operator_count"] = workload_summary.get("operator_count")
        summary["ok_count"] = workload_summary.get("ok_count")
        summary["failed_count"] = workload_summary.get("failed_count")
        results = workload_summary.get("results", [])
        for result in results:
            baseline = baseline_by_name.get(result.get("name"))
            if result.get("status") == "ok" and baseline:
                result.update(
                    _timing_fields(
                        baseline["baseline_elapsed_s"],
                        result["profiled_elapsed_s"],
                        result.get("iters", args.iters),
                    ))
                result["baseline_source"] = "separate_process_no_profiler"
        summary["results"] = results
        summary["failures"] = workload_summary.get("failures", [])
        summary["timing"] = _summarize_timing(results)
        summary["overhead_method"] = "separate_process_no_profiler_baseline"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _make_arg_parser().parse_args()
    if args.baseline_only:
        return _run_baseline(args)
    if args.workload:
        return _run_workload(args)
    return _run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
