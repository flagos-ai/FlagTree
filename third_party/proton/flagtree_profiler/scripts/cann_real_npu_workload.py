#!/usr/bin/env python3
"""Run a small real NPU workload under Proton CANN profiling.

This script is intended to be launched by external msprof, for example:

    msprof --msproftx=on --output=/tmp/proton_cann_real/msprof \
      python third_party/proton/flagtree_profiler/scripts/cann_real_npu_workload.py \
        --name /tmp/proton_cann_real/profile_run \
        --vendor-output /tmp/proton_cann_real/msprof

The process writes Proton artifacts during proton.finalize(). External msprof
exports CSV files after this process exits, so use cann_post_import_msprof.py
afterwards to import those exported CSV files.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import triton
import triton.language as tl
import triton.profiler as proton
from triton._C.libproton import proton as libproton


@triton.jit
def _real_triton_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _load_torch_npu():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for the real NPU workload.") from exc

    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch_npu is required for the real NPU workload.") from exc

    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is unavailable after importing torch_npu.")
    if not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed, but torch.npu.is_available() is false.")
    return torch


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Base path for Proton artifacts.")
    parser.add_argument(
        "--vendor-output",
        required=True,
        help="Directory also passed to msprof --output.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    return parser


def main() -> int:
    args = _make_arg_parser().parse_args()
    base = pathlib.Path(args.name)
    vendor_output = pathlib.Path(args.vendor_output)
    base.parent.mkdir(parents=True, exist_ok=True)
    vendor_output.mkdir(parents=True, exist_ok=True)
    os.chmod(vendor_output, 0o700)

    try:
        torch = _load_torch_npu()
    except RuntimeError as exc:
        print(f"ran_npu_kernel=False {exc!r}", file=sys.stderr)
        return 2

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    mode = ("runtime_base:"
            "vendor_metrics=aicore,bandwidth:"
            f"aclprof_output_path={vendor_output}:"
            "runtime_host_timing_fallback=true:"
            "aclprof_runtime_enabled=false:"
            "aclprof_auto_export=false:"
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

    n_elements = args.size * args.size
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size), )
    a = torch.randn((n_elements, ), device=device, dtype=torch.float32)
    b = torch.randn((n_elements, ), device=device, dtype=torch.float32)
    out = torch.empty_like(a)

    for _ in range(args.warmup):
        _real_triton_add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=block_size)
    torch.npu.synchronize()

    scope_id = libproton.record_scope()
    scope_name = "proton_cann_real_triton_vector_add"
    libproton.enter_op(scope_id, scope_name)
    start = time.time()
    try:
        for _ in range(args.iters):
            _real_triton_add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=block_size)
        torch.npu.synchronize()
        elapsed_s = time.time() - start
    finally:
        libproton.exit_op(scope_id, scope_name)

    checksum = float(out.float().mean().cpu().item())
    proton.finalize(session_id)

    summary = {
        "ran_npu_kernel": True,
        "ran_triton_kernel": True,
        "device": str(device),
        "size": args.size,
        "n_elements": n_elements,
        "iters": args.iters,
        "elapsed_s": elapsed_s,
        "checksum": checksum,
        "base": str(base),
        "vendor_json": str(base.with_suffix(".vendor.json")),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
