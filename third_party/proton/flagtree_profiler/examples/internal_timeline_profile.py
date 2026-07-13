#!/usr/bin/env python3
"""Single-kernel internal timeline example.

Run:

    cd /workspace/FlagTree
    python3 third_party/proton/flagtree_profiler/examples/internal_timeline_profile.py

Outputs:

    /tmp/flagtree_profiler_examples/internal_timeline/profile.hatchet
    /tmp/flagtree_profiler_examples/internal_timeline/profile.meta.json
    /tmp/flagtree_profiler_examples/internal_timeline/profile.timeline.json
    /tmp/flagtree_profiler_examples/internal_timeline/profile.vendor.json

Expected internal timeline:

    The kernel is one vector add kernel.  profile.timeline.json contains
    flagtree.kernel_internal events for non-constant tracked Triton IR ops
    inside the kernel, for each logical program instance.  The most important
    events are:

        tt.load   # load x
        tt.load   # load y
        arith.addf
        tt.store

    The full trace also includes lightweight IR ops such as program_id,
    make_range, splat, addptr and mask comparisons.  Some of these may show
    duration_cycle=0 when the measured interval is below SYS_CNT resolution.

    Example trace event shape:

        {
          "name": "tt.load load",
          "cat": "flagtree.kernel_internal",
          "ph": "X",
          "pid": 0,
          "tid": 0,
          "ts": 16,
          "dur": 1,
          "args": {
            "timestamp_unit": "SYS_CNT cycles",
            "op_id": 12,
            "logical_instance_id": 0,
            "mlir_op": "tt.load",
            "triton_statement": "tt.load",
            "source_loc": "loc(\"...internal_timeline_profile.py\":65:16)",
            "role": "load",
            "start_cycle": 168537204706139,
            "end_cycle": 168537204706140,
            "duration_cycle": 1
          }
        }

    profile.hatchet also contains IR-op children under the kernel node with
    flagtree.internal.* cycle metrics.  hook="instrumentation" does not create
    additional output files; it enriches the same four files used by the
    normal CANN profiler path.
"""

from __future__ import annotations

import os
import pathlib
import shutil

import triton
import triton.language as tl
import triton.profiler as proton


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, z_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)


def _load_torch_npu():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    return torch


def main() -> int:
    out_dir = pathlib.Path("/tmp/flagtree_profiler_examples/internal_timeline")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_base = out_dir / "profile"

    torch = _load_torch_npu()
    device_id = 1
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")

    n = 4096
    block_size = 1024
    x = torch.randn((n, ), device=device, dtype=torch.float32)
    y = torch.randn((n, ), device=device, dtype=torch.float32)
    z = torch.empty_like(x)

    _vector_add_kernel[(triton.cdiv(n, block_size), )](x, y, z, n, BLOCK_SIZE=block_size)
    torch.npu.synchronize()

    sid = proton.start(
        name=str(profile_base),
        context="shadow",
        data="tree",
        backend="cann",
        hook="instrumentation",
        mode=("runtime_base:"
              f"device_id={device_id}:"
              "vendor_metrics=aicore,bandwidth:"
              "mstx_enabled=true:"
              "mstx_domain=proton"),
    )
    try:
        _vector_add_kernel[(triton.cdiv(n, block_size), )](x, y, z, n, BLOCK_SIZE=block_size)
        torch.npu.synchronize()
    finally:
        proton.finalize(sid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
