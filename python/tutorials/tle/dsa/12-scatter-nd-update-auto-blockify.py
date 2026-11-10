# Copyright 2026- Xcoresigma Technology Co., Ltd

# Demonstrate the performance impact of Ascend `enable_auto_blockify`, using
# the original ScatterNdUpdate kernel and its BF16/FP32 shapes. Like
# 11-reduce-sum-strength.py, compare explicit option off/on on the same input.
# True/False overrides TRITON_ALL_BLOCKS_PARALLEL; None inherits the environment.
#
# Each logical program writes one row of updates to a unique destination.
# Auto-blockify groups many short programs into loops on fewer runtime blocks,
# reducing repeated block scheduling/control work. The benefit depends on the
# grid, workload and device: enabling this option is not always faster.
# Historical Ascend 910B3 measurements at N=8160 (FlagTree 95ff4bb40) showed
# 17.585-17.658x for BF16/D=512 and 14.442-14.579x for FP32/D=2048.
# Those are device-profiler results, not portable performance assertions.
#
# Usage (requires the Ascend toolkit environment):
#   ASCEND_RT_VISIBLE_DEVICES=7 \
#   python python/tutorials/tle/dsa/12-scatter-nd-update-auto-blockify.py
# Full original grid sweep:
#   python python/tutorials/tle/dsa/12-scatter-nd-update-auto-blockify.py \
#       --ns 64 256 1024 2041 4096 8160

import argparse
import math
import tempfile
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton import knobs
from triton.backends.ascend.testing import do_bench_npu


@triton.jit
def _scatter_nd_update_kernel(
    updates_ptr,
    indices_ptr,
    var_ptr,
    num_tokens,
    s0,
    s1,
    s2,
    s3,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    lin = 0
    for d in tl.static_range(K):
        idx = tl.load(indices_ptr + pid * K + d)
        if d == 0:
            lin = lin + idx * s0
        elif d == 1:
            lin = lin + idx * s1
        elif d == 2:
            lin = lin + idx * s2
        else:
            lin = lin + idx * s3
    off = tl.arange(0, BLOCK_D)
    mask = off < D
    val = tl.load(updates_ptr + pid * D + off, mask=mask)
    tl.store(var_ptr + lin + off, val, mask=mask)


def scatter_nd_update(var, indices, updates, enable):
    n, k = indices.shape
    d = math.prod(var.shape[k:])
    strides = [var.stride(i) for i in range(k)] + [0] * (4 - k)
    return _scatter_nd_update_kernel[(n, )](
        updates,
        indices,
        var,
        n,
        *strides,
        K=k,
        D=d,
        BLOCK_D=triton.next_power_of_2(d),
        enable_auto_blockify=enable,
    )


def compare(n, var_shape, dtype, warmup, active, profile_root):
    # Generate inputs once, outside timing. Unique indices avoid competing
    # writes; OFF and ON reuse exactly the same tensors and fixed tile size.
    generator = torch.Generator().manual_seed(0)
    slots = torch.randperm(var_shape[0] * var_shape[1], generator=generator)[:n]
    indices = torch.stack((slots // var_shape[1], slots % var_shape[1]), dim=1).to(device="npu", dtype=torch.int32)
    d = math.prod(var_shape[2:])
    updates = torch.randn((n, d), generator=generator, dtype=dtype).to("npu")
    var = torch.zeros(var_shape, dtype=dtype, device="npu")
    expected = var.clone()
    expected.view(var_shape[0], var_shape[1], d)[indices[:, 0].long(), indices[:, 1].long()] = updates
    saved_indices, saved_updates = indices.clone(), updates.clone()

    times = []
    for enable in (False, True):
        var.zero_()
        fn = lambda: scatter_nd_update(var, indices, updates, enable)
        compiled = fn()  # Compile before profiling; no auto-tuning search.
        torch.npu.synchronize()
        assert torch.equal(var, expected), f"incorrect output before profiling: {enable=}"
        # prof_dir forces torch_npu.profiler; the name selects kernel Duration
        # rather than event elapsed time. do_bench_npu returns milliseconds.
        milliseconds = do_bench_npu(
            fn,
            warmup=warmup,
            active=active,
            prof_dir=str(profile_root / f"d{d}_n{n}_{int(enable)}"),
            target_kernel_name=compiled.metadata.name,
        )
        torch.npu.synchronize()
        assert torch.equal(var, expected), f"incorrect output after profiling: {enable=}"
        assert torch.equal(indices, saved_indices) and torch.equal(updates, saved_updates)
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise RuntimeError(f"invalid profiler duration: {milliseconds}")
        times.append(milliseconds * 1000)

    print(
        f"{str(dtype):14s} N={n:5d} D={d:4d}: "
        f"OFF={times[0]:9.3f} us ON={times[1]:9.3f} us "
        f"OFF/ON={times[0] / times[1]:.3f}x; full output and inputs checked", flush=True)


def main():
    parser = argparse.ArgumentParser(description="ScatterNdUpdate: auto-blockify OFF/ON device-profiler comparison")
    parser.add_argument("--ns", type=int, nargs="+", default=[64, 8160])
    parser.add_argument("--dtype", choices=["bf16", "fp32", "all"], default="all")
    parser.add_argument("--device", type=int, default=0, help="ordinal among visible NPUs")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--active", type=int, default=50)
    args = parser.parse_args()
    cases = [((58841, 32, 1, 512), torch.bfloat16, "bf16"), ((58841, 2, 2048), torch.float32, "fp32")]
    cases = [case for case in cases if args.dtype in ("all", case[2])]
    if args.warmup < 0 or args.active <= 0:
        parser.error("warmup must be nonnegative and active must be positive")
    if any(n <= 0 or n > shape[0] * shape[1] for shape, _, _ in cases for n in args.ns):
        parser.error("each N must fit the number of unique destination rows")

    torch.npu.set_device(args.device)
    print("Device-profiler mean durations; fixed configuration, no auto-tuning. OFF/ON > 1 means ON is faster.",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="scatter_auto_blockify_") as tmp, knobs.cache.scope():
        # Explicit options have distinct cache keys. A temporary cache excludes
        # historical binaries and is removed, with profiler files, on exit.
        knobs.cache.dir = str(Path(tmp) / "cache")
        for shape, dtype, _ in cases:
            for n in args.ns:
                compare(n, shape, dtype, args.warmup, args.active, Path(tmp) / "profiles")


if __name__ == "__main__":
    main()
