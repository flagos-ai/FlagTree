# Copyright 2026- Xcoresigma Technology Co., Ltd

# Minimal demo of the Ascend `enable_reduce_sum_strength` option:
# one kernel, one shape. Launches the same kernel twice -- option off/on --
# and prints the speedup.
#
# Usage:
#   PATH=/usr/local/Ascend/cann-9.0.0-beta.2/bin:$PATH \
#   ASCEND_RT_VISIBLE_DEVICES=7 \
#   python -u python/tutorials/tle/dsa/11-reduce-sum-strength.py
#
# Note: the option value is not part of the triton cache key, so each variant
# gets its own cache dir (knobs.cache.dir) to avoid reusing the other
# variant's compiled kernel.

import tempfile

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.backends.ascend.testing import do_bench_npu

N = 16 * 1024 * 1024  # fp32 elements; BLOCK=512 reduction is the sweet spot
BLOCK = 512


@triton.jit
def sum_1d_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)
    start = 0
    while start < n_elements:
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0)
        acc += tl.sum(x, axis=0)
        start += BLOCK
    tl.store(y_ptr, acc.to(y_ptr.type.element_ty))


def bench(enable: bool) -> float:
    knobs.cache.dir = tempfile.mkdtemp(prefix=f"minbench_{'opt' if enable else 'base'}_")
    x = torch.randn(N, dtype=torch.float32, device="npu")
    out = torch.zeros(1, dtype=torch.float32, device="npu")
    fn = lambda: sum_1d_kernel[(1, )](x, out, N, BLOCK=BLOCK, enable_reduce_sum_strength=enable)
    fn()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu()[0], x.sum().cpu(), rtol=1e-3, atol=1.0)
    return do_bench_npu(fn, warmup=5, active=30)


if __name__ == "__main__":
    base_ms = bench(enable=False)
    opt_ms = bench(enable=True)
    print(f"sum_1d fp32 N={N} BLOCK={BLOCK}:")
    print(f"  option off: {base_ms:.4f} ms")
    print(f"  option on : {opt_ms:.4f} ms")
    print(f"  speedup   : {base_ms / opt_ms:.3f}x")
