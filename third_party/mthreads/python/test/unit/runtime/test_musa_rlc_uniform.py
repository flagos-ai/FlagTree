"""Regression for S5000 uniform rounding and live RLC JIT cache identity."""

import pytest
import triton
import triton.language as tl

from triton._C import libtriton

pytestmark = pytest.mark.skipif(
    not hasattr(libtriton, "mthreads"), reason="MThreads backend is required"
)


@triton.jit
def flaggems_uniform_kernel(
    out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 = c0 + i4
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)
    scale = 2.3283064365386963e-10
    width = to - from_
    r0 = r0.to(tl.float32) * scale * width + from_
    r1 = r1.to(tl.float32) * scale * width + from_
    r2 = r2.to(tl.float32) * scale * width + from_
    r3 = r3.to(tl.float32) * scale * width + from_
    off0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK
    tl.store(out_ptr + off0, r0, mask=off0 < N)
    tl.store(out_ptr + off1, r1, mask=off1 < N)
    tl.store(out_ptr + off2, r2, mask=off2 < N)
    tl.store(out_ptr + off3, r3, mask=off3 < N)



def test_uniform_rounding_and_live_rlc_cache(monkeypatch, tmp_path):
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    # n=1025, seed=11, offset=12 reproduces the rounding issue at index 479.
    outputs, kernels = [], []
    for enabled in (False, True):
        with monkeypatch.context() as env:
            env.setenv("FLAGTREE_MUSA_RLC_ENHANCE", str(int(enabled)))
            env.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", "15")
            env.setenv("FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY", "1")
            env.setenv("TRITON_CACHE_DIR", str(tmp_path / str(enabled)))
            out = torch.empty((1025,), device="musa", dtype=torch.float16)
            kernel = flaggems_uniform_kernel[(1,)](
                out, 1025, 11, 12, -1.0, 1.0, 1024, num_warps=16, num_stages=1
            )
            outputs.append(out.cpu())
            kernels.append(kernel)

    assert kernels[0] is not kernels[1]
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert kernels[0].asm["ttgir"].count("ttg.convert_layout") == 4
    assert kernels[1].asm["ttgir"].count("ttg.convert_layout") == 0
