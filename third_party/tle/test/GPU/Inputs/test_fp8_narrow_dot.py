"""Hopper narrow FP8 dots preserve native K32 accumulation.

Run with pytest on an otherwise idle Hopper GPU. All FP8 data is created on
the GPU; the test compares the same eight columns in N8 and N16 instructions.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _narrow_dot(a, b, out, N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, 64)
    reduction = tl.arange(0, K)
    columns = tl.arange(0, N)
    lhs = tl.load(a + rows[:, None] * K + reduction[None, :])
    rhs = tl.load(b + reduction[:, None] * 16 + columns[None, :])
    result = tl.dot(lhs, rhs, out_dtype=tl.float32)
    tl.store(out + rows[:, None] * N + columns[None, :], result)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("reduction", [32, 128, 256])
def test_native_fp8_n8_accumulation(dtype, warps, reduction):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires Hopper FP8 WGMMA")
    torch.manual_seed(1293)
    a = torch.randn((64, reduction), device="cuda", dtype=torch.bfloat16).to(dtype)
    b = torch.randn((reduction, 16), device="cuda", dtype=torch.bfloat16).to(dtype)
    narrow = torch.empty((64, 8), device="cuda", dtype=torch.float32)
    wide = torch.empty((64, 16), device="cuda", dtype=torch.float32)
    kernel = _narrow_dot[(1,)](a, b, narrow, 8, reduction, num_warps=warps, enable_fp_fusion=False)
    _narrow_dot[(1,)](a, b, wide, 16, reduction, num_warps=warps, enable_fp_fusion=False)
    assert torch.equal(narrow.view(torch.int32), wide[:, :8].contiguous().view(torch.int32))
    assert "wgmma.mma_async.sync.aligned.m64n8k32.f32" in kernel.asm["ptx"]
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16" not in kernel.asm["ptx"]
