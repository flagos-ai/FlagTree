"""CPU compilation regressions for the restored qualified layout and barrier paths."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, require_mthreads_libtriton
from register_order import RegisterOrder

require_mthreads_libtriton()


@triton.jit
def _layout(x, y, z, A: tl.constexpr, B: tl.constexpr, M: tl.constexpr):
    m, n = tl.arange(0, M), tl.arange(0, 128)
    value = tle.gpu.set_layout(tl.load(x + m[:, None] * 128 + n[None, :]), A)
    tl.store(y + m[:, None] * 128 + n[None, :], value)
    value = tle.gpu.set_layout(value, B)
    tl.store(z + m[:, None] * 128 + n[None, :], value)


@pytest.mark.parametrize("dtype", ["*fp32", "*i32"])
@pytest.mark.parametrize("kdim", [32, 64])
def test_accumulator_k_change_uses_no_scratch(dtype, kdim):
    a = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])
    b = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, kdim])
    k = compile_musa(
        _layout,
        dict(x=dtype, y=dtype, z=dtype, A="constexpr", B="constexpr", M="constexpr"),
        dict(A=a, B=b, M=64),
        options=dict(num_warps=8),
    )
    assert k.metadata.shared == 0
    assert "addrspace(3)" not in k.asm["llir"]


def _single_mixed_register_byte_layouts():
    a = RegisterOrder(
        [[0, 8], [0, 16], [0, 32], [16, 0], [0, 64]],
        [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]],
        [[4, 0], [8, 0], [32, 0], [64, 0], [128, 0]],
        [256, 128],
    )
    # Exactly one mixed register/lane transposition exercises the byte shuffle
    # lowering. The original three-transposition fixture correctly chose shared
    # memory under cvtNeedsWarpShuffle (<2); its no-scratch assertion was invalid.
    b = RegisterOrder(
        [[0, 1], [0, 16], [0, 32], [16, 0], [0, 64]],
        [[0, 8], [0, 2], [0, 4], [1, 0], [2, 0]],
        [[4, 0], [8, 0], [32, 0], [64, 0], [128, 0]],
        [256, 128],
    )
    return a, b


@pytest.mark.parametrize("dtype", ["*i8", "*fp8e4nv"])
def test_single_mixed_register_byte_layout(dtype):
    a, b = _single_mixed_register_byte_layouts()
    k = compile_musa(
        _layout,
        dict(x=dtype, y=dtype, z=dtype, A="constexpr", B="constexpr", M="constexpr"),
        dict(A=a, B=b, M=256),
        options=dict(num_warps=32),
    )
    assert k.metadata.shared == 0
    assert "addrspace(3)" not in k.asm["llir"]


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA device required")
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_single_mixed_register_byte_layout_runtime(dtype):
    a, b = _single_mixed_register_byte_layouts()
    rows, cols = torch.arange(256)[:, None], torch.arange(128)[None, :]
    original = ((rows * 67 + cols * 29 + (rows ^ cols) * 3) & 255).to(torch.uint8)
    if dtype == torch.float8_e4m3fn:
        original = torch.where(original == 127, 126, torch.where(original == 255, 254, original)).to(torch.uint8)
    x = original.to("musa").view(dtype)
    y, z = torch.empty_like(x), torch.empty_like(x)
    for _ in range(3):
        y.view(torch.uint8).fill_(255)
        z.view(torch.uint8).fill_(255)
        kernel = _layout[(1, )](x, y, z, a, b, 256, num_warps=32)
        assert kernel.metadata.shared == 0
        assert "addrspace(3)" not in kernel.asm["llir"]
        assert torch.equal(y.view(torch.uint8).cpu(), original)
        assert torch.equal(z.view(torch.uint8).cpu(), original)
