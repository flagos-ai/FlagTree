"""Native/linear register permutations must bypass shared carrier lowering."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.types import distributed_encoding
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


class RegisterOrder(distributed_encoding):

    @property
    def rank(self):
        return 2

    def to_ir(self, builder):
        return builder.get_tle_linear_encoding(
            [[0, 8], [0, 16], [16, 0], [0, 32], [0, 64]],
            [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]],
            [[4, 0], [8, 0], [32, 0]],
            [],
            [64, 128],
        )

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) is type(other)


@triton.jit
def _reorder(x, y, z, A: tl.constexpr, B: tl.constexpr):
    m, n = tl.arange(0, 64), tl.arange(0, 128)
    v = tle.gpu.set_layout(tl.load(x + m[:, None] * 128 + n[None, :]), A)
    tl.store(y + m[:, None] * 128 + n[None, :], v)
    v = tle.gpu.set_layout(v, B)
    tl.store(z + m[:, None] * 128 + n[None, :], v)


@pytest.mark.parametrize("dtype", ["*fp32", "*i32", "*i8", "*fp8e4nv"])
def test_register_permutation_has_no_scratch(dtype):
    a = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])
    k = compile_musa(
        _reorder,
        {"x": dtype, "y": dtype, "z": dtype, "A": "constexpr", "B": "constexpr"},
        {"A": a, "B": RegisterOrder()},
        options={"num_warps": 8},
    )
    assert k.metadata.shared == 0
    assert "addrspace(3)" not in k.asm["llir"]


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA device required")
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_register_byte_permutation_runtime(dtype):
    a = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])
    # Make the byte pattern sensitive to row, lane and register order. For
    # FP8, avoid numerical conversions and compare the raw stored bits.
    rows, cols = torch.arange(64)[:, None], torch.arange(128)[None, :]
    raw = ((rows * 67 + cols * 29 + (rows ^ cols) * 3) & 255).to(torch.uint8)
    if dtype == torch.float8_e4m3fn:
        raw = torch.where(raw == 127, 126, torch.where(raw == 255, 254, raw)).to(torch.uint8)
    dx = raw.to("musa").view(dtype)
    dy, dz = torch.empty_like(dx), torch.empty_like(dx)
    for _ in range(3):
        dy.view(torch.uint8).fill_(255)
        dz.view(torch.uint8).fill_(255)
        result = _reorder[(1, )](dx, dy, dz, a, RegisterOrder(), num_warps=8)
        assert result.metadata.shared == 0
        assert torch.equal(dy.view(torch.uint8).cpu(), raw)
        assert torch.equal(dz.view(torch.uint8).cpu(), raw)
