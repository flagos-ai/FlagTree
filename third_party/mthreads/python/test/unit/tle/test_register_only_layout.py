"""Native/linear register permutations must bypass shared carrier lowering."""

import pytest
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
