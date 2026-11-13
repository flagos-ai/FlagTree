"""Reduction inference must bridge an explicit result-layout/ABI contract."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.types import distributed_encoding
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


class _RegisterOrder(distributed_encoding):
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
def _hard_result(x, out, L: tl.constexpr):
    m, n = tl.arange(0, 64), tl.arange(0, 128)
    value = tle.gpu.set_layout(tl.load(x + m[:, None] * 128 + n[None, :]), L)
    maximum = tl.max(value.reshape(64, 16, 8), 1)
    maximum = tle.gpu.set_layout(maximum, L).to(tl.float32)
    k = tl.arange(0, 8)
    tl.store(out + m[:, None] * 8 + k[None, :], maximum)


@pytest.mark.parametrize("dtype", ["*i32", "*fp32"])
@pytest.mark.parametrize("linear", [False, True])
def test_reduce_result_contract_is_materialized(dtype, linear):
    layout = _RegisterOrder() if linear else tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])
    result = compile_musa(
        _hard_result, {"x": dtype, "out": "*fp32", "L": "constexpr"}, {"L": layout}, options={"num_warps": 8}
    )
    assert result.module is None and result.function is None
    assert "unrealized_conversion_cast" not in result.asm["ttgir"]
