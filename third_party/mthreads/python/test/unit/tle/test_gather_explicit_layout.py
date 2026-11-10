import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from register_order import RegisterOrder
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _gather(X, Y, A: tl.constexpr, B: tl.constexpr, MASK: tl.constexpr):
    m, n = tl.arange(0, 64), tl.arange(0, 8)
    v = tle.gpu.set_layout(tl.load(X + m[:, None] * 8 + n[None, :]), A)
    index = tle.gpu.set_layout(tl.broadcast_to((n ^ MASK)[None, :], (64, 8)), B)
    result = tle.gpu.set_layout(tl.gather(v, index, axis=1), A)
    tl.store(Y + m[:, None] * 8 + n[None, :], result)


@pytest.mark.parametrize("dtype", ["*fp32", "*i32"])
@pytest.mark.parametrize("mask", [1, 2])
def test_gather_preserves_independent_index_layout(dtype, mask):
    a = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])
    b = RegisterOrder([[16, 0]], [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]], [[4, 0], [8, 0], [32, 0]], [64, 8])
    k = compile_musa(
        _gather,
        {"X": dtype, "Y": dtype, "A": "constexpr", "B": "constexpr", "MASK": "constexpr"},
        {"A": a, "B": b, "MASK": mask},
        options={"num_warps": 8},
    )
    assert k.module is None and k.function is None
