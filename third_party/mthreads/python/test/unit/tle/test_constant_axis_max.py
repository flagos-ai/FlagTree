"""Idempotent max over a proven constant axis needs no lane exchange."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _max(X, Y, L: tl.constexpr, HINT: tl.constexpr):
    m, n = tl.arange(0, 64), tl.arange(0, 8)
    v = tle.gpu.set_layout(tl.load(X + m[:, None] * 8 + n[None, :]), L)
    if HINT: v = tl.max_constancy(v, [1, 8])
    y = tl.max(v, 1)
    tl.store(Y + m, y)


@pytest.mark.parametrize('hint', [False, True])
def test_full_axis_constancy(hint):
    k = compile_musa(_max, {'X': '*fp32', 'Y': '*fp32', 'L': 'constexpr', 'HINT': 'constexpr'},
                     {'L': tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128]), 'HINT': hint},
                     options={'num_warps': 8, 'local_max_vector_width': 4})
    assert ('musa.reduce_full_axis_constant' in k.asm['ttgir']) == hint
    assert ('shfl' in k.asm['llir']) != (hint)
