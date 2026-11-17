"""Idempotent max over a proven constant axis needs no lane exchange."""
import os

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


@triton.jit
def _scoped_max(X, Y, Z, MODE, L: tl.constexpr):
    m, n = tl.arange(0, 64), tl.arange(0, 8)
    v = tl.load(X + m[:, None] * 8 + n[None, :])
    # MODE is loaded at run time. The input is row-constant only when MODE
    # is true. Neither the sibling nor the post-branch use is annotated.
    if tl.load(MODE) != 0:
        hinted = tl.max_constancy(tle.gpu.set_layout(v, L), [1, 8])
        tl.store(Y + m, tl.max(hinted, 1))
    else:
        tl.store(Y + m, tl.max(v, 1))
    tl.store(Z + m, tl.max(v, 1))


@pytest.mark.parametrize('width', [1, 4])
def test_branch_constancy_is_not_written_back_to_shared_producer(width):
    k = compile_musa(_scoped_max, {'X': '*fp32', 'Y': '*fp32', 'Z': '*fp32', 'MODE': '*i32', 'L': 'constexpr'},
                     {'L': tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])},
                     options={'num_warps': 8, 'local_max_vector_width': width})
    ttgir = k.asm['ttgir']
    assert 'scf.if' in ttgir
    assert ttgir.count('"tt.reduce"') == 3
    assert ttgir.count('musa.reduce_full_axis_constant') == 1
    assert 'tt.constancy = dense<[1, 8]>' not in ttgir
    # The unannotated reductions still need lane exchange with the fast path
    # enabled, while the original single-use regression checks the positive.
    assert 'shfl' in k.asm['llir']


@pytest.mark.skipif(
    os.environ.get('TRITON_MUSA_RUN_NUMERICAL_DEVICE_TESTS') != '1',
    reason='requires explicit MUSA device-test opt-in')
@pytest.mark.parametrize('width', [1, 4])
def test_branch_constancy_device(width):
    import torch

    layout = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])
    rows = torch.arange(64, dtype=torch.float32).reshape(-1, 1)
    # Different maxima locations by row, not a single fortunate lane.
    cols = torch.arange(8, dtype=torch.float32)
    for take_hint, data in ((True, rows.expand(-1, 8).contiguous()),
                            (False, rows + (cols - rows).remainder(8))):
        x = data.to('musa')
        y = torch.empty(64, device='musa', dtype=torch.float32)
        z = torch.empty_like(y)
        mode = torch.tensor([int(take_hint)], device='musa', dtype=torch.int32)
        _scoped_max[(1,)](x, y, z, mode, layout, num_warps=8, local_max_vector_width=width)
        ref = data.max(dim=1).values
        torch.testing.assert_close(y.cpu(), ref, rtol=0, atol=0)
        torch.testing.assert_close(z.cpu(), ref, rtol=0, atol=0)
