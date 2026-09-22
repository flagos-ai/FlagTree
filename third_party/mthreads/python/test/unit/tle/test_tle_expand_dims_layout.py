"""Explicit 2D layouts must support their inferred sliced arange inputs."""

import pytest
import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource

from test_tle_utils import musa_target, require_mthreads_libtriton

require_mthreads_libtriton()
_LAYOUT = tl.constexpr(tle.gpu.BlockEncoding([1, 4], [4, 8], [4, 1], [1, 0]))
_LAYOUT8 = tl.constexpr(tle.gpu.BlockEncoding([1, 4], [4, 8], [8, 1], [1, 0]))


@triton.jit
def _matrix(out, WARPS: tl.constexpr):
    r = tle.gpu.set_layout(tl.broadcast_to(tl.arange(0, 64)[:, None], (64, 64)), _LAYOUT8 if WARPS == 8 else _LAYOUT)
    c = tle.gpu.set_layout(tl.broadcast_to(tl.arange(0, 64)[None, :], (64, 64)), _LAYOUT8 if WARPS == 8 else _LAYOUT)
    value = r * 1024 + c
    row_sum = tl.sum(value, 1)
    # A row reduction has a slice at axis 1; use it as a column vector.
    # Keep this result outside the original explicit layout domain.
    result = tl.broadcast_to(row_sum[None, :], (64, 64))
    target = tl.arange(0, 4096).reshape((64, 64))
    tl.store(out + target, result)


@triton.jit
def _marker(out, INDEX: tl.constexpr):
    tl.store(out + INDEX, INDEX + 7)


@triton.jit
def _kernel(out, markers, WS: tl.constexpr):
    if WS:
        tle.gpu.warp_specialize([(_marker, (markers, 0)), (_matrix, (out, 8)),
                                 (_marker, (markers, 1))],
                                worker_num_warps=[8, 4], worker_num_regs=[32, 32])
    else:
        _matrix(out, 4)


@pytest.mark.parametrize('ws', [False, True])
def test_expand_dims_layout_compile(ws):
    source = ASTSource(_kernel, {'out': '*i32', 'markers': '*i32', 'WS': 'constexpr'},
                       constexprs={'WS': ws})
    compiled = triton.compile(source, target=musa_target(), options={'num_warps': 8 if ws else 4})
    assert compiled.metadata.num_warps == (20 if ws else 4)


@pytest.mark.skipif(not torch.musa.is_available(), reason='MUSA device required')
@pytest.mark.parametrize('ws', [False, True])
def test_expand_dims_layout_runtime(ws):
    out = torch.empty((64, 64), dtype=torch.int32, device='musa')
    markers = torch.zeros((2,), dtype=torch.int32, device='musa')
    expected = torch.arange(64)[:, None] * 1024 + torch.arange(64)[None, :]
    sums = expected.sum(1)
    expected = sums[None, :].expand(64, 64)
    for _ in range(3):
        out.fill_(-1)
        _kernel[(1,)](out, markers, ws, num_warps=8 if ws else 4)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().long(), expected, atol=0, rtol=0)
        if ws:
            torch.testing.assert_close(markers.cpu(), torch.tensor([7, 8], dtype=torch.int32), atol=0, rtol=0)
