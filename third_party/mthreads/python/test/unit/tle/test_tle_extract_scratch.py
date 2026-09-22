"""The register-only extract path must not reserve unused shared memory."""

import pytest
import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource

from test_tle_utils import musa_target, require_mthreads_libtriton

require_mthreads_libtriton()

_FLAT = tl.constexpr(tle.gpu.BlockEncoding([1], [32], [4], [0]))
_MATRIX = tl.constexpr(tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0]))


@triton.jit(do_not_specialize=['index'])
def _extract(inp, out, index, STATIC: tl.constexpr, INDEX: tl.constexpr):
    x = tle.gpu.set_layout(tl.arange(0, 1024), _FLAT)
    x = tle.gpu.set_layout(x.reshape((32, 32)), _MATRIX)
    value = tl.load(inp + x)
    tile = tle.extract_tile(value, index=INDEX if STATIC else index, tile_shape=(16, 32))
    y = tle.gpu.set_layout(tl.arange(0, 512), _FLAT)
    y = tle.gpu.set_layout(y.reshape((16, 32)), _MATRIX)
    tl.store(out + y, tile)


@pytest.mark.parametrize('static', [True, False])
@pytest.mark.parametrize('index', [0, 1])
def test_extract_scratch_compile(static, index):
    source = ASTSource(_extract,
                       {'inp': '*fp32', 'out': '*fp32', 'index': 'i32', 'STATIC': 'constexpr', 'INDEX': 'constexpr'},
                       constexprs={'STATIC': static, 'INDEX': index})
    compiled = triton.compile(source, target=musa_target(), options={'num_warps': 4})
    assert compiled.metadata.shared == (0 if static else 2048)
    if static:
        assert 'llvm.musa.syncthreads.lm' not in compiled.asm['llir']


@pytest.mark.skipif(not torch.musa.is_available(), reason='MUSA device required')
@pytest.mark.parametrize('static', [True, False])
def test_extract_scratch_runtime(static):
    inp = torch.arange(1024, dtype=torch.float32, device='musa')
    out = torch.empty((16, 32), dtype=torch.float32, device='musa')
    for index in (0, 1, 0):
        compiled = _extract[(1, )](inp, out, index, static, index, num_warps=4)
        torch.musa.synchronize()
        torch.testing.assert_close(out, inp.reshape(32, 32)[index * 16:(index + 1) * 16], atol=0, rtol=0)
        assert compiled.metadata.shared == (0 if static else 2048)
