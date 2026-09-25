"""Partition synchronization must retain local-memory completion semantics."""

import re

import pytest
import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource

from test_tle_utils import musa_target, require_mthreads_libtriton

require_mthreads_libtriton()

_LAYOUT = tl.constexpr(tle.gpu.BlockEncoding([4], [32], [8], [0]))


@triton.jit
def _marker(markers, SLOT: tl.constexpr):
    tl.store(markers + tl.program_id(0) * 2 + SLOT, SLOT + 7)


@triton.jit
def _exchange(x, out, shared, ITERS: tl.constexpr):
    index = tle.gpu.set_layout(tl.arange(0, 2048), _LAYOUT)
    values = tl.load(x + tl.program_id(0) * 2048 + index)
    total = tl.full((2048, ), 0, tl.int32)
    for step in range(ITERS):
        tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(shared), _LAYOUT), values + step)
        # Exchange across warps; the compiler must synchronize shared stores.
        value = tl.load(tle.gpu.local_ptr(shared, (index ^ 128, )))
        total += value
    tl.store(out + tl.program_id(0) * 2048 + index, total)


@triton.jit
def _kernel(x, out, markers, WORKER: tl.constexpr, ITERS: tl.constexpr):
    shared = tle.gpu.alloc((2048, ), tl.int32, nv_mma_shared_layout=False)
    if WORKER:
        tle.gpu.warp_specialize([
            (_marker, (markers, 0)),
            (_exchange, (x, out, shared, ITERS)),
            (_marker, (markers, 1)),
        ], worker_num_warps=[8, 4], worker_num_regs=[64, 32])
    else:
        tle.gpu.warp_specialize([
            (_exchange, (x, out, shared, ITERS)),
            (_marker, (markers, 0)),
            (_marker, (markers, 1)),
        ], worker_num_warps=[8, 4], worker_num_regs=[64, 32])


@pytest.mark.parametrize('worker', [False, True], ids=['default', 'worker'])
def test_ws_shared_completion_compile(worker):
    source = ASTSource(_kernel,
                       {'x': '*i32', 'out': '*i32', 'markers': '*i32', 'WORKER': 'constexpr', 'ITERS': 'constexpr'},
                       constexprs={'WORKER': worker, 'ITERS': 17})
    compiled = triton.compile(source, target=musa_target(), options={'num_warps': 8})
    llir = compiled.asm['llir']
    arrivals = re.findall(r'^.*\bcall\b.*@llvm\.musa\.async\.arrive(?:\.none\.phaseid)?\(.*$', llir, re.MULTILINE)
    assert arrivals, 'test must exercise compiler-generated partition barriers'
    fenced = re.findall(r'call void @llvm\.musa\.lma\.wait\(\)', llir)
    assert len(fenced) == len(arrivals), 'partition barrier lost local-memory completion'
    assert not re.search(r'call i32 @llvm\.musa\.async\.arrive\(', llir), 'phase must persist explicitly'
    assert 'alloca ' not in llir, 'phase state must be promoted to registers'


@pytest.mark.skipif(not torch.musa.is_available(), reason='MUSA device required')
@pytest.mark.parametrize('worker', [False, True], ids=['default', 'worker'])
def test_ws_shared_completion_runtime(worker):
    torch.manual_seed(1041)
    cpu = torch.randint(-100, 100, (3, 2048), dtype=torch.int32)
    x = cpu.to('musa')
    out = torch.empty_like(x)
    markers = torch.empty((3, 2), dtype=torch.int32, device='musa')
    iters = 257
    expected = cpu[:, torch.arange(2048) ^ 128] * iters + iters * (iters - 1) // 2
    for _ in range(5):
        _kernel[(3, )](x, out, markers, worker, iters, num_warps=8)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(markers.cpu(), torch.tensor([7, 8], dtype=torch.int32).expand(3, 2), atol=0, rtol=0)
