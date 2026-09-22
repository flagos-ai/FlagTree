"""SQMMA must recover shared allocations captured by an isolated WS worker."""

import pytest
import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource

from test_tle_utils import musa_target, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _marker(markers, ROLE: tl.constexpr):
    tl.store(markers + tl.program_id(0) * 2 + ROLE, ROLE + 31)


@triton.jit
def _consumer(a, b, out, sa, sb, ready, N: tl.constexpr, K: tl.constexpr, TRANS_B: tl.constexpr):
    pid = tl.program_id(0)
    m = tl.arange(0, 64)[:, None]
    k = tl.arange(0, K)[None, :]
    av = tl.load(a + pid * 64 * K + m * K + k)
    if TRANS_B:
        bv = tl.load(b + pid * N * K + tl.arange(0, N)[:, None] * K + k)
    else:
        bv = tl.load(b + pid * N * K + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    tl.store(tle.gpu.local_ptr(sa), av)
    tl.store(tle.gpu.local_ptr(sb), bv)
    tle.gpu.barrier_arrive(ready, phaseIdx=0)
    tle.gpu.barrier_wait(ready, phaseIdx=0)
    acc = tle.gpu.wgmma(sa, sb, tl.zeros((64, N), tl.float32), trans_b=TRANS_B)
    acc = tle.gpu.wgmma_wait(0, acc)
    tl.store(out + pid * 64 * N + m * N + tl.arange(0, N)[None, :], acc)


@triton.jit
def _captured_sqmma(a, b, out, markers, N: tl.constexpr, K: tl.constexpr, TRANS_B: tl.constexpr):
    sa = tle.gpu.alloc((64, K), tl.bfloat16)
    sb = tle.gpu.alloc((N, K) if TRANS_B else (K, N), tl.bfloat16)
    ready = tle.gpu.alloc_barrier(arrive_count=8)
    tle.gpu.warp_specialize(
        [(_marker, (markers, 0)), (_consumer, (a, b, out, sa, sb, ready, N, K, TRANS_B)),
         (_marker, (markers, 1))],
        worker_num_warps=[8, 4], worker_num_regs=[128, 32],
    )


@pytest.mark.parametrize('n,k,trans_b', [(64, 256, True), (128, 64, False)], ids=['qk', 'pv'])
def test_sqmma_worker_capture_compile(n, k, trans_b):
    signature = {'a': '*bf16', 'b': '*bf16', 'out': '*fp32', 'markers': '*i32',
                 'N': 'constexpr', 'K': 'constexpr', 'TRANS_B': 'constexpr'}
    source = ASTSource(_captured_sqmma, signature, constexprs={'N': n, 'K': k, 'TRANS_B': trans_b})
    compiled = triton.compile(source, target=musa_target(), options={'num_warps': 8, 'num_stages': 1})
    assert compiled.metadata.num_warps == 20
    assert 'llvm.musa.sqmma.bfmma.' in compiled.asm['llir']
    assert 'llvm.musa.barrier0' not in compiled.asm['llir']
    assert 'alloca ' not in compiled.asm['llir'], 'phase state must remain promotable at entry'


@pytest.mark.skipif(not hasattr(torch, 'musa') or not torch.musa.is_available(), reason='MUSA device required')
@pytest.mark.parametrize('n,k,trans_b', [(64, 256, True), (128, 64, False)], ids=['qk', 'pv'])
def test_sqmma_worker_capture_runtime(n, k, trans_b):
    torch.manual_seed(31)
    ctas = 3
    a_cpu = torch.randint(-8, 9, (ctas, 64, k)).float().div(8).to(torch.bfloat16)
    b_cpu = torch.randint(-8, 9, (ctas, n, k) if trans_b else (ctas, k, n)).float().div(8).to(torch.bfloat16)
    expected = a_cpu.float() @ (b_cpu.float().transpose(-1, -2) if trans_b else b_cpu.float())
    a, b = a_cpu.to('musa'), b_cpu.to('musa')
    out = torch.empty((ctas, 64, n), dtype=torch.float32, device='musa')
    markers = torch.empty((ctas, 2), dtype=torch.int32, device='musa')
    for _ in range(3):
        out.fill_(float('nan'))
        markers.fill_(-1)
        _captured_sqmma[(ctas,)](a, b, out, markers, n, k, trans_b, num_warps=8, num_stages=1)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(markers.cpu(), torch.tensor([31, 32], dtype=torch.int32).expand(ctas, 2),
                                   rtol=0, atol=0)
