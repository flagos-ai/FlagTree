"""Explicit SQMMA accumulators may use a different instruction layout."""

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
def _accumulator_layout(a, b, c, out, ACC: tl.constexpr):
    pid = tl.program_id(0)
    sa = tle.gpu.alloc((64, 64), tl.bfloat16)
    sb = tle.gpu.alloc((64, 128), tl.bfloat16)
    COPY: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [4, 8], [8, 1], [1, 0])
    ax = tle.gpu.set_layout(tl.arange(0, 4096).reshape(64, 64), COPY)
    bx = tle.gpu.set_layout((tl.arange(8192, 16384) - 8192).reshape(64, 128), COPY)
    tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(sa), COPY), tl.load(a + pid * 4096 + ax))
    tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(sb), COPY), tl.load(b + pid * 8192 + bx))
    cr = tl.arange(1024, 1088) - 1024
    cn = tl.arange(2048, 2176) - 2048
    acc = tle.gpu.set_layout(tl.load(c + pid * 8192 + cr[:, None] * 128 + cn[None, :]), ACC)
    value = tle.gpu.wgmma(sa, sb, acc)
    value = tle.gpu.wgmma_wait(0, value)
    rr = tl.arange(4096, 4160) - 4096
    rc = tl.arange(8192, 8320) - 8192
    tl.store(out + pid * 8192 + rr[:, None] * 128 + rc[None, :], value)


def _layout(n):
    return tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, n, 64])


@pytest.mark.parametrize('input_n', [64, 128])
def test_sqmma_accumulator_layout_compile(input_n):
    signature = {'a': '*bf16', 'b': '*bf16', 'c': '*fp32', 'out': '*fp32', 'ACC': 'constexpr'}
    source = ASTSource(_accumulator_layout, signature, constexprs={'ACC': _layout(input_n)})
    compiled = triton.compile(source, target=musa_target(), options={'num_warps': 8, 'num_stages': 1})
    assert 'llvm.musa.sqmma.bfmma.m32n128k64.mma' in compiled.asm['llir']


@pytest.mark.skipif(not hasattr(torch, 'musa') or not torch.musa.is_available(), reason='MUSA required')
@pytest.mark.parametrize('input_n', [64, 128])
def test_sqmma_accumulator_layout_runtime(input_n):
    torch.manual_seed(219)
    a_cpu = torch.randint(-8, 9, (3, 64, 64)).float().div(8).to(torch.bfloat16)
    b_cpu = torch.randint(-8, 9, (3, 64, 128)).float().div(8).to(torch.bfloat16)
    c_cpu = torch.randint(-64, 65, (3, 64, 128)).float().div(64)
    expected = a_cpu.float() @ b_cpu.float() + c_cpu
    a, b, c = (x.to('musa') for x in (a_cpu, b_cpu, c_cpu))
    out = torch.empty_like(c)
    for _ in range(3):
        out.fill_(float('nan'))
        _accumulator_layout[(3, )](a, b, c, out, _layout(input_n), num_warps=8, num_stages=1)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
