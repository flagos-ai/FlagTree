"""Subgroup barrier safety for PPU inter-warp reductions.

Direct inter-warp gather may replace CTA barriers with hardware subgroup
barriers only when upper warp bits index disjoint shared-memory slices.
Broadcast warp bits (e.g. a 64-element 1D reduction on 16 warps, or a 2D
layout with a replicated warp dimension) must fall back to a CTA barrier so
non-representative warps cannot race ahead of representative writers.
"""

import re

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

SUBGROUP_BARRIER_RE = re.compile(r"__ppukernel_barrier_sync_cnt_void\(i32 %")


@triton.jit
def broadcast_1d_reduce_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    # Pre-pollute shared memory via a small reduction, then branch on a 1D sum
    # where warps 2..15 are broadcast copies of warps 0..1.
    offs = tl.arange(0, N)
    dirty = tl.load(x_ptr + offs)
    dirty_sum = tl.sum(dirty, axis=0)
    flag = tl.load(y_ptr + offs)
    if tl.sum(flag, axis=0) > 0:
        tl.store(out_ptr + offs, dirty + dirty_sum)
    else:
        tl.store(out_ptr + offs, tl.zeros((N, ), dtype=tl.int32))


@triton.jit
def broadcast_2d_dot_reduce_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(a_ptr + m[:, None] * K + k[None, :])
    b = tl.load(b_ptr + k[:, None] * N + n[None, :])
    s = tl.dot(a, b).to(tl.float32)
    row_max = tl.max(s, axis=0)
    probs = tl.exp(s - row_max[None, :])
    row_sum = tl.sum(probs, axis=0)
    tl.store(out_ptr + n, row_sum)


@triton.jit
def disjoint_2d_softmax_kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + m * N + n)
    mx = tl.max(x, axis=1)
    ex = tl.exp(x - mx[:, None])
    sm = tl.sum(ex, axis=1)
    tl.store(y_ptr + m * N + n, ex / sm[:, None])


def test_broadcast_warp_1d_reduce_uses_cta_barrier():
    src = triton.compiler.ASTSource(
        fn=broadcast_1d_reduce_kernel,
        signature={"x_ptr": "*i32", "y_ptr": "*i32", "out_ptr": "*i32", "N": "constexpr"},
        constexprs={"N": 64},
    )
    llir = triton.compile(src, target=GPUTarget("cuda", 80, 32), options={"num_warps": 16}).asm["llir"]
    assert not SUBGROUP_BARRIER_RE.search(llir), llir

    if torch.cuda.is_available():
        x = torch.full((64, ), 7, device="cuda", dtype=torch.int32)
        y = torch.zeros((64, ), device="cuda", dtype=torch.int32)
        out = torch.full((64, ), -1, device="cuda", dtype=torch.int32)
        broadcast_1d_reduce_kernel[(1024, )](x, y, out, N=64, num_warps=16)
        torch.cuda.synchronize()
        assert torch.equal(out, torch.zeros_like(out))


def test_broadcast_warp_2d_reduce_uses_cta_barrier():
    src = triton.compiler.ASTSource(
        fn=broadcast_2d_dot_reduce_kernel,
        signature={
            "a_ptr": "*i8",
            "b_ptr": "*i8",
            "out_ptr": "*fp32",
            "M": "constexpr",
            "N": "constexpr",
            "K": "constexpr",
        },
        constexprs={"M": 128, "N": 16, "K": 64},
    )
    compiled = triton.compile(src, target=GPUTarget("cuda", 80, 32), options={"num_warps": 16})
    llir = compiled.asm["llir"]
    assert not SUBGROUP_BARRIER_RE.search(llir), llir

    if torch.cuda.is_available():
        torch.manual_seed(0)
        a = torch.randint(-4, 5, (128, 64), device="cuda", dtype=torch.int8)
        b = torch.randint(-4, 5, (64, 16), device="cuda", dtype=torch.int8)
        out = torch.empty((16, ), device="cuda", dtype=torch.float32)
        broadcast_2d_dot_reduce_kernel[(1, )](a, b, out, M=128, N=16, K=64, num_warps=16)
        torch.cuda.synchronize()
        s = (a.float() @ b.float())
        ref = torch.exp(s - s.max(dim=0, keepdim=True).values).sum(dim=0)
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_disjoint_2d_softmax_uses_subgroup_barriers():
    src = triton.compiler.ASTSource(
        fn=disjoint_2d_softmax_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
        constexprs={"M": 8, "N": 512},
        attrs={(0, ): [["tt.divisibility", 16]], (1, ): [["tt.divisibility", 16]]},
    )
    llir = triton.compile(src, target=GPUTarget("cuda", 80, 32), options={"num_warps": 8}).asm["llir"]
    assert len(SUBGROUP_BARRIER_RE.findall(llir)) >= 3, llir
    assert "barrier.cta.sync.aligned.all" not in llir, llir
    assert "llvm.nvvm.barrier0" not in llir, llir

    if torch.cuda.is_available():
        torch.manual_seed(0)
        x = torch.randn((8, 512), device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        disjoint_2d_softmax_kernel[(1, )](x, y, M=8, N=512, num_warps=8)
        torch.cuda.synchronize()
        ref = torch.softmax(x, dim=-1)
        torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)
