# SPDX-License-Identifier: MIT
"""Draft-only IQ4_XS projection over device-resident vocabulary row IDs."""
import triton as tr
import triton.language as tl

from .iq4_q8_kernels import dp4a_m1


@tr.jit
def candidate_projection(Q, W, IDS, Y, N: tl.constexpr, K: tl.constexpr):
    """Keep the dense M1 arithmetic and reduction; gather only weight rows.

    The dense helper uses program_id(0) for both weight and output indexing.
    Rebase W so that its effective weight row is IDS[program_id(0)], while
    the output remains compact. No weight repacking or vocabulary scan occurs.
    """
    index = tl.program_id(0)
    row = tl.load(IDS + index).to(tl.int64)
    gathered = W + (row - index.to(tl.int64)) * (K // 256) * 136
    dp4a_m1(Q, gathered, gathered, Y, Y, N=N, K=K, MODE=0)


@tr.jit
def candidate_argmax(Y, IDS, PM, PI, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    values = tl.load(Y + i, i < N, other=-float('inf'))
    ids = tl.load(IDS + i, i < N, other=2147483647)
    best = tl.max(values, 0)
    token = tl.min(tl.where(values == best, ids, 2147483647), 0)
    tl.store(PM + tl.program_id(0), best)
    tl.store(PI + tl.program_id(0), token)


@tr.jit
def candidate_argmax_merge(PM, PI, NEXT, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    values = tl.load(PM + i, i < N, other=-float('inf'))
    ids = tl.load(PI + i, i < N, other=2147483647)
    best = tl.max(values, 0)
    token = tl.min(tl.where(values == best, ids, 2147483647), 0)
    tl.store(NEXT, token)
