"""Warp layout of chained dots (S = Q K^T feeding O = P V) on PPU.

AcceleratePPUMatmul lets a dot share the warp layout of the dot it feeds, so
the two can exchange P without a layout conversion. When that layout puts more
warps along M or N than S has 16-wide MMA tiles, warps would compute duplicate
tiles, so S uses its own heuristic instead -- unless that one replicates work
as well, in which case S keeps the shared layout.
"""

import re

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def _attention_tile(q_ptr, k_ptr, v_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    m = tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + m[:, None] * D + d[None, :])
    k = tl.load(k_ptr + n[:, None] * D + d[None, :])
    v = tl.load(v_ptr + n[:, None] * D + d[None, :])
    s = tl.dot(q, tl.trans(k))
    p = tl.exp(s - tl.max(s, 1)[:, None]).to(tl.bfloat16)
    o = tl.dot(p, v)
    tl.store(o_ptr + m[:, None] * D + d[None, :], o)


def _dot_warps(bm, bn, d, num_warps):
    """warpsPerCTA of the S and O dot results in the compiled ttgir."""
    src = triton.compiler.ASTSource(
        fn=_attention_tile,
        signature={
            "q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "o_ptr": "*fp32", "BM": "constexpr", "BN":
            "constexpr", "D": "constexpr"
        },
        constexprs={"BM": bm, "BN": bn, "D": d},
    )
    ttgir = triton.compile(src, target=GPUTarget("cuda", 80, 32), options={"num_warps": num_warps}).asm["ttgir"]
    aliases = {
        name: [int(w)
               for w in warps.split(",")]
        for name, warps in re.findall(r"^#(\w+) = #\S+<\{.*?warpsPerCTA = \[([\d, ]+)\]", ttgir, re.M)
    }
    results = re.findall(r"= tt\.dot .*-> tensor<\w+, #(\w+)>", ttgir)
    assert len(results) == 2, ttgir
    return aliases[results[0]], aliases[results[1]]


def _fits(warps, rows, cols):
    return warps[0] * 16 <= rows and warps[1] * 16 <= cols


def test_s_avoids_replicating_shared_layout():
    # O (64x512) takes [2, 8]; sharing it would put 8 warps on S's 64 columns.
    s, o = _dot_warps(64, 64, 512, 16)
    assert not _fits(o, 64, 64)
    assert s == [4, 4], (s, o)


@pytest.mark.parametrize("bm,bn,d,num_warps", [(64, 16, 64, 8)])
def test_s_keeps_shared_layout_when_no_layout_fits(bm, bn, d, num_warps):
    # S is 64x16: neither O's layout nor S's own fallback fits, so S keeps O's
    # layout and P needs no layout conversion.
    s, o = _dot_warps(bm, bn, d, num_warps)
    assert not _fits(o, bm, bn)
    assert s == o, (s, o)
