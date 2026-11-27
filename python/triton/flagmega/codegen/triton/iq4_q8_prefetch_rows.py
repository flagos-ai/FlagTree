# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Single-weight GLU loader: same dp4a/FMA sequence, paired weight reads.

The qualified full payload word offsets fit signed int32 (including logits).
The original four-warp partial reduction and interleaved stores live in the
shared template, outside this helper.
"""
import triton as tr
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
from .iq4_q8_kernels import E, packets_cached_tables

@tr.jit
def accumulate_prefetched_group(W, Q, row, kbx, g, active, acc, K: tl.constexpr, ROWS: tl.constexpr):
    base = row.to(tl.int32) * (K // 256) * 34 + kbx.to(tl.int32) * 34
    WP = W.to(tl.pointer_type(tl.uint32))
    QP = Q.to(tl.pointer_type(tl.uint32))
    qb = (8 * kbx.to(tl.int32) + g) * 9
    I = ()
    for r in tl.static_range(ROWS):
        I += (gpu.set_layout(tl.full((4, 32), 0, tl.int32), E),)
    # Stream one nibble word through every token before decoding the next.
    # This shares the decode without keeping eight decoded packets live.
    for pair_index in tl.static_range(2):
        pair = gpu.set_layout(tl.load((WP + base + 2 + 4 * g + 2 * pair_index).to(tl.pointer_type(tl.uint64)), active, 0, eviction_policy='evict_first'), E)
        for half in tl.static_range(2):
            j = 2 * pair_index + half
            a, b = packets_cached_tables((pair >> (32 * half)).to(tl.uint32))
            next_I = ()
            for r in tl.static_range(ROWS):
                qr = qb + r * (K // 32) * 9
                q0 = gpu.set_layout(tl.load(QP + qr + 1 + j, active, 0).to(tl.int32, bitcast=True), E)
                q1 = gpu.set_layout(tl.load(QP + qr + 5 + j, active, 0).to(tl.int32, bitcast=True), E)
                Ir = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r',
                                              args=[a, q0, I[r]], dtype=tl.int32, is_pure=True, pack=1)
                Ir = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r',
                                              args=[b, q1, Ir], dtype=tl.int32, is_pure=True, pack=1)
                next_I += (Ir,)
            I = next_I
    header = gpu.set_layout(tl.load((WP + base).to(tl.pointer_type(tl.uint64)), active, 0, eviction_policy='evict_first'), E)
    h, s = header.to(tl.uint32), (header >> 32).to(tl.uint32)
    sg = (((s >> (8 * (g // 2) + 4 * (g % 2))) & 15)
          | (((h >> (16 + 2 * g)) & 3) << 4)).to(tl.int32) - 32
    dw = (h & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    result = ()
    for r in tl.static_range(ROWS):
        dq = gpu.set_layout(tl.load(QP + qb + r * (K // 32) * 9, active, 0), E)
        dx = (dq & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        J = (I[r] * sg).to(tl.float32)
        D = tl.inline_asm_elementwise('mul.rn.f32 $0,$1,$2;', constraints='=f,f,f',
                                      args=[dw, dx], dtype=tl.float32, is_pure=True, pack=1)
        pn = tl.inline_asm_elementwise('fma.rn.f32 $0,$1,$2,$3;', constraints='=f,f,f,f',
                                       args=[D, J, acc[r]], dtype=tl.float32, is_pure=True, pack=1)
        result += (gpu.set_layout(tl.where(active, pn, acc[r]), E),)
    return result


