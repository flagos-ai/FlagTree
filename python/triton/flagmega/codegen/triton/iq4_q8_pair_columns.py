# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Static concurrent output-column schedule; original K partition and arithmetic."""

import triton as tr
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
from triton.flagmega.codegen.triton.iq4_q8_kernels import E, typed_reduce


@tr.constexpr_function
def column_layout(COLS: tl.constexpr):
    return gpu.BlockEncoding([COLS, 1, 1], [1, 1, 32], [1, 4, 1], [2, 1, 0])


@tr.jit
def vector_packets(word, COLS: tl.constexpr):
    word = gpu.set_layout(word, column_layout(COLS))
    t0 = gpu.set_layout(tl.full((COLS, 4, 32), 0xBFAD9881, tl.uint32), column_layout(COLS))
    t1 = gpu.set_layout(tl.full((COLS, 4, 32), 0xF6EADDCF, tl.uint32), column_layout(COLS))
    t2 = gpu.set_layout(tl.full((COLS, 4, 32), 0x26190D01, tl.uint32), column_layout(COLS))
    t3 = gpu.set_layout(tl.full((COLS, 4, 32), 0x71594535, tl.uint32), column_layout(COLS))
    a, b = tl.inline_asm_elementwise(
        """{
.reg .b32 sel,idx,lo,hi,t0,t1;
shr.u32 idx,$2,16;
and.b32 sel,$2,0x88888888; shr.u32 sel,sel,1; or.b32 sel,sel,0x32103210;
and.b32 lo,$2,0x7777; prmt.b32 lo,$3,$4,lo;
and.b32 hi,$2,0x7777; prmt.b32 hi,$5,$6,hi;
prmt.b32 t0,lo,hi,sel;
shr.u32 sel,sel,16;
and.b32 lo,idx,0x7777; prmt.b32 lo,$3,$4,lo;
and.b32 hi,idx,0x7777; prmt.b32 hi,$5,$6,hi;
prmt.b32 t1,lo,hi,sel;
prmt.b32 $0,t0,t1,0x6420; prmt.b32 $1,t0,t1,0x7531;
}""",
        constraints="=r,=r,r,r,r,r,r",
        args=[word, t0, t1, t2, t3],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )
    return gpu.set_layout(a, column_layout(COLS)), gpu.set_layout(b, column_layout(COLS))


@tr.jit
def vector_group(W, Q, row, kbx, g, active, acc, K: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    # Two-column payloads fit in signed 32-bit word offsets. Keep the accepted
    # four-column address calculation for the vocabulary projection.
    if COLS == 2:
        base = row.to(tl.int32) * (K // 256) * 34 + kbx.to(tl.int32) * 34
    else:
        base = row.to(tl.int64) * (K // 256) * 34 + kbx.to(tl.int64) * 34
    WP = W.to(tl.pointer_type(tl.uint32))
    QP = Q.to(tl.pointer_type(tl.uint32))
    if COLS == 2:
        qb = (8 * kbx.to(tl.int32) + g) * 9
    else:
        qb = (8 * kbx.to(tl.int64) + g) * 9
    I = ()
    for r in tl.static_range(ROWS):
        I += (gpu.set_layout(tl.full((COLS, 4, 32), 0, tl.int32), column_layout(COLS)),)
    # Stream one nibble word through every token before decoding the next.
    # This shares the decode without keeping eight decoded packets live.
    for pair_index in tl.static_range(2):
        pair = gpu.set_layout(tl.load((WP + base + 2 + 4*g + 2*pair_index).to(tl.pointer_type(tl.uint64)),active,0,eviction_policy="evict_first"),column_layout(COLS))
        for half in tl.static_range(2):
            j=2*pair_index+half
            word=(pair>>(32*half)).to(tl.uint32)
            a, b = vector_packets(word, COLS)
            next_I = ()
            for r in tl.static_range(ROWS):
                qr = qb + r * (K // 32) * 9
                q0 = gpu.set_layout(tl.load(QP + qr + 1 + j, active, 0).to(tl.int32, bitcast=True), column_layout(COLS))
                q1 = gpu.set_layout(tl.load(QP + qr + 5 + j, active, 0).to(tl.int32, bitcast=True), column_layout(COLS))
                Ir = tl.inline_asm_elementwise(
                    "dp4a.s32.s32 $0,$1,$2,$3;",
                    constraints="=r,r,r,r",
                    args=[a, q0, I[r]],
                    dtype=tl.int32,
                    is_pure=True,
                    pack=1,
                )
                Ir = tl.inline_asm_elementwise(
                    "dp4a.s32.s32 $0,$1,$2,$3;",
                    constraints="=r,r,r,r",
                    args=[b, q1, Ir],
                    dtype=tl.int32,
                    is_pure=True,
                    pack=1,
                )
                next_I += (Ir,)
            I = next_I
    header = gpu.set_layout(tl.load((WP+base).to(tl.pointer_type(tl.uint64)),active,0,eviction_policy="evict_first"),column_layout(COLS))
    h,s = header.to(tl.uint32),(header>>32).to(tl.uint32)
    sg = (((s >> (8 * (g // 2) + 4 * (g % 2))) & 15) | (((h >> (16 + 2 * g)) & 3) << 4)).to(tl.int32) - 32
    dw = (h & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    result = ()
    for r in tl.static_range(ROWS):
        dq = gpu.set_layout(tl.load(QP + qb + r * (K // 32) * 9, active, 0), column_layout(COLS))
        dx = (dq & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        J = (I[r] * sg).to(tl.float32)
        D = tl.inline_asm_elementwise(
            "mul.rn.f32 $0,$1,$2;", constraints="=f,f,f", args=[dw, dx], dtype=tl.float32, is_pure=True, pack=1
        )
        pn = tl.inline_asm_elementwise(
            "fma.rn.f32 $0,$1,$2,$3;",
            constraints="=f,f,f,f",
            args=[D, J, acc[r]],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        result += (gpu.set_layout(tl.where(active, pn, acc[r]), column_layout(COLS)),)
    return result


@tr.jit
def _dp4a_columns_impl(
    Q, W, U, Y, Z, N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr
):
    tl.static_assert((ROWS > 1) & (ROWS <= 5))
    tl.static_assert((COLS == 2) | (COLS == 4))
    tl.static_assert((MODE == 0) | (MODE == 2))
    tl.static_assert(K % 256 == 0)
    tl.static_assert(N * (K // 256) * 34 < 2147483648)
    ce: tl.constexpr = column_layout(COLS)
    c = gpu.set_layout(tl.broadcast_to(tl.arange(0, COLS)[:, None, None], (COLS, 4, 32)), ce)
    w = gpu.set_layout(tl.broadcast_to(tl.arange(0, 4)[None, :, None], (COLS, 4, 32)), ce)
    l = gpu.set_layout(tl.broadcast_to(tl.arange(0, 32)[None, None, :], (COLS, 4, 32)), ce)
    row = tl.program_id(0) * COLS + c
    tid = w * 32 + l
    g = tid % 8
    p = ()
    for r in tl.static_range(ROWS):
        p += (gpu.set_layout(tl.full((COLS, 4, 32), 0.0, tl.float32), ce),)
    for it in range(tl.cdiv(K // 256, 16)):
        kbx = tid // 8 + 16 * it
        active = (row < N) & (kbx < K // 256)
        p = vector_group(W, Q, row, kbx, g, active, p, K, ROWS, COLS)
    w2 = gpu.set_layout(tl.broadcast_to(tl.arange(0, 4)[:, None], (4, 32)), E)
    l2 = gpu.set_layout(tl.broadcast_to(tl.arange(0, 32)[None, :], (4, 32)), E)
    for col in tl.static_range(COLS):
        for r in tl.static_range(ROWS):
            v = gpu.set_layout(tl.gather(p[r], tl.full((1, 4, 32), col, tl.int32), 0).reshape((4, 32)), E)
            v, u = typed_reduce(v, v, w2, l2, False)
            output_col = tl.program_id(0) * COLS + col
            if MODE == 2:
                out = tl.where(output_col % 512 // 256 == 0, Y, Z)
                offset = r * (N // 2) + output_col // 512 * 256 + output_col % 256
            else:
                out = Y
                offset = r * N + output_col
            tl.store(out + tl.full((4, 32), offset, tl.int32), v, (output_col < N) & (w2 == 0) & (l2 == 0))
            tl.debug_barrier()


@tr.jit
def dp4a_pair_columns(Q,W,U,Y,Z,N:tl.constexpr,K:tl.constexpr,MODE:tl.constexpr,ROWS:tl.constexpr,COLS:tl.constexpr):
    _dp4a_columns_impl(Q,W,U,Y,Z,N,K,MODE,ROWS,COLS)
