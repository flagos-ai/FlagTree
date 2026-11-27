# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""R031 t3 arithmetic, static four-warp CTA, shared weights across token rows.

The signed helper below is preserved apart from its symbol. The multirow
extension unrolls tokens inside each decoded word, preserving each token's
integer and floating-point operation order and the original butterfly. Paired
The first four FP32 pairs live in uint64 shared slots between K steps;
the fifth pair stays in registers.
this keeps the measured four-warp CTA at 40 registers without ptxas spills.
"""
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
from .iq4_q8_kernels import E, packets_cached_tables as packets
from .iq4_q8_batch_reduce import batch_typed_reduce
from .iq4_q8_persistent_reduce import persistent_typed_reduce as typed_reduce

@triton.jit
def row_ptr(W, byte, row=0):
    return W[0] + byte

@triton.jit
def dot_group_single(W, Q, row, kbx, g, active, K: tl.constexpr, WEIGHT_BASE = 0):
    base = (kbx.to(tl.int32) - WEIGHT_BASE) * 34
    QP = Q.to(tl.pointer_type(tl.uint32))
    I = gpu.set_layout(tl.full((4, 32), 0, tl.int32), E)
    for pair_index in tl.static_range(2):
        pair = gpu.set_layout(tl.load(row_ptr(W, 4 * (base + 2 + 4 * g + 2 * pair_index), row).to(tl.pointer_type(tl.uint64, 1)), active, 0, eviction_policy='evict_first'), E)
        for half in tl.static_range(2):
            j = 2 * pair_index + half
            word = gpu.set_layout((pair >> (32 * half)).to(tl.uint32), E)
            q0 = gpu.set_layout(tl.load(QP + (8 * kbx.to(tl.int32) + g) * 9 + 1 + j, active, 0).to(tl.int32, bitcast=True), E)
            q1 = gpu.set_layout(tl.load(QP + (8 * kbx.to(tl.int32) + g) * 9 + 5 + j, active, 0).to(tl.int32, bitcast=True), E)
            (a, b) = packets(word)
            I = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r', args=[a, q0, I], dtype=tl.int32, is_pure=True, pack=1)
            I = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r', args=[b, q1, I], dtype=tl.int32, is_pure=True, pack=1)
    header = gpu.set_layout(tl.load(row_ptr(W, 4 * base, row).to(tl.pointer_type(tl.uint64, 1)), active, 0, eviction_policy='evict_first'), E)
    h = (header & 0xffffffff).to(tl.uint32)
    s = (header >> 32).to(tl.uint32)
    sg = s >> 8 * (g // 2) + 4 * (g % 2) & 15 | (h >> 16 + 2 * g & 3) << 4
    sg = sg.to(tl.int32) - 32
    dq = gpu.set_layout(tl.load(QP + (8 * kbx.to(tl.int32) + g) * 9, active, 0, eviction_policy='evict_last'), E)
    dw = (h & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    dx = (dq & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    D = tl.inline_asm_elementwise('mul.rn.f32 $0,$1,$2;', constraints='=f,f,f', args=[dw, dx], dtype=tl.float32, is_pure=True, pack=1)
    return (gpu.set_layout((I * sg).to(tl.float32), E), gpu.set_layout(D, E))

@triton.jit
def transport_t3(Q,W,U,Y,Z,N:tl.constexpr,K:tl.constexpr,ROWS:tl.constexpr=1):
    tl.static_assert(N == 17408 and K == 5120 and ROWS >= 1 and ROWS <= 5)
    w = gpu.set_layout(tl.broadcast_to(tl.arange(0,4)[:,None],(4,32)),E)
    l = gpu.set_layout(tl.broadcast_to(tl.arange(0,32)[None,:],(4,32)),E)
    tid = w*32+l
    g = tid%8
    row_index = tl.program_id(0)
    row = gpu.set_layout(tl.full((4,32),row_index,tl.int32),E)
    # Same L2 sectors as the signed producer, issued by warp zero in this CTA.
    # A one-column CTA has no future columns; the signed window is never full.
    for bank in tl.static_range(2):
        ptr = (W if bank == 0 else U)+(row_index.to(tl.int32)*2720).to(tl.int64)+l*128
        tl.inline_asm_elementwise('{ .reg .pred p; setp.ne.u32 p,$1,0; @p prefetch.global.L2 [$2]; mov.u32 $0,0; }',constraints='=r,r,l',args=[(w==0)&(l<22),ptr],dtype=tl.int32,is_pure=False,pack=1)
    if ROWS == 1:
        p = gpu.set_layout(tl.full((4,32),0.,tl.float32),E)
        u = gpu.set_layout(tl.full((4,32),0.,tl.float32),E)
        for it in range(tl.cdiv(K//256,16)):
            kbx = tid//8+16*it
            active = (row<N)&(kbx<K//256)
            J,D = dot_group_single((W+(row_index.to(tl.int32)*2720).to(tl.int64),),Q,row,kbx,g,active,K)
            pn = tl.inline_asm_elementwise('fma.rn.f32 $0,$1,$2,$3;',constraints='=f,f,f,f',args=[D,J,p],dtype=tl.float32,is_pure=True,pack=1)
            p = tl.where(active,pn,p)
            J,D = dot_group_single((U+(row_index.to(tl.int32)*2720).to(tl.int64),),Q,row,kbx,g,active,K)
            un = tl.inline_asm_elementwise('fma.rn.f32 $0,$1,$2,$3;',constraints='=f,f,f,f',args=[D,J,u],dtype=tl.float32,is_pure=True,pack=1)
            u = tl.where(active,un,u)
        p,u = typed_reduce(p,u,w,l,True)
        p = p/(1.+tl.exp(-p))*u
        tl.store(Y+row,p,(row<N)&(w==0)&(l==0))
    else:
        SROWS:tl.constexpr = 4 if ROWS == 5 else ROWS
        S = gpu.alloc((SROWS*128,),tl.uint64,layout=None,nv_mma_shared_layout=False,alignment_bytes=16)
        B = gpu.alloc_barrier(arrive_count=128)
        for token in tl.static_range(SROWS):
            tl.store(gpu.local_ptr(S,(token*128+tid,),shape=(4,32)),tl.full((4,32),0,tl.uint64))
        p5 = gpu.set_layout(tl.full((4,32),0.,tl.float32),E)
        u5 = gpu.set_layout(tl.full((4,32),0.,tl.float32),E)
        for it in range(tl.cdiv(K//256,16)):
            kbx = tid//8+16*it
            active = (row<N)&(kbx<K//256)
            p5,u5 = accumulate_pair(W,U,Q,row,kbx,g,active,S,tid,p5,u5,K,ROWS)
        gpu.barrier_wait(B)
        token = w
        gate,up = batch_typed_reduce(S,token,w,l,SROWS)
        value = gate/(1.+tl.exp(-gate))*up
        tl.store(Y+token*N+row,value,(token<ROWS)&(row<N)&(l==0))
        gpu.barrier_wait(B)
        if ROWS == 5:
            gate,up = typed_reduce(p5,u5,w,l,True)
            value = gate/(1.+tl.exp(-gate))*up
            tl.store(Y+4*N+row,value,(row<N)&(w==0)&(l==0))


@triton.jit
def accumulate_pair(W, U, Q, row, kbx, g, active, S, tid, p5, u5, K: tl.constexpr, ROWS: tl.constexpr):
    base = row.to(tl.int32) * (K // 256) * 34 + kbx.to(tl.int32) * 34
    WP = W.to(tl.pointer_type(tl.uint32))
    UP = U.to(tl.pointer_type(tl.uint32))
    QP = Q.to(tl.pointer_type(tl.uint32))
    qb = (8 * kbx.to(tl.int32) + g) * 9
    I = ()
    H = ()
    for r in tl.static_range(ROWS):
        I += (gpu.set_layout(tl.full((4, 32), 0, tl.int32), E),)
        H += (gpu.set_layout(tl.full((4, 32), 0, tl.int32), E),)
    for pair_index in tl.static_range(2):
        pair = gpu.set_layout(tl.load((WP + base + 2 + 4 * g + 2 * pair_index).to(tl.pointer_type(tl.uint64)), active, 0, eviction_policy='evict_first'), E)
        upair = gpu.set_layout(tl.load((UP + base + 2 + 4 * g + 2 * pair_index).to(tl.pointer_type(tl.uint64)), active, 0, eviction_policy='evict_first'), E)
        for half in tl.static_range(2):
            j = 2 * pair_index + half
            a, b = packets((pair >> (32 * half)).to(tl.uint32))
            c, d = packets((upair >> (32 * half)).to(tl.uint32))
            next_I = ()
            next_H = ()
            for r in tl.static_range(ROWS):
                qr = qb + r * (K // 32) * 9
                q0 = gpu.set_layout(tl.load(QP + qr + 1 + j, active, 0).to(tl.int32, bitcast=True), E)
                q1 = gpu.set_layout(tl.load(QP + qr + 5 + j, active, 0).to(tl.int32, bitcast=True), E)
                Ir = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r', args=[a, q0, I[r]], dtype=tl.int32, is_pure=True, pack=1)
                Hr = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r', args=[c, q0, H[r]], dtype=tl.int32, is_pure=True, pack=1)
                Ir = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r', args=[b, q1, Ir], dtype=tl.int32, is_pure=True, pack=1)
                Hr = tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;', constraints='=r,r,r,r', args=[d, q1, Hr], dtype=tl.int32, is_pure=True, pack=1)
                next_I += (Ir,)
                next_H += (Hr,)
            I = next_I
            H = next_H
    header = gpu.set_layout(tl.load((WP + base).to(tl.pointer_type(tl.uint64)), active, 0, eviction_policy='evict_first'), E)
    uheader = gpu.set_layout(tl.load((UP + base).to(tl.pointer_type(tl.uint64)), active, 0, eviction_policy='evict_first'), E)
    h, s = header.to(tl.uint32), (header >> 32).to(tl.uint32)
    uh, us = uheader.to(tl.uint32), (uheader >> 32).to(tl.uint32)
    sg = (((s >> (8 * (g // 2) + 4 * (g % 2))) & 15) | (((h >> (16 + 2 * g)) & 3) << 4)).to(tl.int32) - 32
    usg = (((us >> (8 * (g // 2) + 4 * (g % 2))) & 15) | (((uh >> (16 + 2 * g)) & 3) << 4)).to(tl.int32) - 32
    dw = (h & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    udw = (uh & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    for r in tl.static_range(ROWS):
        dq = gpu.set_layout(tl.load(QP + qb + r * (K // 32) * 9, active, 0), E)
        dx = (dq & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        J = (I[r] * sg).to(tl.float32)
        UJ = (H[r] * usg).to(tl.float32)
        D = tl.inline_asm_elementwise('mul.rn.f32 $0,$1,$2;', constraints='=f,f,f', args=[dw, dx], dtype=tl.float32, is_pure=True, pack=1)
        UD = tl.inline_asm_elementwise('mul.rn.f32 $0,$1,$2;', constraints='=f,f,f', args=[udw, dx], dtype=tl.float32, is_pure=True, pack=1)
        if r < 4:
            ptr = gpu.local_ptr(S,(r*128+tid,),shape=(4,32))
            packed = tl.load(ptr)
            p = packed.to(tl.uint32).to(tl.float32,bitcast=True)
            u = (packed >> 32).to(tl.uint32).to(tl.float32,bitcast=True)
        else:
            p,u = p5,u5
        pn = tl.inline_asm_elementwise('fma.rn.f32 $0,$1,$2,$3;', constraints='=f,f,f,f', args=[D, J, p], dtype=tl.float32, is_pure=True, pack=1)
        un = tl.inline_asm_elementwise('fma.rn.f32 $0,$1,$2,$3;', constraints='=f,f,f,f', args=[UD, UJ, u], dtype=tl.float32, is_pure=True, pack=1)
        p = tl.where(active,pn,p)
        u = tl.where(active,un,u)
        if r < 4:
            value = p.to(tl.uint32,bitcast=True).to(tl.uint64)|(u.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)
            tl.store(ptr,value)
        else:
            p5,u5 = p,u
    return p5,u5
