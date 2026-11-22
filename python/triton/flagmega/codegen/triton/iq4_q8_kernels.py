# SPDX-License-Identifier: MIT
# Copyright 2026 FlagOS Contributors
# R003-P0R repository-owned kernel; deliberately not registered for model rollout.
# Arithmetic/mapping translated from the retained P0 typed source / llama.cpp 5ce06fc.
# Normal gpu.alloc/local_ptr lowering owns the shared backing, never inline PTX.
PLAN_SHA256 = "eaa88d099e7d4c9cb33b24793abbaa6e1b42159900c032c40b958ffd1d8bebc4"
NUMERICAL_SUBJECT_SHA256 = "2d0483edc3787d3dab325f3665e205bec952216a146937152300d3fc4a26ccb1"
PARTIAL_ABI = "H003Partial-f32-3x32/v1"
PARTIAL_LAYOUTS = {
    "ordinary": {"payload_bytes": 384, "backing_bytes": 384,
                 "padding_range_bytes": (384, 384), "alignment_bytes": 16,
                 "alias_offset_bytes": 0, "plane_view_offsets_bytes": (0,)},
    "glu": {"payload_bytes": 768, "backing_bytes": 768,
            "padding_range_bytes": (768, 768), "alignment_bytes": 16,
            "alias_offset_bytes": 0, "plane_view_offsets_bytes": (0, 384)},
}
import triton as tr
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.gpu as gpu
E = tl.constexpr(gpu.BlockEncoding([1,1],[1,32],[4,1],[1,0]))

@tr.jit
def packets(word):
    word = gpu.set_layout(word,E)
    a,b = tl.inline_asm_elementwise('''{
.reg .b32 sel,idx,lo,hi,t0,t1;
shr.u32 idx,$2,16;
and.b32 sel,$2,0x88888888; shr.u32 sel,sel,1; or.b32 sel,sel,0x32103210;
and.b32 lo,$2,0x7777; prmt.b32 lo,0xbfad9881,0xf6eaddcf,lo;
and.b32 hi,$2,0x7777; prmt.b32 hi,0x26190d01,0x71594535,hi;
prmt.b32 t0,lo,hi,sel;
shr.u32 sel,sel,16;
and.b32 lo,idx,0x7777; prmt.b32 lo,0xbfad9881,0xf6eaddcf,lo;
and.b32 hi,idx,0x7777; prmt.b32 hi,0x26190d01,0x71594535,hi;
prmt.b32 t1,lo,hi,sel;
prmt.b32 $0,t0,t1,0x6420; prmt.b32 $1,t0,t1,0x7531;
}''', constraints='=r,=r,r',args=[word],dtype=(tl.int32,tl.int32),is_pure=True,pack=1)
    return gpu.set_layout(a,E),gpu.set_layout(b,E)

@tr.jit
def dot_group(W,Q,row,kbx,g,active,K:tl.constexpr):
    base = row.to(tl.int64)*(K//256)*34 + kbx.to(tl.int64)*34
    WP=W.to(tl.pointer_type(tl.uint32))
    QP=Q.to(tl.pointer_type(tl.uint32))
    h=gpu.set_layout(tl.load(WP+base,active,0),E)
    s=gpu.set_layout(tl.load(WP+base+1,active,0),E)
    sg=((s >> (8*(g//2)+4*(g%2)))&15)|(((h>>(16+2*g))&3)<<4)
    sg=sg.to(tl.int32)-32
    dq=gpu.set_layout(tl.load(QP+(8*kbx.to(tl.int64)+g)*9,active,0),E)
    dw=(h&65535).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)
    dx=(dq&65535).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)
    I=tl.full((4,32),0,tl.int32); I=gpu.set_layout(I,E)
    for j in tl.static_range(4):
        word=gpu.set_layout(tl.load(WP+base+2+4*g+j,active,0),E)
        a,b=packets(word)
        q0=gpu.set_layout(tl.load(QP+(8*kbx.to(tl.int64)+g)*9+1+j,active,0).to(tl.int32,bitcast=True),E)
        q1=gpu.set_layout(tl.load(QP+(8*kbx.to(tl.int64)+g)*9+5+j,active,0).to(tl.int32,bitcast=True),E)
        I=tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;',constraints='=r,r,r,r',args=[a,q0,I],dtype=tl.int32,is_pure=True,pack=1)
        I=tl.inline_asm_elementwise('dp4a.s32.s32 $0,$1,$2,$3;',constraints='=r,r,r,r',args=[b,q1,I],dtype=tl.int32,is_pure=True,pack=1)
    D=tl.inline_asm_elementwise('mul.rn.f32 $0,$1,$2;',constraints='=f,f,f',args=[dw,dx],dtype=tl.float32,is_pure=True,pack=1)
    return gpu.set_layout((I*sg).to(tl.float32),E),gpu.set_layout(D,E)

@tr.jit
def partial_store(p, S, w, l, plane: tl.constexpr):
    # Warp0 has no payload store. Its materialized pointer remains in range.
    index = plane * 96 + 32 * tl.maximum(w - 1, 0) + l
    ptr = gpu.set_layout(gpu.local_ptr(S, (index,), shape=(4, 32)), E)
    tl.store(ptr, p, w > 0)


@tr.jit
def partial_read(p, S, w, l, plane: tl.constexpr):
    for t in tl.static_range(3):
        index = plane * 96 + 32 * t + l
        ptr = gpu.set_layout(gpu.local_ptr(S, (index,), shape=(4, 32)), E)
        v = tl.load(ptr, w == 0, 0)
        p = tl.inline_asm_elementwise(
            "add.rn.f32 $0,$1,$2;", constraints="=f,f,f",
            args=[p, v], dtype=tl.float32, is_pure=True, pack=1)
    return gpu.set_layout(p, E)


@tr.jit
def warp0_sum(p, w, l):
    # Uniform warp branch; every lane in warp0 reaches all five full-mask shuffles.
    for off in tl.static_range(5):
        p = tl.inline_asm_elementwise("""{
.reg .pred z; .reg .b32 v;
setp.eq.u32 z,$2,0; @!z bra DONE;
shfl.sync.bfly.b32 v,$1,$3,0x1f,0xffffffff;
add.rn.f32 $0,$1,v; bra END;
DONE: mov.b32 $0,0; END:
}""", constraints="=f,f,r,r", args=[p, w, 16 >> off],
            dtype=tl.float32, is_pure=False, pack=1)
    return gpu.set_layout(tl.where((w == 0) & (l == 0), p, 0.), E)


@tr.jit
def typed_reduce(p, u, w, l, GLU: tl.constexpr):
    # Physical affine arena: exact effective capacity, no compiler padding.
    S = gpu.alloc((192 if GLU else 96,), tl.float32, layout=None,
                  nv_mma_shared_layout=False, alignment_bytes=16)
    partial_store(p, S, w, l, 0)
    if GLU:
        partial_store(u, S, w, l, 1)
    tl.debug_barrier()
    p = partial_read(p, S, w, l, 0)
    if GLU:
        u = partial_read(u, S, w, l, 1)
    p = warp0_sum(p, w, l)
    if GLU:
        u = warp0_sum(u, w, l)
    return p, gpu.set_layout(tl.where((w == 0) & (l == 0), u, 0.), E)


@tr.jit
def q8_quantize_lanes(x):
    # H003-N-v2: pinned SM120a binary arithmetic, not RN division.
    # E assigns one contiguous K32 group to each full warp. Return d/r/v
    # as well as bytes/header so diagnostics can expose the actual stage bits.
    x=gpu.set_layout(x,E)
    a=tl.inline_asm_elementwise('''{
.reg .b32 b; and.b32 b,$1,0x7fffffff;
add.rn.ftz.f32 $0,b,0f80000000;
}''',constraints='=f,f',args=[x],dtype=tl.float32,is_pure=True,pack=1)
    a=gpu.set_layout(a,E)
    for i in tl.static_range(5):
        a=tl.inline_asm_elementwise('''{
.reg .b32 b;
shfl.sync.bfly.b32 b,$1,$2,0x1f,0xffffffff;
max.ftz.f32 $0,$1,b;
}''',constraints='=f,f,r',args=[a,16>>i],dtype=tl.float32,is_pure=False,pack=1)
        a=gpu.set_layout(a,E)
    d,r,v,q=tl.inline_asm_elementwise('''{
.reg .pred zero; .reg .b32 bits,h; .reg .f32 dd,rr,vv,t,f;
.reg .s32 qq;
setp.eq.f32 zero,$4,0f00000000; @zero bra ZERO;
mul.rn.ftz.f32 dd,$4,0f3C010204;
rcp.approx.ftz.f32 rr,dd;
mul.rn.ftz.f32 vv,rr,$5;
mov.b32 bits,vv; and.b32 h,bits,0x80000000; or.b32 h,h,0x3f000000;
add.rz.ftz.f32 t,vv,h;
cvt.rzi.f32.f32 f,t;
// Identity bit transport prevents ptxas from folding FRND + F2I into
// a single conversion and dropping the frozen F2I.FTZ instruction.
shfl.sync.bfly.b32 bits,f,0,0x1f,0xffffffff;
mov.b32 f,bits;
cvt.rzi.ftz.s32.f32 qq,f;
// Sum the sign-extended published byte, including on unsupported inputs.
shl.b32 qq,qq,24; shr.s32 qq,qq,24;
// Publish outputs only after consuming both inputs: no early-clobber alias.
mov.b32 $0,dd; mov.b32 $1,rr; mov.b32 $2,vv; mov.b32 $3,qq;
bra END;
ZERO: mov.b32 $0,0; mov.b32 $1,0; mov.b32 $2,0; mov.b32 $3,0;
END:
}''',constraints='=f,=f,=f,=r,f,f',args=[a,x],
        dtype=(tl.float32,tl.float32,tl.float32,tl.int32),is_pure=False,pack=1)
    d=gpu.set_layout(d,E); r=gpu.set_layout(r,E)
    v=gpu.set_layout(v,E); q=gpu.set_layout(q,E)
    sq=q
    for i in tl.static_range(5):
        sq=tl.inline_asm_elementwise('''{
.reg .b32 b;
shfl.sync.bfly.b32 b,$1,$2,0x1f,0xffffffff;
add.s32 $0,$1,b;
}''',constraints='=r,r,r',args=[sq,16>>i],dtype=tl.int32,is_pure=False,pack=1)
        sq=gpu.set_layout(sq,E)
    # PTX f16x2 takes the high source first: dsum high, original F32 d low.
    # Half conversion is RN-even with gradual underflow and no saturation.
    header=tl.inline_asm_elementwise('''{
.reg .f32 s,ds;
cvt.rn.f32.s32 s,$2;
mul.rn.ftz.f32 ds,$1,s;
cvt.rn.f16x2.f32 $0,ds,$1;
}''',constraints='=r,f,r',args=[d,sq],dtype=tl.uint32,is_pure=True,pack=1)
    return a,d,r,v,q,sq,gpu.set_layout(header,E)


@tr.jit
def q8_prepare(X,Q,K:tl.constexpr):
    w=gpu.set_layout(tl.broadcast_to(tl.arange(0,4)[:,None],(4,32)),E)
    l=gpu.set_layout(tl.broadcast_to(tl.arange(0,32)[None,:],(4,32)),E)
    group=4*tl.program_id(0)+w; k=32*group+l
    x=gpu.set_layout(tl.load(X+k.to(tl.int64),k<K,0).to(tl.float32),E)
    a,d,r,v,q,sq,header=q8_quantize_lanes(x)
    tl.store(Q+36*group.to(tl.int64)+4+l,q.to(tl.uint8),group<tl.cdiv(K,32))
    tl.store(Q.to(tl.pointer_type(tl.uint32))+9*group.to(tl.int64),header,(group<tl.cdiv(K,32))&(l==0))

@tr.jit
def dp4a_m1(Q,W,U,Y,Z,N:tl.constexpr,K:tl.constexpr,MODE:tl.constexpr):
    w=gpu.set_layout(tl.broadcast_to(tl.arange(0,4)[:,None],(4,32)),E)
    l=gpu.set_layout(tl.broadcast_to(tl.arange(0,32)[None,:],(4,32)),E)
    tid=w*32+l; g=tid%8; row=gpu.set_layout(tl.full((4,32),tl.program_id(0),tl.int64),E)
    p=gpu.set_layout(tl.full((4,32),0.,tl.float32),E)
    u=gpu.set_layout(tl.full((4,32),0.,tl.float32),E)
    for it in range(tl.cdiv(K//256,16)):
        kbx=tid//8+16*it; active=(row<N)&(kbx<K//256)
        J,D=dot_group(W,Q,row,kbx,g,active,K)
        pn=tl.inline_asm_elementwise('fma.rn.f32 $0,$1,$2,$3;',constraints='=f,f,f,f',args=[D,J,p],dtype=tl.float32,is_pure=True,pack=1)
        p=tl.where(active,pn,p)
        if MODE==1:
            J,D=dot_group(U,Q,row,kbx,g,active,K)
            un=tl.inline_asm_elementwise('fma.rn.f32 $0,$1,$2,$3;',constraints='=f,f,f,f',args=[D,J,u],dtype=tl.float32,is_pure=True,pack=1)
            u=tl.where(active,un,u)
    tl.static_assert(K % 256 == 0, "R003 requires complete original K256 blocks")
    p,u=typed_reduce(p,u,w,l,MODE==1)
    if MODE==1:
        p=(p/(1.+tl.exp(-p)))*u
    writer=(row<N)&(w==0)&(l==0)
    if MODE==2:
        out=tl.where((row%512)//256==0,Y,Z)
        offset=(row//512)*256+row%256
        tl.store(out+offset,p.to(tl.bfloat16),writer)
    else:
        tl.store(Y+row,p,writer)

