# Copyright (c) 2026 FlagOS Contributors. MIT license.
import triton
import triton.language as tl

@triton.jit
def split_mm(A,B,P,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,S:tl.constexpr,
             AS0:tl.constexpr,AS1:tl.constexpr,BS0:tl.constexpr,BS1:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    sk=tl.program_id(2)
    steps:tl.constexpr=triton.cdiv(triton.cdiv(K,BK),S)
    k=sk*steps*BK+tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for t in range(steps):
        kk=k+t*BK
        a=tl.load(A+m[:,None]*AS0+kk[None,:]*AS1,(m[:,None]<M)&(kk[None,:]<K),other=0.)
        b=tl.load(B+n[None,:]*BS0+kk[:,None]*BS1,(n[None,:]<N)&(kk[:,None]<K),other=0.)
        acc=tl.dot(a,b,acc)
    if S==1:
        tl.store(C+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
    else:
        tl.store(P+sk*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def finish_split(P,C,MN:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    s=tl.arange(0,S)
    x=tl.load(P+s[:,None]*MN+i[None,:],i[None,:]<MN,other=0.)
    tl.store(C+i,tl.sum(x,0),i<MN)

@triton.jit
def quant_block128(X,Q,S,ROWS:tl.constexpr,H:tl.constexpr,GR:tl.constexpr,
                   XS0:tl.constexpr,XS1:tl.constexpr,MR:tl.constexpr):
    r=tl.program_id(0)*MR+tl.arange(0,MR)
    g=tl.program_id(1)
    k=tl.arange(0,128)+g*128
    x=tl.load(X+r[:,None]*XS0+k[None,:]*XS1,(r[:,None]<ROWS)&(k[None,:]<H),other=0.).to(tl.float32)
    amax=tl.max(tl.abs(x),1)
    scale=tl.where(amax==0.,1.,tl.div_rn(amax,448.))
    value=tl.div_rn(x,scale[:,None])
    tl.store(Q+r[:,None]*(GR*128)+k[None,:],value.to(tl.float8e4nv),r[:,None]<ROWS)
    tl.store(S+r*GR+g,scale,r<ROWS)
