"""The frozen R940 per-thread INT8/FP8 contract, using public TLE layout hints.

BN128 and the final masked tile are kept as quantization boundaries. This is
independent of the fixed-scale, BN64 software-pipe capability probes.
"""
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def step(qq,qscale,k,v,ks,kb,maximum,den,acc,
         NQ:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,BM:tl.constexpr,
         LAYOUT:tl.constexpr,FULL:tl.constexpr):
    d=tl.arange(0,128);j=tl.arange(0,128);key=kb*128+j
    if LAYOUT:
        enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
        d=tle.gpu.set_layout(d,tle.gpu.SlicedEncoding(0,enc))
        j=tle.gpu.set_layout(j,tle.gpu.SlicedEncoding(0,enc))
        key=kb*128+j
    if LAYOUT:
        drow=tle.gpu.set_layout(d,tle.gpu.SlicedEncoding(1,enc))
        keyrow=tle.gpu.set_layout(key,tle.gpu.SlicedEncoding(1,enc))
    else:
        drow=d;keyrow=key
    kdim=drow; kkey=key
    if LAYOUT:
        kenc:tl.constexpr=tle.gpu.BlockEncoding([16,1],[4,16],[1,4],[0,1])
        kdim=tle.gpu.set_layout(drow,tle.gpu.SlicedEncoding(1,kenc))
        kkey=tle.gpu.set_layout(key,tle.gpu.SlicedEncoding(0,kenc))
    if FULL:
        # The public host wrapper proves aligned complete allocations below
        # 2^48; the FULL loop proves the descriptor tile lies in bounds.
        origin=k+kb*128*(H*128)
        tl.assume((origin.to(tl.uint64)>>48)==0)
        tl.assume((origin.to(tl.uint64)&15)==0)
        tl.assume(((origin.to(tl.uint64)+127*(H*128)+127)>>48)==0)
        tl.assume(kb>=0)
        tl.assume(kb*128<=NK-128)
        kd=tl.make_tensor_descriptor(origin,shape=[128,128],strides=[H*128,1],block_shape=[128,128])
        kk=kd.load([0,0]).T
    else:
        kk=tl.load(k+kkey[None,:]*(H*128)+kdim[:,None],kkey[None,:]<NK,other=0)
    rawint=tl.dot(qq,kk,tl.full((BM,128),0x4b400000,tl.int32),out_dtype=tl.int32)
    if LAYOUT:rawint=tle.gpu.set_layout(rawint,enc)
    kscale=tl.load(ks+kb*4+(j%8)//2)
    scale=qscale[:,None]*kscale[None,:]
    if FULL:scale=scale*0.12751743232459868
    else:
        raw=(rawint-0x4b400000).to(tl.float32)
        raw=raw*scale
        scale=tl.full((BM,128),0.12751743232459868,tl.float32)
        if LAYOUT:scale=tle.gpu.set_layout(scale,enc)
    if FULL:
        group_max=tl.max(tl.max(rawint.reshape(BM,8,4,2,2),4),1)
        group_scale=tl.max(tl.max(scale.reshape(BM,8,4,2,2),4),1)
        group_score=tl.fma(group_max.to(tl.float32,bitcast=True)-12582912.,group_scale,-8.807)
        mx=tl.max(tl.max(group_score,2),1)
        updated=tl.maximum(maximum,mx)
    else:
        candidate=tl.fma(raw,scale,-8.807)
        candidate=tl.where(key[None,:]<NK,candidate,-float('inf'))
        updated=tl.maximum(maximum,tl.max(candidate,1))
    if FULL:raw=rawint.to(tl.float32,bitcast=True)-12582912.
    alpha=tl.exp2(maximum-updated)
    p=tl.exp2(tl.fma(raw,scale,-updated[:,None]))
    if not FULL:p=tl.where(key[None,:]<NK,p,0.)
    den=tl.fma(den,alpha,tl.sum(p,1))
    vd=d; vkey=keyrow
    if LAYOUT:
        venc:tl.constexpr=tle.gpu.BlockEncoding([16,1],[4,16],[1,4],[0,1])
        vd=tle.gpu.set_layout(d,tle.gpu.SlicedEncoding(0,venc))
        vkey=tle.gpu.set_layout(keyrow,tle.gpu.SlicedEncoding(1,venc))
    # Input V is logically [key, dim], physically [head, keyTile, dim, keyInTile].
    vorig=v+kb*16384
    tl.assume((vorig.to(tl.uint64)>>48)==0)
    tl.assume((vorig.to(tl.uint64)&15)==0)
    tl.assume(((vorig.to(tl.uint64)+16383)>>48)==0)
    vdesc=tl.make_tensor_descriptor(vorig,shape=[128,128],strides=[128,1],block_shape=[128,128])
    vv=vdesc.load([0,0]).T
    # phys(k)=32*(k//32)+8*((k%16)//4)+4*((k%32)//16)+k%4.
    # Paired V rows use the same physical order; P quantization is unchanged.
    paired=p.reshape(BM,4,2,4,4).permute(0,1,3,2,4).reshape(BM,128)
    pv=tl.dot(paired.to(tl.float8e4nv),vv,out_dtype=tl.float32,max_num_imprecise_acc=0)
    if LAYOUT:pv=tle.gpu.set_layout(pv,enc)
    acc=tl.fma(acc,alpha[:,None],pv)
    return updated,den,acc

@triton.jit
def sage(q,k,v,qs,ks,vs,out,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
         BM:tl.constexpr=128,LAYOUT:tl.constexpr=True):
    h=tl.program_id(1);kh=h//(HQ//HK);batch=tl.program_id(2)
    r=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,128)
    if LAYOUT:
        enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
        r=tle.gpu.set_layout(r,tle.gpu.SlicedEncoding(1,enc))
        d=tle.gpu.set_layout(d,tle.gpu.SlicedEncoding(0,enc))
    qbase=q+batch*NQ*HQ*128+h*128
    kbase=k+batch*NK*HK*128+kh*128;vbase=v+(batch*HK+kh)*tl.cdiv(NK,128)*16384
    NQS:tl.constexpr=tl.cdiv(NQ,64)*32;NKS:tl.constexpr=tl.cdiv(NK,128)*4
    qscale=tl.load(qs+(batch*HQ+h)*NQS+(r//16)*8+r%8,r<NQ,other=0)
    qq=tl.load(qbase+r[:,None]*(HQ*128)+d[None,:],r[:,None]<NQ,other=0)
    maximum=tl.full((BM,),-5000000.,tl.float32);den=tl.full((BM,),0.,tl.float32)
    acc=tl.full((BM,128),0.,tl.float32)
    if LAYOUT:
        maximum=tle.gpu.set_layout(maximum,tle.gpu.SlicedEncoding(1,enc))
        den=tle.gpu.set_layout(den,tle.gpu.SlicedEncoding(1,enc))
        acc=tle.gpu.set_layout(acc,enc)
    for kb in range(NK//128):
        maximum,den,acc=step(qq,qscale,kbase,vbase,ks+(batch*HK+kh)*NKS,kb,maximum,den,acc,NQ,NK,HK,BM,LAYOUT,True)
    if NK%128:
        maximum,den,acc=step(qq,qscale,kbase,vbase,ks+(batch*HK+kh)*NKS,NK//128,maximum,den,acc,NQ,NK,HK,BM,LAYOUT,False)
    vscale=tl.load(vs+(batch*HK+kh)*128+d)
    result=(acc/den[:,None])*vscale[None,:]
    tl.store(out+(batch*NQ*HQ+r[:,None]*HQ+h)*128+d[None,:],result,r[:,None]<NQ)
