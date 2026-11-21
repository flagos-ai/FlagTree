# Copyright 2026 FlagOS Contributors
# Quantization grouping adapted from SageAttention (c) 2024 SageAttention team.
# Licensed under the Apache License, Version 2.0.
"""Our INT8 per-thread QK / FP8 PV Sage contract, generalized over D and GQA.

P is quantized once per 128 keys. FP32 recurrence uses FMA(acc, alpha, PV),
including on alpha==1. Full and masked final tiles retain distinct scale order.
"""
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def q_quant(X,Y,S,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,NS:tl.constexpr):
    b=tl.program_id(2);h=tl.program_id(1);g=tl.program_id(0)*8+tl.arange(0,8);i=tl.arange(0,2*BD)
    n=g[:,None]//8*16+g[:,None]%8+i[None,:]//BD*8;d=i%BD
    x=tl.load(X+((b*N+n)*H+h)*D+d[None,:],(n<N)&(d[None,:]<D),other=0.).to(tl.float32)
    scale=tl.div_rn(tl.max(tl.abs(x),1),127.)+1e-7
    y=tl.div_rn(x,scale[:,None]);y=y+tl.where(y>=0.,.5,-.5)
    tl.store(Y+((b*N+n)*H+h)*D+d[None,:],y.to(tl.int8),(n<N)&(d[None,:]<D))
    tl.store(S+(b*H+h)*NS+g,scale,g<NS)

@triton.jit
def k_quant(X,MEAN,Y,S,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,NS:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1);b=tl.program_id(2)
    n=g//4*128+tl.arange(0,16)*8+(g%4)*2;d=tl.arange(0,BD)
    mean=tl.load(MEAN+(b*H+h)*D+d,d<D,other=0.).to(tl.float32)
    a=tl.load(X+((b*N+n[:,None])*H+h)*D+d[None,:],(n[:,None]<N)&(d[None,:]<D),other=0.).to(tl.float32)
    z=tl.load(X+((b*N+n[:,None]+1)*H+h)*D+d[None,:],(n[:,None]+1<N)&(d[None,:]<D),other=0.).to(tl.float32)
    a=tl.where((n[:,None]<N)&(d[None,:]<D),(a-mean[None,:]).to(X.dtype.element_ty).to(tl.float32),0.)
    z=tl.where((n[:,None]+1<N)&(d[None,:]<D),(z-mean[None,:]).to(X.dtype.element_ty).to(tl.float32),0.)
    scale=tl.div_rn(tl.maximum(tl.max(tl.abs(a)),tl.max(tl.abs(z))),127.)+1e-7
    qa=tl.div_rn(a,scale);qz=tl.div_rn(z,scale)
    qa=qa+tl.where(qa>=0.,.5,-.5);qz=qz+tl.where(qz>=0.,.5,-.5)
    tl.store(Y+((b*N+n[:,None])*H+h)*D+d[None,:],qa.to(tl.int8),(n[:,None]<N)&(d[None,:]<D))
    tl.store(Y+((b*N+n[:,None]+1)*H+h)*D+d[None,:],qz.to(tl.int8),(n[:,None]+1<N)&(d[None,:]<D))
    tl.store(S+(b*H+h)*NS+g,scale)

@triton.jit
def v_scales(MAX,S,REC,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);x=tl.load(MAX+i,i<COUNT,other=0.).to(tl.float32)
    s=tl.where(x>0.,tl.div_rn(x,448.),1.);r=tl.where(x>0.,tl.div_rn(448.,x),1.)
    tl.store(S+i,s,i<COUNT);tl.store(REC+i,r,i<COUNT)

@triton.jit
def v_quant(X,REC,Y,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);d=i%D;h=i//D%H;b=i//(D*H*N)
    x=tl.load(X+i,i<COUNT,other=0.).to(tl.float32);r=tl.load(REC+(b*H+h)*D+d,i<COUNT,other=1.)
    tl.store(Y+i,(x*r).to(tl.float8e4nv),i<COUNT)

@triton.jit
def sage_tile(q,qs,k,ks,v,m,den,acc,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
              BD:tl.constexpr,BM:tl.constexpr,b,h,kh,row,kb,FULL:tl.constexpr):
    d=tl.arange(0,BD);j=tl.arange(0,128);key=kb*128+j
    qq=tl.load(q+((b*NQ+row[:,None])*HQ+h)*D+d[None,:],(row[:,None]<NQ)&(d[None,:]<D),other=0)
    kk=tl.load(k+((b*NK+key[None,:])*HK+kh)*D+d[:,None],(key[None,:]<NK)&(d[:,None]<D),other=0)
    qi=tl.load(qs+(b*HQ+h)*(tl.cdiv(NQ,64)*32)+row//16*8+row%8,row<NQ,other=0.)
    ki=tl.load(ks+(b*HK+kh)*(tl.cdiv(NK,128)*4)+kb*4+(j%8)//2)
    raw=tl.dot(qq,kk,out_dtype=tl.int32).to(tl.float32)
    scale=qi[:,None]*ki[None,:]
    sm:tl.constexpr=D**-.5*1.4426950408889634
    if FULL:scale=scale*sm
    else:raw=raw*scale;scale=tl.full((BM,128),sm,tl.float32)
    score=tl.fma(raw,scale,-8.807)
    score=tl.where(key[None,:]<NK,score,-float('inf'))
    updated=tl.maximum(m,tl.max(score,1));alpha=tl.exp2(m-updated)
    p=tl.exp2(tl.fma(raw,scale,-updated[:,None]));p=tl.where(key[None,:]<NK,p,0.)
    den=tl.fma(den,alpha,tl.sum(p,1))
    vv=tl.load(v+((b*NK+key[:,None])*HK+kh)*D+d[None,:],(key[:,None]<NK)&(d[None,:]<D),other=0.)
    pv=tl.dot(p.to(tl.float8e4nv),vv,out_dtype=tl.float32,max_num_imprecise_acc=0)
    acc=tl.fma(acc,alpha[:,None],pv)
    return updated,den,acc

@triton.jit
def sage_general(Q,QS,K,KS,V,VS,O,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr):
    b=tl.program_id(2);h=tl.program_id(1);kh=h//(HQ//HK);row=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,BD)
    m=tl.full((BM,),-5000000.,tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
    for kb in range(NK//128):m,den,acc=sage_tile(Q,QS,K,KS,V,m,den,acc,NQ,NK,HQ,HK,D,BD,BM,b,h,kh,row,kb,True)
    if NK%128:m,den,acc=sage_tile(Q,QS,K,KS,V,m,den,acc,NQ,NK,HQ,HK,D,BD,BM,b,h,kh,row,NK//128,False)
    scale=tl.load(VS+(b*HK+kh)*D+d,d<D,other=0.)
    result=(acc/den[:,None])*scale[None,:]
    tl.store(O+((b*NQ+row[:,None])*HQ+h)*D+d[None,:],result,(row[:,None]<NQ)&(d[None,:]<D))

class RawSagePlan:
    """Owned buffers; quantization statistics use the input dtype contract."""
    def __init__(self,q,k,v,*,bm=None,fast_d128=True):
        b,nq,hq,d=q.shape;bk,nk,hk,dk=k.shape
        if min(b,nq,nk,hq,hk)<1 or b!=bk or d!=dk or v.shape!=k.shape or hq%hk or d not in (64,96,128,192,256):raise ValueError('matching dense NHD MHA/GQA and D64/96/128/192/256 required')
        if q.dtype not in (torch.bfloat16,torch.float16)or q.dtype!=k.dtype or q.dtype!=v.dtype:raise ValueError('matching BF16/FP16 required')
        if any(not x.is_cuda or not x.is_contiguous()or x.device!=q.device for x in(q,k,v)):raise ValueError('contiguous GPU NHD inputs required')
        self.q,self.k,self.v=q,k,v;self.bm=bm or (128 if d<=128 else 64);self.bd=triton.next_power_of_2(d);self.fast=fast_d128 and d==128
        self.qi=torch.empty_like(q,dtype=torch.int8);self.ki=torch.empty_like(k,dtype=torch.int8);self.vi=torch.empty_like(v,dtype=torch.float8_e4m3fn)
        self.qs=torch.empty((b,hq,triton.cdiv(nq,64)*32),device=q.device,dtype=torch.float32);self.ks=torch.empty((b,hk,triton.cdiv(nk,128)*4),device=q.device,dtype=torch.float32)
        self.mean=torch.empty((b,1,hk,d),device=q.device,dtype=q.dtype);self.vabs=torch.empty_like(v);self.vmax=torch.empty_like(self.mean)
        self.vs=torch.empty((b,hk,d),device=q.device,dtype=torch.float32);self.vrec=torch.empty_like(self.vs)
        self.output=torch.empty(q.shape,device=q.device,dtype=torch.float32);self.cast_output=torch.empty_like(q)
        if self.fast:self.packed=torch.empty((b*hk*triton.cdiv(nk,128)*16384,),device=q.device,dtype=torch.float8_e4m3fn)
    def quantize(self):
        q,k,v=self.q,self.k,self.v;b,nq,hq,d=q.shape;nk=k.shape[1];hk=k.shape[2]
        torch.mean(k,dim=1,keepdim=True,out=self.mean);torch.abs(v,out=self.vabs);torch.amax(self.vabs,dim=1,keepdim=True,out=self.vmax)
        opts=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
        q_quant[(triton.cdiv(self.qs.shape[-1],8),hq,b)](q,self.qi,self.qs,nq,hq,d,self.bd,self.qs.shape[-1],**opts)
        k_quant[(self.ks.shape[-1],hk,b)](k,self.mean,self.ki,self.ks,nk,hk,d,self.bd,self.ks.shape[-1],**opts)
        v_scales[(triton.cdiv(self.vs.numel(),256),)](self.vmax,self.vs,self.vrec,self.vs.numel(),256,**opts)
        v_quant[(triton.cdiv(v.numel(),1024),)](v,self.vrec,self.vi,nk,hk,d,v.numel(),1024,**opts)
    def pack(self):
        if self.fast:
            from .paired_layout_safe import pack_v_paired_gpu
            pack_v_paired_gpu(self.vi,self.vi.shape,self.packed)
    def core(self):
        q,k=self.q,self.k;b,nq,hq,d=q.shape;nk=k.shape[1];hk=k.shape[2]
        opts=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=16,hcu_sched_model='gfx938-v2',hcu_packed_f32='fma',mmac_issue_group_size=2)
        if self.fast:
            from .sage_descriptor_gqa_r54 import sage
            self.compiled=sage[(triton.cdiv(nq,128),hq,b)](self.qi,self.ki,self.packed,self.qs,self.ks,self.vs,self.output,nq,nk,hq,hk,128,True,hcu_descriptor_mls_load=True,**opts)
        else:self.compiled=sage_general[(triton.cdiv(nq,self.bm),hq,b)](self.qi,self.qs,self.ki,self.ks,self.vi,self.vs,self.output,nq,nk,hq,hk,d,self.bd,self.bm,**opts)
        return self.output
    def full(self):self.quantize();self.pack();return self.core()
    def full_cast(self):self.full();self.cast_output.copy_(self.output);return self.cast_output
    def __call__(self):return self.full()
