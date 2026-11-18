"""D256 extension of the validated per-thread INT8 / paired FP8 Sage path.

A single128-key probability quantization is retained. K is consumed as two
D128 integer dots before softmax; V as two output-dimension halves after the
same P conversion. No split-K softmax and no activation requantization.
"""
import torch,triton
import triton.language as tl
import triton.experimental.tle.language as tle
from .sage_general_r54 import RawSagePlan as OriginalPlan,sage_general

@triton.jit
def load_k_half(K,kb,NK:tl.constexpr,H:tl.constexpr,COL:tl.constexpr,FULL:tl.constexpr):
    if FULL:
        origin=K+kb*128*(H*256)+COL
        tl.assume((origin.to(tl.uint64)&15)==0);tl.assume((origin.to(tl.uint64)>>48)==0)
        tl.assume(((origin.to(tl.uint64)+127*(H*256)+127)>>48)==0)
        tl.assume(kb>=0);tl.assume(kb*128<=NK-128)
        kd=tl.make_tensor_descriptor(origin,shape=[128,128],strides=[H*256,1],block_shape=[128,128])
        kk=kd.load([0,0]).T
    else:
        enc:tl.constexpr=tle.gpu.BlockEncoding([16,1],[4,16],[1,4],[0,1])
        dim=tle.gpu.set_layout(tl.arange(0,128),tle.gpu.SlicedEncoding(1,enc))
        key=tle.gpu.set_layout(kb*128+tl.arange(0,128),tle.gpu.SlicedEncoding(0,enc))
        kk=tl.load(K+key[None,:]*(H*256)+COL+dim[:,None],key[None,:]<NK,other=0)
    return kk

@triton.jit
def load_v_half(V,kb,COL:tl.constexpr):
    origin=V+kb*32768+COL*128
    tl.assume((origin.to(tl.uint64)&15)==0);tl.assume((origin.to(tl.uint64)>>48)==0)
    tl.assume(((origin.to(tl.uint64)+16383)>>48)==0)
    td=tl.make_tensor_descriptor(origin,shape=[128,128],strides=[128,1],block_shape=[128,128])
    return td.load([0,0]).T

@triton.jit
def step_wide(q0,q1,qscale,K,V,KS,kb,m,z,a0,a1,NK:tl.constexpr,H:tl.constexpr,BM:tl.constexpr,FULL:tl.constexpr):
    enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
    rawint=tl.dot(q0,load_k_half(K,kb,NK,H,0,FULL),tl.full((BM,128),0x4b400000,tl.int32),out_dtype=tl.int32)
    rawint=tl.dot(q1,load_k_half(K,kb,NK,H,128,FULL),rawint,out_dtype=tl.int32)
    rawint=tle.gpu.set_layout(rawint,enc)
    j=tle.gpu.set_layout(tl.arange(0,128),tle.gpu.SlicedEncoding(0,enc))
    key=kb*128+j;kscale=tl.load(KS+kb*4+(j%8)//2)
    scale=qscale[:,None]*kscale[None,:]
    SM:tl.constexpr=256**-.5*1.4426950408889634
    if FULL:scale=scale*SM
    else:
        raw=(rawint-0x4b400000).to(tl.float32);raw=raw*scale
        scale=tle.gpu.set_layout(tl.full((BM,128),SM,tl.float32),enc)
    if FULL:
        imax=tl.max(tl.max(rawint.reshape(BM,8,4,2,2),4),1)
        smax=tl.max(tl.max(scale.reshape(BM,8,4,2,2),4),1)
        score=tl.fma(imax.to(tl.float32,bitcast=True)-12582912.,smax,-8.807)
        update=tl.maximum(m,tl.max(tl.max(score,2),1))
        raw=rawint.to(tl.float32,bitcast=True)-12582912.
    else:
        score=tl.fma(raw,scale,-8.807);score=tl.where(key[None,:]<NK,score,-float('inf'))
        update=tl.maximum(m,tl.max(score,1))
    alpha=tl.exp2(m-update);p=tl.exp2(tl.fma(raw,scale,-update[:,None]))
    if not FULL:p=tl.where(key[None,:]<NK,p,0.)
    z=tl.fma(z,alpha,tl.sum(p,1))
    paired=p.reshape(BM,4,2,4,4).permute(0,1,3,2,4).reshape(BM,128).to(tl.float8e4nv)
    pv0=tle.gpu.set_layout(tl.dot(paired,load_v_half(V,kb,0),out_dtype=tl.float32,max_num_imprecise_acc=0),enc)
    a0=tl.fma(a0,alpha[:,None],pv0)
    pv1=tle.gpu.set_layout(tl.dot(paired,load_v_half(V,kb,128),out_dtype=tl.float32,max_num_imprecise_acc=0),enc)
    a1=tl.fma(a1,alpha[:,None],pv1)
    return update,z,a0,a1

@triton.jit
def sage_wide(Q,K,V,QS,KS,VS,O,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,BM:tl.constexpr):
    h=tl.program_id(1);kh=h//(HQ//HK);b=tl.program_id(2)
    enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
    row=tle.gpu.set_layout(tl.program_id(0)*BM+tl.arange(0,BM),tle.gpu.SlicedEncoding(1,enc))
    d=tle.gpu.set_layout(tl.arange(0,128),tle.gpu.SlicedEncoding(0,enc))
    qptr=Q+(b*NQ*HQ+h)*256+row[:,None]*(HQ*256)+d[None,:]
    q0=tl.load(qptr,row[:,None]<NQ,other=0);q1=tl.load(qptr+128,row[:,None]<NQ,other=0)
    nqs:tl.constexpr=tl.cdiv(NQ,64)*32;nks:tl.constexpr=tl.cdiv(NK,128)*4
    qs=tl.load(QS+(b*HQ+h)*nqs+row//16*8+row%8,row<NQ,other=0.)
    k=K+(b*NK*HK+kh)*256;v=V+(b*HK+kh)*tl.cdiv(NK,128)*32768;ks=KS+(b*HK+kh)*nks
    m=tle.gpu.set_layout(tl.full((BM,),-5000000.,tl.float32),tle.gpu.SlicedEncoding(1,enc))
    z=tle.gpu.set_layout(tl.zeros((BM,),tl.float32),tle.gpu.SlicedEncoding(1,enc))
    a0=tle.gpu.set_layout(tl.zeros((BM,128),tl.float32),enc);a1=tle.gpu.set_layout(tl.zeros((BM,128),tl.float32),enc)
    for kb in range(NK//128):m,z,a0,a1=step_wide(q0,q1,qs,k,v,ks,kb,m,z,a0,a1,NK,HK,BM,True)
    if NK%128:m,z,a0,a1=step_wide(q0,q1,qs,k,v,ks,NK//128,m,z,a0,a1,NK,HK,BM,False)
    s0=tl.load(VS+(b*HK+kh)*256+d);s1=tl.load(VS+(b*HK+kh)*256+128+d)
    ptr=O+((b*NQ+row[:,None])*HQ+h)*256+d[None,:]
    tl.store(ptr,(a0/z[:,None])*s0[None,:],row[:,None]<NQ)
    tl.store(ptr+128,(a1/z[:,None])*s1[None,:],row[:,None]<NQ)

@triton.jit
def pack_wide(X,Y,NK:tl.constexpr,H:tl.constexpr):
    bh=tl.program_id(2);h=bh%H;b=bh//H
    row=tl.program_id(0)*64+tl.arange(0,64);d=tl.program_id(1)*64+tl.arange(0,64)
    val=tl.load(X+((b*NK+row[:,None])*H+h)*256+d[None,:],row[:,None]<NK,other=0)
    k=row%128;physical=k//32*32+(k%16//4)*8+(k%32//16)*4+k%4
    dst=((bh*tl.cdiv(NK,128)+row[:,None]//128)*256+d[None,:])*128+physical[:,None]
    tl.store(Y+dst,val)

class RawSagePlan(OriginalPlan):
    def __init__(self,q,k,v,*,bm=None,wide=True,priority=None):
        super().__init__(q,k,v,bm=bm)
        self.wide=bool(wide and q.shape[-1]==256);self.priority=priority
        if self.wide:
            self.bm=bm or 64
            self.packed=torch.empty((q.shape[0]*k.shape[2]*triton.cdiv(k.shape[1],128)*32768,),device=q.device,dtype=torch.float8_e4m3fn)
    def pack(self):
        if self.wide:
            b,n,h,d=self.vi.shape
            pack_wide[(triton.cdiv(n,128)*2,4,b*h)](self.vi.view(torch.uint8),self.packed.view(torch.uint8),n,h,num_warps=4,num_stages=1)
        else:super().pack()
    def core(self):
        q,k=self.q,self.k;b,nq,hq,d=q.shape;nk=k.shape[1];hk=k.shape[2]
        opts=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=16,hcu_sched_model='gfx938-v2',hcu_packed_f32='fma',mmac_issue_group_size=2)
        if self.priority is not None:opts['hcu_mmac_priority']=self.priority
        if self.wide:
            self.compiled=sage_wide[(triton.cdiv(nq,self.bm),hq,b)](self.qi,self.ki,self.packed,self.qs,self.ks,self.vs,self.output,nq,nk,hq,hk,self.bm,hcu_descriptor_mls_load=True,**opts)
        elif self.fast:
            from .sage_descriptor_gqa_r54 import sage
            self.compiled=sage[(triton.cdiv(nq,128),hq,b)](self.qi,self.ki,self.packed,self.qs,self.ks,self.vs,self.output,nq,nk,hq,hk,128,True,hcu_descriptor_mls_load=True,**opts)
        else:self.compiled=sage_general[(triton.cdiv(nq,self.bm),hq,b)](self.qi,self.qs,self.ki,self.ks,self.vi,self.vs,self.output,nq,nk,hq,hk,d,self.bd,self.bm,**opts)
        return self.output
