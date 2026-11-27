"""D128 MLS panel FA generalized to GQA and bounded split KV.

Only the partial final tile is padded; full tiles read original K/V.
"""
import torch,triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def panel_load(P,STRIDE:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    a=P.to(tl.uint64)
    tl.assume((a&15)==0);tl.assume((a>>48)==0)
    tl.assume(((a+(R-1)*STRIDE*2+C*2-1)>>48)==0)
    td=tl.make_tensor_descriptor(P,shape=[R,C],strides=[STRIDE,1],block_shape=[R,C])
    return td.load([0,0]).T

@triton.jit
def native_v_load(P,STRIDE:tl.constexpr):
    a=P.to(tl.uint64)
    tl.assume((a&15)==0);tl.assume((a>>48)==0)
    tl.assume(((a+31*STRIDE*2+256-1)>>48)==0)
    td=tl.make_tensor_descriptor(P,shape=[32,128],strides=[STRIDE,1],block_shape=[32,128])
    return td.load([0,0])

@triton.jit
def quarters(x,BM:tl.constexpr):
    lo,hi=tl.split(x.reshape(BM,2,64).permute(0,2,1))
    a,b=tl.split(lo.reshape(BM,2,32).permute(0,2,1))
    c,d=tl.split(hi.reshape(BM,2,32).permute(0,2,1))
    return a,b,c,d

@triton.jit
def step128(q0,q1,q2,q3,K,V,m,den,acc,start,offset,NK:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,BM:tl.constexpr,TAIL:tl.constexpr):
    k0=panel_load(K+offset*KS,KS,128,32)
    score=tl.dot(q0,k0).to(tl.float32)
    k1=panel_load(K+offset*KS+32,KS,128,32)
    score=tl.dot(q1,k1,score).to(tl.float32)
    k2=panel_load(K+offset*KS+64,KS,128,32)
    score=tl.dot(q2,k2,score).to(tl.float32)
    k3=panel_load(K+offset*KS+96,KS,128,32)
    score=tl.dot(q3,k3,score).to(tl.float32)
    if TAIL:score=tl.where(start+tl.arange(0,128)[None,:]<NK,score,-float('inf'))
    update=tl.maximum(m,tl.max(score,1));alpha=tl.exp2((m-update)*0.12751743232459868)
    p=tl.exp2(score*0.12751743232459868-update[:,None]*0.12751743232459868)
    den=den*alpha+tl.sum(p,1);acc=acc*alpha[:,None]
    p0,p1,p2,p3=quarters(p,BM)
    v0=native_v_load(V+(offset+0)*VS,VS)
    acc=tl.dot(p0.to(q0.dtype),v0,acc)
    v1=native_v_load(V+(offset+32)*VS,VS)
    acc=tl.dot(p1.to(q0.dtype),v1,acc)
    v2=native_v_load(V+(offset+64)*VS,VS)
    acc=tl.dot(p2.to(q0.dtype),v2,acc)
    v3=native_v_load(V+(offset+96)*VS,VS)
    acc=tl.dot(p3.to(q0.dtype),v3,acc)
    return update,den,acc

@triton.jit
def panel_gqa_fwd(Q,K,V,O,L,KT,VT,PART,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,QB:tl.constexpr,KB:tl.constexpr,VB:tl.constexpr,OB:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,OS:tl.constexpr,VH:tl.constexpr,BM:tl.constexpr,SPLITS:tl.constexpr,STATIC_SPLIT:tl.constexpr,OUTV:tl.constexpr):
    bh=tl.program_id(1);b=bh//HQ;h=bh%HQ;kh=h//(HQ//HK);split=tl.program_id(2);row=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,128)
    if STATIC_SPLIT and SPLITS==1:split=0
    enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
    row=tle.gpu.set_layout(row,tle.gpu.SlicedEncoding(1,enc))
    d=tle.gpu.set_layout(d,tle.gpu.SlicedEncoding(0,enc))
    q=tl.load(Q+b*QB+h*128+row[:,None]*QS+d[None,:],row[:,None]<NQ,other=0.)
    q0,q1,q2,q3=quarters(q,BM)
    m=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,128),tl.float32)
    kp=K+b*KB+kh*128;vp=V+b*VB+kh*VH
    for start in range((NK//128*split//SPLITS)*128,(NK//128*(split+1)//SPLITS)*128,128):m,den,acc=step128(q0,q1,q2,q3,kp,vp,m,den,acc,start,start,NK,KS,VS,BM,False)
    if NK%128 and split==SPLITS-1:
        kt=KT+b*(128*HK*128)+kh*128;vt=VT+b*(128*HK*128)+kh*128
        m,den,acc=step128(q0,q1,q2,q3,kt,vt,m,den,acc,NK//128*128,0,NK,HK*128,HK*128,BM,True)
    if SPLITS==1:
        inverse=tl.div_rn(tl.full((BM,),1.,tl.float32),den)
        out_enc:tl.constexpr=tle.gpu.BlockEncoding([1,OUTV],[16,4],[4,1],[0,1])
        out_row=tle.gpu.set_layout(tl.program_id(0)*BM+tl.arange(0,BM),tle.gpu.SlicedEncoding(1,out_enc))
        out_d=tle.gpu.set_layout(tl.arange(0,128),tle.gpu.SlicedEncoding(0,out_enc))
        result=tle.gpu.set_layout(acc*inverse[:,None],out_enc)
        tl.store(O+b*OB+h*128+out_row[:,None]*OS+out_d[None,:],result,out_row[:,None]<NQ)
        lse_value=tle.gpu.set_layout((m*.12751743232459868+tl.log2(den))*.6931471805599453,tle.gpu.SlicedEncoding(1,out_enc))
        tl.store(L+(b*HQ+h)*NQ+out_row,lse_value,out_row<NQ)
    else:
        base=((b*HQ+h)*SPLITS+split)*NQ+row
        tl.store(PART+base[:,None]*130+d[None,:],acc,row[:,None]<NQ)
        tl.store(PART+base*130+128,m*.12751743232459868,row<NQ)
        tl.store(PART+base*130+129,den,row<NQ)


@triton.jit
def prepare_tail(K,V,KT,VT,NK:tl.constexpr,H:tl.constexpr,KB:tl.constexpr,VB:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);b=tl.program_id(1)
    row=i//(H*128);col=i%(H*128);start=NK//128*128
    k=tl.load(K+b*KB+(start+row)*KS+col,(row<NK%128)&(i<128*H*128),other=0.)
    v=tl.load(V+b*VB+(start+row)*VS+col,(row<NK%128)&(i<128*H*128),other=0.)
    tl.store(KT+b*(128*H*128)+i,k,i<128*H*128)
    tl.store(VT+b*(128*H*128)+i,v,i<128*H*128)

from .panel32_gqa_r22 import PanelGQAPlan
from .dense_attention_r41 import _CachedLaunch
class PrefillPlan(PanelGQAPlan):
    def __init__(self,q,k,v,*,static_split=True,outvec=8):
        super().__init__(q,k,v,splits=1)
        if outvec not in (4,8):raise ValueError('outvec4/8 required')
        self.static_split=static_split;self.outvec=outvec;self._options=None
        self.config['hcu_panel_unroll2']=k.shape[1]//128>=3 and q.shape[1]>512
    def _build(self):
        if self._options==self.config:return
        q,k,v=self.q,self.k,self.v;b,nq,hq,d=q.shape;nk=k.shape[1];hk=k.shape[2]
        self._prepare=None
        if nk%128:
            self._prepare=_CachedLaunch(self,prepare_tail,(triton.cdiv(128*hk*128,2048),b,1),('k','v','kpad','packed'),(nk,hk,k.stride(0),v.stride(0),k.stride(1),v.stride(1),2048),dict(num_warps=4,num_stages=1))
        strides=(q.stride(0),k.stride(0),v.stride(0),self.output.stride(0),q.stride(1),k.stride(1),v.stride(1),self.output.stride(1))
        self._main=_CachedLaunch(self,panel_gqa_fwd,(triton.cdiv(nq,128),b*hq,1),('q','k','v','output','lse','kpad','packed','partial'),(nq,nk,hq,hk,*strides,128,128,1,self.static_split,self.outvec),self.config)
        self.compiled=self._main.kernel;self._options=dict(self.config)
    def pack(self):
        self._build()
        if self._prepare is not None:self._prepare(self)
    def core(self):
        self._build();self._main(self);return self.output
    def full(self):self.pack();return self.core()
