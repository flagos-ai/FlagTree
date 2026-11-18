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
def panel_gqa_fwd(Q,K,V,O,L,KT,VT,PART,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,QB:tl.constexpr,KB:tl.constexpr,VB:tl.constexpr,OB:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,OS:tl.constexpr,VH:tl.constexpr,BM:tl.constexpr,SPLITS:tl.constexpr):
    bh=tl.program_id(1);b=bh//HQ;h=bh%HQ;kh=h//(HQ//HK);split=tl.program_id(2);row=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,128)
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
        result=tle.gpu.set_layout(acc*inverse[:,None],enc)
        tl.store(O+b*OB+h*128+row[:,None]*OS+d[None,:],result,row[:,None]<NQ)
        tl.store(L+(b*HQ+h)*NQ+row,(m*.12751743232459868+tl.log2(den))*.6931471805599453,row<NQ)
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

from .dense_fa_flex_r18 import merge_kv

class PanelGQAPlan:
    def __init__(self,q,k,v,*,splits=1):
        if any(x.ndim!=4 or not x.is_cuda for x in (q,k,v)):raise ValueError('GPU NHD tensors required')
        b,nq,h,d=q.shape;bk,nk,hk,dk=k.shape
        if d!=128 or dk!=128 or b!=bk or v.shape!=k.shape or h%hk or min(b,nq,nk,h,hk)<1:raise ValueError('D128 MHA/GQA required')
        if q.dtype not in (torch.bfloat16,torch.float16)or q.dtype!=k.dtype or q.dtype!=v.dtype:raise ValueError('matching BF16/FP16 required')
        for x in (q,k,v):
            if x.device!=q.device or x.stride(3)!=1 or x.stride(2)!=128 or x.stride(1)<x.shape[2]*128 or x.stride(1)%8:raise ValueError('NHD layout required')
            span=(sum((n-1)*stride for n,stride in zip(x.shape,x.stride()))+1)*x.element_size()
            if x.data_ptr()%16 or x.data_ptr()+span>=2**48:raise ValueError('aligned complete low48 span required')
        if splits<1 or (splits>1 and splits>nk//128):raise ValueError('split requires a nonempty full tile per partition')
        self.q,self.k,self.v=q,k,v;self.splits=splits;self.output=torch.empty(q.shape,device=q.device,dtype=q.dtype)
        self.lse=torch.empty((b,h,nq),device=q.device,dtype=torch.float32)
        self.kpad=torch.empty((b,128,hk,128),device=q.device,dtype=q.dtype);self.packed=torch.empty_like(self.kpad)
        self.partial=torch.empty((b,h,splits,nq,130),device=q.device,dtype=torch.float32)if splits>1 else self.lse
        self.config=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=16,hcu_descriptor_mls_load=True,hcu_sched_model='default',hcu_packed_f32='vector',mmac_issue_group_size=2,hcu_fold_equivalent_loop_state=True,hcu_panel_native_war=True,hcu_panel_pipeline=True,hcu_identity_rescale=True,hcu_panel_compact_k=True,hcu_panel_cross_iteration=True,hcu_panel_unroll2=True)
    def pack(self):
        q,k,v=self.q,self.k,self.v
        if k.shape[1]%128:prepare_tail[(triton.cdiv(128*k.shape[2]*128,2048),q.shape[0])](k,v,self.kpad,self.packed,k.shape[1],k.shape[2],k.stride(0),v.stride(0),k.stride(1),v.stride(1),2048,num_warps=4,num_stages=1)
    def core(self):
        q,k,v=self.q,self.k,self.v;o=self.output
        self.compiled=panel_gqa_fwd[(triton.cdiv(q.shape[1],128),q.shape[0]*q.shape[2],self.splits)](q,k,v,o,self.lse,self.kpad,self.packed,self.partial,q.shape[1],k.shape[1],q.shape[2],k.shape[2],q.stride(0),k.stride(0),v.stride(0),o.stride(0),q.stride(1),k.stride(1),v.stride(1),o.stride(1),128,128,self.splits,**self.config)
        if self.splits>1:merge_kv[(q.shape[1],q.shape[0]*q.shape[2])](self.partial,o,self.lse,q.shape[1],q.shape[2],128,128,self.splits,triton.next_power_of_2(self.splits),num_warps=4,num_stages=1)
        return o
    def full(self):self.pack();return self.core()
    def __call__(self):return self.full()
