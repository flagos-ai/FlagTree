"""General head dimensions using the verified D32 K / key32 V MLS panels."""
import torch,triton
import triton.language as tl
import triton.experimental.tle.language as tle
from .panel32_compiled_tail_r17 import panel_load,quarters

@triton.jit
def v_panel(P,S:tl.constexpr,C:tl.constexpr):
    a=P.to(tl.uint64);tl.assume((a&15)==0);tl.assume((a>>48)==0)
    tl.assume(((a+31*S*2+C*2-1)>>48)==0)
    desc=tl.make_tensor_descriptor(P,shape=[32,C],strides=[S,1],block_shape=[32,C])
    return desc.load([0,0])

@triton.jit
def panel_step(qs,K,V,m,den,a0,a1,start,offset,NK:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,
               D:tl.constexpr,DP:tl.constexpr,BM:tl.constexpr,VC:tl.constexpr,KP:tl.constexpr,TAIL:tl.constexpr):
    score=tl.dot(qs[0],panel_load(K+offset*KS,KS,64,32)).to(tl.float32)
    for i in tl.static_range(1,KP):score=tl.dot(qs[i],panel_load(K+offset*KS+i*32,KS,64,32),score).to(tl.float32)
    if TAIL:score=tl.where(start+tl.arange(0,64)[None,:]<NK,score,-float('inf'))
    sm:tl.constexpr=D**-.5*1.4426950408889634
    update=tl.maximum(m,tl.max(score,1));alpha=tl.exp2((m-update)*sm)
    p=tl.exp2(score*sm-update[:,None]*sm);den=den*alpha+tl.sum(p,1)
    a0=a0*alpha[:,None]
    if DP==256:a1=a1*alpha[:,None]
    ps=tl.split(p.reshape(BM,2,32).permute(0,2,1))
    for j in tl.static_range(2):
        vv=v_panel(V+(offset+j*32)*VS,VS,VC)
        a0=tl.dot(ps[j].to(qs[0].dtype),vv,a0)
        if DP==256:
            vv=v_panel(V+(offset+j*32)*VS+128,VS,128)
            a1=tl.dot(ps[j].to(qs[0].dtype),vv,a1)
    return update,den,a0,a1

@triton.jit
def panel_dense(Q,K,V,KT,VT,O,L,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                D:tl.constexpr,DP:tl.constexpr,STORED:tl.constexpr,BM:tl.constexpr,VC:tl.constexpr,KP:tl.constexpr):
    b=tl.program_id(2);h=tl.program_id(1);kh=h//(HQ//HK);r=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,DP)
    enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
    r=tle.gpu.set_layout(r,tle.gpu.SlicedEncoding(1,enc));d=tle.gpu.set_layout(d,tle.gpu.SlicedEncoding(0,enc))
    q=tl.load(Q+((b*NQ+r[:,None])*HQ+h)*D+d[None,:],(r[:,None]<NQ)&(d[None,:]<D),other=0.)
    if DP==64:qs=tl.split(q.reshape(BM,2,32).permute(0,2,1))
    elif DP==128:qs=quarters(q,BM)
    else:
        lo,hi=tl.split(q.reshape(BM,2,128).permute(0,2,1));a,bq,c,e=quarters(lo,BM);f,g,i,j=quarters(hi,BM);qs=(a,bq,c,e,f,g,i,j)
    m=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32)
    a0=tl.zeros((BM,VC),tl.float32);a1=tl.zeros((BM,128),tl.float32)
    kp=K+(b*STORED*HK+kh)*DP;vp=V+(b*STORED*HK+kh)*DP
    for start in range(0,NK//64*64,64):m,den,a0,a1=panel_step(qs,kp,vp,m,den,a0,a1,start,start,NK,HK*DP,HK*DP,D,DP,BM,VC,KP,False)
    if NK%64:
        kt=KT+(b*64*HK+kh)*DP;vt=VT+(b*64*HK+kh)*DP
        m,den,a0,a1=panel_step(qs,kt,vt,m,den,a0,a1,NK//64*64,0,NK,HK*DP,HK*DP,D,DP,BM,VC,KP,True)
    inverse=tl.div_rn(tl.full((BM,),1.,tl.float32),den)
    out_enc:tl.constexpr=tle.gpu.BlockEncoding([1,8],[16,4],[4,1],[0,1])
    out_row=tle.gpu.set_layout(tl.program_id(0)*BM+tl.arange(0,BM),tle.gpu.SlicedEncoding(1,out_enc))
    col=tle.gpu.set_layout(tl.arange(0,VC),tle.gpu.SlicedEncoding(0,out_enc))
    result=tle.gpu.set_layout(a0*inverse[:,None],out_enc)
    tl.store(O+((b*NQ+out_row[:,None])*HQ+h)*D+col[None,:],result,(out_row[:,None]<NQ)&(col[None,:]<D))
    if DP==256:
        result=tle.gpu.set_layout(a1*inverse[:,None],out_enc)
        tl.store(O+((b*NQ+out_row[:,None])*HQ+h)*D+128+col[None,:],result,(out_row[:,None]<NQ)&(col[None,:]+128<D))
    lse_value=tle.gpu.set_layout((m*(D**-.5*1.4426950408889634)+tl.log2(den))*.6931471805599453,tle.gpu.SlicedEncoding(1,out_enc))
    tl.store(L+(b*HQ+h)*NQ+out_row,lse_value,out_row<NQ)


@triton.jit
def prepare_kv_dims(K,V,KP,VP,NK:tl.constexpr,NP:tl.constexpr,H:tl.constexpr,D:tl.constexpr,DP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);b=tl.program_id(1);d=i%DP;h=i//DP%H;n=i//(DP*H)
    valid=(n<NK)&(d<D);idx=((b*NK+n)*H+h)*D+d
    k=tl.load(K+idx,valid,other=0.);v=tl.load(V+idx,valid,other=0.)
    tl.store(KP+b*NP*H*DP+i,k,i<NP*H*DP);tl.store(VP+b*NP*H*DP+i,v,i<NP*H*DP)

@triton.jit
def prepare_kv_tail(K,V,KT,VT,NK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,DP:tl.constexpr,STORED:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);b=tl.program_id(1);r=i//(H*DP);c=i%(H*DP);start=NK//64*64
    k=tl.load(K+(b*STORED+start+r)*(H*DP)+c,(r<NK%64)&(i<64*H*DP),other=0.)
    v=tl.load(V+(b*STORED+start+r)*(H*DP)+c,(r<NK%64)&(i<64*H*DP),other=0.)
    tl.store(KT+b*64*H*DP+i,k,i<64*H*DP);tl.store(VT+b*64*H*DP+i,v,i<64*H*DP)

class GeneralPanelPlan:
    def __init__(self,q,k,v,*,bm=None,pipeline=True):
        b,nq,hq,d=q.shape;bk,nk,hk,dk=k.shape
        if d not in(64,96,128,192,256)or b!=bk or d!=dk or v.shape!=k.shape or hq%hk:raise ValueError('D64/96/128/192/256 dense MHA/GQA required')
        if q.dtype not in(torch.bfloat16,torch.float16)or q.dtype!=k.dtype or q.dtype!=v.dtype or any(not x.is_contiguous()for x in(q,k,v)):raise ValueError('matching contiguous BF16/FP16 NHD required')
        self.q,self.k,self.v=q,k,v;self.dp=triton.next_power_of_2(d);self.bm=bm or(64 if d>128 else 128);self.vc=min(self.dp,128)
        self.stored=triton.cdiv(nk,64)*64 if self.dp!=d else nk
        self.kmain=torch.empty((b,self.stored,hk,self.dp),device=q.device,dtype=q.dtype)if self.dp!=d else k
        self.vmain=torch.empty_like(self.kmain)if self.dp!=d else v
        self.ktail=torch.empty((b,64,hk,self.dp),device=q.device,dtype=q.dtype);self.vtail=torch.empty_like(self.ktail)
        self.output=torch.empty_like(q);self.lse=torch.empty((b,hq,nq),device=q.device,dtype=torch.float32)
        for tensor in [self.kmain,self.vmain,self.ktail,self.vtail]:
            if tensor.data_ptr()%16 or tensor.data_ptr()+tensor.numel()*tensor.element_size()>=2**48:raise ValueError('MLS requires an aligned complete low48 span')
        self.kp=4 if d==96 else d//32
        self.config=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=16,hcu_descriptor_mls_load=True,hcu_sched_model='default',hcu_packed_f32='vector',mmac_issue_group_size=2,hcu_fold_equivalent_loop_state=True,hcu_panel_pipeline=pipeline,hcu_panel_native_war=pipeline,hcu_identity_rescale=pipeline,hcu_panel_cross_iteration=pipeline,hcu_panel_compact_k=pipeline,hcu_panel_unroll2=pipeline)
    def pack(self):
        q,k,v=self.q,self.k,self.v;hk=k.shape[2];nk=k.shape[1];b=q.shape[0]
        if self.dp!=q.shape[3]:prepare_kv_dims[(triton.cdiv(self.stored*hk*self.dp,2048),b)](k,v,self.kmain,self.vmain,nk,self.stored,hk,q.shape[3],self.dp,2048,num_warps=4,num_stages=1)
        if nk%64:prepare_kv_tail[(triton.cdiv(64*hk*self.dp,2048),b)](self.kmain,self.vmain,self.ktail,self.vtail,nk,hk,q.shape[3],self.dp,self.stored,2048,num_warps=4,num_stages=1)
    def core(self):
        q,k=self.q,self.k
        self.compiled=panel_dense[(triton.cdiv(q.shape[1],self.bm),q.shape[2],q.shape[0])](q,self.kmain,self.vmain,self.ktail,self.vtail,self.output,self.lse,q.shape[1],k.shape[1],q.shape[2],k.shape[2],q.shape[3],self.dp,self.stored,self.bm,self.vc,self.kp,**self.config)
        return self.output
    def full(self):self.pack();return self.core()
    def __call__(self):return self.full()
