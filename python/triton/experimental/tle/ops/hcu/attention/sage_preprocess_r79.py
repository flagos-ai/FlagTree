"""Same finite Sage quantization; fuse V statistics and consumer-layout writes.

K mean and Q/K rounded-division quantizers are unchanged. Logical V remains
produced for validation. Max reduction is exact for finite BF16/FP16 inputs.
"""
import torch,triton
import triton.language as tl
from .sage_padded_r77 import RawSagePlan as Base,pad_i8
from .sage_general_r54 import q_quant,k_quant

@triton.jit
def v_partial(X,P,NK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,PARTS:tl.constexpr,ROWS:tl.constexpr):
    bh=tl.program_id(2);b=bh//H;h=bh%H
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS);d=tl.program_id(1)*64+tl.arange(0,64)
    x=tl.load(X+((b*NK+row[:,None])*H+h)*D+d[None,:],(row[:,None]<NK)&(d[None,:]<D),other=0.).to(tl.float32)
    maximum=tl.max(tl.abs(x),0)
    tl.store(P+(bh*PARTS+tl.program_id(0))*D+d,maximum,d<D)

@triton.jit
def v_finalize_pack(X,P,VI,PACKED,MAX,VS,REC,NK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,DP:tl.constexpr,PARTS:tl.constexpr,NP2:tl.constexpr):
    bh=tl.program_id(2);b=bh//H;h=bh%H
    d=tl.program_id(1)*64+tl.arange(0,64);part=tl.arange(0,NP2)
    a=tl.load(P+(bh*PARTS+part[:,None])*D+d[None,:],(part[:,None]<PARTS)&(d[None,:]<D),other=0.)
    maximum=tl.max(a,0)
    scale=tl.where(maximum>0.,tl.div_rn(maximum,448.),1.)
    reciprocal=tl.where(maximum>0.,tl.div_rn(448.,maximum),1.)
    if tl.program_id(0)==0:
        tl.store(MAX+bh*D+d,maximum,d<D);tl.store(VS+bh*D+d,scale,d<D);tl.store(REC+bh*D+d,reciprocal,d<D)
    row=tl.program_id(0)*64+tl.arange(0,64)
    x=tl.load(X+((b*NK+row[:,None])*H+h)*D+d[None,:],(row[:,None]<NK)&(d[None,:]<D),other=0.).to(tl.float32)
    y=tl.where((row[:,None]<NK)&(d[None,:]<D),x*reciprocal[None,:],0.).to(tl.float8e4nv)
    tl.store(VI+((b*NK+row[:,None])*H+h)*D+d[None,:],y,(row[:,None]<NK)&(d[None,:]<D))
    k=row%128;physical=k//32*32+(k%16//4)*8+(k%32//16)*4+k%4
    dst=((bh*tl.cdiv(NK,128)+row[:,None]//128)*DP+d[None,:])*128+physical[:,None]
    tl.store(PACKED+dst,y)

class RawSagePlan(Base):
    def __init__(self,q,k,v,*,bm=None,stats_rows=512):
        if stats_rows not in (256,512):raise ValueError('stats_rows must be256 or512')
        super().__init__(q,k,v,bm=bm)
        self.wide=q.shape[-1]==256;self.priority=None
        self.stats_rows=stats_rows;self.parts=triton.cdiv(k.shape[1],stats_rows)
        self.vparts=torch.empty((q.shape[0]*k.shape[2],self.parts,q.shape[-1]),device=q.device,dtype=torch.float32)
        self.vabs=None
    def quantize(self):
        q,k,v=self.q,self.k,self.v;b,nq,hq,d=q.shape;nk=k.shape[1];hk=k.shape[2]
        torch.mean(k,dim=1,keepdim=True,out=self.mean)
        opts=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
        q_quant[(triton.cdiv(self.qs.shape[-1],8),hq,b)](q,self.qi,self.qs,nq,hq,d,self.bd,self.qs.shape[-1],**opts)
        k_quant[(self.ks.shape[-1],hk,b)](k,self.mean,self.ki,self.ks,nk,hk,d,self.bd,self.ks.shape[-1],**opts)
        v_partial[(self.parts,triton.cdiv(d,64),b*hk)](v,self.vparts,nk,hk,d,self.parts,self.stats_rows,**opts)
        v_finalize_pack[(triton.cdiv(nk,128)*2,self.dp//64,b*hk)](v,self.vparts,self.vi,self.packed,self.vmax,self.vs,self.vrec,nk,hk,d,self.dp,self.parts,triton.next_power_of_2(self.parts),**opts)
    def pack(self):
        if self.pad_needed:
            b,n,h,d=self.vi.shape
            pad_i8[(triton.cdiv(self.qpad.numel(),1024),)](self.qi,self.qpad,self.q.shape[1],self.q.shape[2],d,self.dp,self.qpad.numel(),1024,num_warps=4,num_stages=1)
            pad_i8[(triton.cdiv(self.kpad.numel(),1024),)](self.ki,self.kpad,n,h,d,self.dp,self.kpad.numel(),1024,num_warps=4,num_stages=1)

    def core(self):
        if self.wide:
            from .sage_wide_r66 import RawSagePlan as Wide
            return Wide.core(self)
        return super().core()
