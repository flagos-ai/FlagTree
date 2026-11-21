"""Public same-contract Sage: descriptor chunks and finite V preprocessing.

Q/K rounding, K smoothing, BN128 P quantization and FP32 FMA recurrence are
unchanged. Short V statistics are fused; longer sequences finalize once per
channel and reuse the reciprocal rather than repeating reductions per tile.
Logical vi is always produced. No vendor FA kernel is called.
"""
import torch,triton
import triton.language as tl
from .sage_preprocess_r79 import RawSagePlan as Base,v_partial,v_finalize_pack
from .sage_general_r54 import q_quant,k_quant

@triton.jit
def v_final_stats(P,MAX,VS,REC,H:tl.constexpr,D:tl.constexpr,PARTS:tl.constexpr,NP2:tl.constexpr):
    bh=tl.program_id(1);d=tl.program_id(0)*64+tl.arange(0,64);part=tl.arange(0,NP2)
    a=tl.load(P+(bh*PARTS+part[:,None])*D+d[None,:],(part[:,None]<PARTS)&(d[None,:]<D),other=0.)
    maximum=tl.max(a,0);scale=tl.where(maximum>0.,tl.div_rn(maximum,448.),1.);reciprocal=tl.where(maximum>0.,tl.div_rn(448.,maximum),1.)
    tl.store(MAX+bh*D+d,maximum,d<D);tl.store(VS+bh*D+d,scale,d<D);tl.store(REC+bh*D+d,reciprocal,d<D)

@triton.jit
def v_quant_paired(X,REC,VI,PACKED,NK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,DP:tl.constexpr):
    bh=tl.program_id(2);b=bh//H;h=bh%H
    row=tl.program_id(0)*64+tl.arange(0,64);d=tl.program_id(1)*64+tl.arange(0,64)
    reciprocal=tl.load(REC+bh*D+d,d<D,other=1.)
    x=tl.load(X+((b*NK+row[:,None])*H+h)*D+d[None,:],(row[:,None]<NK)&(d[None,:]<D),other=0.).to(tl.float32)
    y=tl.where((row[:,None]<NK)&(d[None,:]<D),x*reciprocal[None,:],0.).to(tl.float8e4nv)
    tl.store(VI+((b*NK+row[:,None])*H+h)*D+d[None,:],y,(row[:,None]<NK)&(d[None,:]<D))
    k=row%128;physical=k//32*32+(k%16//4)*8+(k%32//16)*4+k%4
    tl.store(PACKED+((bh*tl.cdiv(NK,128)+row[:,None]//128)*DP+d[None,:])*128+physical[:,None],y)

class RawSagePlan(Base):
    def __init__(self,q,k,v,*,bm=None,separate_stats=None):
        super().__init__(q,k,v,bm=bm,stats_rows=512)
        self.separate_stats=k.shape[1]>=16384 if separate_stats is None else separate_stats
    def quantize(self):
        if not self.separate_stats:return super().quantize()
        q,k,v=self.q,self.k,self.v;b,nq,hq,d=q.shape;nk=k.shape[1];hk=k.shape[2]
        torch.mean(k,dim=1,keepdim=True,out=self.mean)
        opts=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
        q_quant[(triton.cdiv(self.qs.shape[-1],8),hq,b)](q,self.qi,self.qs,nq,hq,d,self.bd,self.qs.shape[-1],**opts)
        k_quant[(self.ks.shape[-1],hk,b)](k,self.mean,self.ki,self.ks,nk,hk,d,self.bd,self.ks.shape[-1],**opts)
        v_partial[(self.parts,triton.cdiv(d,64),b*hk)](v,self.vparts,nk,hk,d,self.parts,self.stats_rows,**opts)
        v_final_stats[(triton.cdiv(d,64),b*hk)](self.vparts,self.vmax,self.vs,self.vrec,hk,d,self.parts,triton.next_power_of_2(self.parts),**opts)
        v_quant_paired[(triton.cdiv(nk,128)*2,self.dp//64,b*hk)](v,self.vrec,self.vi,self.packed,nk,hk,d,self.dp,**opts)
