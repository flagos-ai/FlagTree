"""General dense FA source variants: ownership, tile geometry and split KV.

All variants compute the same noncausal forward, with FP32 state and input
dtype probabilities. There is no vendor FA call or edited LLVM in this module.
"""
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def flex_fwd(Q,K,V,O,L,PART,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
             QB:tl.constexpr,KB:tl.constexpr,VB:tl.constexpr,OB:tl.constexpr,
             QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,OS:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr,OWN:tl.constexpr,SPLITS:tl.constexpr):
    bh=tl.program_id(1);b=bh//HQ;h=bh%HQ;kh=h//(HQ//HK);split=tl.program_id(2)
    row=tl.program_id(0)*BM+tl.arange(0,BM);dim=tl.arange(0,BD);key=tl.arange(0,BN)
    enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
    if OWN==1 or OWN==2:
        row=tle.gpu.set_layout(row,tle.gpu.SlicedEncoding(1,enc))
        dim=tle.gpu.set_layout(dim,tle.gpu.SlicedEncoding(0,enc))
    q=tl.load(Q+b*QB+h*D+row[:,None]*QS+dim[None,:],(row[:,None]<NQ)&(dim[None,:]<D),other=0.)
    m=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
    if OWN==2:
        m=tle.gpu.set_layout(m,tle.gpu.SlicedEncoding(1,enc));den=tle.gpu.set_layout(den,tle.gpu.SlicedEncoding(1,enc));acc=tle.gpu.set_layout(acc,enc)
    total:tl.constexpr=tl.cdiv(NK,BN)
    for it in range(total*split//SPLITS,total*(split+1)//SPLITS):
        col=it*BN+key
        k=tl.load(K+b*KB+kh*D+dim[:,None]+col[None,:]*KS,(dim[:,None]<D)&(col[None,:]<NK),other=0.)
        score=tl.dot(q,k).to(tl.float32)*(D**-.5*1.4426950408889634)
        if OWN==2:score=tle.gpu.set_layout(score,enc)
        score=tl.where(col[None,:]<NK,score,-float('inf'))
        updated=tl.maximum(m,tl.max(score,1));alpha=tl.exp2(m-updated)
        p=tl.exp2(score-updated[:,None]);den=den*alpha+tl.sum(p,1)
        v=tl.load(V+b*VB+kh*D+col[:,None]*VS+dim[None,:],(col[:,None]<NK)&(dim[None,:]<D),other=0.)
        acc=acc*alpha[:,None];acc=tl.dot(p.to(v.dtype),v,acc)
        if OWN==2:acc=tle.gpu.set_layout(acc,enc)
        m=updated
    if SPLITS==1:
        out=acc/den[:,None]
        if OWN==1 or OWN==2:out=tle.gpu.set_layout(out,enc)
        tl.store(O+b*OB+h*D+row[:,None]*OS+dim[None,:],out,(row[:,None]<NQ)&(dim[None,:]<D))
        tl.store(L+(b*HQ+h)*NQ+row,(m+tl.log2(den))*.6931471805599453,row<NQ)
    else:
        # FP32 unnormalized partials and their base-2 max/denominator.
        base=((b*HQ+h)*SPLITS+split)*NQ+row
        tl.store(PART+base[:,None]*(BD+2)+dim[None,:],acc,row[:,None]<NQ)
        tl.store(PART+base*(BD+2)+BD,m,row<NQ)
        tl.store(PART+base*(BD+2)+BD+1,den,row<NQ)

@triton.jit
def merge_kv(P,O,L,NQ:tl.constexpr,HQ:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0);bh=tl.program_id(1);s=tl.arange(0,BS);d=tl.arange(0,BD)
    base=(bh*SPLITS+s)*NQ+row
    m=tl.load(P+base*(BD+2)+BD,s<SPLITS,other=-float('inf'))
    den=tl.load(P+base*(BD+2)+BD+1,s<SPLITS,other=0.)
    mx=tl.max(m,0);a=tl.exp2(m-mx);z=tl.sum(den*a,0)
    acc=tl.load(P+base[:,None]*(BD+2)+d[None,:],s[:,None]<SPLITS,other=0.)
    out=tl.sum(acc*a[:,None],0)/z
    b=bh//HQ;h=bh%HQ
    tl.store(O+((b*NQ+row)*HQ+h)*D+d,out,d<D)
    tl.store(L+bh*NQ+row,(mx+tl.log2(z))*.6931471805599453)

class FlexFAPlan:
    def __init__(self,q,k,v,*,bm=64,bn=64,warps=4,instr=16,own=0,splits=1,sched='default',packed='auto'):
        b,nq,hq,d=q.shape;bk,nk,hk,dk=k.shape
        assert b==bk and d==dk and v.shape==k.shape and hq%hk==0
        assert q.dtype in (torch.float16,torch.bfloat16)and q.dtype==k.dtype==v.dtype
        assert own==0 or warps==4
        assert 1<=splits<=triton.cdiv(nk,bn)
        self.q,self.k,self.v=q,k,v;self.bm,self.bn,self.splits,self.own=bm,bn,splits,own
        self.bd=max(16,triton.next_power_of_2(d));self.output=torch.empty(q.shape,device=q.device,dtype=q.dtype)
        self.lse=torch.empty((b,hq,nq),device=q.device,dtype=torch.float32)
        self.partial=torch.empty((b,hq,splits,nq,self.bd+2),device=q.device,dtype=torch.float32)if splits>1 else self.lse
        self.config=dict(num_warps=warps,num_stages=1,matrix_instr_nonkdim=instr,hcu_sched_model=sched,hcu_packed_f32=packed,mmac_issue_group_size=2,hcu_fold_equivalent_loop_state=True)
    def core(self):
        q,k,v=self.q,self.k,self.v;o=self.output
        self.compiled=flex_fwd[(triton.cdiv(q.shape[1],self.bm),q.shape[0]*q.shape[2],self.splits)](q,k,v,o,self.lse,self.partial,q.shape[1],k.shape[1],q.shape[2],k.shape[2],q.shape[3],q.stride(0),k.stride(0),v.stride(0),o.stride(0),q.stride(1),k.stride(1),v.stride(1),o.stride(1),self.bm,self.bn,self.bd,self.own,self.splits,**self.config)
        if self.splits>1:merge_kv[(q.shape[1],q.shape[0]*q.shape[2])](self.partial,o,self.lse,q.shape[1],q.shape[2],q.shape[3],self.bd,self.splits,triton.next_power_of_2(self.splits),num_warps=4,num_stages=1)
        return o
    def full(self):return self.core()
    def __call__(self):return self.full()
