"""Self-written TLE dense noncausal FA for common head dimensions and GQA.

This path uses the same public compiler, without vendor FA or frozen LLVM.
The specialized equal-head D128 path remains a separate measured choice.
"""
import torch,triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def generic_fwd(Q,K,V,O,L,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
                QB:tl.constexpr,KB:tl.constexpr,VB:tl.constexpr,OB:tl.constexpr,
                QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,OS:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    b=tl.program_id(2);h=tl.program_id(1);kh=h//(HQ//HK)
    row=tl.program_id(0)*BM+tl.arange(0,BM);dim=tl.arange(0,BD);key=tl.arange(0,BN)
    enc:tl.constexpr=tle.gpu.BlockEncoding([1,4],[16,4],[4,1],[0,1])
    q=tl.load(Q+b*QB+h*D+row[:,None]*QS+dim[None,:],(row[:,None]<NQ)&(dim[None,:]<D),other=0.)
    m=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
    m=tle.gpu.set_layout(m,tle.gpu.SlicedEncoding(1,enc));den=tle.gpu.set_layout(den,tle.gpu.SlicedEncoding(1,enc));acc=tle.gpu.set_layout(acc,enc)
    for start in range(tl.cdiv(NK,BN)):
        col=start*BN+key
        k=tl.load(K+b*KB+kh*D+dim[:,None]+col[None,:]*KS,(dim[:,None]<D)&(col[None,:]<NK),other=0.)
        score=tl.dot(q,k).to(tl.float32)*(D**-.5*1.4426950408889634)
        score=tle.gpu.set_layout(score,enc)
        score=tl.where(col[None,:]<NK,score,-float('inf'))
        updated=tl.maximum(m,tl.max(score,1));alpha=tl.exp2(m-updated)
        p=tl.exp2(score-updated[:,None]);den=tl.fma(den,alpha,tl.sum(p,1))
        v=tl.load(V+b*VB+kh*D+col[:,None]*VS+dim[None,:],(col[:,None]<NK)&(dim[None,:]<D),other=0.)
        acc=acc*alpha[:,None];acc=tl.dot(p.to(v.dtype),v,acc)
        acc=tle.gpu.set_layout(acc,enc);m=updated
    out=tle.gpu.set_layout(acc/den[:,None],enc)
    tl.store(O+b*OB+h*D+row[:,None]*OS+dim[None,:],out,(row[:,None]<NQ)&(dim[None,:]<D))
    tl.store(L+(b*HQ+h)*NQ+row,(m+tl.log2(den))*.6931471805599453,row<NQ)

class GenericDenseFlashAttentionPlan:
    def __init__(self,q,k,v):
        if any(x.ndim!=4 or not x.is_cuda for x in (q,k,v)):raise ValueError('GPU NHD [B,N,H,D] tensors required')
        b,nq,hq,d=q.shape;bk,nk,hk,dk=k.shape
        if min(b,nq,nk,hq,hk)<1 or b!=bk or d!=dk or v.shape!=k.shape or hq%hk:raise ValueError('matching positive B/N/D and Hq divisible by Hkv required')
        if d<8 or d>256 or d%8:raise ValueError('head dimension must be a multiple of8 in[8,256]')
        if q.dtype not in (torch.bfloat16,torch.float16) or k.dtype!=q.dtype or v.dtype!=q.dtype:raise ValueError('matching BF16 or FP16 inputs required')
        if any(x.device!=q.device or x.stride(3)!=1 or x.stride(2)!=d for x in (q,k,v)):raise ValueError('same device and contiguous head/dimension axes required')
        self.q,self.k,self.v=q,k,v;self.output=torch.empty(q.shape,device=q.device,dtype=q.dtype)
        self.lse=torch.empty((b,hq,nq),device=q.device,dtype=torch.float32)
        self.bm=128 if d<=64 else 64;self.bn=64;self.bd=max(16,triton.next_power_of_2(d))
        self.config=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=16,hcu_packed_f32='vector',hcu_sched_model='default',mmac_issue_group_size=2,hcu_fold_equivalent_loop_state=True)
    def core(self):
        q,k,v=self.q,self.k,self.v;o=self.output
        self.compiled=generic_fwd[(triton.cdiv(q.shape[1],self.bm),q.shape[2],q.shape[0])](q,k,v,o,self.lse,q.shape[1],k.shape[1],q.shape[2],k.shape[2],q.shape[3],q.stride(0),k.stride(0),v.stride(0),o.stride(0),q.stride(1),k.stride(1),v.stride(1),o.stride(1),self.bm,self.bn,self.bd,**self.config)
        return o
    def full(self):return self.core()
    def __call__(self):return self.full()
