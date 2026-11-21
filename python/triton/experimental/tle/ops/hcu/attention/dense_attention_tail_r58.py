"""Owned FA plans with source compilation and cached public Triton launches."""
from operator import attrgetter
import triton
import triton.language as tl
from .panel32_compiled_tail_r40 import PanelFAPlan,panel_fwd,prepare_tail
from .panel32_gqa_r22 import PanelGQAPlan,panel_gqa_fwd
from .panel_dense_r58 import GeneralPanelPlan,panel_dense,prepare_kv_dims,prepare_kv_tail,prepare_last_logical
IMPLEMENTATION_MODULES=('triton.experimental.tle.ops.hcu.attention.panel32_compiled_tail_r40', 'triton.experimental.tle.ops.hcu.attention.panel32_gqa_r22', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r58')

@triton.jit
def merge_rows(P,O,L,NQ:tl.constexpr,HQ:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0)*BR+tl.arange(0,BR);bh=tl.program_id(1);s=tl.arange(0,BS);d=tl.arange(0,BD)
    base=(bh*S+s[None,:])*NQ+row[:,None]
    maxima=tl.load(P+base*(BD+2)+BD,(row[:,None]<NQ)&(s[None,:]<S),other=-float('inf'))
    maximum=tl.max(maxima,1);den=tl.zeros((BR,),tl.float32);acc=tl.zeros((BR,BD),tl.float32)
    for split in tl.static_range(S):
        index=(bh*S+split)*NQ+row
        m=tl.load(P+index*(BD+2)+BD,row<NQ,other=0.)
        z=tl.load(P+index*(BD+2)+BD+1,row<NQ,other=0.);a=tl.exp2(m-maximum)
        v=tl.load(P+index[:,None]*(BD+2)+d[None,:],row[:,None]<NQ,other=0.)
        den=den+z*a;acc=acc+v*a[:,None]
    b=bh//HQ;h=bh%HQ
    inverse=tl.div_rn(tl.full((BR,),1.,tl.float32),den)
    tl.store(O+((b*NQ+row[:,None])*HQ+h)*D+d[None,:],acc*inverse[:,None],(row[:,None]<NQ)&(d[None,:]<D))
    tl.store(L+bh*NQ+row,(maximum+tl.log2(den))*.6931471805599453,row<NQ)

class _CachedLaunch:
    def __init__(self,p,fn,grid,names,constants,options):
        self.pointers=attrgetter(*names);self.constants=tuple(constants)
        self.kernel=fn.warmup(*self.pointers(p),*self.constants,grid=grid,**options)
        self.runner=self.kernel[grid]
    def __call__(self,p):self.runner(*self.pointers(p),*self.constants)

class DenseFlashAttentionPlan:
    """Shapes, layouts and compile options belong to the plan; values may change."""
    def __init__(self,q,k,v):
        if any(x.ndim!=4 or not x.is_cuda or not x.is_contiguous()for x in(q,k,v)):raise ValueError('contiguous GPU NHD tensors required')
        if min(*q.shape,*k.shape)<1 or q.shape[0]!=k.shape[0]or q.shape[3]!=k.shape[3]or v.shape!=k.shape or q.shape[2]%k.shape[2]:raise ValueError('matching positive dense MHA/GQA shapes required')
        if any(x.device!=q.device or x.dtype!=q.dtype for x in(k,v)):raise ValueError('matching input devices/dtypes required')
        d=q.shape[3];nq=q.shape[1];nk=k.shape[1]
        if d==128 and nq<=512 and nk>=4096:
            splits=8 if nq==1 or q.shape[2]<=8 else 4
            inner=PanelGQAPlan(q,k,v,splits=min(splits,nk//128))
        elif d==128 and q.shape[2]!=k.shape[2]:inner=PanelGQAPlan(q,k,v)
        elif d==128 and nq>=512:inner=PanelFAPlan(q,k,v)
        else:inner=GeneralPanelPlan(q,k,v,bm=64 if d==128 else None)
        inner.config['hcu_panel_unroll2']=d in(96,128,256)and nk//128>=3 and nq>512
        object.__setattr__(self,'_inner',inner);object.__setattr__(self,'_options',None)
    def __getattr__(self,name):return getattr(self._inner,name)
    def __setattr__(self,name,value):
        if name.startswith('_'):object.__setattr__(self,name,value)
        else:setattr(self._inner,name,value)
    def _build(self):
        p=self._inner
        if self._options==p.config:return
        q,k,v=p.q,p.k,p.v;b,nq,hq,d=q.shape;nk=k.shape[1];hk=k.shape[2]
        launches=[];merge=None;simple=dict(num_warps=4,num_stages=1)
        if isinstance(p,GeneralPanelPlan):
            if p.dp!=d and not p.tail_only:launches.append(_CachedLaunch(p,prepare_kv_dims,(triton.cdiv(p.stored*hk*p.dp,2048),b,1),('k','v','kmain','vmain'),(nk,p.stored,hk,d,p.dp,2048),simple))
            if p.tail_only:launches.append(_CachedLaunch(p,prepare_last_logical,(triton.cdiv(128*hk*p.dp,2048),b,1),('k','v','ktail','vtail'),(nk,hk,d,p.dp,2048),simple))
            if nk%128 and not p.tail_only:launches.append(_CachedLaunch(p,prepare_kv_tail,(triton.cdiv(128*hk*p.dp,2048),b,1),('kmain','vmain','ktail','vtail'),(nk,hk,d,p.dp,p.stored,2048),simple))
            core=_CachedLaunch(p,panel_dense,(triton.cdiv(nq,p.bm),hq,b),('q','kmain','vmain','ktail','vtail','output','lse'),(nq,nk,hq,hk,d,p.dp,p.stored,p.bm,p.vc,p.kp),p.config)
        else:
            if nk%128:launches.append(_CachedLaunch(p,prepare_tail,(triton.cdiv(128*hk*128,2048),b,1),('k','v','kpad','packed'),(nk,hk,k.stride(0),v.stride(0),k.stride(1),v.stride(1),2048),simple))
            opts={key:val for key,val in p.config.items()if key not in('BM','BN')}
            strides=(q.stride(0),k.stride(0),v.stride(0),p.output.stride(0),q.stride(1),k.stride(1),v.stride(1),p.output.stride(1))
            if isinstance(p,PanelGQAPlan):
                core=_CachedLaunch(p,panel_gqa_fwd,(triton.cdiv(nq,128),b*hq,p.splits),('q','k','v','output','lse','kpad','packed','partial'),(nq,nk,hq,hk,*strides,128,128,p.splits),opts)
                if p.splits>1:
                    br=1 if nq<=16 else 8
                    merge=_CachedLaunch(p,merge_rows,(triton.cdiv(nq,br),b*hq,1),('partial','output','lse'),(nq,hq,128,128,p.splits,triton.next_power_of_2(p.splits),br),simple)
            else:core=_CachedLaunch(p,panel_fwd,(triton.cdiv(nq,128),hq,b),('q','k','v','output','lse','kpad','packed'),(nq,nk,hq,*strides,128,128),opts)
        object.__setattr__(self,'_preparation',launches);object.__setattr__(self,'_main',core);object.__setattr__(self,'_merge',merge);object.__setattr__(self,'_options',dict(p.config));p.compiled=core.kernel
    def pack(self):
        self._build()
        for call in self._preparation:call(self._inner)
    def core(self):
        self._build();self._main(self._inner)
        if self._merge is not None:self._merge(self._inner)
        return self.output
    def full(self):self.pack();return self.core()
    def __call__(self):return self.full()

def dense_attention(q,k,v):return DenseFlashAttentionPlan(q,k,v)()
