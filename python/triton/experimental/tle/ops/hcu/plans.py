# Copyright (c) 2026 FlagOS Contributors. MIT license.
"""Plans allocate once, then launch without CPU tensor reads or allocations.

Each plan owns reusable outputs/workspace; use separate plans for concurrent
calls. Graph replay follows captured addresses; in-place input refresh is valid.
"""
import torch
import triton
from .kernels import split_mm, finish_split, quant_block128


def _signature(x):
    return (tuple(x.shape),tuple(x.stride()),x.dtype,x.device)


def _input(x,dtype):
    if x.ndim!=2 or x.dtype!=dtype or x.device.type!='cuda':
        raise ValueError(f'expected a two-dimensional GPU tensor of dtype {dtype}')
    # The validated buffer path has a 0x7ffffffe-byte resource extent.
    if x.numel() and (sum((n-1)*s for n,s in zip(x.shape,x.stride()))+1)*x.element_size()>0x7ffffffe:
        raise ValueError('tensor address span exceeds the validated buffer extent')


class SmallMBF16GemmPlan:
    """Compute A[M,K] @ B[N,K].T with FP32 reduction and BF16 output.

    Auto Split-K is an empirical gfx938 policy for small-M/large-K. It changes
    FP32 summation order. ``split_k=1`` retains the unsplit reduction path.
    Arbitrary input strides are retained; output is contiguous and plan-owned.
    """
    def __init__(self,a,b,*,split_k=None):
        _input(a,torch.bfloat16);_input(b,torch.bfloat16)
        if a.device!=b.device or a.shape[1]!=b.shape[1]:
            raise ValueError('A/B must share device and K dimension')
        self.a,self.b=a,b
        self.signatures=(_signature(a),_signature(b))
        self.m,self.k=a.shape;self.n=b.shape[0]
        self.bm=16 if self.m<=16 else 32
        self.bn,self.bk=64,64
        if split_k is None:
            split_k=1
            if self.m and self.n and self.m<=256 and self.k>=4096:
                cu=torch.cuda.get_device_properties(a.device).multi_processor_count
                target=(4 if self.m<=16 else 8)*cu
                base=triton.cdiv(self.m,self.bm)*triton.cdiv(self.n,self.bn)
                split_k=min(32,triton.next_power_of_2(triton.cdiv(target,base)),
                            triton.next_power_of_2(triton.cdiv(triton.cdiv(self.k,self.bk),4)))
        if not isinstance(split_k,int) or split_k<1 or split_k>32 or split_k&(split_k-1):
            raise ValueError('split_k must be a power of two in [1,32]')
        self.split_k=split_k
        if self.m*self.n*2>0x7ffffffe:
            raise ValueError('output exceeds the validated buffer extent')
        count=(split_k*self.m*self.n) if split_k>1 else 1
        if count*4>0x7ffffffe:
            raise ValueError('workspace exceeds the validated buffer extent')
        self.output=torch.empty((self.m,self.n),device=a.device,dtype=torch.bfloat16)
        self.workspace=torch.empty((count,),device=a.device,dtype=torch.float32)
        self.last_main=None;self.last_reduce=None

    def __call__(self,a=None,b=None):
        a=self.a if a is None else a;b=self.b if b is None else b
        if (_signature(a),_signature(b))!=self.signatures:
            raise ValueError('input shape, stride, dtype and device must match the plan')
        if not self.m or not self.n:return self.output
        if not self.k:
            self.output.zero_();return self.output
        self.last_main=split_mm[(triton.cdiv(self.m,self.bm),triton.cdiv(self.n,self.bn),self.split_k)](
            a,b,self.workspace,self.output,self.m,self.n,self.k,self.split_k,
            *a.stride(),*b.stride(),self.bm,self.bn,self.bk,
            num_warps=4,num_stages=1,matrix_instr_nonkdim=16)
        if self.split_k>1:
            self.last_reduce=finish_split[(triton.cdiv(self.m*self.n,256),)](
                self.workspace,self.output,self.m*self.n,self.split_k,256,num_warps=4,num_stages=1)
        return self.output


class BlockFP8QuantPlan:
    """BF16 input, block128 RN scaling, E4M3FN output and FP32 scales.

    Output is [rows,ceil(H/128)*128] with zero padded columns. Scale is1 for
    all-zero groups. No reciprocal approximation is substituted for RN division.
    """
    def __init__(self,x):
        _input(x,torch.bfloat16)
        self.x=x;self.signature=_signature(x)
        self.rows,self.width=x.shape;self.groups=triton.cdiv(self.width,128)
        self.rows_per_cta,self.waves=(1,1) if self.rows<8 else (8,4)
        if self.rows*self.groups*128>0x7ffffffe:
            raise ValueError('quantized output exceeds the validated buffer extent')
        self.output=torch.empty((self.rows,self.groups*128),device=x.device,dtype=torch.float8_e4m3fn)
        self.scales=torch.empty((self.rows,self.groups),device=x.device,dtype=torch.float32)
        self.last_kernel=None

    def __call__(self,x=None):
        x=self.x if x is None else x
        if _signature(x)!=self.signature:
            raise ValueError('input shape, stride, dtype and device must match the plan')
        if self.rows and self.width:
            self.last_kernel=quant_block128[(triton.cdiv(self.rows,self.rows_per_cta),self.groups)](
                x,self.output,self.scales,self.rows,self.width,self.groups,*x.stride(),self.rows_per_cta,
                num_warps=self.waves,num_stages=1)
        return self.output,self.scales
