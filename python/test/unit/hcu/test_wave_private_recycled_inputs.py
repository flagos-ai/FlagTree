"""Recycled LDS inputs may elide CTA edges only with equal physical wave owners."""
import re

import pytest
import triton
import triton.language as tl
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@triton.jit
def _recycled_inputs(A, B, C, BM:tl.constexpr, BN:tl.constexpr):
    part=tl.arange(0,4)
    mi=tl.arange(0,BM)
    ni=tl.arange(0,BN)
    ki=part[:,None]*64+tl.arange(0,64)[None,:]
    acc=tl.zeros((4,BM,BN),tl.float32)
    for start in range(0,8):
        kk=start*256+ki
        a=tl.load(A+mi[None,:,None]*2048+kk[:,None,:])
        b=tl.load(B+ni[None,None,:]*2048+kk[:,:,None])
        acc=tl.dot(a,b,acc,allow_tf32=False)
    pairs=tl.reshape(tl.permute(acc,(1,2,0)),(BM*BN,2,2))
    even,odd=tl.split(pairs)
    lo,hi=tl.split(even+odd)
    value=tl.reshape(lo+hi,(BM,BN))
    tl.store(C+mi[:,None]*BN+ni[None,:],value.to(C.dtype.element_ty))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("bm,bn", [(16,16),(32,32),(16,32),(32,16)])
@pytest.mark.parametrize("alignment", [2,16])
def test_recycled_wave_partition_sync(dtype,bm,bn,alignment):
    signature={name:('constexpr'if name in ('BM','BN')else'*'+dtype)
               for name in _recycled_inputs.arg_names}
    source=ASTSource(_recycled_inputs,signature,dict(BM=bm,BN=bn),
                     attrs={(i,):[('tt.divisibility',alignment),('tt.pointer_range',32)]
                            for i in range(3)})
    compiled=triton.compile(source,target=GPUTarget('hip','gfx936',64),
        options=dict(num_warps=4,num_stages=1,kpack=2,mmac_transposed=False,
                     wave_private_sync=True,padded_slot_intervals=True,
                     disable_llvm_loop_unroll=True,loop_unroll_stage='post-pipeline'))
    asm=compiled.asm['amdgcn']
    loop=re.search(r'(?P<label>\.LBB\d+_\d+):[^\n]*Inner Loop.*?s_cbranch_\w+\s+(?P=label)',asm,re.S)
    assert loop, 'test must exercise a real steady loop'
    barriers=loop.group().count('s_barrier')
    # A and B are separate IDs recycled at the same address. Equal plane
    # pitches preserve wave ownership; unequal pitches must retain CTA edges.
    assert (barriers==0)==(bm==bn)
    # Cross-wave output reduction remains synchronized in both cases.
    assert asm.count('s_barrier')>barriers
