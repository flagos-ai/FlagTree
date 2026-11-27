# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
from copy import deepcopy
import ast
from pathlib import Path
import pytest
from triton.flagmega.codegen.triton.kernel_programs import _q8_program_options
from triton.flagmega.errors import CodegenError


def subject():
    call=dict(family='iq4_q8_projection',variant='r040_down_pair_columns_v1',
              jit_module='triton.flagmega.codegen.triton.iq4_q8_pair_columns',jit_entry='dp4a_pair_columns',
              constexprs=dict(N=5120,K=17408,ROWS=5,MODE=0,COLS=2),grid=[2560,1,1],
              compile_options=dict(num_warps=4,num_ctas=1,num_stages=3,maxnreg=80,
                                   enable_fp_fusion=False,enable_static_shared=True,ptx_options='--minnctapersm=1'))
    launch=dict(program_kind='ordinary',num_warps=4,maxnreg=80,enable_static_shared=True,grid=[2560,1,1])
    return call,launch


def test_qualified_long_pair_retains_80_register_upper_bound():
    call,launch=subject();_q8_program_options([call],launch)
    assert launch['maxnreg']==80
    launch['maxnreg']=56
    with pytest.raises(CodegenError):_q8_program_options([call],launch)


@pytest.mark.parametrize('key,value',[('N',6144),('K',5120),('ROWS',1),('ROWS',4),('COLS',4),('MODE',1)])
def test_pair_rejects_unmeasured_buckets(key,value):
    call,launch=subject();call['constexprs'][key]=value
    with pytest.raises(CodegenError,match='qualified schedule'):_q8_program_options([call],launch)


@pytest.mark.parametrize('mutation',['module','entry','grid','budget'])
def test_pair_preserves_exact_native_contract(mutation):
    call,launch=subject()
    if mutation=='module':call['jit_module']='wrong'
    if mutation=='entry':call['jit_entry']='dp4a_two_columns'
    if mutation=='grid':launch['grid']=[1280,1,1]
    if mutation=='budget':call['compile_options']['maxnreg']=104
    with pytest.raises(CodegenError):_q8_program_options([call],launch)


def test_pair_renderer_has_no_environment_selector():
    from triton.flagmega.codegen.triton import iq4_q8_pair_renderer
    tree=ast.parse(Path(iq4_q8_pair_renderer.__file__).read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('environ','getenv') for n in ast.walk(tree))


@pytest.mark.parametrize('dtype_name',['float32','bfloat16'])
def test_long_pair_preserves_sequential_bits_and_spill_gate(dtype_name):
    import torch
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    from triton.flagmega.codegen.triton.iq4_q8_kernels import q8_prepare,dp4a_m1
    from triton.flagmega.codegen.triton.iq4_q8_pair_columns import dp4a_pair_columns
    torch.manual_seed(40);n,k,rows=5120,17408,5
    x=torch.randn((rows,k),device='cuda',dtype=torch.bfloat16)
    w=torch.randint(0,256,(n,k//256*136),device='cuda',dtype=torch.uint8)
    w.view(torch.float16).reshape(n,-1)[:,::68]=1/256
    q=torch.empty((rows,k//32,36),device='cuda',dtype=torch.uint8)
    q8_prepare[(k//128,)](x,q,K=k,ROWS=rows,num_warps=4)
    dtype=getattr(torch,dtype_name)
    y=torch.full((rows,n),float('nan'),device='cuda',dtype=dtype);ref=torch.empty_like(y)
    opts=dict(num_warps=4,num_stages=3,maxnreg=80,enable_fp_fusion=False,
              enable_static_shared=True,ptx_options='--minnctapersm=1')
    for r in range(rows):dp4a_m1[(n,)](q[r],w,w,ref[r],ref[r],N=n,K=k,ROWS=1,MODE=0,**opts)
    kernel=dp4a_pair_columns[(n//2,)](q,w,w,y,y,N=n,K=k,ROWS=rows,MODE=0,COLS=2,**opts)
    assert bool(torch.isfinite(y).all())
    bits=torch.int32 if dtype_name=='float32' else torch.int16
    assert torch.equal(y.view(bits),ref.view(bits))
    assert kernel.n_spills==0
    assert kernel.metadata.ptxas_spill_store_bytes==0
    assert kernel.metadata.ptxas_spill_load_bytes==0
