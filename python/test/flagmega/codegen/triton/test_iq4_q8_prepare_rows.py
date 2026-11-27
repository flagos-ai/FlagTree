# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
import ast
from pathlib import Path
import pytest
import torch
from triton.flagmega.codegen.triton import iq4_q8_prepare_rows as module
from triton.flagmega.codegen.triton.iq4_q8_kernels import q8_prepare


def test_static_prepare_uses_frozen_quantizer():
    source=Path(module.__file__).read_text()
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('environ','getenv') for n in ast.walk(tree))
    assert 'q8_quantize_lanes(x)' in source


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
@pytest.mark.parametrize('rows',[2,3,4,5])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_exact_q8_record_bytes_across_token_owners(rows,dtype):
    # Different magnitudes/signs in adjacent token rows detect owner/stride bugs;
    # zero and tiny groups exercise the frozen zero and FTZ branches.
    torch.manual_seed(40)
    x=torch.randn((rows,5120),device='cuda',dtype=dtype)
    x[0,:32]=0
    x[-1,32:64]=torch.linspace(-1e-37,1e-37,32,device='cuda').to(dtype)
    x[:,64:96]*=128
    q=torch.full((rows,160,36),0xa5,device='cuda',dtype=torch.uint8)
    ref=torch.full_like(q,0x5a)
    q8_prepare[(40,)](x,ref,K=5120,ROWS=rows,num_warps=4,enable_fp_fusion=False)
    compiled=module.q8_prepare_rows[(rows*40,)](x,q,K=5120,ROWS=rows,num_warps=4,enable_fp_fusion=False)
    assert torch.equal(q,ref)
    assert compiled.n_spills==0
    assert compiled.metadata.ptxas_spill_store_bytes==0
    assert compiled.metadata.ptxas_spill_load_bytes==0


def test_program_rejects_incomplete_token_launch():
    from triton.flagmega.codegen.triton.kernel_programs import _q8_program_options
    from triton.flagmega.errors import CodegenError
    options=dict(num_warps=4,num_ctas=1,num_stages=3,maxnreg=80,
                 enable_fp_fusion=False,enable_static_shared=False,ptx_options='--minnctapersm=1')
    call=dict(family='q8_1_prepare',variant='r040_q8_prepare_rows_v1',jit_entry='q8_prepare_rows',
              jit_module='triton.flagmega.codegen.triton.iq4_q8_prepare_rows',
              constexprs=dict(K=5120,ROWS=5),grid=[200,1,1],compile_options=options)
    launch=dict(program_kind='ordinary',num_warps=4,maxnreg=80,enable_static_shared=False,grid=[40,1,1])
    with pytest.raises(CodegenError,match='every token row'):
        _q8_program_options([call],launch)
    launch['grid']=[200,1,1]
    _q8_program_options([call],launch)
    assert launch['compile_options']==options
    call['jit_entry']='q8_prepare'
    with pytest.raises(CodegenError,match='entry differs'):
        _q8_program_options([call],launch)
    call['jit_entry']='q8_prepare_rows';call['variant']='q8_r003_v2'
    with pytest.raises(CodegenError,match='qualified schedule'):
        _q8_program_options([call],launch)


def test_prepare_templates_keep_versioned_entries_separate():
    import jinja2
    from triton.flagmega.codegen.triton import iq4_q8_renderer,iq4_q8_prepare_renderer
    folder=Path(iq4_q8_renderer.__file__).parent/'kernels/q8_1_prepare'
    for variant,entry in [('q8_r003_v2','q8_prepare'),('r040_q8_prepare_rows_v1','q8_prepare_rows')]:
        source=jinja2.Template((folder/(variant+'.py.jinja')).read_text()).render(render_calls=[dict(
            family='q8_1_prepare',variant=variant,symbol='prepare',signature='X,Q',
            pointers=['X','Q'],constexprs=dict(K=5120,ROWS=5))])
        tree=ast.parse(source)
        imports={n.asname or n.name:n.name for node in tree.body if isinstance(node,ast.ImportFrom) for n in node.names}
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        call=next(n for n in ast.walk(function) if isinstance(n,ast.Call))
        assert imports[call.func.id]==entry
    tree=ast.parse(Path(iq4_q8_prepare_renderer.__file__).read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('environ','getenv') for n in ast.walk(tree))
