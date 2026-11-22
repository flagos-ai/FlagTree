"""Causal registration/address/arithmetic checks for the r000 native half."""
import ast
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent_optimizations.iq4_streaming import native
from agent_optimizations.iq4_streaming.qualify import FMA
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.codegen.triton.renderer import render_triton_package
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.errors import CodegenError


@pytest.mark.parametrize('family',native.FAMILIES)
def test_public_selection_and_typed_encoder(tmp_path,family):
    class Graph(fm.Module):
        def forward(self):
            n=12288 if family.endswith('interleaved') else 16
            x=self.input('x',fm.tensor_type('bfloat16',(1,256)))
            w=self.input('w',fm.tensor_type('uint8',(n,136)))
            scale=self.input('scale',fm.tensor_type('float32',((n+127)//128,2)))
            attrs=dict(weight_block_n=128,weight_block_k=128,weight_encoding='IQ4_XS')
            if family.endswith('interleaved'):
                out=fm.F.nn.interleaved_projection(x,w,scale,feature_shape=(24,256),output_lanes=((8,),(8,)),**attrs)
                outputs=fm.F.tensors.get_items(out,0,1)
            elif family.endswith('glu'):
                attrs.pop('weight_encoding')
                out=fm.F.nn.matmul_glu(x,w,w,scale,scale,activation='silu',gate_weight_encoding='IQ4_XS',up_weight_encoding='IQ4_XS',**attrs)
                outputs=(out,)
            else:
                out=fm.F.math.block_scaled_matmul(x,w,scale,**attrs);outputs=(out,)
            self.function('main',(x,w,scale),outputs)
    compiler=Compiler(CompileOptions(target='nvidia-sm120',codegen_backend='triton'))
    from agent_optimizations.iq4_streaming.graph import configure_target
    compiler.target=configure_target(compiler.target)
    module=compiler.compile(Graph(dialect='high_level',stage='imported',entry='main').build()).module
    descriptor=describe_tir_package(module)
    call=next(c for c in descriptor['render_calls'] if c['family']==family)
    assert call['schema']==native.SCHEMA and call['tile_n']==4 and call['local_n_capacity']==n_for(family)
    assert call['internal_grid_barriers']==0
    render_triton_package(module,tmp_path/'generated')
    source=(tmp_path/'generated/generated_kernels.py').read_text();tree=ast.parse(source)
    helpers={node.name for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}
    helper_calls={node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call)
                  and isinstance(node.func,ast.Name) and node.func.id.startswith('_iq4_')}
    assert helper_calls <= helpers, f"Unbound native helper calls: {helper_calls-helpers}"
    assert 'tl.program_id(0).to(tl.int64) * 4' in source
    assert 'prmt.b32' in source and 'fma.rn.f32' in source
    bad=deepcopy(call);bad['parameters']['tile_n']=2
    with pytest.raises(CodegenError,match='R4'):native.validate_raw(bad)
    bad=deepcopy(call);bad['parameters']['residency_encoding']['threads_per_warp']=[4,8]
    with pytest.raises(CodegenError,match='E32'):native.validate_raw(bad)
    bad=deepcopy(call);bad['parameters']['plan_hash']='r000'
    with pytest.raises(CodegenError,match='E32'):native.validate_raw(bad)


@pytest.mark.parametrize('r',(8,16))
def test_h001_rejects_other_row_tiles(r):
    with pytest.raises(CodegenError,match='R4'):
        native.implementations(row_tile=r)


def n_for(family):return 12288 if family.endswith('interleaved') else 16


def test_virtual_tree_and_fma_cancellation_witnesses():
    # A change to ordinary scalar summation or the other balanced tree fails
    # these exact witnesses. No tolerance masks a different contract.
    leaves=np.array([2**24,0,1,0,-2**24,0,1,0],np.float32)
    p=np.array([np.float32(leaves[i]+leaves[i+1]) for i in range(0,8,2)])
    correct=np.float32(np.float32(p[0]+p[2])+np.float32(p[1]+p[3]))
    changed=np.float32(np.float32(p[0]+p[1])+np.float32(p[2]+p[3]))
    assert correct==2 and changed!=correct
    a=np.float32(1+2**-23);b=np.float32(1-2**-23);c=np.float32(-1)
    contracted=np.float32(FMA(float(a),float(b),float(c)))
    separate=np.float32(np.float32(a*b)+c)
    assert contracted.view(np.uint32)!=separate.view(np.uint32)


@pytest.mark.parametrize('r',(4,8,16))
def test_q_writer_and_compute_address_census(r):
    rows=(np.arange((12288+r-1)//r)[:,None]*r+np.arange(r)[None,:]).ravel()
    rows=rows[rows<12288]
    assert np.array_equal(np.bincount(rows,minlength=12288),np.ones(12288))
    field=rows%512//256;address=rows//512*256+rows%256
    for f in (0,1):
        assert np.array_equal(np.bincount(address[field==f],minlength=6144),np.ones(6144))
