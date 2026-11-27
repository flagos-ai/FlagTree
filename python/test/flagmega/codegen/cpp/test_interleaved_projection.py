# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Normal tuple lowering, native row/lane mapping and compiler-only probes."""
import ast
from copy import deepcopy
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import os
import sys

import numpy as np
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.codegen.cpp import describe_cpp_package, render_cpp_source_package
from triton.flagmega.codegen.cpp.build import build_cpp_package, toolchain_available
from triton.flagmega.codegen.triton.renderer import render_triton_package
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import packed_rows, bf16, f32, render, library, run


def graph(encoding='IQ4_XS', lanes=((2,4),(8,)), policy=None, dtype='bfloat16', heads=6, width=16, tokens=2, k=512, placement=None):
    placement = placement or fm.Placement((2,), 'x', 'b')
    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type(dtype, (tokens, k)))
            w = self.input('w', fm.tensor_type('uint8', (heads*2*width, (136 if encoding=='IQ4_XS' else 144)*(k//256))))
            s = self.input('s', fm.tensor_type('float32', ((heads*2*width+127)//128, (k+127)//128)))
            values = (x,w,s)
            if policy is not None:
                broadcast = fm.SBP.broadcast()
                values = tuple(fm.F.distributed.boxing(n, fm.DistributedType(n.type, policies, placement))
                               for n,policies in zip(values, ((broadcast,broadcast),(policy,broadcast),(broadcast,broadcast))))
            p = fm.F.nn.interleaved_projection(*values, weight_block_n=128, weight_block_k=128,
                    weight_encoding=encoding, feature_shape=(heads,width), output_lanes=lanes, name='projection')
            self.function('main',(x,w,s), fm.F.tensors.get_items(p,0,1))
    return Graph(dialect='high_level', stage='distributed' if policy is not None else 'imported', entry='main',
                 metadata={'auto_distribution':{'placement':placement.to_data()}} if policy is not None else {}).build()


@lru_cache(None)
def compiled(encoding='IQ4_XS', lanes=((2,4),(8,)), split='contiguous', dtype='bfloat16', target='nvidia-sm120'):
    policy = (fm.SBP.split_contiguous((0,)) if split=='contiguous'
              else fm.SBP.split_block_cyclic((0,),32) if split=='cyclic' else None)
    return Compiler(CompileOptions(target=target, bufferize_opt_level='fast')).compile(graph(encoding,lanes,policy,dtype)).module


@pytest.mark.parametrize('target', ['nvidia-sm120','iluvatar-bi-v150'])
def test_normal_tuple_candidate_has_two_complete_write_fields(tmp_path,target):
    module = compiled(target=target)
    cpp = describe_cpp_package(module)
    calls = [c for c in cpp['render_calls'] if c['family']=='interleaved_projection']
    assert len(calls)==1
    call=calls[0]
    assert call['implementation']=='tir.interleaved_projection.simt'
    assert [p['formal'] for p in call['outputs']]==['result_0','result_1']
    assert not call['workspaces']
    assert not any(n.op in {'tensors.slice','tensors.pack','tensors.unpack'} for n in module.nodes)
    # The executable dispatch, not a frontend tuple adapter, owns both writes.
    from triton.flagmega.ir import execution_calls_of
    dispatch=next(c for c in execution_calls_of(module.execution_function_map['main']) if getattr(c,'callee',None)==call['callee'])
    assert len(dispatch.results)==2
    assert len(dispatch.writes)==2
    render_cpp_source_package(module,tmp_path/'cpp')
    render_triton_package(module,tmp_path/'triton')
    cpp_source=(tmp_path/'cpp/generated_kernels.cpp').read_text()
    triton_source=(tmp_path/'triton/generated_kernels.py').read_text()
    assert cpp_source.count('const float accumulator = ftt::quantized::dot<')==1
    assert 'destination = tl.where(' in triton_source
    assert 'TILE_N: tl.constexpr' in triton_source
    ast.parse(triton_source)


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
@pytest.mark.parametrize('split',['contiguous','cyclic'])
@pytest.mark.parametrize('lanes',[((),()),((2,4),(8,)),((),(8,))])
@pytest.mark.parametrize('storage',['canonical','compact'])
def test_generated_cpp_writes_every_original_row_to_its_field(tmp_path,encoding,split,lanes,storage):
    descriptor=describe_cpp_package(compiled(encoding,lanes,split))
    call=deepcopy(next(c for c in descriptor['render_calls'] if c['family']=='interleaved_projection'))
    # Execute the normal descriptor on CPU. External pointers keep canonical
    # strides, while the whole-head owner maps and masks are untouched.
    types={'x':'ftt::bfloat16_t','w':'unsigned char','s':'float','a':'ftt::bfloat16_t','b':'ftt::bfloat16_t'}
    for name,parameter in zip(types,(*call['inputs'],*call['outputs'])):
        binding=parameter['buffers'][0]
        binding['runtime_argument']=name
        abi=binding['abi']
        abi.update(storage='input',pooled=False,pool_byte_offset=0,pool_scope_stride_bytes=0,
                   component_stride_scalar_elements=0,coordinate_space='canonical_global')
        logical=abi['logical_shape'];lanes_count=abi['scalar_lane_count']
        abi['scalar_storage_strides']=tuple(int(np.prod(logical[axis+1:]))*lanes_count for axis in range(len(logical)))
        if storage=='compact' and name in ('w','a','b'):
            local=abi['local_capacity_shape']
            abi.update(coordinate_space='local',storage_kind='compact_per_owner',owner_index_expression='shard_index',
                       component_stride_scalar_elements=int(np.prod(local))*lanes_count)
            abi['scalar_storage_strides']=tuple(int(np.prod(local[axis+1:]))*lanes_count for axis in range(len(local)))
    call=prepare_kernel_calls([call],function_name='main')[0]
    mesh=tuple(descriptor['mesh_hierarchy'])
    lib=library(tmp_path,render(call,types,mesh=mesh),types,mesh=mesh)
    weight,decoded=packed_rows(encoding,rows=192,blocks=2)
    rng=np.random.default_rng(51)
    x=bf16(rng.standard_normal((2,512)).astype(np.float32)/16)
    owner_heads=[np.arange(owner*3,(owner+1)*3) if split=='contiguous' else np.arange(owner,6,2) for owner in range(2)]
    if storage=='compact':
        weight=np.concatenate([weight.reshape(6,32,-1)[h].reshape(-1,weight.shape[1]) for h in owner_heads])
    actual=[np.full((2,2,3,16) if storage=='compact' else (2,6,16),0x7fc0,np.uint16) for _ in range(2)]
    run(lib,x,weight,np.ones((2,4),np.float32),*actual)
    # Match the documented sequential FP32 native reference accumulation;
    # each row has distinct quantized headers/codes and covers the tail owner.
    accum=np.zeros((2,192),np.float32)
    for k in range(512):
        accum=np.float32(accum+f32(x)[:,k,None]*decoded[None,:,k])
    expected=bf16(accum).reshape(2,6,2,16)
    for field in range(2):
        oracle=np.stack([expected[:,h,field,:] for h in owner_heads]) if storage=='compact' else expected[:,:,field,:]
        np.testing.assert_array_equal(actual[field],oracle)


@pytest.mark.parametrize('change',[ 'output_lane', 'output_dtype', 'packed_input', 'k_stride', 'placeholder_scale', 'head_owner' ])
def test_renderer_rejects_inconsistent_native_field_abi(change):
    call=deepcopy(next(c for c in describe_cpp_package(compiled())['render_calls'] if c['family']=='interleaved_projection'))
    if change=='output_lane':call['outputs'][0]['buffers'][0]['abi']['scalar_lane_shape']=(8,)
    if change=='output_dtype':call['outputs'][1]['buffers'][0]['abi']['scalar_dtype']='float32'
    if change=='packed_input':call['inputs'][0]['buffers'][0]['abi']['scalar_lane_count']=8
    if change=='k_stride':call['inputs'][0]['buffers'][0]['abi']['scalar_storage_strides']=(512,2)
    if change=='placeholder_scale':call['inputs'][2]['buffers'][0]['abi']['logical_shape']=(1,1)
    if change=='head_owner':call['outputs'][1]['buffers'][0]['abi']['distributed_type']['axis_policies'][1]={'kind':'broadcast'}
    with pytest.raises(CodegenError,match='InterleavedProjection'):
        prepare_kernel_calls([call],function_name='main')


@pytest.mark.parametrize('encoding,dtype',[('IQ4_XS','bfloat16'),('Q4_K','float32')])
def test_normal_native_projection_nvcc_compiles_without_launch(tmp_path,encoding,dtype):
    if not toolchain_available('nvidia-sm120'):pytest.skip('NVCC is required')
    render_cpp_source_package(compiled(encoding,dtype=dtype),tmp_path)
    assert build_cpp_package(tmp_path/'generated_kernels.cpp',tmp_path/'kernel.so',target='nvidia-sm120').stat().st_size


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
def test_normal_native_projection_runs_in_triton_cpu_interpreter(tmp_path,encoding):
    module=Compiler(CompileOptions(target='nvidia-sm120',bufferize_opt_level='fast')).compile(
        graph(encoding,heads=2)).module
    render_triton_package(module,tmp_path)
    weight,decoded=packed_rows(encoding,rows=64,blocks=2)
    expected=bf16(np.stack([decoded[:,257]*.5,decoded[:,385]*-.75])).reshape(2,2,2,16)
    np.savez(tmp_path/'data.npz',weight=weight,expected=expected)
    script=tmp_path/'interpret.py'
    script.write_text('''import importlib.util
from pathlib import Path
import numpy as np
import torch
p=Path(__file__).parent
spec=importlib.util.spec_from_file_location('native_projection',p/'generated_kernels.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data=np.load(p/'data.npz')
x=torch.zeros(2,512,dtype=torch.bfloat16);x[0,257]=.5;x[1,385]=-.75
a=torch.full((2,2,16),float('nan'),dtype=torch.bfloat16);b=a.clone()
m.flagmega_main[(1,)](x,torch.from_numpy(data['weight']),torch.ones(1,4),a,b)
expected=torch.from_numpy(data['expected']).view(torch.bfloat16)
assert torch.equal(a,expected[:,:,0,:])
assert torch.equal(b,expected[:,:,1,:])
print('CPU interpreter: both native fields exact')
''')
    process=subprocess.run([sys.executable,str(script)],env={**os.environ,'TRITON_INTERPRET':'1'},
                           capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stdout+process.stderr
