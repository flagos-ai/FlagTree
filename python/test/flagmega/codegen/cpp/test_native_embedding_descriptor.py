# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native decoded packets preserve original byte rows, padding and subspans."""
import ast
from copy import deepcopy
from functools import lru_cache
import numpy as np
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.codegen.cpp import describe_cpp_package, render_cpp_source_package
from triton.flagmega.codegen.triton.renderer import render_triton_package
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import packed_rows, bf16, render, library, run


@lru_cache(None)
def compiled(encoding,lanes):
    class Graph(fm.Module):
        def forward(self):
            ids=self.input('ids',fm.tensor_type('int32',(3,)))
            weight=self.input('weight',fm.tensor_type('uint8',(5,(136 if encoding=='IQ4_XS' else 144)*2)))
            result=fm.F.nn.embedding(ids,weight,weight_encoding=encoding,output_lanes=lanes,padding_idx=-5,name='embedding')
            self.function('main',(ids,weight),(result,))
    m=Graph(dialect='high_level',stage='imported',entry='main').build()
    return Compiler(CompileOptions(target='nvidia-sm120',bufferize_opt_level='fast')).compile(m).module


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
@pytest.mark.parametrize('lanes',[(8,),(2,4)])
def test_normal_packet_embedding_renders_both_backends(tmp_path,encoding,lanes):
    module=compiled(encoding,lanes)
    d=describe_cpp_package(module)
    c=next(c for c in d['render_calls'] if c['family']=='embedding')
    assert c['padding_idx']==0
    assert '* 8 +' in c['global_feature']
    assert c['inputs'][1]['buffers'][0]['abi']['scalar_lane_shape']==()
    assert c['outputs'][0]['buffers'][0]['abi']['scalar_lane_shape']==lanes
    assert c['native_weight_row_offset']
    render_cpp_source_package(module,tmp_path/'cpp')
    render_triton_package(module,tmp_path/'triton')
    source=(tmp_path/'triton/generated_kernels.py').read_text()
    ast.parse(source)
    assert ', 0,' in source


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
@pytest.mark.parametrize('lanes',[(8,),(2,4)])
def test_normal_native_embedding_respects_byte_row_stride_and_subspan(tmp_path,encoding,lanes):
    d=describe_cpp_package(compiled(encoding,lanes))
    c=deepcopy(next(c for c in d['render_calls'] if c['family']=='embedding'))
    names=('ids','weight','result')
    for name,p in zip(names,(*c['inputs'],*c['outputs'])):
        binding=p['buffers'][0];binding['runtime_argument']=name
        binding['abi'].update(storage='input',pooled=False,pool_scope_stride_bytes=0,
                              component_stride_scalar_elements=0,coordinate_space='canonical_global',pool_byte_offset=0)
    ids_abi=c['inputs'][0]['buffers'][0]['abi'];ids_abi['pool_byte_offset']=8
    weight_abi=c['inputs'][1]['buffers'][0]['abi']
    row_bytes=weight_abi['logical_shape'][1]
    weight_abi['pool_byte_offset']=24
    weight_abi['scalar_storage_strides']=(row_bytes+8,1)
    result_abi=c['outputs'][0]['buffers'][0]['abi']
    result_abi['pool_byte_offset']=16
    result_abi['scalar_storage_strides']=(528,8)
    c=prepare_kernel_calls([c],function_name='main')[0]
    types={'ids':'int','weight':'unsigned char','result':'ftt::bfloat16_t'}
    mesh=tuple(d['mesh_hierarchy'])
    lib=library(tmp_path,render(c,types,mesh),types,mesh)
    packed,decoded=packed_rows(encoding,rows=5)
    weight=np.full(24+5*(row_bytes+8),0xee,np.uint8)
    for row in range(5):weight[24+row*(row_bytes+8):24+row*(row_bytes+8)+row_bytes]=packed[row]
    ids=np.array([99,99,4,0,2],np.int32)
    result=np.full(8+3*528,0x7fc0,np.uint16)
    run(lib,ids,weight,result)
    expected=bf16(decoded[[4,0,2]]);expected[1]=0
    for row in range(3):np.testing.assert_array_equal(result[8+row*528:8+row*528+512],expected[row])
    touched=np.zeros_like(result,dtype=bool)
    for row in range(3):touched[8+row*528:8+row*528+512]=True
    assert np.all(result[~touched]==0x7fc0)


@pytest.mark.parametrize('field,value',[('scalar_lane_shape',()),('scalar_dtype','float32'),('scalar_lane_count',4)])
def test_native_embedding_descriptor_rejects_output_mismatch(field,value):
    c=deepcopy(next(c for c in describe_cpp_package(compiled('IQ4_XS',(8,)))['render_calls'] if c['family']=='embedding'))
    c['outputs'][0]['buffers'][0]['abi'][field]=value
    with pytest.raises(CodegenError,match='Native embedding'):
        prepare_kernel_calls([c],function_name='main')
