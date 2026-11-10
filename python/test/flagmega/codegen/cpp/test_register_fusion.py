# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""C++ fusion preserves typed register boundaries and rejects unknown programs."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.cpp.renderer import describe_cpp_package, render_cpp_source_package
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.options import CompileOptions
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import (
    FTT_INCLUDE, bf16, f32, library, render, run,
)

GATE_LINES = (
    'pre_rhs_0 = (tl.sigmoid((rhs_value).to(tl.float32))).to(tl.bfloat16)',
    'base_result = ((lhs_value * pre_rhs_0)).to(tl.bfloat16)',
)


def gate_module():
    class Graph(fm.Module):
        def forward(self):
            kind = fm.tensor_type('bfloat16', (1, 64))
            lhs, rhs = self.input('lhs', kind), self.input('rhs', kind)
            gate = fm.F.math.sigmoid(rhs, name='gate')
            output = fm.F.math.mul(lhs, gate, name='result')
            self.function('main', (lhs,rhs), (output,))
    return Compiler(CompileOptions(target='nvidia-sm120', bufferize_opt_level='fast')).compile(
        Graph(dialect='high_level', stage='imported', entry='main').build()).module


@pytest.fixture(scope='module')
def compiled_gate():
    return gate_module()


def test_normal_cpp_package_preserves_sigmoid_bf16_fusion(compiled_gate, tmp_path):
    package = render_cpp_source_package(compiled_gate, tmp_path)
    descriptor = describe_cpp_package(compiled_gate)
    fused = [call for call in descriptor['render_calls'] if call.get('fusion_lines')]
    assert len(fused) == 1
    assert fused[0]['family'] == 'elementwise'
    source = (tmp_path / package['source']).read_text()
    assert 'ftt::sigmoid' in source
    assert 'ftt::to_bfloat16' in source
    assert 'pre_rhs_0' in source
    assert 'base_result' in source


@pytest.mark.skipif(shutil.which('nvcc') is None, reason='NVCC is required; no GPU execution')
def test_nvcc_builds_normal_fused_cpp_package(compiled_gate, tmp_path):
    from triton.flagmega.codegen.cpp.build import build_cpp_package
    package = render_cpp_source_package(compiled_gate, tmp_path)
    output = build_cpp_package(tmp_path/package['source'], tmp_path/'fused.so', target='nvidia-sm120')
    assert output.is_file()


def _host_gate_call(lines=GATE_LINES, value='base_result'):
    return dict(family='elementwise', variant='mul', lhs='lhs',rhs='rhs', result='result',
                local_capacity=64, active='True', lhs_offset='local_offsets',rhs_offset='local_offsets',
                result_offset='local_offsets', fusion_lines=lines, fusion_value=value, math_mode='fast')


@pytest.mark.skipif(shutil.which('c++') is None, reason='A host C++ compiler is required')
def test_host_fused_gate_has_bf16_boundary_before_multiply(tmp_path):
    types = {'lhs':'ftt::bfloat16_t','rhs':'ftt::bfloat16_t','result':'ftt::bfloat16_t'}
    body = render(_host_gate_call(),types,mesh=(1,))
    lib = library(tmp_path,body,types,mesh=(1,))
    lhs = bf16(np.linspace(0.6,1.5,64,dtype=np.float32))
    rhs = bf16(np.linspace(-2,2,64,dtype=np.float32))
    sigmoid = 1/(1+np.exp(-f32(rhs)))
    expected = bf16(f32(lhs) * f32(bf16(sigmoid)))
    missing_boundary = bf16(f32(lhs) * sigmoid)
    assert np.count_nonzero(expected != missing_boundary) >= 8
    actual = np.zeros(64,np.uint16)
    run(lib,lhs,rhs,actual)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('lines,value',[
    (('pre_rhs_0 = arbitrary(rhs_value)', 'base_result = lhs_value * pre_rhs_0'),'base_result'),
    (('base_result = tl.load(lhs_value)',),'base_result'),
    (('base_result = (lhs_value).to(tl.float16)',),'base_result'),
    (('base_result = lhs_value + uninitialized',),'base_result'),
    (('base_result = lhs_value; tl.store(rhs_value, 1)',),'base_result'),
    (('lhs_value = rhs_value', 'base_result = lhs_value'),'base_result'),
    (GATE_LINES, 'base_result; __builtin_trap()'),
])
def test_unknown_register_programs_fail_closed(lines,value):
    types={'lhs':'ftt::bfloat16_t','rhs':'ftt::bfloat16_t','result':'ftt::bfloat16_t'}
    with pytest.raises(CodegenError):
        render(_host_gate_call(lines,value),types,mesh=(1,))


@pytest.mark.skipif(shutil.which('c++') is None, reason='A host C++ compiler is required')
def test_host_pre_and_post_casts_keep_every_register_boundary(tmp_path):
    lines = (
        'pre_lhs_0 = (lhs_value).to(tl.float32)',
        'pre_lhs_1 = (pre_lhs_0 * tl.full((), 1.00390625, tl.float32)).to(tl.bfloat16)',
        'base_result = tl.div_rn(pre_lhs_1, (rhs_value).to(tl.float32)).to(tl.float32)',
        'post_result_0 = (base_result).to(tl.bfloat16)',
        'post_result_1 = (post_result_0).to(tl.float32)',
    )
    call = dict(_host_gate_call(lines,'post_result_1'),variant='div')
    types = {'lhs':'ftt::bfloat16_t','rhs':'ftt::bfloat16_t','result':'float'}
    lib = library(tmp_path,render(call,types,mesh=(1,)),types,mesh=(1,))
    lhs = bf16(np.linspace(0.7,1.6,64,dtype=np.float32))
    rhs = bf16(np.linspace(0.8,1.1,64,dtype=np.float32))
    expected = f32(bf16(f32(bf16(f32(lhs)*np.float32(1.00390625)))/f32(rhs)))
    late_cast = f32(bf16((f32(lhs)*np.float32(1.00390625))/f32(rhs)))
    assert np.count_nonzero(expected != late_cast) >= 8
    actual = np.zeros(64,np.float32)
    run(lib,lhs,rhs,actual)
    np.testing.assert_array_equal(actual,expected)


def test_feature_validation_rejects_fusion_in_an_unimplemented_owner():
    from triton.flagmega.codegen.cpp.fusion import validate_call
    with pytest.raises(CodegenError,match='no owner'):
        validate_call(dict(_host_gate_call(),family='softmax'))
    with pytest.raises(CodegenError,match='fusion_post_lines'):
        validate_call(dict(family='softmax',fusion_post_lines=['post_result_0 = (value).to(tl.bfloat16)']))
    with pytest.raises(CodegenError,match='fusion_post_value'):
        validate_call(dict(family='softmax',fusion_post_value='(value).to(tl.bfloat16)'))
    with pytest.raises(CodegenError,match='no register program'):
        validate_call(dict(family='elementwise',variant='mul',fusion_value='base_result'))
    with pytest.raises(CodegenError,match='no implementing kernel owner'):
        validate_call(dict(family='dense_matmul',variant='gemv',addend_cast_types=['tl.bfloat16']))


@pytest.mark.parametrize('casts,projection',[
    ((), 'tl.bfloat16'),
    (('tl.bfloat16','tl.float32'), 'tl.bfloat16'),
    (('tl.float32','tl.bfloat16','tl.float32'), 'tl.float32'),
])
@pytest.mark.skipif(shutil.which('c++') is None, reason='A host C++ compiler is required')
def test_dense_norm_stats_preserves_projection_and_addend_rounding(tmp_path,casts,projection):
    types={'source':'ftt::bfloat16_t','weight':'ftt::bfloat16_t','residual':'float','result':'float','stats':'float'}
    call=dict(family='dense_matmul',variant='packed_k_major_gemv_norm_stats',symbol='kernel',arguments=list(types),
              source='source',weight='weight',residual='residual',result='result',stats='stats',
              local_m_capacity=1,local_n_capacity=8,local_k_capacity=2,weight_k_lane=0,k_streams=1,
              weight_block_n=0,weight_block_k=0,weight_group_stride=0,
              global_n='dense_local_n_offsets',global_k='dense_local_k_offsets',
              source_active='True',result_active='True',source_offset='dense_local_k_offsets',
              weight_offset='dense_local_n_offsets*2+dense_local_k_offsets',
              residual_offset='dense_local_n_offsets',result_offset='dense_local_n_offsets',
              result_writer_active='True',stats_writer_active='True',stats_offset='0',
              projection_type=projection,addend_cast_types=casts)
    context=dict(render_calls=[call],argument_types={n:{'type':t,'pointer':True} for n,t in types.items()},
                 mesh_hierarchy=[1],distributed_entry=False,block_threads=1,warp_size=1,warps=1,
                 sme=False,sme_slots=0,sme_pad=0,shared_arena_reuse=False)
    body=CppTemplateRegistry().render('kernels/dense_matmul/packed_k_major_gemv_norm_stats.cpp.jinja',context)
    source_code=f'''#include <cmath>
#include <cstdint>
#include <utility>
#define __shared__ static
struct float4 {{float x,y,z,w;}};
#include <ftt/ftt.h>
{body}
extern "C" void run(ftt::bfloat16_t *source,ftt::bfloat16_t *weight,float *residual,float *result,float *stats) {{kernel(source,weight,residual,result,stats);}}
'''
    cpp=tmp_path/'epilogue.cpp'; cpp.write_text(source_code)
    so=tmp_path/'epilogue.so'
    subprocess.run(['c++','-std=c++17','-O1','-shared','-fPIC','-DFTT_ARCH_ILUVATAR',f'-I{FTT_INCLUDE}',str(cpp),'-o',str(so)],check=True,capture_output=True,text=True)
    lib=ctypes.CDLL(str(so))
    source=bf16([1,1]); weight=bf16(np.tile([1,1/256],(8,1)))
    residual=np.array([-1,-.999,-1.0008,-.99,-.5,-.499,-.1,-.099],np.float32)
    result=np.full(8,np.nan,np.float32);stats=np.full(1,np.nan,np.float32)
    run(lib,source,weight,residual,result,stats)
    expected_residual=residual.copy()
    for dtype in casts:
        if dtype=='tl.bfloat16': expected_residual=f32(bf16(expected_residual))
    projection_value=np.float32(1+1/256)
    if projection=='tl.bfloat16': projection_value=f32(bf16(projection_value))
    expected=projection_value+expected_residual
    np.testing.assert_array_equal(result,expected)
    expected_stat = np.float32(0)
    for value in expected:
        expected_stat = np.float32(expected_stat + np.float32(value*value))
    np.testing.assert_array_equal(stats[0],expected_stat)
