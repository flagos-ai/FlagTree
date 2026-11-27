# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit direct dot-operand shared loads preserve the packed address contract."""
from copy import deepcopy
import pytest
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.errors import CodegenError


def raw_call(module):
    binding=describe_function_runtime_binding(module)
    raw=deepcopy(next(v for v in binding['call_abi']['kernel_calls'] if v['family']=='dense_matmul'))
    raw['parameters'].update(accumulation_mode='ordered_mma',mma_block_k=128)
    return raw


def test_explicit_direct_dot_is_encoded(packed_descriptor_pipeline_module):
    raw=raw_call(packed_descriptor_pipeline_module)
    assert 'mma_rhs_load' not in _dense_matmul_call(raw)
    raw['parameters']['mma_rhs_load']='direct_dot'
    assert _dense_matmul_call(raw)['mma_rhs_load']=='direct_dot'


@pytest.mark.parametrize('value',[None,True,[],{},'blocked','automatic'])
def test_invalid_direct_dot_selection_rejects(packed_descriptor_pipeline_module,value):
    raw=raw_call(packed_descriptor_pipeline_module);raw['parameters']['mma_rhs_load']=value
    with pytest.raises(CodegenError,match='RHS load'):
        _dense_matmul_call(raw)


@pytest.mark.parametrize('mutation',['simt','float16','unpacked_variant'])
def test_direct_dot_requires_ordered_bf16_packed_descriptor(packed_descriptor_pipeline_module,mutation):
    raw=raw_call(packed_descriptor_pipeline_module);raw['parameters']['mma_rhs_load']='direct_dot'
    if mutation=='simt':raw['parameters']['accumulation_mode']='simt'
    elif mutation=='float16':raw['inputs'][0]['buffers'][0]['abi']['scalar_dtype']='float16'
    else:raw['variant']='tensor_descriptor_smem_pipeline_gemv'
    with pytest.raises(CodegenError):_dense_matmul_call(raw)


def test_direct_rhs_coordinates_cover_original_slot():
    old={(k//16,n//8,((n%8)*16+k%16)//64,((n%8)*16+k%16)%64)
         for n in range(16) for k in range(1024)}
    parts=[{(k//16,n//8,((n%8)*16+k%16)//64,((n%8)*16+k%16)%64)
           for k in range(start,start+128) for n in range(16)} for start in range(0,1024,128)]
    assert len(old)==16384 and all(len(x)==2048 for x in parts)
    assert set.union(*parts)==old and sum(map(len,parts))==len(old)


def test_direct_dot_helper_source_keeps_math_and_default(packed_descriptor_pipeline_module):
    package=describe_tir_package(packed_descriptor_pipeline_module)
    call=next(v for v in package['render_calls'] if v['family']=='dense_matmul')
    call.update(accumulation_mode='ordered_mma',mma_block_k=128)
    before=render_tir_package(package,'direct-dot-test')
    call['mma_rhs_load']='direct_dot'
    direct=render_tir_package(package,'direct-dot-test')
    assert 'dense_stage_k[:, None] // 16, dense_stage_n[None, :] // 8' in direct
    assert 'dense_b = tle.gpu.set_layout(tl.load(dense_pointer),' in direct
    assert 'tl.trans(dense_weight)' not in direct
    assert 'dense_accumulator = tl.dot(dense_a, dense_b, dense_accumulator)' in direct
    del call['mma_rhs_load']
    assert render_tir_package(package,'direct-dot-test')==before


def test_norm_stats_include_uses_same_explicit_contract():
    from .test_dense_lhs_staging import direct_norm_stats_staged_module
    module=direct_norm_stats_staged_module()
    binding=describe_function_runtime_binding(module)
    raw=deepcopy(next(v for v in binding['call_abi']['kernel_calls'] if v['family']=='dense_matmul'))
    raw['parameters'].update(accumulation_mode='ordered_mma',mma_block_k=128,mma_rhs_load='direct_dot')
    call=_dense_matmul_call(raw)
    assert call['mma_rhs_load']=='direct_dot'
    package=describe_tir_package(module)
    encoded,=package['render_calls']
    encoded.update(accumulation_mode='ordered_mma',mma_block_k=128)
    before=render_tir_package(package,'norm-direct-dot-test')
    encoded['mma_rhs_load']='direct_dot'
    source=render_tir_package(package,'norm-direct-dot-test')
    assert 'dense_b = tle.gpu.set_layout(tl.load(dense_pointer),' in source
    assert 'tl.trans(dense_weight)' not in source
    del encoded['mma_rhs_load']
    assert render_tir_package(package,'norm-direct-dot-test')==before
