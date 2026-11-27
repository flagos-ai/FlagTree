# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Function-forwarded reference fields keep their physical scalar ABI."""
import pytest
from dataclasses import replace
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.codegen.cpp import describe_cpp_package,render_cpp_source_package
from triton.flagmega.codegen.cpp.build import build_cpp_package,toolchain_available
from triton.flagmega.codegen.cpp.expressions import cpp_scalar_type
from triton.flagmega.codegen.cpp.renderer import _argument_types
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.errors import CodegenError
from python.test.flagmega.passes.tir.bufferize.test_ref_slice import state_slice_graph


@pytest.fixture(scope='module')
def forwarded_state():
    graph=replace(state_slice_graph(index=1,reusable=True),stage='imported')
    return Compiler(CompileOptions(target='nvidia-sm120',bufferize_opt_level='fast')).compile(
        graph).module


def test_normal_forwarded_and_unused_ref_fields_are_typed_from_buffer_plan(forwarded_state,tmp_path):
    module=forwarded_state
    descriptor=describe_cpp_package(module)
    plan=fm.verify_buffer_plan(module)
    root=describe_function_runtime_binding(module)
    reference=[v for v in root['values'] if v['value']=='entry_state' and v['role']=='parameter']
    assert len(reference)==2
    assert {describe_local_buffer_abi(plan.buffer_map[v['buffer']],plan)['scalar_dtype'] for v in reference}=={'bfloat16','float32'}
    kernel_roots={b['runtime_argument'] for c in descriptor['render_calls'] for p in (*c['inputs'],*c['outputs']) for b in p['buffers']}
    assert any(v['runtime_argument'] not in kernel_roots for v in reference)
    for v in reference:
        abi=describe_local_buffer_abi(plan.buffer_map[v['buffer']],plan)
        assert descriptor['argument_types'][v['runtime_argument']]=={'type':cpp_scalar_type(abi['scalar_dtype']),'pointer':True}
    render_cpp_source_package(module,tmp_path)
    source=(tmp_path/'generated_kernels.cpp').read_text()
    assert 'float *entry_state_recurrent' in source or 'float *entry_state_1' in source
    if toolchain_available('nvidia-sm120'):
        assert build_cpp_package(tmp_path/'generated_kernels.cpp',tmp_path/'state.so',target='nvidia-sm120').stat().st_size


def test_unbound_parameter_has_no_byte_pointer_fallback(forwarded_state):
    descriptor=describe_cpp_package(forwarded_state)
    descriptor['signature_arguments']=[*descriptor['signature_arguments'],'invented_argument']
    with pytest.raises(CodegenError,match='invented_argument.*physical buffer ABI'):
        _argument_types(descriptor,forwarded_state)
