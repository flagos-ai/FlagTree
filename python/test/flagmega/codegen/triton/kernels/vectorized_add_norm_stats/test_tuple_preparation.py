# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compiler-owned tuple fields survive normal kernel materialization and PreOps checks."""

from copy import deepcopy

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import CodegenError
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.codegen.triton.call_abi import plain_package_value
from triton.flagmega.codegen.triton.vectorized_add_norm_stats_renderer import vectorized_add_norm_stats_call
from triton.flagmega.passes.tir.materialize_kernel_definitions import _kernel_definition

from python.test.flagmega.codegen.triton.kernels.vectorized_add_norm_stats.test_local16_contract import abi, parameter, ORDER


def materialized_call(round_lhs):
    mesh = fm.Placement((8, 16), 'yx', 'bb')
    split = lambda unit: (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), unit))
    lhs_type = fm.DistributedType(fm.tensor_type(fm.vector_type('float32', (4, )), (1, 512)), split(4), mesh)
    rhs_type = fm.DistributedType(fm.tensor_type('bfloat16', (1, 2048)), split(16), mesh)
    inputs = tuple(
        fm.Node(name, 'builtin.var', (), kind, attrs={'name': name})
        for name, kind in (('lhs', lhs_type), ('rhs', rhs_type)))

    @fm.fusion(rhs_type)
    def widen(x):
        return fm.F.tensors.cast(x, 'float32')

    attrs = {
        'binary_op': 'add', 'rhs_pack': {'axes': (1, ), 'lanes': (4, )}, 'axis': -1, 'use_mean': False,
        'local_reduction_order': ORDER, 'pre_ops': {'rhs': widen}
    }
    if round_lhs:

        @fm.fusion(lhs_type)
        def rounded(x):
            return fm.F.tensors.cast(fm.F.tensors.cast(x, fm.vector_type('bfloat16', (4, ))),
                                     fm.vector_type('float32', (4, )))

        attrs['pre_ops']['lhs'] = rounded
    prepared = fm.get_definition('ntt.vectorized_add_norm_stats').prepare(inputs, attrs)
    parameters = {
        'family': 'vectorized_add_norm_stats', 'variant': 'local16_xor_fma', 'tile': 16, 'vector_schedule':
        {'contract': {'kind': 'typed_elements', 'lanes': (4, )}, 'physical': {'elements_per_program': 16}}
    }
    call = fm.Node(
        'paired', 'tir.call', ('lhs', 'rhs'), prepared.result_type, prepared.effect, {
            'semantic_op': 'ntt.vectorized_add_norm_stats', 'semantic_attrs': prepared.attrs, 'candidate':
            'tir.vectorized_add_norm_stats.local16_xor_fma', 'parameters': parameters, 'facts': {}
        })
    module = fm.IRModule('tir', 'selected', (*inputs, call), (fm.Function('main', ('lhs', 'rhs'), ('paired', )), ),
                         'main')
    definition, _ = _kernel_definition(call, module)
    assert definition.dispatch.outputs == ('result_0', 'result_1')
    # Build physical local ABIs with the actual typed parameters and the output
    # formals emitted by normal materialization, not invented semantic names.
    kinds = [abi((1, 4), (4, )), abi((1, 16)), abi((1, 4), (4, )), abi((1, 1, 1), partial=True)]
    bindings = []
    for p, kind in zip(definition.parameters, kinds, strict=True):
        kind['distributed_type'] = p.type.to_data()
        kind['logical_shape'] = tuple(d.fixed_value for d in p.type.tensor.shape)
        dtype = p.type.tensor.dtype
        scalar = dtype.elem_type if isinstance(dtype, fm.VectorType) else dtype
        kind['scalar_dtype'] = scalar.value
        kind['scalar_itemsize'] = scalar.itemsize
        bindings.append(parameter(p.name, kind))
    raw = {
        'call': 'paired', 'family': parameters['family'], 'variant': parameters['variant'], 'semantic_op':
        definition.dispatch.semantic_op, 'semantic_attrs': plain_package_value(definition.dispatch.semantic_attrs),
        'parameters': parameters, 'facts': {}, 'inputs': bindings[:2], 'outputs': bindings[2:], 'workspaces': ()
    }
    return raw, prepared.result_type


@pytest.mark.parametrize('round_lhs', [False, True])
def test_normal_materialization_names_reach_renderer(round_lhs):
    raw, _ = materialized_call(round_lhs)
    call = vectorized_add_norm_stats_call(raw)
    assert call['fusion_lines']
    assert 'bfloat16' in '\n'.join(call['fusion_lines']) if round_lhs else True


@pytest.mark.parametrize('round_lhs', [False, True])
@pytest.mark.parametrize('reverse_outputs', [False, True])
def test_prepare_validates_whole_tuple_and_preserves_field_order(round_lhs, reverse_outputs):
    raw, _ = materialized_call(round_lhs)
    if reverse_outputs:
        raw['outputs'].reverse()
    before = deepcopy(raw)
    calls = prepare_kernel_calls((raw, ), function_name='main')
    assert len(calls) == 1 and calls[0]['family'] == 'vectorized_add_norm_stats'
    assert raw == before


def test_tuple_preparation_rejects_missing_or_duplicate_fields():
    raw, _ = materialized_call(False)
    for outputs in ([raw['outputs'][0]], [raw['outputs'][0], raw['outputs'][0]]):
        with pytest.raises(CodegenError, match='outputs|result|formal'):
            prepare_kernel_calls(({**raw, 'outputs': outputs}, ), function_name='main')
