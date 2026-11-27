# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Frozen C native provider/definitions/rendering; CPU only."""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.simple import _gdn_applicable
from triton.flagmega.codegen.triton.kernel_call_renderers import _r004_gdn_phase_call, prepare_kernel_calls
from triton.flagmega.codegen.triton.kernels.gdn_recurrent.r004_contract_v4 import NATIVE_ARGUMENTS
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry, KernelTemplateSpec
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.tir import KernelDefinition, KernelDispatch, PrimParameter, PrimParameterRole, Return, ReturnBinding, ValueRef
from triton.flagmega.ir.tir.verify import verify_prim_function
from triton.flagmega.passes.tir.materialize_kernel_definitions import r004_phase_definitions
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import reference_phase_plan
from triton.flagmega.targets.nvidia.gdn import R004_REFERENCE_ID, r004_reference_applicable
from triton.flagmega.targets.portable_triton_implementations import portable_triton_implementation_model
from triton.flagmega.targets import NvidiaSm120Target


def raw_call(entry):
    plan = reference_phase_plan()
    attrs = dict(num_key_heads=16, num_value_heads=48, key_head_dim=128, value_head_dim=128,
                 epsilon=1e-6, b_weight_encoding='IQ4_XS', a_weight_encoding='IQ4_XS', value_head_order='tiled')
    inputs = []
    for formal, i in (('qkv', 0), ('z', 1), ('projection_input', 2), ('b_weight', 6),
                      ('a_weight', 7), ('a_log', 8), ('dt_bias', 9), ('norm_weight', 10)):
        r = plan['roots'][i]
        inputs.append(dict(formal=formal, buffers=[dict(runtime_argument=r['name'], abi=dict(
            logical_shape=r['shape'], scalar_dtype=r['dtype'], scalar_lane_count=1))]))
    return dict(call='coarse', family='gdn_recurrent', variant='r004_reference_v4',
                implementation=R004_REFERENCE_ID, semantic_attrs=attrs, inputs=inputs, outputs=[], workspaces=[],
                arguments=list(NATIVE_ARGUMENTS[entry['phase']]), argument_bindings=entry['argument_bindings'],
                facts=dict(internal_grid_barriers=0), parameters=dict(phase_plan=plan, r004_phase=entry['phase'],
                program_grid=entry['grid'], program_owner_mapping='x-fast'))


def coarse_definition():
    tensor = fm.tensor_type('float32', (1,))
    return KernelDefinition('coarse', 'triton', (
        PrimParameter('input', tensor), PrimParameter('output', tensor, PrimParameterRole.OUTPUT)),
        KernelDispatch('nn.gdn_recurrent_core', arguments=('input',), outputs=('output',),
            candidate=R004_REFERENCE_ID, parameters=dict(family='gdn_recurrent', variant='r004_reference_v4',
            phase_plan=reference_phase_plan()), writes=('output',)),
        Return((ReturnBinding(ValueRef('output', tensor), 'output'),)))


def test_registered_standalone_C_T_and_original_default():
    model = portable_triton_implementation_model()
    assert model.implementation(R004_REFERENCE_ID).variant == 'r004_reference_v4'
    assert model.implementation('tir.gdn_recurrent.r004_column_v4').variant == 'r004_column_v4'
    assert 'tir.gdn_recurrent.r004_column_v4' not in [i.id for i in model.find('gdn_recurrent')]
    assert [i.id for i in model.find('gdn_recurrent', standalone_schema='r004.gdn.standalone/v4',
                                   standalone_module=model.implementation('tir.gdn_recurrent.r004_column_v4').contract['standalone_module'])] == ['tir.gdn_recurrent.r004_column_v4']
    assert model.implementation('tir.gdn_recurrent.r004_reference_v3') is None
    assert model.choose_default('gdn_recurrent', ('tir.gdn_recurrent.persistent', R004_REFERENCE_ID)) == 'tir.gdn_recurrent.persistent'
    assert R004_REFERENCE_ID not in [i.id for i in model.find('gdn_recurrent')]
    assert [i.id for i in model.find('gdn_recurrent', standalone_schema='r004.gdn.standalone/v4',
                                    standalone_module='r004_C_full_p5_v4')] == [R004_REFERENCE_ID]
    assert model.preferences['gdn_recurrent'] == ('tir.gdn_recurrent.persistent',
        'tir.gdn_recurrent.persistent_k256', 'tir.gdn_recurrent.persistent_k512')


def test_standalone_request_only_and_iq4_provider():
    model = portable_triton_implementation_model()
    impl = model.implementation(R004_REFERENCE_ID)
    node = fm.Node('gdn', 'nn.gdn_recurrent_core', (), fm.tensor_type('float32', (1,)),
                   attrs=dict(b_weight_encoding='IQ4_XS', key_head_dim=128, phase_plan=reference_phase_plan()))
    ctx = SimpleNamespace(module=SimpleNamespace(metadata={}, stage='target_proposed'))
    assert not r004_reference_applicable(node, ctx, impl)
    assert not _gdn_applicable(node, ctx, impl)
    ctx.module.metadata = dict(r004_standalone_request=dict(schema='r004.gdn.standalone/v4',
                               module='r004_C_full_p5_v4', readonly_layer=3))
    assert r004_reference_applicable(node, ctx, impl)
    # The ordinary registry still rejects C. Ntt's explicit standalone facade
    # selects it from the real catalog without changing the default provider.
    assert not _gdn_applicable(node, ctx, impl)
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import configure_standalone_phases
    standalone = configure_standalone_phases(standalone_module())
    proposed = NvidiaSm120Target().propose_tir(standalone)
    point = next(p for p in proposed.selection_points if p.id == 'tir.recurrent')
    assert point.default_candidate == R004_REFERENCE_ID
    assert [c.id for c in point.candidates] == [R004_REFERENCE_ID]
    ordinary_module = standalone_module()
    ordinary_module = replace(ordinary_module, metadata={})
    ordinary_proposed = NvidiaSm120Target().propose_tir(ordinary_module)
    default_point = next(p for p in ordinary_proposed.selection_points if p.id == 'tir.recurrent')
    assert default_point.default_candidate == 'tir.gdn_recurrent.persistent'
    assert R004_REFERENCE_ID not in [c.id for c in default_point.candidates]
    ordinary = replace(node, attrs=dict(b_weight_encoding='IQ4_XS', key_head_dim=128))
    ctx.module.metadata = {}
    assert _gdn_applicable(ordinary, ctx, model.implementation('tir.gdn_recurrent.persistent'))
    assert not _gdn_applicable(ordinary, ctx, impl)


def standalone_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('r004_graph_fixture',Path(__file__).with_name('test_r004_phase_bindings.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return fixture.boundary()


def test_normal_materializer_retains_one_coarse_call_and_phase_arena():
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import configure_standalone_phases
    target = NvidiaSm120Target()
    selected = target.lower_to_tir(target.propose_tir(configure_standalone_phases(standalone_module())))
    assert sum(n.op == 'tir.call' for n in selected.nodes) == 1
    assert all(e['symbol'] in selected.kernel_callable_map for e in reference_phase_plan()['entries'])
    call = next(n for n in selected.nodes if n.op == 'tir.call')
    definition = selected.kernel_callable_map[call.attrs['callee']]
    assert len(definition.dispatch.workspaces) == 1
    workspace = definition.dispatch.workspaces[0]
    assert workspace.name == 'phase_pool'
    assert workspace.type == fm.tensor_type('uint8', (3212800,))
    assert workspace.memory_space == 'workspace'
    assert workspace.alignment == 256


def test_materialized_named_native_definitions_geometry_effects():
    coarse = coarse_definition()
    coarse = replace(coarse, dispatch=replace(coarse.dispatch, parameters={**coarse.dispatch.parameters,
        'distribution': {'input_types': [None] * 9}}))
    definitions = r004_phase_definitions(coarse)
    plan = reference_phase_plan()
    assert [d.name for d in definitions] == [e['symbol'] for e in plan['entries']]
    for d, e in zip(definitions, plan['entries'], strict=True):
        verify_prim_function(d)
        assert tuple(d.attrs['native_parameter_order']) == NATIVE_ARGUMENTS[e['phase']]
        assert set(d.parameter_map) == set(NATIVE_ARGUMENTS[e['phase']])
        assert list(d.dispatch.parameters['program_grid']) == e['grid']
        assert d.dispatch.parameters['program_owner_mapping'] == 'x-fast'
        assert 'distribution' not in d.dispatch.parameters
        assert d.attrs['kernel_num_warps'] == e['num_warps']
        assert d.attrs['kernel_maxnreg'] == 80
        assert d.attrs['kernel_enable_static_shared'] == e['enable_static_shared']
        assert d.attrs['depends_on'] == e['depends_on']
        assert d.dispatch.facts['internal_grid_barriers'] == 0
        assert d.attrs['phase_pool'] == plan['pool']
    core = next(d for d in definitions if d.name == "r004_C_core")
    assert core.parameter_map['StateParent'].role == PrimParameterRole.OUTPUT
    from triton.flagmega.ir import MemoryEffect
    assert dict(core.dispatch.memory_effects)['StateParent'] == MemoryEffect.WRITE
    assert dict(core.dispatch.memory_effects)['GatheredState'] == MemoryEffect.READ
    assert core.dispatch.writes == ('StateParent', 'Core')
    assert 'StateParent' not in core.dispatch.reads
    assert 'GatheredState' in core.dispatch.reads
    clear = next(d for d in definitions if d.name == 'r004_C_clear')
    assert clear.parameter_map['StateParent'].role == PrimParameterRole.OUTPUT
    assert dict(clear.dispatch.memory_effects)['StateParent'] == MemoryEffect.READ_WRITE
    assert clear.dispatch.writes == ('StateParent',)
    assert tuple(v.storage for v in clear.results.values) == ("StateParent",)
    for name in ('Layer','Scale','Bias'):
        assert clear.parameter_map[name].role == PrimParameterRole.INPUT
        assert clear.parameter_map[name].type.shape == ()
    assert clear.parameter_map['Scale'].type == fm.tensor_type('float32', ())
    assert clear.parameter_map['Bias'].type == fm.tensor_type('float32', ())
    assert all('StateParent' not in d.dispatch.writes for d in definitions if d not in (core,clear))


@pytest.mark.parametrize('entry', reference_phase_plan()['entries'], ids=lambda e: e['phase'])
def test_native_renderer_and_wrapper(entry):
    raw = raw_call(entry)
    call = _r004_gdn_phase_call(raw)
    assert call['arguments'] == list(NATIVE_ARGUMENTS[entry['phase']])
    assert call['argument_bindings'] == entry['argument_bindings']
    assert call['dynamic_argument_indices'] == entry['dynamic_argument_indices']
    assert call['requires_grid_wide_sync'] is False
    assert call['program_kind'] == 'ordinary'
    rendered = prepare_kernel_calls([raw], function_name='main')[0]
    source = TritonTemplateRegistry().render_kernel(KernelTemplateSpec('gdn_recurrent', 'r004_reference_v4'),
                                                    dict(render_calls=[rendered])).source
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == rendered['symbol'])
    assert [a.arg for a in fn.args.args] == call['arguments']
    assert [a.id for a in fn.body[0].value.args] == call['arguments']
    assert all(a.annotation is None for a in fn.args.args)
    bad = deepcopy(raw)
    bad['argument_bindings'][-1] = dict(kind='immediate', value=entry['argument_bindings'][-1])
    with pytest.raises(CodegenError):
        _r004_gdn_phase_call(bad)
    bad = deepcopy(raw)
    bad['inputs'][0]['buffers'][0]['abi']['scalar_dtype'] = 'float32'
    with pytest.raises(CodegenError):
        _r004_gdn_phase_call(bad)


def test_core_and_register_helpers_AST_fidelity():
    registry = TritonTemplateRegistry()
    source = registry.render_kernel(KernelTemplateSpec('gdn_recurrent', 'r004_reference_v4'), {}).source
    previous = registry.render_kernel(KernelTemplateSpec('gdn_recurrent', 'r004_reference_v3'), {}).source
    a = {n.name:n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    b = {n.name:n for n in ast.parse(previous).body if isinstance(n, ast.FunctionDef)}
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,n):
            return ast.copy_location(ast.Name(id=n.id.replace('_v4','_v3'),ctx=n.ctx),n)
    core = a['_r004_reference_core_v4']
    old = b['_r004_reference_core_v3']
    body = deepcopy(core.body)
    inserted = body.pop(5)
    assert inserted.targets[0].id == 'state_input_ptr'
    assert 'GatheredState' in ast.unparse(inserted)
    assert 'StateParent' not in ast.unparse(inserted)
    assert 'state_input_ptr' in ast.unparse(body[5])
    body[5] = old.body[5]
    assert ast.dump(Normalize().visit(ast.Module(body=body,type_ignores=[]))) == ast.dump(ast.Module(body=old.body,type_ignores=[]))
    for oldname,newname in [('extract_r','extract_r'),('local4_square','local4_square'),
                           ('gdn_reduce8_xor_v3','gdn_reduce8_xor_v4'),('reduce32_xor_v3','reduce32_xor_v4')]:
        assert ast.dump(Normalize().visit(ast.Module(body=deepcopy(a[newname].body),type_ignores=[]))) == ast.dump(ast.Module(body=b[oldname].body,type_ignores=[]))
    assignments = lambda src:{n.targets[0].id:ast.dump(n.value) for n in ast.parse(src).body if isinstance(n,ast.Assign)}
    for name,value in assignments(previous).items():
        assert assignments(source)[name] == value
    clear = a['_r004_reference_clear_v4']
    assert [arg.arg for arg in clear.args.args] == ['StateParent','Layer','Scale','Bias']
    assert ast.literal_eval(clear.decorator_list[0].keywords[0].value) == ['Scale','Bias']


def test_frozen_epsilon_checks_exact_F32_bits():
    from struct import pack,unpack
    raw=raw_call(reference_phase_plan()['entries'][0])
    bits=unpack('I',pack('f',1e-6))[0]
    raw['semantic_attrs']['epsilon']=unpack('f',pack('I',bits))[0]
    _r004_gdn_phase_call(raw)
    for adjacent in (bits-1,bits+1):
        raw['semantic_attrs']['epsilon']=unpack('f',pack('I',adjacent))[0]
        with pytest.raises(CodegenError):
            _r004_gdn_phase_call(raw)
