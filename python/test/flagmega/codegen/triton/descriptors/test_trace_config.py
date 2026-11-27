# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed diagnostic rendering retains the ordinary model and caller ABI."""
from copy import deepcopy
from types import SimpleNamespace
import ast

import pytest

from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.codegen.triton.renderer import render_triton_package
from triton.flagmega.codegen.triton.trace_config import TracePlan, digest, validate_option
from triton.flagmega.errors import CodegenError


def config(module, call, *, layer=17):
    return {'schema': 'flagmega.triton-trace/v1', 'base_semantic_hash': module.semantic_hash,
            'layer_bindings': {'main': {'constant': layer}},
            'detailed_calls': [{'function': 'main', 'call': call}],
            'typed_ctas': [0], 'capacity': 2048, 'coarse_calls': True}


@pytest.mark.parametrize('fixture,pipe_sites', [
    ('packed_glu_paired_full_lhs_descriptor_pipeline_module', 3),
    ('packed_glu_direct_lhs_descriptor_pipeline_module', 6),
])
def test_normal_render_threads_context_and_preserves_none_bytes(request, fixture, pipe_sites, tmp_path):
    module = request.getfixturevalue(fixture)
    before = module.semantic_hash
    plain = render_triton_package(module, tmp_path/'plain')
    explicit_none = render_triton_package(module, tmp_path/'none', triton_trace=None)
    assert plain == explicit_none
    assert (tmp_path/'plain/generated_kernels.py').read_bytes() == (tmp_path/'none/generated_kernels.py').read_bytes()
    call, = (c for c in describe_tir_package(module)['render_calls'] if c['family'] == 'dense_matmul_glu')
    options = config(module, call['call'])
    traced = render_triton_package(module, tmp_path/'traced', triton_trace=options)
    source = (tmp_path/'traced/generated_kernels.py').read_text()
    compile(source, 'diagnostic.py', 'exec')
    entry = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == traced['symbol'])
    calls = [node for node in ast.walk(entry) if isinstance(node, ast.Call)]
    configure, = [node for node in calls if ast.unparse(node.func) == 'tle.gpu.trace_configure']
    assert {kw.arg: ast.literal_eval(kw.value) for kw in configure.keywords} == {
        'capacity': options['capacity'], 'typed_ctas': options['typed_ctas']}
    specialize = next(node for node in calls if ast.unparse(node.func) == 'tle.gpu.warp_specialize')
    assert configure.lineno < specialize.lineno
    assert 'trace_configure' not in (tmp_path/'plain/generated_kernels.py').read_text()
    assert 'tle.gpu.trace_event(272,' in source and 'tle.gpu.trace_event(273,' in source
    assert 'trace=(' in source and '_flagmega_trace_layer_id' in source
    assert 'tl.full((), 17, tl.int32)' in source
    assert 'ProfileScratch' not in source and 'globaltimer' not in source
    metadata = traced['triton_trace']
    assert metadata['native_contract']['runtime_stream_filtering'] is False
    assert metadata['native_contract']['typed_ctas'] == [0]
    assert metadata['native_contract']['capacity'] == 2048
    assert metadata['config_sha256'] == digest(options)
    assert metadata['dictionary_sha256'] == digest(metadata['dictionary'])
    sites = metadata['dictionary']['sites']
    assert len([r for r in sites if r['key']['kind'] == 'pipe_boundary']) == pipe_sites
    assert len({r['site_id'] for r in sites}) == len(sites)
    assert module.semantic_hash == before
    assert render_triton_package(module, tmp_path/'again') == plain


@pytest.mark.parametrize('field,value,message', [
    ('typed_ctas', [0, 0], 'allowlist'), ('typed_ctas', [-1], 'allowlist'),
    ('typed_ctas', [False], 'allowlist'), ('typed_ctas', [], 'allowlist'),
    ('capacity', 0, 'capacity'), ('coarse_calls', False, 'complete caller'),
    ('schema', 'unknown', 'schema'),
])
def test_invalid_configuration_is_rejected(field, value, message):
    options = config(SimpleNamespace(semantic_hash='a'*64), 'glu')
    options[field] = value
    with pytest.raises(CodegenError, match=message):
        validate_option(options, target='nvidia-sm90')


@pytest.mark.parametrize('fixture', [
    'packed_glu_paired_full_lhs_descriptor_pipeline_module',
    'packed_glu_direct_lhs_descriptor_pipeline_module',
])
def test_minimal_ring_configures_complete_coarse_scope(request, fixture, tmp_path):
    module = request.getfixturevalue(fixture)
    call, = (c for c in describe_tir_package(module)['render_calls'] if c['family'] == 'dense_matmul_glu')
    options = config(module, call['call'])
    options['detailed_calls'] = []
    package = render_triton_package(module, tmp_path, triton_trace=options)
    source = (tmp_path/'generated_kernels.py').read_text()
    assert source.count('tle.gpu.trace_configure(') == 1
    assert 'tle.gpu.trace_configure(capacity=2048, typed_ctas=[0])' in source
    assert 'tle.gpu.trace_event(272,' in source and 'tle.gpu.trace_event(273,' in source
    assert '_flagmega_trace_layer_id' not in source
    assert package['triton_trace']['native_contract']['typed_ctas'] == options['typed_ctas']


@pytest.mark.parametrize('target,backend', [('amd-gfx1201', 'triton'), ('nvidia-sm90', 'cpp')])
def test_unsupported_backend_is_rejected(target, backend):
    with pytest.raises(CodegenError, match='SM90 Triton'):
        validate_option(config(SimpleNamespace(semantic_hash='a'*64), 'glu'), target=target, codegen_backend=backend)


def nested_plan(*, dtype='int32'):
    options = config(SimpleNamespace(semantic_hash='a'*64), 'glu')
    options['detailed_calls'] = []
    options['layer_bindings']['decoder'] = {'formal': 'layer_value'}
    bindings = {'main': {'arguments': []}, 'decoder': {'arguments': [
        {'buffer': 'layer_value', 'name': 'real_layer', 'runtime_value_kind': 'scalar', 'scalar_dtype': dtype}]}}
    return TracePlan(SimpleNamespace(semantic_hash='a'*64, entry='main'), bindings, {}, options)


def nested_descriptor():
    return {'platform': 'nvidia', 'target_architecture': 'sm90', 'grid': [2,1,1],
            'pipeline_schedule': {'consumer_events': [
                {'kind':'function_call','call':'first','callee':'decoder',
                 'argument_expressions':('endpoint', 'tl.full((), 17, tl.int32)')},
                {'kind':'function_call','call':'second','callee':'decoder',
                 'argument_expressions':('endpoint', 'tl.full((), 29, tl.int32)')} ]},
            'device_functions': [{'function':'decoder','signature_arguments':['real_layer'], 'schedule': {
                'consumer_events':[{'kind':'pipeline_kernel_call','call':'gate', 'family':'dense_matmul_glu',
                                    'variant':'packed_tensor_descriptor_smem_pipeline_gemv'}]}}]}


def test_shared_function_uses_actual_dynamic_layer_and_separate_call_sites():
    plan = nested_plan()
    descriptor = nested_descriptor()
    result = plan.finish(descriptor)
    first, second = result['pipeline_schedule']['consumer_events']
    assert first['trace_layer'] == 'tl.full((), 17, tl.int32)'
    assert second['trace_layer'] == 'tl.full((), 29, tl.int32)'
    assert first['trace_site'] != second['trace_site']
    leaf, = result['device_functions'][0]['schedule']['consumer_events']
    assert leaf['trace_layer'] == 'real_layer'
    assert len(result['triton_trace']['dictionary']['sites']) == 3
    replay = nested_plan().finish(nested_descriptor())
    assert result['triton_trace']['dictionary_sha256'] == replay['triton_trace']['dictionary_sha256']


def test_non_i32_semantic_layer_is_not_implicitly_narrowed():
    with pytest.raises(CodegenError, match='scalar int32'):
        nested_plan(dtype='int64')


@pytest.mark.parametrize('change,message', [('predicate','uniform'), ('loop','structured-loop'), ('grid','outside')])
def test_unproven_schedule_forms_are_rejected(change, message):
    plan = nested_plan(); descriptor = nested_descriptor()
    if change == 'predicate':
        descriptor['pipeline_schedule']['consumer_events'][0]['participation_active'] = 'active'
    elif change == 'loop':
        descriptor['pipeline_schedule']['consumer_events'].append({'kind':'loop'})
    else:
        plan.config['typed_ctas'] = [2]
    with pytest.raises(CodegenError, match=message):
        plan.finish(descriptor)


def test_site_hash_collision_is_an_error(monkeypatch):
    from triton.flagmega.codegen.triton import trace_config
    plan = nested_plan()
    values = iter(['00000001'+'a'*56, '00000001'+'b'*56])
    monkeypatch.setattr(trace_config, 'digest', lambda value: next(values))
    plan.site('decoder','consumer','one','device_call_envelope')
    with pytest.raises(CodegenError, match='collision'):
        plan.site('decoder','consumer','two','device_call_envelope')


def test_refresh_cannot_silently_drop_diagnostic_configuration(monkeypatch, tmp_path):
    import importlib
    artifact = importlib.import_module('triton.flagmega.artifacts.module')
    from triton.flagmega.errors import ArtifactError
    manifest = {'status':'executable', 'codegen_backend':'triton',
                'flagmega_artifact_version':artifact.ARTIFACT_VERSION, 'codegen':{'triton_trace':{'schema':'test'}}}
    monkeypatch.setattr(artifact, 'load_artifact', lambda *args, **kwargs: (manifest, None))
    with pytest.raises(ArtifactError, match='fresh emission'):
        artifact.refresh_artifact_codegen(tmp_path)
