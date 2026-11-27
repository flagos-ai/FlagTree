# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Public frozen specialization dispatch and exact immutable identity checks."""
import inspect
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import triton.language as tl
from triton.flagmega.errors import ArtifactError, RuntimeContractError

def test_public_loader_has_explicit_frozen_native_keyword():
    from triton.flagmega.runtime.loader import load
    assert 'frozen_native' in inspect.signature(load).parameters

def _pin(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'nbytes': path.stat().st_size}

def _write(path, value):
    path.write_text(json.dumps(value) + '\n')
    return _pin(path)

class Pointer:
    dtype = tl.uint8
    def __init__(self, address=256):
        self.address = address
    def data_ptr(self):
        return self.address

@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from triton.flagmega.runtime import frozen_native as frozen
    from triton.compiler.compiler import ASTSource
    root = tmp_path
    source = root / 'entry.py'
    source.write_text('import triton\n@triton.jit\ndef entry(a, b):\n    pass\n')
    spec = importlib.util.spec_from_file_location('_frozen_fixture_' + root.name.replace('-', '_'), source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.entry
    attrs = {(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}
    signature = {'a': '*u8', 'b': '*u8'}
    ast = ASTSource(kernel, signature=signature, attrs=attrs)
    codegen = dict(kind='tir_call_graph/v1', source='entry.py', symbol='entry', grid=[1, 1, 1],
                   num_warps=4, dynamic_argument_indices=[0])
    manifest = dict(status='executable', semantic_hash='a' * 64, target='nvidia-sm90',
                    codegen_backend='triton', codegen=codegen)
    artifact = _write(root / 'artifact.json', manifest)
    driver = root / 'CPU-runtime-library-identity-carrier.so'
    driver.write_bytes(b'Explicit CPU identity carrier; not a loaded GPU runtime')
    monkeypatch.setattr(frozen, '_runtime_library_path', lambda: driver.resolve())
    monkeypatch.setattr(frozen, '_mapped_library_identity',
                        lambda path: [(driver.stat().st_dev, driver.stat().st_ino)])
    key = 'b' * 64
    metadata = dict(name='entry', hash=key, target=dict(backend='cuda', arch=90, warp_size=32),
                    num_warps=4, num_ctas=1, shared=0, cluster_dims=[1, 1, 1],
                    global_scratch_size=0, global_scratch_align=4,
                    profile_scratch_size=0, profile_scratch_align=1,
                    ptxas_stack_frame_bytes=0, ptxas_spill_store_bytes=0,
                    ptxas_spill_load_bytes=0, launch_cooperative_grid=False,
                    extern_libs=[['libdevice', '/CPU/libdevice.bc'], ['support', '/CPU/support.bc']],
                    ptx_options='--minnctapersm=1')
    group = {}
    for suffix in ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']:
        path = root / ('entry.' + suffix)
        payload = ('Explicit CPU assembly identity carrier ' + suffix).encode()
        if suffix == 'ptx':
            payload = b'.visible .entry entry(.param .u64 a, .param .u64 b, .param .u64 global_scratch, .param .u64 profile_scratch) {}\n'
        path.write_bytes(payload)
        group[path.name] = _pin(path)
    meta = _write(root / 'entry.json', metadata)
    group['entry.json'] = meta
    data = dict(schema='flagmega.frozen-native-specializations/v1', artifact_identity=artifact,
                semantic_hash=manifest['semantic_hash'], target='nvidia-sm90', codegen_backend='triton',
                expected_runtime_driver_identity=_pin(driver),
                compiler_provenance=dict(observed_library_identity=None,
                                         qualification='CPU fixture; historical compiler unobserved',
                                         evidence_refs=[artifact]),
                entries=[dict(symbol='entry', source_identity=_pin(source), native_key=key,
                              ast=dict(hash=ast.hash(), function_argument_names=['a', 'b'], signature=signature,
                                       attrs=[dict(key=list(k), value=v) for k, v in attrs.items()], constexprs=[]),
                              native_metadata_identity=meta, metadata_group=group,
                              program_binding=dict(grid=[1, 1, 1], dynamic_argument_indices=[0],
                                                   compute_num_warps=4))])
    return root, manifest, data, kernel

def _bundle(data):
    from triton.flagmega.runtime.frozen_native import FrozenNativeSpecializations
    return FrozenNativeSpecializations.from_data(data)

def _bound(fixture):
    root, manifest, data, kernel = fixture
    result = _bundle(data).bind(root, manifest)
    result.validate_kernels((kernel,))
    return result

def _resource_carrier(bound):
    # CompiledKernel itself is real. Only the absent live CU handle and counters
    # are explicit CPU carriers; no CUDA function or launcher is invoked.
    compiled = bound._compiled
    compiled.module = object()
    compiled.function = object()
    compiled._run = lambda *args: None
    compiled.n_regs, compiled.n_spills = 24, 0
    return compiled

def test_real_lazy_compiled_constructor_preserves_native_key(fixture):
    bound = _bound(fixture)
    assert bound._compiled.hash == fixture[2]['entries'][0]['native_key']
    assert bound._compiled.module is None and bound._compiled.function is None and bound._compiled._run is None
    assert hashlib.sha256(bound._compiled.kernel).hexdigest() == fixture[2]['entries'][0]['metadata_group']['entry.cubin']['sha256']

def test_bundle_does_not_follow_mutation_of_caller_data(fixture):
    root, manifest, data, kernel = fixture
    value = _bundle(data)
    data['entries'].clear()
    bound = value.bind(root, manifest)
    bound.validate_kernels((kernel,))

@pytest.mark.parametrize('field', ['source_identity', 'native_metadata_identity'])
def test_changed_pinned_source_or_metadata_rejected(fixture, field):
    root, manifest, data, kernel = fixture
    path = Path(data['entries'][0][field]['path'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ArtifactError, match='changed file'):
        _bundle(data).bind(root, manifest)

@pytest.mark.parametrize('suffix', ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin'])
def test_every_changed_assembly_member_rejected(fixture, suffix):
    root, manifest, data, kernel = fixture
    path = Path(data['entries'][0]['metadata_group']['entry.' + suffix]['path'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ArtifactError, match='changed file'):
        _bundle(data).bind(root, manifest)

@pytest.mark.parametrize('change', ['key', 'symbol', 'ast_hash', 'signature', 'attrs', 'constexprs',
                                  'dynamic', 'grid', 'compute', 'semantic', 'target', 'missing', 'extra'])
def test_exact_specialization_mismatches_reject(fixture, change):
    root, manifest, data, kernel = fixture
    data = copy.deepcopy(data)
    entry = data['entries'][0]
    if change == 'key': entry['native_key'] = 'c' * 64
    if change == 'symbol': entry['symbol'] = 'foreign'
    if change == 'ast_hash': entry['ast']['hash'] = 'c' * 64
    if change == 'signature': entry['ast']['signature']['a'] = '*fp32'
    if change == 'attrs': entry['ast']['attrs'][0]['value'][0][1] = 8
    if change == 'constexprs': entry['ast']['constexprs'] = [dict(key=[0], value=1)]
    if change == 'dynamic': entry['program_binding']['dynamic_argument_indices'] = [1]
    if change == 'grid': entry['program_binding']['grid'] = [2, 1, 1]
    if change == 'compute': entry['program_binding']['compute_num_warps'] = 8
    if change == 'semantic': data['semantic_hash'] = 'c' * 64
    if change == 'target': data['target'] = 'nvidia-sm120'
    if change == 'missing': data['entries'] = []
    if change == 'extra': data['entries'].append(copy.deepcopy(entry))
    with pytest.raises((ArtifactError, ValueError, TypeError)):
        value = _bundle(data).bind(root, manifest)
        value.validate_kernels((kernel,))

@pytest.mark.parametrize('value', [True, 90.0])
def test_raw_metadata_target_numeric_aliases_reject(fixture, value):
    root, manifest, data, kernel = fixture
    path = Path(data['entries'][0]['native_metadata_identity']['path'])
    metadata = json.loads(path.read_text());metadata['target']['arch'] = value
    new_pin = _write(path, metadata)
    data['entries'][0]['native_metadata_identity'] = new_pin
    data['entries'][0]['metadata_group']['entry.json'] = new_pin
    with pytest.raises(ArtifactError, match='target'):
        _bundle(data).bind(root, manifest)

def test_loaded_runtime_path_or_mapping_mismatch_rejects(fixture, monkeypatch):
    from triton.flagmega.runtime import frozen_native as frozen
    monkeypatch.setattr(frozen, '_runtime_library_path', lambda: Path('/different/runtime.so'))
    with pytest.raises(ArtifactError, match='loaded extension'):
        _bound(fixture)
    monkeypatch.setattr(frozen, '_runtime_library_path', lambda: Path(fixture[2]['expected_runtime_driver_identity']['path']))
    monkeypatch.setattr(frozen, '_mapped_library_identity', lambda path: [])
    with pytest.raises(ArtifactError, match='inode/device'):
        _bound(fixture)

def test_prepare_uses_existing_resource_and_static_alignment_checks(fixture):
    from triton.flagmega.runtime.prepared import ResourceContract, PreparedKernel
    bound = _bound(fixture)
    compiled = _resource_carrier(bound)
    result = bound.prepare(fixture[3], (Pointer(), Pointer()), (0,), grid=(1, 1, 1),
                           contract=ResourceContract(4, 1), extern_libs={'support': '/CPU/support.bc'})
    assert isinstance(result, PreparedKernel) and result.compiled_kernel is compiled
    with pytest.raises(RuntimeContractError, match='16-byte'):
        result.launch(Pointer(260), stream=1)
    compiled.n_regs = 256
    with pytest.raises(RuntimeContractError, match='registers/thread'):
        bound.prepare(fixture[3], (Pointer(), Pointer()), (0,), grid=(1, 1, 1), contract=ResourceContract(4, 1))

@pytest.mark.parametrize('argument', [Pointer(260), SimpleNamespace(dtype=tl.float32, data_ptr=lambda: 256)])
def test_projected_live_specialization_mismatch_rejects_before_resource_loading(fixture, argument):
    from triton.flagmega.runtime.prepared import ResourceContract
    bound = _bound(fixture)
    with pytest.raises(ArtifactError, match='projected live'):
        bound.prepare(fixture[3], (argument, Pointer()), (0,), grid=(1, 1, 1), contract=ResourceContract(4, 1))
    assert bound._compiled.module is None

def _public_fixture(fixture, monkeypatch):
    from triton.flagmega.runtime import loader, module as runtime
    from triton.flagmega.runtime.prepared import ResourceContract
    root, manifest, data, kernel = fixture
    model = runtime.GeneratedTirCallGraphModule.__new__(runtime.GeneratedTirCallGraphModule)
    model.artifact, model.manifest, model.codegen = root, manifest, manifest['codegen']
    model.ir_module = SimpleNamespace(metadata={})
    model.kernels = (kernel,)
    model.runtime_binding = dict(pools=[])
    model._pool_values = {}
    model._needs_device_dptr = False
    model._function_offset_values = ()
    model._device, model._prepared, model.prepare_count = 'CPU-carrier', None, 0
    model._require_loaded = lambda: None
    model._validate_external_arguments = lambda args: None
    model._materialize_tensor_descriptors = lambda args: ()
    model._mark_prepared = lambda: None
    factory = lambda *args: model
    monkeypatch.setattr(loader, 'load_artifact', lambda *args, **kwargs: (manifest, None))
    monkeypatch.setattr(loader, 'create_tir_runtime', factory)
    monkeypatch.setattr(loader.package_registry, 'resolve', lambda *args: SimpleNamespace(factory=factory))
    monkeypatch.setattr(loader, '_import_source', lambda *args: SimpleNamespace(entry=kernel))
    monkeypatch.setattr(runtime, '_requires_grid_wide_sync', lambda ir: False)
    monkeypatch.setattr(runtime, '_resource_contract', lambda *args, **kwargs: ResourceContract(4, 1))
    monkeypatch.setattr(runtime, 'compilation_options', lambda *args, **kwargs: {'num_warps': 4})
    return loader, model

def test_public_runtime_load_and_program_projection_use_frozen_route(fixture, monkeypatch):
    from triton.flagmega.runtime import module as runtime
    root, manifest, data, kernel = fixture
    loader, model = _public_fixture(fixture, monkeypatch)
    monkeypatch.setattr(runtime, 'prepare_jit_kernel', lambda *args, **kwargs: pytest.fail('selected path called JIT'))
    loaded = loader.load(root, frozen_native=_bundle(data))
    _resource_carrier(loaded._frozen_native)
    loaded._prepare_external((Pointer(), Pointer()))
    assert loaded._prepared[0].compiled_kernel.hash == data['entries'][0]['native_key']
    assert loaded.prepare_count == 1

def test_default_public_route_retains_normal_jit_preparation(fixture, monkeypatch):
    root, manifest, data, kernel = fixture
    loader, model = _public_fixture(fixture, monkeypatch)
    compiled = _resource_carrier(_bound(fixture))
    calls = []
    monkeypatch.setattr(kernel, 'run', lambda *args, **kwargs: (calls.append(kwargs), compiled)[1])
    loaded = loader.load(root)
    loaded._prepare_external((Pointer(), Pointer()))
    assert calls and calls[0]['warmup'] is True
    assert not hasattr(loaded, '_frozen_native')
    assert loaded._prepared[0].compiled_kernel is compiled

@pytest.mark.parametrize('change', ['cpp', 'sequence', 'schema'])
def test_unsupported_package_or_program_sequence_rejects_before_factory(fixture, monkeypatch, change):
    from triton.flagmega.runtime import loader
    root, manifest, data, kernel = fixture
    if change == 'cpp': manifest['codegen']['kind'] = 'tir_call_graph_cpp/v1'
    if change == 'sequence': manifest['codegen']['kernels'] = [dict(symbol='entry')]
    if change == 'schema': manifest['codegen']['kernel_program_schema'] = 'flagmega.kernel-programs/v4-functions'
    data['artifact_identity'] = _write(root / 'artifact.json', manifest)
    data['compiler_provenance']['evidence_refs'] = [data['artifact_identity']]
    monkeypatch.setattr(loader, 'load_artifact', lambda *args, **kwargs: (manifest, None))
    monkeypatch.setattr(loader.package_registry, 'resolve', lambda *args: pytest.fail('adapter factory reached'))
    with pytest.raises(ArtifactError, match='unsupported artifact/package/program'):
        loader.load(root, device='cuda:0', frozen_native=_bundle(data))

def test_source_alias_and_extra_metadata_member_reject(fixture):
    root, manifest, data, kernel = fixture
    alias = root / 'alias.py';alias.write_bytes(Path(data['entries'][0]['source_identity']['path']).read_bytes())
    changed = copy.deepcopy(data);changed['entries'][0]['source_identity'] = _pin(alias)
    with pytest.raises(ArtifactError, match='source path'):
        _bundle(changed).bind(root, manifest)
    changed = copy.deepcopy(data);changed['entries'][0]['metadata_group']['extra.json'] = data['entries'][0]['native_metadata_identity']
    with pytest.raises(ArtifactError, match='complete exact'):
        _bundle(changed)

def test_changed_runtime_identity_rejected_after_binding(fixture):
    bound = _bound(fixture)
    driver = Path(fixture[2]['expected_runtime_driver_identity']['path'])
    driver.write_bytes(driver.read_bytes() + b'changed')
    with pytest.raises(ArtifactError, match='changed file'):
        bound.verify_artifact(fixture[0], fixture[1], fixture[1]['codegen'])

def test_frozen_options_do_not_silently_change_the_selected_binary(fixture):
    from triton.flagmega.runtime.prepared import ResourceContract
    bound = _bound(fixture)
    with pytest.raises(ArtifactError, match='external library'):
        bound.prepare(fixture[3], (Pointer(), Pointer()), (0,), grid=(1, 1, 1), contract=ResourceContract(4, 1),
                      extern_libs={'support': '/different/library.bc'})
    with pytest.raises(ArtifactError, match='unknown frozen preparation option'):
        bound.prepare(fixture[3], (Pointer(), Pointer()), (0,), grid=(1, 1, 1), contract=ResourceContract(4, 1),
                      unknown=None)

@pytest.mark.parametrize('field,value', [('global_scratch_align', 0), ('profile_scratch_align', 3),
                                       ('shared', 0.0), ('ptxas_spill_store_bytes', True),
                                       ('cluster_dims', [1.0, 1, 1])])
def test_native_metadata_has_no_silent_identity_defaults(fixture, field, value):
    root, manifest, data, kernel = fixture
    path = Path(data['entries'][0]['native_metadata_identity']['path'])
    metadata = json.loads(path.read_text());metadata[field] = value
    new_pin = _write(path, metadata)
    data['entries'][0]['native_metadata_identity'] = new_pin
    data['entries'][0]['metadata_group']['entry.json'] = new_pin
    with pytest.raises(ArtifactError):
        _bundle(data).bind(root, manifest)

def test_native_entry_scratch_parameter_abi_is_verified(fixture):
    root, manifest, data, kernel = fixture
    path = Path(data['entries'][0]['metadata_group']['entry.ptx']['path'])
    path.write_bytes(b'.visible .entry entry(.param .u64 a, .param .u64 b) {}\n')
    data['entries'][0]['metadata_group']['entry.ptx'] = _pin(path)
    with pytest.raises(ArtifactError, match='native pointer ABI'):
        _bundle(data).bind(root, manifest)

def test_serving_consumes_selector_before_backend_constructor(monkeypatch):
    import triton.flagmega.runtime as runtime
    from triton.flagmega.serving.backend import ArtifactBackend
    events = []
    model = SimpleNamespace(close=lambda: events.append('close'))
    def load(path, **kwargs):
        events.append(kwargs)
        return model
    monkeypatch.setattr(runtime, 'load', load)
    class Carrier(ArtifactBackend):
        def __init__(self, received, **kwargs):
            assert received is model and 'frozen_native' not in kwargs
            self.kwargs = kwargs
    selector = object()
    result = Carrier.load('artifact', device='CPU-carrier', frozen_native=selector, cuda_graph=False)
    assert events == [dict(device='CPU-carrier', frozen_native=selector)]
    assert result.kwargs == dict(cuda_graph=False)

def test_public_loader_rejects_untyped_selector_before_artifact_or_device(monkeypatch):
    from triton.flagmega.runtime import loader
    monkeypatch.setattr(loader, 'load_artifact', lambda *args, **kwargs: pytest.fail('artifact decoded'))
    with pytest.raises(ArtifactError, match='typed FrozenNative'):
        loader.load('/missing/artifact', device='cuda:0', frozen_native={})
