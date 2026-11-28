# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit immutable precompiled specializations for the normal TIR runtime.

Native/compiler provenance describes the captured binary; the expected loaded
runtime library is a separate, mandatory identity. No cache lookup, JIT compile,
binary rewrite, global driver replacement or alternative launch path occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

from triton.flagmega.artifacts.manifest import resolve_artifact_path
from triton.flagmega.artifacts.program_sources import program_source_records
from triton.flagmega.errors import ArtifactError

_SCHEMA = 'flagmega.frozen-native-specializations/v1'
_SEAL = object()


def _error(message):
    raise ArtifactError('Frozen native specialization: ' + message)


def _object(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        _error(f'invalid {label} fields.')


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ArtifactError('Frozen native input requires finite JSON data.') from error


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        _error('requires complete SHA256 identities.')


def _pin(value):
    _object(value, {'path', 'sha256', 'nbytes'}, 'file identity')
    _sha(value['sha256'])
    if type(value['path']) is not str or not Path(value['path']).is_absolute():
        _error('file identities require absolute paths.')
    if type(value['nbytes']) is not int or value['nbytes'] < 0:
        _error('file byte counts require nonnegative integers.')


def _stat(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@lru_cache(maxsize=128)
def _digest(path, identity):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if _stat(Path(path)) != identity:
        _error('file changed while verifying its digest.')
    return digest.hexdigest()


def _verify_file(value):
    _pin(value)
    path = Path(value['path'])
    try:
        identity = _stat(path)
        if identity[2] != value['nbytes'] or _digest(str(path), identity) != value['sha256']:
            _error(f'changed file identity {path}.')
        if _stat(path) != identity:
            _error(f'file changed during verification {path}.')
    except OSError as error:
        raise ArtifactError(f'Cannot read frozen native identity {path}.') from error
    return path


def _json_file(value):
    path = _verify_file(value)
    try:
        data = json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as error:
        raise ArtifactError('Frozen native identity must contain JSON.') from error
    _verify_file(value)
    if type(data) is not dict:
        _error('JSON identity must contain an object.')
    return data


def _runtime_library_path():
    from triton._C import libtriton
    return Path(libtriton.__file__).resolve()


def _mapped_library_identity(path):
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5].endswith(' (deleted)'):
            continue
        if Path(fields[5]).resolve() == path:
            major, minor = (int(value, 16) for value in fields[3].split(':'))
            yield os.makedev(major, minor), int(fields[4])


def _verify_runtime_driver(value):
    path = _verify_file(value).resolve()
    if _runtime_library_path() != path:
        _error('expected runtime library differs from the loaded extension path.')
    stat = path.stat()
    if (stat.st_dev, stat.st_ino) not in set(_mapped_library_identity(path)):
        _error('expected runtime library inode/device is not mapped in this process.')
    _verify_file(value)


def _indices(value, label):
    if (type(value) is not list or any(type(item) is not int or item < 0 for item in value)
            or len(set(value)) != len(value)):
        _error(f'invalid {label}.')


def _validate(data):
    _object(
        data, {
            'schema', 'artifact_identity', 'semantic_hash', 'target', 'codegen_backend',
            'expected_runtime_driver_identity', 'compiler_provenance', 'entries'
        }, 'bundle')
    if data['schema'] != _SCHEMA or data['target'] != 'nvidia-sm90' or data['codegen_backend'] != 'triton':
        _error('unsupported schema, target or backend.')
    _pin(data['artifact_identity'])
    _sha(data['semantic_hash'])
    _pin(data['expected_runtime_driver_identity'])
    provenance = data['compiler_provenance']
    _object(provenance, {'observed_library_identity', 'qualification', 'evidence_refs'}, 'compiler provenance')
    if provenance['observed_library_identity'] is not None:
        _pin(provenance['observed_library_identity'])
    if type(provenance['qualification']) is not str or not provenance['qualification'].strip():
        _error('compiler provenance requires an explicit qualification.')
    if type(provenance['evidence_refs']) is not list or not provenance['evidence_refs']:
        _error('compiler provenance requires pinned evidence.')
    for item in provenance['evidence_refs']:
        _pin(item)
    if type(data['entries']) is not list or len(data['entries']) != 1:
        _error('requires exactly the supported single entry, with no missing or extra specialization.')
    for entry in data['entries']:
        _object(
            entry, {
                'symbol', 'source_identity', 'native_key', 'ast', 'native_metadata_identity', 'metadata_group',
                'program_binding'
            }, 'entry')
        if type(entry['symbol']) is not str or not entry['symbol'].isidentifier():
            _error('invalid entry symbol.')
        _sha(entry['native_key'])
        _pin(entry['source_identity'])
        _pin(entry['native_metadata_identity'])
        ast = entry['ast']
        _object(ast, {'hash', 'function_argument_names', 'signature', 'attrs', 'constexprs'}, 'AST')
        _sha(ast['hash'])
        names, signature = ast['function_argument_names'], ast['signature']
        if (type(names) is not list or not names or any(type(name) is not str for name in names)
                or len(set(names)) != len(names) or type(signature) is not dict or set(signature) != set(names)):
            _error('AST argument names/signature must cover the complete source ABI.')
        if any(type(value) is not str or not value.startswith('*') for value in signature.values()):
            _error('the supported frozen entry requires the captured pointer ABI.')
        for label in ['attrs', 'constexprs']:
            records = ast[label]
            if type(records) is not list:
                _error(f'invalid AST {label}.')
            seen = set()
            for record in records:
                _object(record, {'key', 'value'}, f'AST {label} record')
                _indices(record['key'], f'AST {label} key')
                key = tuple(record['key'])
                if not key or key in seen or key[0] >= len(names):
                    _error(f'invalid or duplicate AST {label} key.')
                seen.add(key)
        binding = entry['program_binding']
        _object(binding, {'grid', 'dynamic_argument_indices', 'compute_num_warps'}, 'program binding')
        _indices(binding['dynamic_argument_indices'], 'dynamic argument indices')
        if any(index >= len(names) for index in binding['dynamic_argument_indices']):
            _error('dynamic argument index exceeds the source ABI.')
        if (type(binding['grid']) is not list or len(binding['grid']) != 3
                or any(type(value) is not int or value <= 0 for value in binding['grid'])):
            _error('requires three positive grid dimensions.')
        if type(binding['compute_num_warps']) is not int or binding['compute_num_warps'] <= 0:
            _error('compute warps must be a positive integer.')
        group = entry['metadata_group']
        extensions = {'source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'json'}
        required = {entry['symbol'] + '.' + extension for extension in extensions}
        if type(group) is not dict or set(group) != required:
            _error('metadata group must contain the complete exact source/assembly membership.')
        for name, item in group.items():
            _pin(item)
            if Path(item['path']).name != name:
                _error('metadata group member filename differs from its identity.')
        if group[entry['symbol'] + '.json'] != entry['native_metadata_identity']:
            _error('raw metadata identity differs from the group member.')


@dataclass(frozen=True)
class FrozenNativeSpecializations:
    """Verified JSON value; edits to caller input cannot change this binding."""
    _serialized: str
    _seal: object

    @classmethod
    def from_data(cls, data):
        serialized = _canonical(data)
        _validate(json.loads(serialized))
        return cls(serialized, _SEAL)

    @classmethod
    def from_manifest(cls, identity):
        return cls.from_data(_json_file(identity))

    def bind(self, artifact, manifest):
        if self._seal is not _SEAL:
            _error('requires a binding constructed through from_data/from_manifest.')
        data = json.loads(self._serialized)
        _validate(data)
        path = Path(artifact).resolve()
        if Path(data['artifact_identity']['path']).resolve() != path / 'artifact.json':
            _error('artifact path does not match the selected bundle.')
        if _canonical(_json_file(data['artifact_identity'])) != _canonical(manifest):
            _error('loaded artifact metadata differs from its exact identity.')
        codegen = manifest.get('codegen', {})
        if (manifest.get('semantic_hash') != data['semantic_hash'] or manifest.get('target') != data['target']
                or manifest.get('codegen_backend', 'triton') != 'triton' or codegen.get('kind') != 'tir_call_graph/v1'
                or codegen.get('kernels') is not None or codegen.get('kernel_program_schema') is not None):
            _error('unsupported artifact/package/program sequence or semantic identity.')
        entry = data['entries'][0]
        if codegen.get('symbol') != entry['symbol']:
            _error('entry symbol does not match the artifact.')
        records = program_source_records(codegen)
        record = next((item for item in records if item['symbol'] == entry['symbol']), None)
        source = resolve_artifact_path(path, record['source'] if record else codegen.get('source', ''))
        if source.resolve() != Path(entry['source_identity']['path']).resolve():
            _error('source path does not match the artifact program.')
        binding = entry['program_binding']
        if (binding['grid'] != list(codegen.get('grid', ()))
                or binding['dynamic_argument_indices'] != list(codegen.get('dynamic_argument_indices', ()))
                or binding['compute_num_warps'] != codegen.get('num_warps')):
            _error('static/dynamic/grid/compute-warp program binding differs.')
        result = BoundFrozenNative(data, path, manifest, _SEAL)
        result.verify_artifact(path, manifest, codegen)
        return result


class BoundFrozenNative:

    def __init__(self, data, artifact, manifest, seal):
        if seal is not _SEAL:
            _error('requires an authenticated bundle binding.')
        self._data, self._artifact, self._manifest = data, artifact, _canonical(manifest)
        self._compiled = None

    def verify_artifact(self, artifact, manifest, codegen):
        if (Path(artifact).resolve() != self._artifact or _canonical(manifest) != self._manifest
                or _canonical(codegen) != _canonical(manifest['codegen'])):
            _error('bound runtime artifact/codegen changed.')
        _json_file(self._data['artifact_identity'])
        _verify_runtime_driver(self._data['expected_runtime_driver_identity'])
        provenance = self._data['compiler_provenance']
        for pin in provenance['evidence_refs']:
            _verify_file(pin)
        if provenance['observed_library_identity'] is not None:
            _verify_file(provenance['observed_library_identity'])
        entry = self._data['entries'][0]
        _verify_file(entry['source_identity'])
        for pin in entry['metadata_group'].values():
            _verify_file(pin)
        metadata = _json_file(entry['native_metadata_identity'])
        if metadata.get('hash') != entry['native_key'] or metadata.get('name') != entry['symbol']:
            _error('native key/entry differs from raw metadata.')
        if metadata.get('target') != {'backend': 'cuda', 'arch': 90, 'warp_size': 32}:
            _error('raw native target differs from the selected target.')
        if (type(metadata['target']['arch']) is not int or type(metadata['target']['warp_size']) is not int):
            _error('raw native target requires exact integer architecture/warp size.')
        for key in [
                'num_warps', 'num_ctas', 'shared', 'global_scratch_size', 'global_scratch_align',
                'profile_scratch_size', 'profile_scratch_align', 'ptxas_stack_frame_bytes', 'ptxas_spill_store_bytes',
                'ptxas_spill_load_bytes'
        ]:
            if type(metadata.get(key)) is not int or metadata[key] < 0:
                _error(f'raw native {key} requires exact integer metadata.')
        if metadata['num_warps'] <= 0 or metadata['num_ctas'] <= 0:
            _error('raw native warp/CTA count must be positive.')
        for key in ['global_scratch_align', 'profile_scratch_align']:
            value = metadata[key]
            if value <= 0 or value & (value - 1):
                _error('raw native scratch alignment must be a positive power of two.')
        cluster = metadata.get('cluster_dims')
        if (type(cluster) is not list or len(cluster) != 3
                or any(type(value) is not int or value <= 0 for value in cluster)):
            _error('raw native cluster dimensions require exact positive integers.')
        ptx = Path(entry['metadata_group'][entry['symbol'] + '.ptx']['path']).read_text()
        declaration = re.search(r'\.entry\s+' + re.escape(entry['symbol']) + r'\s*\((.*?)\)', ptx, re.S)
        # The supported NVIDIA pointer ABI appends global/profile scratch in
        # the ordinary launcher, independently of whether profile size is zero.
        parameter_count = len(entry['ast']['function_argument_names']) + 2
        if declaration is None or len(re.findall(r'\.param\s+\.u64\b', declaration[1])) != parameter_count:
            _error('PTX entry/native pointer ABI differs from source plus ordinary scratch arguments.')

    def validate_kernels(self, kernels):
        if len(kernels) != 1:
            _error('selected runtime must contain exactly one program.')
        from triton.compiler.compiler import ASTSource, CompiledKernel
        entry, kernel = self._data['entries'][0], kernels[0]
        ast = entry['ast']
        if list(kernel.arg_names) != ast['function_argument_names']:
            _error('source argument names differ from captured AST.')
        source = ASTSource(kernel, signature=ast['signature'],
                           attrs={tuple(row['key']): row['value']
                                  for row in ast['attrs']},
                           constexprs={tuple(row['key']): row['value']
                                       for row in ast['constexprs']})
        if source.hash() != ast['hash']:
            _error('source/signature/attrs/constexpr AST hash differs.')
        if self._compiled is not None and self._compiled.src.fn is kernel:
            return
        compiled = CompiledKernel(source, {name: item['path']
                                           for name, item in entry['metadata_group'].items()}, entry['native_key'])
        if hashlib.sha256(compiled.kernel).hexdigest() != entry['metadata_group'][entry['symbol'] + '.cubin']['sha256']:
            _error('constructed binary differs from its frozen CUBIN identity.')
        self._compiled = compiled

    def prepare(self, kernel, arguments, dynamic_indices, *, grid, contract, allocation_device=None, **options):
        self.verify_artifact(self._artifact, json.loads(self._manifest), json.loads(self._manifest)['codegen'])
        self.validate_kernels((kernel, ))
        entry = self._data['entries'][0]
        binding = entry['program_binding']
        if (list(grid) != binding['grid'] or list(dynamic_indices) != binding['dynamic_argument_indices']
                or contract.compute_num_warps != binding['compute_num_warps']):
            _error('prepared grid/dynamic/compute-warp binding differs.')
        for key, value in options.items():
            if key == 'num_warps':
                expected = binding['compute_num_warps']
            else:
                expected = getattr(self._compiled.metadata, key, None)
                if key == 'extern_libs':
                    expected = dict(expected or ())
                    if type(value) is not dict or any(expected.get(name) != path for name, path in value.items()):
                        _error('requested external library differs from the frozen metadata.')
                    continue
                if not hasattr(self._compiled.metadata, key):
                    _error(f'unknown frozen preparation option {key}.')
            if value != expected or (type(expected) in (int, bool) and type(value) is not type(expected)):
                _error(f'requested preparation option {key} differs from the frozen native metadata.')
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.compiler import CUDABackend
        from triton.compiler.compiler import ASTSource
        from triton.runtime.jit import create_function_from_signature
        backend = CUDABackend(GPUTarget('cuda', 90, 32))
        binder = create_function_from_signature(kernel.signature, kernel.params, backend)
        bound, specializations, raw_options = binder(*arguments, **options)
        _, signature, constants, attrs = kernel._pack_args(backend, options, bound, specializations, raw_options)
        live = ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs)
        if live.hash() != entry['ast']['hash']:
            _error('projected live arguments do not match the captured specialization.')
        from triton.flagmega.runtime.prepared import prepare_compiled_kernel
        result = prepare_compiled_kernel(self._compiled, arguments, dynamic_indices, grid=grid, contract=contract,
                                         allocation_device=allocation_device)
        self.verify_artifact(self._artifact, json.loads(self._manifest), json.loads(self._manifest)['codegen'])
        return result
