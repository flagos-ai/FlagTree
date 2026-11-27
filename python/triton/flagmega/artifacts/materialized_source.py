# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Verified reuse of complete, previously materialized readonly outputs.

The checkpoint retains its real identity. This module proves byte equality of
closed typed recipe outputs; it never evaluates recipes or reads image bytes.
The only additional equivalence is a legal full-logical sharded_view, including
its propagation through the logical-array bitcast evaluator. Boxing is retained.
"""
from __future__ import annotations

import ast
import hashlib
from functools import lru_cache
import json
import math
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from triton.flagmega.errors import ArtifactError
from triton.flagmega.ir import ConstantRecipe, DistributedType, VectorType, verify_module, verify_buffer_plan
from triton.flagmega.ir.constant_recipe import _plain
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import sharded_view_error
from triton.flagmega.ir.ops.core import get_definition, tensor_nbytes
from triton.flagmega.artifacts.source_verification import file_identity

SCHEMA = 'flagmega.materialized-output-source/v1'
_OPS = {'builtin.weight', 'builtin.splat_const', 'distributed.sharded_view',
        'distributed.boxing', 'math.vectorized_binary', 'tensors.bitcast', 'tensors.pack'}
# These annotations are not consumed by the pinned materializer/evaluator.
_ANNOTATIONS = frozenset('source_location consumer input_index realization source_type target_type callee introduced_by parameter_index vectorization_internal vectorization_role vectorization_root distributed_candidate distributed_placement distributed_reason lifted_constant_from_function prepared_from_vectorization selected_vector_axes selected_vector_lanes selected_vectorization vector_axes vector_lanes vectorization_attrs vectorization_candidate vectorization_inputs vectorization_rule vectorized_from rewritten_by'.split())
_NUMERIC_FILES = ('evaluator/numpy_backend.py', 'evaluator/recipe_scope.py', 'evaluator/torch_backend.py', 'evaluator/context.py',
                  'ir/ops/core.py', 'ir/distributed_inference.py', 'ir/distributed_type.py',
                  'ir/ops/builtin/weight.py', 'ir/ops/builtin/splat_const.py',
                  'ir/ops/distributed/sharded_view.py', 'ir/ops/distributed/boxing.py',
                  'ir/ops/math/vectorized_binary.py', 'ir/ops/tensors/bitcast.py',
                  'ir/ops/tensors/pack.py')
_TOKEN = object()


class _FrozenDict(dict):
    def _deny(self, *args, **kwargs):
        raise ArtifactError('Materialized source metadata is immutable')
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny
    def __deepcopy__(self, memo):
        from copy import deepcopy
        return {k: deepcopy(v, memo) for k, v in self.items()}


class _FrozenList(list):
    def _deny(self, *args, **kwargs):
        raise ArtifactError('Materialized source metadata is immutable')
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _deny
    def __deepcopy__(self, memo):
        from copy import deepcopy
        return [deepcopy(v, memo) for v in self]


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict((k, _freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze(v) for v in value)
    return value


def require(condition, message):
    if not condition:
        raise ArtifactError('Materialized source: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_pinned_json(pin):
    path = Path(pin['path'])
    require(path.suffix == '.json', 'only JSON metadata may be opened as evidence')
    raw = path.read_bytes()
    require(len(raw) == pin['nbytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'metadata evidence pin changed: ' + str(path))
    return json.loads(raw)


@lru_cache(maxsize=4)
def numerical_contract(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    text = (root / 'artifacts/rdata.py').read_text()
    tree = ast.parse(text)
    writers = {'_write_chip_sharded_buffer', '_extract_local_shard', '_as_numpy_array', '_range_numpy_array', '_tensor_buffer', '_write_rdata_entry', '_write_rdata_range'}
    functions = {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest() for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in writers}
    require(set(functions) == writers, 'numerical writer definitions missing')
    return {'byteorder': sys.byteorder, 'files': {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in _NUMERIC_FILES}, 'writer_functions': functions}


def _literal(node):
    attrs = _plain(node.attrs)
    if node.op == 'builtin.splat_const':
        value = attrs['value']; dtype = tensor_of(node.type).dtype
        scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
        formats = {'float32': '<f', 'float64': '<d', 'float16': '<e', 'int32': '<i', 'int64': '<q', 'uint8': '<B'}
        require(scalar.value in formats and isinstance(value, (int, float)) and math.isfinite(value), 'unsupported literal storage bits')
        attrs['value'] = {'storage_bits': struct.pack(formats[scalar.value], value).hex(), 'dtype': scalar.value}
    return attrs


def _dense_type(t):
    tensor = tensor_of(t)
    require(all(d.is_fixed and d.fixed_value > 0 for d in tensor.shape), 'dynamic/empty tensor')
    require(tensor.layout.tag == 'dense' and not tensor.layout.order and not tensor.layout.strides, 'noncanonical logical layout')
    if isinstance(t, DistributedType):
        require(t.partial is None and t.exclusive is None, 'partial or exclusive value')
    return tensor


class RecipeGraph:
    """One typed/effect/signature pass over the shared recipe DAG."""
    def __init__(self, recipes):
        self.recipes = tuple(recipes)
        self.nodes = {}; self.output_recipe = {}; self.exact = {}; self.logical = {}; self.views = {}
        self.data = []
        for recipe in self.recipes:
            self.data.append(recipe.to_data())
            for node in recipe.nodes:
                if node.id in self.nodes:
                    require(self.nodes[node.id] == node, 'conflicting shared node')
                    continue
                require(node.op in _OPS, 'unsupported op ' + node.op)
                require(set(node.metadata) <= _ANNOTATIONS, 'unknown execution metadata')
                require(not any(k in node.attrs for k in ('ops', 'fused_ops')), 'fused op attrs')
                require(all(x in self.nodes for x in node.inputs), 'unclosed or reordered dependency')
                inputs = tuple(self.nodes[x] for x in node.inputs)
                definition = get_definition(node.op)
                require(node.effect.to_data() == {'kind': 'pure', 'resource': None}, 'non-pure effect')
                require(definition.infer_effect(inputs, node.attrs) == node.effect, 'inferred effect differs')
                tensor = _dense_type(node.type)
                for source in inputs: _dense_type(source.type)
                if node.op not in {'builtin.weight', 'builtin.splat_const'}:
                    require(definition.numpy_materializable or node.op == 'math.vectorized_binary', 'unsupported materializer contract')
                    require(definition.infer_call_type(inputs, node.attrs) == node.type, 'inferred result type differs')
                if node.op == 'math.vectorized_binary':
                    require(set(node.attrs) <= {'binary_op', 'lhs_pack', 'rhs_pack'} and node.attrs['binary_op'] in {'add', 'mul'}, 'unsupported arithmetic contract')
                    require(all(set(node.attrs[k]) == {'axes', 'lanes'} for k in ('lhs_pack', 'rhs_pack') if k in node.attrs), 'unsupported arithmetic packing')
                attrs = _literal(node)
                provenance = _plain(node.metadata.get('source_location')) if node.op == 'builtin.weight' else None
                typed = {'op': node.op, 'attrs': attrs, 'type': node.type.to_data(),
                         'effect': node.effect.to_data(), 'weight_provenance': provenance}
                self.exact[node.id] = digest({**typed, 'inputs': [self.exact[x] for x in node.inputs]})
                rules = tuple(v for x in node.inputs for v in self.views[x])
                if node.op == 'distributed.sharded_view':
                    require(len(inputs) == 1 and set(node.attrs) == {'new_type'}, 'unrecognized view attrs')
                    require(tensor_of(inputs[0].type) == tensor and node.attrs['new_type'] == node.type and sharded_view_error(inputs[0].type, node.type) is None, 'nonidentity full logical view')
                    self.logical[node.id] = self.logical[node.inputs[0]]
                    rules += (node.id,)
                else:
                    if node.op == 'tensors.bitcast':
                        source = tensor_of(inputs[0].type)
                        scalar = lambda t: t.dtype.elem_type if isinstance(t.dtype, VectorType) else t.dtype
                        require(scalar(source) == scalar(tensor), 'bitcast changes scalar storage dtype')
                        # The pinned materializer uses tensor_of only and C-order
                        # view/reshape; all dtype, lane, coordinate facts remain.
                        typed = {**typed, 'type': tensor.to_data()}
                    self.logical[node.id] = digest({**typed, 'inputs': [self.logical[x] for x in node.inputs]})
                self.views[node.id] = tuple(dict.fromkeys(rules))
                self.nodes[node.id] = node
            for output in recipe.outputs:
                require(output in self.nodes and output not in self.output_recipe, 'missing or repeated recipe output')
                self.output_recipe[output] = recipe
        self.data = _freeze(self.data)
        for name in ('nodes', 'output_recipe', 'exact', 'logical', 'views'):
            setattr(self, name, MappingProxyType(getattr(self, name)))
        self._sealed = True

    def __setattr__(self, name, value):
        require(not getattr(self, '_sealed', False), 'typed recipe graph is immutable')
        object.__setattr__(self, name, value)


def physical(buffer):
    require(buffer.storage == 'rdata' and buffer.alias is None and buffer.field is None and not buffer.fused_weight_keys and buffer.rdata_group is None, 'not a complete readonly owner')
    require(buffer.distributed_storage_kind.value == 'canonical_global' and buffer.distributed_backing_type is None and buffer.component_stride_bytes == 0 and not buffer.offset_bindings, 'unsupported distributed storage')
    strides = [math.prod(buffer.shape[i + 1:]) for i in range(len(buffer.shape))]
    require(list(buffer.strides) == strides and buffer.nbytes == math.prod(buffer.shape) * buffer.dtype.itemsize, 'non-dense physical bytes')
    require(buffer.offset >= 0 and buffer.offset % buffer.alignment == 0, 'misaligned owner')
    if buffer.distributed_type is not None: _dense_type(buffer.distributed_type)
    from triton.flagmega.ir.types import data_type_to_data
    return {'dtype': data_type_to_data(buffer.dtype), 'shape': list(buffer.shape), 'strides': strides,
            'nbytes': buffer.nbytes, 'alignment': buffer.alignment, 'storage_kind': 'canonical_global',
            'backing_type': None, 'component_stride_bytes': 0}


def module_owners(module):
    from triton.flagmega.ir import launch_chip_count
    from triton.flagmega.artifacts.rdata import _aliases_rdata_owner
    plan = verify_buffer_plan(module)
    require(launch_chip_count(module) == 1, 'multiple chips')
    owners = sorted((b for b in plan.buffers if b.storage == 'rdata' and not _aliases_rdata_owner(b)), key=lambda b: (b.offset, b.id))
    end = 0
    for b in owners:
        physical(b)
        require(b.offset >= end and b.offset + b.nbytes <= plan.rdata_bytes, 'overlap or out of bounds owner')
        end = b.offset + b.nbytes
    return plan, owners


class VerifiedMaterializedSource:
    """Created only after verifying lineage, typed closure, and source identity."""
    def __init__(self, token, *, path, identity, table, graph, owners, checkpoint_identity):
        require(token is _TOKEN, 'use verify_materialized_source factory')
        self.path = Path(path); self.identity = tuple(identity)
        self.table = _freeze(table); self.graph = graph; self.owners = tuple(owners)
        self.checkpoint_identity = _freeze(checkpoint_identity)
        self._proofs = {}
        self._sealed = True

    def __setattr__(self, name, value):
        require(not getattr(self, '_sealed', False), 'verified source object is immutable')
        object.__setattr__(self, name, value)

    def source_file_digest(self, path):
        require(Path(path) == self.path and file_identity(self.path) == self.identity and not self.path.stat().st_mode & 0o222, 'source version changed or writable')
        return self.table['index']['sha256']


def verify_materialized_source(prior_module, checkpoint, *, lineage_pin, prior_manifest_pin):
    """Admit existing qualified normal-writer lineage, never (path,digest) trust.

    Pinned lineage is the caller's previously reviewed evidence anchor. Every
    image/recipe/cache/key/checkpoint edge is checked before granting a source.
    No unknown version is repaired by reading or hashing the payload.
    """
    from triton.flagmega.artifacts.rdata import _load_rdata_cache_entry, _rdata_cache_key_data, verify_rdata
    verify_module(prior_module)
    lineage = read_pinned_json(lineage_pin)
    require(lineage.get('schema') == 'flagmega.explicit-rdata-current-identity-lineage/v1' and lineage.get('status') == 'pass', 'unknown qualified lineage')
    receipt = read_pinned_json(lineage['prior_materialization'])
    qualification = read_pinned_json(lineage['qualification'])
    manifest = read_pinned_json(prior_manifest_pin)
    key = read_pinned_json(lineage['current_key_data'])
    historical_key = read_pinned_json(lineage['reconstructed_historical_key_data'])
    require(digest(key) == lineage['current_cache_key'] and digest(historical_key) == lineage['prior_cache_key'] == receipt['cache_key'], 'cache preimage mismatch')
    require({**historical_key, 'checkpoint': key['checkpoint']} == key, 'historical noncheckpoint preimage changed')
    before = historical_key['checkpoint']; after = key['checkpoint']
    require({**before, 'device': after['device']} == after, 'unsupported checkpoint identity change')
    require(qualification['same_original_content'] and qualification['sha256'] == lineage['same_original_GGUF_sha256'] == receipt['checkpoint']['sha256'], 'checkpoint content qualification differs')
    require(list(file_identity(Path(qualification['path']))) == qualification['identity'] == lineage['current_checkpoint_stat'], 'current checkpoint version differs')
    require(_rdata_cache_key_data(prior_module, checkpoint) == key, 'prior typed module does not reproduce key data')
    original_index = read_pinned_json(lineage['prior_index'])
    original_cache = read_pinned_json(lineage['prior_cache_manifest'])
    cache_manifest = read_pinned_json(lineage['current_cache_manifest'])
    entry = Path(lineage['current_cache_manifest']['path']).parent
    index = _load_rdata_cache_entry(entry, lineage['current_cache_key'])
    require(index == original_index and index is not None and index['schema'] == 'flagmega.rdata-index/v3' and not index.get('sources') and index['digest_kind'] == 'sha256', 'source must be ordinary fully materialized v3')
    require(original_cache['key'] == lineage['prior_cache_key'] and original_cache['index_sha256'] == lineage['prior_index']['sha256'], 'original cache/index link differs')
    require(cache_manifest['image_sha256'] == original_cache['image_sha256'] == receipt['rdata']['sha256'] == index['sha256'] == lineage['image_sha256'], 'image digest lineage differs')
    require(manifest['semantic_hash'] == prior_module.semantic_hash == receipt['semantic_hash'] and manifest['rdata'] == receipt['rdata'], 'prior artifact does not bind module/image')
    final = next(s for s in manifest['sections'] if s['path'] == manifest['final_ir'])
    final_path = Path(prior_manifest_pin['path']).parent / manifest['final_ir']
    require(hashlib.sha256(final_path.read_bytes()).hexdigest() == final['sha256'], 'prior final IR section changed')
    path = entry / 'rdata.bin'
    identity = file_identity(path)
    require(list(identity) == lineage['fresh_image_stat'] and identity[2] == index['nbytes'] and not path.stat().st_mode & 0o222, 'unproved or writable source image version')
    verify_rdata(entry, module=prior_module, verify_image_bytes=False)
    contract = numerical_contract()
    require(numerical_contract(Path(receipt['source_tree'])/'python/triton/flagmega') == contract, 'historical numerical definitions differ')
    plan, owners = module_owners(prior_module); graph = RecipeGraph(prior_module.constant_recipes)
    table = {'schema': SCHEMA, 'source': 0, 'index': index, 'key_data': key,
             'recipes': graph.data, 'numerical_contract': contract,
             'lineage': {'anchor_sha256': lineage_pin['sha256'], 'receipt': receipt,
                         'manifest': {k: manifest[k] for k in ('semantic_hash', 'rdata', 'final_ir')}, 'cache': cache_manifest, 'original_cache': original_cache,
                         'historical_key': historical_key, 'qualification': qualification,
                         'source_identity': list(identity), 'lineage': lineage,
                         'evidence_json': {
                             'anchor': Path(lineage_pin['path']).read_text(),
                             'receipt': Path(lineage['prior_materialization']['path']).read_text(),
                             'qualification': Path(lineage['qualification']['path']).read_text(),
                             'cache': Path(lineage['current_cache_manifest']['path']).read_text(),
                             'original_cache': Path(lineage['prior_cache_manifest']['path']).read_text()}},
             'owners': [{'id': b.id, 'output': b.weight_key, 'offset': b.offset, 'physical': physical(b)} for b in owners]}
    _validate_embedded_lineage(table)
    return VerifiedMaterializedSource(_TOKEN, path=path, identity=identity, table=table, graph=graph, owners=owners, checkpoint_identity=key['checkpoint'])


def build_materialized_program(buffer, old):
    dtype = buffer.dtype
    shape = [*buffer.shape, *(dtype.lanes if isinstance(dtype, VectorType) else ()), (dtype.elem_type if isinstance(dtype, VectorType) else dtype).itemsize]
    return {'buffer': buffer.id, 'shape': shape, 'concat_shape': shape, 'axis': 0,
            'operands': [{'key': old['output'], 'source': 0, 'offset': old['offset'], 'nbytes': buffer.nbytes, 'extent': shape[0]}],
            'transforms': [], 'ranges': [[buffer.offset, buffer.nbytes]]}


def _prove(module, table, old_graph):
    plan, buffers = module_owners(module); current = RecipeGraph(module.constant_recipes)
    by_signature = {}
    for old in table['owners']:
        output = old['output']
        require(output in old_graph.output_recipe, 'historical output missing')
        by_signature.setdefault(old_graph.logical[output], []).append(old)
    bindings = []; programs = []
    for buffer in buffers:
        output = buffer.weight_key
        require(output in current.output_recipe, 'current owner lacks output closure')
        matches = [b for b in by_signature.get(current.logical[output], []) if b['physical'] == physical(buffer)]
        require(bool(matches), 'no complete typed byte-equivalent output for ' + output)
        matches.sort(key=lambda b: (current.exact[output] != old_graph.exact[b['output']], b['offset'], b['id']))
        old = matches[0]; prior = old['output']; exact = current.exact[output] == old_graph.exact[prior]
        bindings.append({'buffer': buffer.id, 'current_output': output, 'prior_output': prior,
                         'prior_recipe': old_graph.output_recipe[prior].fingerprint,
                         'current_recipe': current.output_recipe[output].fingerprint,
                         'rule': 'exact' if exact else 'full-logical-sharded-view',
                         'current_signature': current.exact[output], 'prior_signature': old_graph.exact[prior],
                         'logical_signature': current.logical[output], 'current_views': list(current.views[output]),
                         'prior_views': list(old_graph.views[prior]), 'physical': old['physical'],
                         'prior_offset': old['offset'], 'current_offset': buffer.offset})
        programs.append(build_materialized_program(buffer, old))
    return {'bindings': bindings, 'programs': programs, 'nbytes': plan.rdata_bytes,
            'data_bytes': sum(b.nbytes for b in buffers), 'padding_bytes': plan.rdata_bytes - sum(b.nbytes for b in buffers),
            'old_nodes': len(old_graph.nodes), 'current_nodes': len(current.nodes)}


def prove_output_reuse(current_module, source):
    require(isinstance(source, VerifiedMaterializedSource), 'unverified source object')
    source.source_file_digest(source.path)
    key = id(current_module)
    if key not in source._proofs:
        verify_module(current_module)
        source._proofs[key] = (current_module, _freeze(_prove(current_module, source.table, source.graph)))
    return source._proofs[key][1]


def _validate_embedded_lineage(table):
    """Recheck every factory evidence edge after serialization and relocation.

    Small receipt JSON is retained verbatim so its original external pin can
    be checked without dependence on JSON key ordering or the old host paths.
    Large key preimages bind through their normal canonical cache keys.
    """
    l = table['lineage']; raw = l['evidence_json']; anchor = l['lineage']
    require(hashlib.sha256(raw['anchor'].encode()).hexdigest() == l['anchor_sha256'] and json.loads(raw['anchor']) == anchor, 'embedded lineage anchor differs')
    require(anchor.get('schema') == 'flagmega.explicit-rdata-current-identity-lineage/v1' and anchor.get('status') == 'pass', 'unqualified embedded lineage')
    for field, pin_field in (('receipt', 'prior_materialization'), ('qualification', 'qualification'),
                             ('cache', 'current_cache_manifest'), ('original_cache', 'prior_cache_manifest')):
        encoded = raw[field].encode(); pin = anchor[pin_field]
        require(len(encoded) == pin['nbytes'] and hashlib.sha256(encoded).hexdigest() == pin['sha256'] and json.loads(raw[field]) == l[field], 'embedded ' + field + ' pin differs')
    key = table['key_data']; prior = l['historical_key']; index = table['index']
    receipt = l['receipt']; qualification = l['qualification']
    require(digest(key) == anchor['current_cache_key'] == l['cache']['key'], 'embedded current cache key differs')
    require(digest(prior) == anchor['prior_cache_key'] == l['original_cache']['key'] == receipt['cache_key'], 'embedded historical cache key differs')
    require({**prior, 'checkpoint': key['checkpoint']} == key and {**prior['checkpoint'], 'device': key['checkpoint']['device']} == key['checkpoint'], 'embedded historical key relation differs')
    require(index['sha256'] == anchor['image_sha256'] == l['cache']['image_sha256'] == l['original_cache']['image_sha256'] == receipt['rdata']['sha256'], 'embedded image digest chain differs')
    require(index['nbytes'] == l['cache']['image_nbytes'] == l['original_cache']['image_nbytes'] == receipt['rdata']['nbytes'] == l['source_identity'][2], 'embedded image size chain differs')
    require(l['source_identity'] == anchor['fresh_image_stat'], 'embedded source version differs')
    require(l['manifest']['semantic_hash'] == receipt['semantic_hash'] and l['manifest']['rdata'] == receipt['rdata'], 'embedded historical artifact differs')
    require(qualification['same_original_content'] and qualification['sha256'] == anchor['same_original_GGUF_sha256'] == receipt['checkpoint']['sha256'], 'embedded checkpoint digest differs')
    require(qualification['identity'] == anchor['current_checkpoint_stat'] and qualification['path'] == key['checkpoint']['path'], 'embedded checkpoint identity differs')
    require([key['checkpoint'][k] for k in ('device', 'inode', 'size', 'mtime_ns')] == qualification['identity'][:4], 'checkpoint key identity differs')
    require(l['cache']['index_sha256'] == l['original_cache']['index_sha256'] == anchor['prior_index']['sha256'], 'embedded source index pin differs')


def validate_materialized_source_table(index, module):
    tables = index.get('materialized_sources')
    require(isinstance(tables, list) and len(tables) == 1, 'one materialized source required')
    table = tables[0]
    require(table.get('schema') == SCHEMA and table.get('source') == 0 and len(index['sources']) == 1, 'invalid table/source identity')
    require(table['numerical_contract'] == numerical_contract(), 'numerical implementation contract changed')
    _validate_embedded_lineage(table)
    old_index = table['index']; key = table['key_data']; lineage = table['lineage']
    require(old_index['schema'] == 'flagmega.rdata-index/v3' and not old_index.get('sources') and old_index['digest_kind'] == 'sha256', 'recursive or nonmaterialized image')
    require(index['sources'][0]['sha256'] == old_index['sha256'] == lineage['cache']['image_sha256'] == lineage['receipt']['rdata']['sha256'] and index['sources'][0]['nbytes'] == old_index['nbytes'], 'source image provenance changed')
    require(digest(key) == lineage['cache']['key'] and digest(lineage['historical_key']) == lineage['original_cache']['key'], 'historical preimage changed')
    require(key['checkpoint'] == index['materialized_checkpoint_identity'], 'current checkpoint differs')
    written_index = hashlib.sha256((json.dumps(old_index, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    require(written_index == lineage['cache']['index_sha256'] == lineage['original_cache']['index_sha256'], 'historical index pin differs')
    require({**lineage['historical_key'], 'checkpoint': key['checkpoint']} == key, 'historical key relation differs')
    require(lineage['manifest']['rdata'] == lineage['receipt']['rdata'], 'historical artifact/image differs')
    require(lineage['qualification']['sha256'] == lineage['receipt']['checkpoint']['sha256'] and lineage['qualification']['same_original_content'], 'checkpoint content lineage differs')
    graph = RecipeGraph(ConstantRecipe.from_data(r) for r in table['recipes'])
    fingerprints = {r['output']: r['fingerprint'] for r in key['recipes']}
    entries = {e['buffer']: e for e in old_index['entries']}
    key_buffers = {b['id']: b for b in key['buffers']}
    require(len(table['owners']) == len(entries) == len(key_buffers) and len({b['id'] for b in table['owners']}) == len(entries), 'old owner coverage differs')
    for owner in table['owners']:
        output = owner['output']; b = key_buffers[owner['id']]; e = entries[owner['id']]
        require(output in graph.output_recipe and graph.output_recipe[output].fingerprint == fingerprints.get(output), 'old output selection/fingerprint differs')
        require(owner['offset'] == b['offset'] == e['offset'] and output == b['key'] == e['key'], 'old output byte selection differs')
        require(all(owner['physical'][field] == b[field] for field in ('dtype', 'shape', 'strides', 'nbytes', 'storage_kind', 'backing_type', 'component_stride_bytes')), 'old physical descriptor differs')
        require(owner['physical']['nbytes'] == e['nbytes'] and owner['offset'] + e['nbytes'] <= old_index['nbytes'], 'old bytes out of bounds')
    if module is not None:
        proof = _prove(module, table, graph)
        require(proof['bindings'] == table['bindings'] and proof['programs'] == index['programs'] and proof['nbytes'] == index['nbytes'], 'current typed output/program proof differs')
    else:
        require(len(table['bindings']) == len(index['programs']), 'binding coverage differs')
    by_output = {b['output']: b for b in table['owners']}
    require(len(by_output) == len(table['owners']), 'duplicate historical output')
    for binding, program in zip(table['bindings'], index['programs'], strict=True):
        old = by_output.get(binding['prior_output'])
        require(old is not None and binding['physical'] == old['physical'] and binding['prior_offset'] == old['offset'], 'binding old byte range differs')
        require(binding['prior_signature'] == graph.exact[old['output']] and binding['logical_signature'] == graph.logical[old['output']], 'binding old typed closure differs')
        p = binding['physical']; dtype = p['dtype']
        from triton.flagmega.ir.types import data_type_from_data
        dt = data_type_from_data(dtype)
        shape = [*p['shape'], *(dt.lanes if isinstance(dt, VectorType) else ()), (dt.elem_type if isinstance(dt, VectorType) else dt).itemsize]
        expected = {'buffer': binding['buffer'], 'shape': shape, 'concat_shape': shape, 'axis': 0, 'operands': [{'key': old['output'], 'source': 0, 'offset': old['offset'], 'nbytes': p['nbytes'], 'extent': shape[0]}], 'transforms': [], 'ranges': [[binding['current_offset'], p['nbytes']]]}
        require(program == expected, 'binding address program differs')
    return table
