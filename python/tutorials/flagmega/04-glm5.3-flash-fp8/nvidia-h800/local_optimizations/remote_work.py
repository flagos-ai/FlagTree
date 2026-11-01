# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Assign proved remote copy or sum tiles once within each chip's output."""
import hashlib
import json
import re
from math import prod
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError

original_prepare = TritonTemplateRegistry.prepare


def byte_span(abi):
    """Bound all canonical coordinates or compact placement components."""
    shape = abi['logical_shape'] if abi['coordinate_space'] == 'canonical_global' else abi['local_capacity_shape']
    strides = abi['scalar_storage_strides']
    offset = abi['pool_byte_offset']
    if (type(offset) is not int or len(shape) != len(strides)
        or any(type(v) is not int or v < 0 for v in (*shape, *strides))):
        return None
    elements = abi['scalar_lane_count'] + sum((n - 1) * stride for n, stride in zip(shape, strides, strict=True))
    owners = abi.get('owner_count', abi['placement_owner_count'])
    elements += (owners - 1) * abi['component_stride_scalar_elements']
    return offset, offset + elements * abi['scalar_itemsize']


def _partition(call, leaf, source_binding, result_binding, context, *, reduction):
    source, result = source_binding['abi'], result_binding['abi']
    if (call['participation_active'] != 'True' or not context['distributed_entry']
        or leaf.get('source_is_scalar') or leaf['reduction'] != reduction
        or 'remote_source' not in leaf or leaf['capacity'] <= leaf['tile']):
        return None
    axes = [i for i, level in enumerate(context['mesh_levels']) if level == 'b']
    blocks = prod(context['mesh_hierarchy'][i] for i in axes)
    if blocks <= 1 or tuple(context['grid']) != (blocks, 1, 1):
        return None
    if (source.get('memory_sharing_scope') != 'chip'
        or source.get('storage') != context.get('device_window_storage')
        or leaf.get('remote_window_pool') != context.get('device_window_storage')
        or result.get('memory_sharing_scope') != 'chip'
        or result.get('coordinate_space') != 'canonical_global'):
        return None
    if source['storage'] == result['storage']:
        if source_binding['runtime_argument'] != result_binding['runtime_argument']:
            return None
        before, after = byte_span(source), byte_span(result)
        if (before is None or after is None
            or before[0] < after[1] and after[0] < before[1]):
            return None
    for abi in (source, result):
        for key in ('distributed_type', 'distributed_backing_type'):
            kind = abi.get(key)
            if kind is None:
                continue
            partial = kind.get('partial')
            if kind.get('exclusive') is not None:
                return None
            if partial is not None:
                if not reduction or abi is result or key != 'distributed_type':
                    return None
                if partial.get('reduce_op') != 'sum' or any(
                    context['mesh_levels'][axis] != 'c' for axis in partial.get('axes', ())
                ):
                    return None
            if abi is result and any(
                axis in axes
                for policy in kind.get('axis_policies', ())
                for stage in policy.get('stages', ())
                for axis in stage.get('hierarchy_axes', ())
            ):
                return None
    # Every lane computes its source owner from the destination coordinate.
    # A private base or a source expression using the executing CTA instead
    # invalidates this proof, even when the output is broadcast.
    forbidden = r'\b(?:shard_index|block_index)\b|program_id|_flagmega_block_scope_base'
    coordinate = r'\bshard_coord(?:' + '|'.join(map(str, axes)) + r')\b'
    # The remote template uses its window base and peer-relative offset.
    # Its separate local-copy source expression is an inactive branch.
    keys = ('result', 'result_offset', 'active', 'remote_source', 'remote_local_base')
    keys += ('source_offset', 'partial_owner') if reduction else ('remote_source_offset', 'remote_owner')
    expressions = [leaf[key] for key in keys]
    if reduction:
        partial = source.get('distributed_type', {}).get('partial')
        if (partial is None or not partial.get('axes') or 'remote_owner' in leaf
            or leaf.get('partial_owner_count') != 1 or leaf.get('partial_owner_tile') != 1
            or leaf.get('accumulator_type') != 'tl.float32'
            or leaf.get('remote_peers') != prod(context['mesh_hierarchy'][axis] for axis in partial['axes'])):
            return None
    if any(re.search(forbidden, value) or re.search(coordinate, value) for value in expressions):
        return None
    writer = leaf['writer_active']
    for axis in axes:
        writer = re.sub(rf'\(shard_coord{axis}\s*==\s*0\)', 'True', writer)
    writer = re.sub(r'\bblock_index\s*==\s*0\b', 'True', writer)
    if re.search(forbidden, writer) or re.search(coordinate, writer):
        return None
    return writer


def gather_partition(call, leaf, source_binding, result_binding, context):
    return _partition(call, leaf, source_binding, result_binding, context, reduction=False)


def reduce_partition(call, leaf, source_binding, result_binding, context):
    """Keep the peer sum loop; partition only nonaliasing shared output tiles."""
    return _partition(call, leaf, source_binding, result_binding, context, reduction=True)


def prepare(self, name, context):
    if name != 'module.py.jinja':
        return original_prepare(self, name, context)
    selected = []
    for call in context['render_calls']:
        if call['family'] != 'distributed_boxing':
            continue
        sources = [b for a in call['inputs'] for b in a['buffers']]
        results = [b for a in call['outputs'] for b in a['buffers']]
        paired = len(sources) == len(results) == len(call['leaves'])
        gather_required = call.get('facts', {}).get('require_cooperative_remote_gather', False)
        reduce_required = call.get('facts', {}).get('require_cooperative_remote_reduce', False)
        required = gather_required or reduce_required
        transitions = call.get('parameters', {}).get('leaf_transitions', ())
        if required and (not paired or len(transitions) != len(call['leaves'])):
            raise CodegenError('Cooperative peer gather requires one proved source/result binding per leaf.')
        for index, leaf in enumerate(call['leaves']):
            partition = reduce_partition if reduce_required else gather_partition
            writer = partition(call, leaf, sources[index], results[index], context) if paired else None
            if required and transitions[index] != 'identity' and writer is None:
                operation = 'reduction' if reduce_required else 'gather'
                raise CodegenError(f'Selected cooperative peer {operation} cannot prove chip-shared disjoint output tiles.')
            leaf['remote_work_partition'] = writer is not None
            leaf['remote_work_writer_active'] = leaf['writer_active'] if writer is None else writer
            if writer is not None:
                selected.append(dict(call=call['call'], leaf=index, capacity=leaf['capacity'], tile=leaf['tile']))
    original_prepare(self, name, context)
    key = 'kernels/distributed_boxing/_function.py.jinja'
    source = self.environment.loader.get_source(self.environment, key)[0]
    marker = '    for boxing_start in tl.range(0, {{ leaf.capacity }}, {{ leaf.tile }}):'
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if leaf.remote_work_partition %}
    for boxing_start in tl.range(tl.program_id(0) * {{ leaf.tile }}, {{ leaf.capacity }}, tl.num_programs(0) * {{ leaf.tile }}):
{% else %}
''' + marker + '\n{% endif %}')
    source = source.replace('leaf.writer_active', 'leaf.remote_work_writer_active')
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps(dict(trial='unique remote copy/sum output tiles per chip', selected=selected,
                          override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)


TritonTemplateRegistry.prepare = prepare
