# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Share independent matmul tiles across proven replicas of one local owner."""
import hashlib
import json
import re
from collections.abc import Mapping
from math import prod
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.kernel_call_renderers import _distributed_unique_writer_active, _pointer
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError

original_prepare = TritonTemplateRegistry.prepare


def _complete_private_read(abi, context, physical_count, *, allow_canonical=False):
    """A dense private input carries the same value on its broadcast axes."""
    kind = abi.get('distributed_type')
    count = abi.get('pool_scope_count')
    stride = abi.get('pool_scope_stride_bytes', 0)
    representation = (abi.get('coordinate_space') == 'local' and abi.get('storage_kind') == 'compact_local'
                      or allow_canonical and abi.get('coordinate_space') == 'canonical_global'
                      and abi.get('storage_kind') == 'canonical_global')
    if (not representation or not abi.get('pooled') or abi.get('distributed_backing_type') is not None
            or 'view_byte_offset' in abi or not isinstance(kind, Mapping)
            or kind.get('tensor', {}).get('kind') != 'tensor'
            or kind.get('partial') is not None or kind.get('exclusive') is not None
            or isinstance(count, bool) or count not in {1, physical_count}
            or count == 1 and stride != 0
            or count != 1 and (not isinstance(stride, int) or stride <= 0
                              or abi.get('pool_scope_index') != 'block_index')):
        return False
    placement = kind.get('placement', {})
    return (tuple(placement.get('hierarchy', ())) == tuple(context['mesh_hierarchy'])
            and placement.get('hierarchy_levels') == ''.join(context['mesh_levels']))


def replica_group(call, context, *, expressions=(), private_canonical_inputs=()):
    """Hold every logical/physical owner axis fixed, vary only CTA replicas."""
    if call['participation_active'] != 'True' or not context['distributed_entry']:
        return None
    physical_count = prod(size for size, level in zip(context['mesh_hierarchy'], context['mesh_levels'], strict=True)
                          if level=='b')
    if tuple(context['grid']) != (physical_count, 1, 1):
        return None
    used = set()
    private_bases = set()
    for group in ('inputs', 'outputs'):
        for argument in call[group]:
            for buffer in argument['buffers']:
                abi = buffer['abi']
                if abi.get('storage')=='scalar' and buffer.get('runtime_value_kind') in {'scalar', 'immediate'}:
                    continue
                if abi.get('memory_sharing_scope') != 'chip':
                    if (group != 'inputs' or argument.get('role') != 'input'
                            or abi.get('memory_sharing_scope') != 'block'
                            or not _complete_private_read(abi, context, physical_count,
                                allow_canonical=argument.get('formal') in private_canonical_inputs)):
                        return None
                    private_bases.add(_pointer(buffer))
                for key in ('distributed_type', 'distributed_backing_type'):
                    kind = abi.get(key)
                    if kind is None:
                        continue
                    if kind.get('exclusive') is not None:
                        return None
                    used.update((kind.get('partial') or {}).get('axes', ()))
                    for policy in kind.get('axis_policies', ()):
                        for stage in policy.get('stages', ()):
                            used.update(stage.get('hierarchy_axes', ()))
    axes = [i for i, level in enumerate(context['mesh_levels']) if level=='b' and i not in used]
    count = prod(context['mesh_hierarchy'][i] for i in axes)
    if count <= 1:
        return None
    # Reject unmodelled private-base/address dependencies conservatively.
    expressions = [*expressions, *(call[key] for key in
                   ('source', 'weight', 'result', 'source_offset', 'weight_offset',
                    'result_offset', 'result_active', 'source_active', 'weight_active')
                   if isinstance(call.get(key), str))]
    # Only the exact pointer emitted from a proven complete input component
    # may vary. Its indices, masks, other operands and all outputs still have
    # to be invariant in the selected replica axes.
    expressions = [value for value in expressions if value not in private_bases]
    forbidden = r'\b(?:shard_index|block_index)\b|program_id|_flagmega_block_scope_base'
    if any(re.search(forbidden, value) for value in expressions):
        return None
    coordinates = [f'shard_coord{axis}' for axis in axes]
    if len(context['mesh_hierarchy']) == 2:
        coordinates.extend(('shard_y', 'shard_x')[axis] for axis in axes)
    if any(re.search(rf'\b{coordinate}\b', value) for coordinate in coordinates for value in expressions):
        return None
    index = '0'
    for axis in axes:
        index = f"({index} * {context['mesh_hierarchy'][axis]} + shard_coord{axis})"
    return dict(axes=axes, count=count, index=index)


def replica_task_writer(abi, group):
    """Publish disjoint tasks without dropping unrelated physical owners."""
    kind = abi.get('distributed_type')
    if kind is not None and (kind.get('partial') is not None or kind.get('exclusive') is not None):
        raise CodegenError('Replica task writers require non-partial, non-exclusive outputs.')
    if abi.get('storage_kind') != 'canonical_global' or kind is None:
        return 'True'
    used = {axis for policy in kind['axis_policies'] for stage in policy.get('stages', ())
            for axis in stage['hierarchy_axes']}
    placement = kind['placement']
    levels = placement.get('hierarchy_levels') or 'b' * len(placement['hierarchy'])
    chip_replicas = tuple(axis for axis, level in enumerate(levels)
                          if level in {'c', 'd'} and axis not in used)
    return _distributed_unique_writer_active(abi, allow_redundant_axes=(*chip_replicas, *group['axes']))


def prepare(self, name, context):
    if name != 'module.py.jinja':
        return original_prepare(self, name, context)
    selected = []
    skipped = []
    for call in context['render_calls']:
        call['replica_n_group'] = None
        if (call['family'] != 'dense_matmul'
            or call['variant'] not in {'packed_k_major_mma', 'packed_k_major_threaded_fma',
                                       'packed_k_major_mma_accumulator'}
            or call['local_n_capacity'] <= call['tile_n']):
            continue
        group = replica_group(call, context)
        if group is not None:
            call['replica_n_group'] = group
            selected.append(dict(call=call['call'], group=group, local_n=call['local_n_capacity'],
                                 tile_n=call['tile_n'], block_k=call['block_k']))
        else:
            skipped.append(dict(call=call['call'], participation=call['participation_active'],
                                distributed_entry=context['distributed_entry'],
                                buffers=[dict(formal=a['formal'], **{key: b['abi'].get(key) for key in
                                         ('memory_sharing_scope', 'coordinate_space', 'distributed_type',
                                          'distributed_backing_type')})
                                         for g in ('inputs','outputs') for a in call[g] for b in a['buffers']],
                                expressions={key: call.get(key) for key in
                                             ('source', 'weight', 'result', 'source_offset', 'weight_offset', 'result_offset')}))
    original_prepare(self, name, context)
    sources = {}
    # Every variant retains its complete K loop and ordered FP32 merges.
    # Only independent N tiles move between proven replicas of one owner.
    for key in ('kernels/dense_matmul/platforms/nvidia/sm90/packed_k_major_mma.py.jinja',
                'kernels/dense_matmul/platforms/nvidia/sm90/packed_k_major_mma_accumulator.py.jinja',
                'kernels/dense_matmul/platforms/nvidia/packed_k_major_threaded_fma.py.jinja'):
        source = self.environment.loader.get_source(self.environment, key)[0]
        marker = '        for dense_local_n_start in tl.range(0, {{ call.local_n_capacity }}, {{ call.tile_n }}):'
        assert source.count(marker)==1, (key, source.count(marker))
        source = source.replace(marker, '''{% if call.replica_n_group %}
        for dense_local_n_start in tl.range(({{ call.replica_n_group.index }}) * {{ call.tile_n }}, {{ call.local_n_capacity }}, {{ call.replica_n_group.count * call.tile_n }}):
{% else %}
''' + marker + '\n{% endif %}')
        sources[key] = source
    self.environment.loader = ChoiceLoader([DictLoader(sources), self.environment.loader])
    print(json.dumps(dict(trial='independent matmul N tiles shared across replicas of the same owner',
                          selected=selected, skipped=skipped,
                          override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)


TritonTemplateRegistry.prepare = prepare
