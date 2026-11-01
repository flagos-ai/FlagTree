# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Assign independent KDA value chunks once across replicas of a head owner."""

import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from .replica_work import replica_group, replica_task_writer as writer_for_value_tasks

original_prepare = TritonTemplateRegistry.prepare


def value_group(call, context):
    """Keep the complete K reduction and every physical owner fixed."""
    expressions = tuple(value for name, value in call.items() if isinstance(value, str) and (
        name.endswith('_head_offset') or name in {
            'query', 'key', 'value', 'gate', 'decay_rate', 'gate_bias', 'beta_logits',
            'state', 'out', 'new_state', 'request_prefill',
        }))
    for group in ('inputs', 'outputs'):
        for argument in call[group]:
            for buffer in argument['buffers']:
                abi = buffer['abi']
                for key in ('distributed_type', 'distributed_backing_type'):
                    kind = abi.get(key)
                    if kind is not None and (kind.get('partial') is not None or kind.get('exclusive') is not None):
                        return None
    return replica_group(call, context, expressions=expressions)


def coalesced_state_io(source):
    """Use row-major memory transactions and the original warp-K arithmetic."""
    marker = '{{ call.symbol }}__v_vector = tl.constexpr(tle.gpu.SlicedEncoding(0, {{ call.symbol }}__warp_k))'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
{% if call.kda_value_replica_group %}
{{ call.symbol }}__state_io_layout = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 4], [4, 8], [{{ num_warps }}, 1], [1, 0],
))
{% endif %}''')
    marker = '''            _fm_h = tl.load({{ call.state }} + ({{ call.state_head_offset }}) + _fm_offsets_k[:, None] * {{ call.value_dim }} + _fm_offsets_v[None, :],
                            mask=_fm_v_active[None, :], other=0.0).to(tl.float32)'''
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if call.kda_value_replica_group and call.kda_reduction_layout == "warp_k" %}
            _fm_state_ptrs = tle.gpu.set_layout(
                {{ call.state }} + ({{ call.state_head_offset }}) + _fm_offsets_k[:, None] * {{ call.value_dim }} + _fm_offsets_v[None, :],
                {{ call.symbol }}__state_io_layout)
            _fm_io_mask = tle.gpu.set_layout(tl.broadcast_to(_fm_v_active[None, :], ({{ call.key_dim }}, 32)),
                                            {{ call.symbol }}__state_io_layout)
            _fm_h = tl.load(_fm_state_ptrs, mask=_fm_io_mask, other=0.0).to(tl.float32)
            _fm_h = tle.gpu.set_layout(_fm_h, {{ call.symbol }}__state_io_layout)
            _fm_h = tle.gpu.set_layout(_fm_h, {{ call.symbol }}__warp_k)
{% else %}
''' + marker + '\n{% endif %}')
    marker = '''            tl.store({{ call.new_state }} + ({{ call.new_state_head_offset }}) + _fm_offsets_k[:, None] * {{ call.value_dim }} + _fm_offsets_v[None, :], _fm_h,
                     mask=_fm_v_active[None, :] & ({{ call.state_writer_active }}))'''
    assert source.count(marker) == 2
    # The first occurrence is the chunk_bf16 branch; only recurrent warp_k
    # needs a conversion between memory and arithmetic layouts.
    start = source.rindex(marker)
    replacement = '''{% if call.kda_value_replica_group and call.kda_reduction_layout == "warp_k" %}
            _fm_new_state_ptrs = tle.gpu.set_layout(
                {{ call.new_state }} + ({{ call.new_state_head_offset }}) + _fm_offsets_k[:, None] * {{ call.value_dim }} + _fm_offsets_v[None, :],
                {{ call.symbol }}__state_io_layout)
            tl.store(_fm_new_state_ptrs, tle.gpu.set_layout(_fm_h, {{ call.symbol }}__state_io_layout),
                     mask=_fm_io_mask & ({{ call.state_writer_active }}))
{% else %}
''' + marker + '\n{% endif %}'
    return source[:start] + replacement + source[start + len(marker):]


def prepare(self, name, context):
    if name != 'module.py.jinja':
        return original_prepare(self, name, context)
    selected = []
    for call in context['render_calls']:
        call['kda_value_replica_group'] = None
        if call['family'] != 'kda_gated_recurrent_core' or call['value_chunks'] <= 1:
            continue
        group = value_group(call, context)
        if group is None:
            continue
        call['kda_value_replica_group'] = group
        outputs = {argument['formal']: argument['buffers'][0]['abi'] for argument in call['outputs']}
        call['out_writer_active'] = writer_for_value_tasks(outputs['result_0'], group)
        call['state_writer_active'] = writer_for_value_tasks(outputs['result_1'], group)
        selected.append(dict(call=call['call'], group=group, value_chunks=call['value_chunks']))
    original_prepare(self, name, context)
    key = 'kernels/kda_gated_recurrent_core/local.py.jinja'
    source = self.environment.loader.get_source(self.environment, key)[0]
    marker = '    for _fm_head in tl.range(0, {{ call.heads }}):'
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if call.kda_value_replica_group %}
    for _fm_head in tl.range(0, tl.where(({{ call.kda_value_replica_group.index }}) < {{ call.value_chunks }}, {{ call.heads }}, 0)):
{% else %}
''' + marker + '\n{% endif %}')
    source = coalesced_state_io(source)
    marker = '        for _fm_v_chunk in tl.range(0, {{ call.value_chunks }}):'
    assert source.count(marker) == 2
    source = source.replace(marker, '''{% if call.kda_value_replica_group %}
        for _fm_v_chunk in tl.range({{ call.kda_value_replica_group.index }}, {{ call.value_chunks }}, {{ call.kda_value_replica_group.count }}):
{% else %}
''' + marker + '\n{% endif %}')
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps(dict(trial='independent KDA value chunks across replicas of the same head owner',
                          selected=selected,
                          override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)


TritonTemplateRegistry.prepare = prepare
