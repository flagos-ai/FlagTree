# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Assign independent DSA head tiles across proven chip-shared CTA replicas."""
import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from .replica_work import replica_group, replica_task_writer

original_prepare = TritonTemplateRegistry.prepare


def head_group(call, context):
    addresses = tuple(call[key] for key in (
        'query', 'topk', 'kv_cache', 'query_start_loc', 'sequence_lengths',
        'slot_mapping', 'block_table', 'layer_id', 'result',
    ) if isinstance(call.get(key), str))
    for kind in ('inputs', 'outputs'):
        for argument in call[kind]:
            for buffer in argument['buffers']:
                for key in ('distributed_type', 'distributed_backing_type'):
                    distribution = buffer['abi'].get(key)
                    if distribution is not None and (
                            distribution.get('partial') is not None or distribution.get('exclusive') is not None):
                        return None
    return replica_group(call, context, expressions=addresses)


def prepare(self, name, context):
    if name != 'module.py.jinja':
        return original_prepare(self, name, context)
    selected = []
    for call in context['render_calls']:
        call['dsa_head_group'] = None
        if (call['family'] != 'dsa_paged_attention' or call['variant'] != 'decode'
                or call['decode_schedule'] != 'bf16_mma' or call['num_query_heads'] <= 16):
            continue
        group = head_group(call, context)
        if group is None:
            continue
        call['dsa_head_group'] = group
        abi, = (buffer['abi'] for value in call['outputs'] for buffer in value['buffers'])
        call['writer_active'] = replica_task_writer(abi, group)
        selected.append(dict(call=call['call'], group=group, heads=call['num_query_heads']))
    original_prepare(self, name, context)
    key = 'kernels/dsa_paged_attention/platforms/nvidia/sm90/online_mma.py.jinja'
    source = self.environment.loader.get_source(self.environment, key)[0]
    guard = '    if {{ call.writer_active }}:'
    loop = '        for _fm_dsa_head_start in tl.range(0, {{ call.num_query_heads }}, 16):'
    assert source.count(guard) == source.count(loop) == 1
    source = source.replace(guard, '''{% if call.dsa_head_group %}
    if ({{ call.writer_active }}) & (({{ call.dsa_head_group.index }}) < {{ (call.num_query_heads + 15) // 16 }}):
{% else %}
''' + guard + '\n{% endif %}')
    source = source.replace(loop, '''{% if call.dsa_head_group %}
        for _fm_dsa_head_start in tl.range(({{ call.dsa_head_group.index }}) * 16, {{ call.num_query_heads }}, {{ call.dsa_head_group.count * 16 }}):
{% else %}
''' + loop + '\n{% endif %}')
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps(dict(trial='independent DSA M16 head tiles across CTA replicas', selected=selected,
                          override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)


TritonTemplateRegistry.prepare = prepare
