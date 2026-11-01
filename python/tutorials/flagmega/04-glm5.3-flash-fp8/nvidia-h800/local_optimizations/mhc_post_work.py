# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Share independent mHC post hidden tiles across proven CTA replicas."""

import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from .replica_work import replica_group, replica_task_writer

original_prepare = TritonTemplateRegistry.prepare


def hidden_group(call, context):
    addresses = tuple(value for name, value in call.items() if isinstance(value, str) and (
        name.endswith('_offset') or name in {'value', 'residual', 'post', 'comb', 'result'}))
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
        call['mhc_post_hidden_group'] = None
        if call['family'] != 'mhc_post':
            continue
        group = hidden_group(call, context)
        if group is None:
            continue
        call['mhc_post_hidden_group'] = group
        abi, = (buffer['abi'] for value in call['outputs'] for buffer in value['buffers'])
        call['writer_active'] = replica_task_writer(abi, group)
        selected.append(dict(call=call['call'], group=group, tile=call['tile'], hidden=call['hidden']))
    original_prepare(self, name, context)
    key = 'kernels/mhc_post/local.py.jinja'
    source = self.environment.loader.get_source(self.environment, key)[0]
    marker = '        for _fm_start in tl.range(0, {{ call.hidden }}, {{ call.tile }}):'
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if call.mhc_post_hidden_group %}
        for _fm_start in tl.range(({{ call.mhc_post_hidden_group.index }}) * {{ call.tile }}, {{ call.hidden }}, {{ call.mhc_post_hidden_group.count * call.tile }}):
{% else %}
''' + marker + '\n{% endif %}')
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps(dict(trial='independent mHC post hidden tiles across CTA replicas', selected=selected,
                          override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)


TritonTemplateRegistry.prepare = prepare
