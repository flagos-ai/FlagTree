# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Cooperate only over CTA replicas of the same complete MLA head owner."""
import hashlib
import json
from pathlib import Path
from jinja2 import ChoiceLoader,DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from .replica_work import replica_group,replica_task_writer

original_prepare=TritonTemplateRegistry.prepare


def prepare(self,name,context):
    if name!='module.py.jinja':return original_prepare(self,name,context)
    selected=[]
    for call in context['render_calls']:
        call['mla_replica_group']=None
        if call['family']!='mla_projection' or call['variant']!='mma':continue
        addresses=tuple(call[k] for k in ('value','weight','scale','result','global_head','head_active',
                                         'source_offset','result_offset','weight_offset','scale_offset')
                        if isinstance(call.get(k),str))
        # The MLA encoder proved all weight loads address complete owned
        # head rows (and scale blocks), including canonical private copies.
        # Every unused block axis therefore sees identical operand values.
        group=replica_group(call,context,expressions=addresses,private_canonical_inputs=('weight','scale'))
        if group is None:
            if tuple(context['grid'])!=(1,1,1):
                diagnostic=dict(call=call,context={k:context[k] for k in ('grid','mesh_hierarchy','mesh_levels','distributed_entry')})
                Path('mla-sharing-rejection.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
                raise CodegenError('Requested cooperative MLA requires proved shared storage and CTA replicas of the same head owner; see mla-sharing-rejection.json.')
            continue
        call['mla_replica_group']=group
        abi,=(b['abi'] for a in call['outputs'] for b in a['buffers'])
        call['writer_active']=replica_task_writer(abi,group)
        selected.append(dict(call=call['call'],heads=call['heads'],group=group))
    original_prepare(self,name,context)
    key='kernels/mla_projection/platforms/nvidia/sm90/mma.py.jinja'
    source=self.environment.loader.get_source(self.environment,key)[0]
    outer='    for _fm_mla_head in tl.range({{ call.heads }}):\n'
    inner='        for _fm_mla_start in tl.range(0, {{ call.output }}, {{ call.tile_n }}):\n'
    closing='{% endif %}\n{% endfor %}\n{% endif %}'
    assert source.count(outer)==source.count(inner)==1
    start=source.index(outer);middle=source.index(inner,start);end=source.rindex(closing)
    head_map=source[start+len(outer):middle]
    body=''.join(line.removeprefix('    ') for line in source[middle+len(inner):end].splitlines(keepends=True))
    cooperative='''{% if call.mla_replica_group %}
    for _fm_mla_task in tl.range(({{ call.mla_replica_group.index }}), {{ call.heads * ((call.output + call.tile_n - 1) // call.tile_n) }}, {{ call.mla_replica_group.count }}):
        _fm_mla_head = _fm_mla_task // {{ (call.output + call.tile_n - 1) // call.tile_n }}
        _fm_mla_start = (_fm_mla_task % {{ (call.output + call.tile_n - 1) // call.tile_n }}) * {{ call.tile_n }}
'''+head_map+body+'{% else %}\n'+source[start:end]+'{% endif %}\n'
    source=source[:start]+cooperative+source[end:]
    self.environment.loader=ChoiceLoader([DictLoader({key:source}),self.environment.loader])
    print(json.dumps(dict(trial='MLA tasks across replicas of each head shard',selected=selected,
                         override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())),flush=True)


TritonTemplateRegistry.prepare=prepare
