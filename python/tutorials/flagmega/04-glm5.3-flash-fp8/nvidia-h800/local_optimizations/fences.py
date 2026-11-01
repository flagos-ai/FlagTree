"""Reuse an entry body while preserving each invocation's device fences.

The uniform immediate mask is derived from the verified physical schedule,
reconstructed by normal runtime verification, and never a model/user input.
Local/axis barriers and their relative positions remain part of entry identity.
"""
from collections import defaultdict
from pathlib import Path

from jinja2 import ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError

MASK='_flagmega_entry_device_fences'
TEMPLATE='entrypoints/call_graph.py.jinja'


def canonicalize(programs,launches):
    definitions={p['symbol']:dict(p) for p in programs if not p.get('entry_alias')}
    groups=defaultdict(list)
    for program,launch in zip(programs,launches,strict=True):
        if launch['entry_function'] is None:
            continue
        definition=definitions[launch['entry_symbol']]
        events=definition['entry_events']
        body=[e for e in events if e['kind']!='barrier']
        if len(body)!=1 or body[0]['kind']!='function_call':
            raise CodegenError('Fence arguments require one complete reusable function call')
        if MASK in definition['signature_arguments']:
            raise CodegenError('Entry fence argument collides with an existing formal')
        skeleton=[];slots=defaultdict(list)
        for event in events:
            if event['kind']=='barrier' and event['scope']=='device':
                slots[len(skeleton)].append(event)
            else:
                skeleton.append(event)
        pattern=tuple(('call',) if e['kind']=='function_call' else
                      ('barrier',e['scope'],e.get('barrier_axis_group_key')) for e in skeleton)
        key=(launch['entry_function'],launch['num_warps'],tuple(launch['grid']),pattern,
             tuple(definition['typed_signature_arguments']),
             tuple((a['name'],a['stride'],a['dtype']) for a in definition.get('scoped_entry_arguments',())),
             body[0].get('participation_active','True'))
        groups[key].append((program,launch,definition,skeleton,slots))
    receipts=[]
    for key,items in groups.items():
        if len({v[1]['entry_symbol'] for v in items})<2:
            continue
        widths={gap:max(len(v[4].get(gap,())) for v in items)
                for gap in range(len(items[0][3])+1)}
        positions=[(gap,index) for gap,width in widths.items() for index in range(width)]
        if not positions:
            raise CodegenError('Entry variants differ outside their proven device-fence slots')
        if len(positions)>31:
            raise CodegenError('Entry fence mask exceeds its uniform int32 ABI')
        bits={position:i for i,position in enumerate(positions)}
        canonical=items[0][0]['symbol']
        masks=[]
        for program,launch,definition,skeleton,slots in items:
            mask=sum(1<<bits[(gap,index)] for gap,events in slots.items() for index in range(len(events)))
            launch['entry_symbol']=canonical
            launch['argument_bindings']=[*launch['argument_bindings'],
                dict(kind='immediate',dtype='int32',value=mask)]
            # The compiled body contains conditional full-grid synchronization,
            # so even a mask-zero invocation needs the common residency ABI.
            launch['requires_grid_wide_sync']=True
            masks.append(mask)
            if program['symbol']!=canonical:
                program['entry_alias']=canonical
                continue
            program.pop('entry_alias',None)
            for field in ('render_calls','device_functions','scoped_entry_arguments',
                          'unspecialized_pointer_arguments'):
                if field in definition:
                    program[field]=definition[field]
            typed=[*definition['typed_signature_arguments'],MASK+': tl.int32']
            program.update(signature=',\n    '.join(typed),
                signature_arguments=[*definition['signature_arguments'],MASK],
                typed_signature_arguments=typed,
                scalar_arguments=[*definition.get('scalar_arguments',()),MASK])
            emitted=[]
            for gap in range(len(skeleton)+1):
                for index in range(widths[gap]):
                    example=next(v[4][gap][index] for v in items if len(v[4].get(gap,()))>index)
                    bit=bits[(gap,index)]
                    emitted.append({**example,'program_fence_bit':bit,
                        'uniform_condition':f'({MASK} & {1<<bit}) != 0'})
                if gap<len(skeleton):
                    emitted.append(skeleton[gap])
            program['entry_events']=emitted
        if len(set(masks))<2:
            raise CodegenError('Fence canonicalization found no actual per-invocation difference')
        receipts.append(dict(function=key[0],entry=canonical,invocations=len(items),
                             masks=sorted(set(masks)),fence_slots=len(positions)))
    return receipts


def activate():
    from triton.flagmega.codegen.triton import function_programs
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous=function_programs.project_function_programs
    if not getattr(previous,'_fence_argument_trial',False):
        def project(module,descriptor,programs,launches):
            result=previous(module,descriptor,programs,launches)
            project.last_receipts=canonicalize(programs,launches) if result else []
            return result
        project._fence_argument_trial=True
        function_programs.project_function_programs=project

    original=TritonTemplateRegistry.render
    if getattr(original,'_persistent_join_renderer',False):
        return
    closure=dict(zip(original.__code__.co_freevars,
                     (cell.cell_contents for cell in original.__closure__ or ())))
    if 'event.uniform_condition' not in closure.get('template','') or not callable(closure.get('original')):
        raise CodegenError('Expected accepted uniform-fence renderer before scoped replacement')
    # Replace the accepted fence wrapper while keeping all selected-kernel
    # descriptor/template preparation wrappers beneath it.
    original=closure['original']
    template=(Path(__import__('triton.flagmega.codegen.triton.templates',fromlist=['x']).__file__).parent/TEMPLATE).read_text()
    old='''{% if event.kind == "barrier" %}
{{ scoped_barrier(event.scope, event.barrier_axis_group_key | default(none)) }}'''
    new='''{% if event.kind == "barrier" %}
{% if event.uniform_condition | default(none) %}
    if {{ event.uniform_condition }}:
{{ scoped_barrier(event.scope, event.barrier_axis_group_key | default(none)) | indent(4, true) }}
{% else %}
{{ scoped_barrier(event.scope, event.barrier_axis_group_key | default(none)) }}
{% endif %}'''
    if template.count(old)!=1:
        raise CodegenError('Entry event renderer changed; uniform fence hook needs review')
    template=template.replace(old,new)
    consumer='{{ pipeline_role_events(definition.schedule.consumer_events) }}'
    replacement='''{% if definition.owner_matched_join | default(none) %}
{% include "join-body.py.jinja" %}
{% else %}
{{ pipeline_role_events(definition.schedule.consumer_events) }}
{% endif %}'''
    if template.count(consumer)!=1:
        raise CodegenError('Complete function consumer override hook changed')
    template=template.replace(consumer,replacement)
    prologue = '{{ entry.mesh_coordinates() }}\n{% if definition.owner_matched_join | default(none) %}'
    snapshot = '{{ entry.mesh_coordinates() }}\n{% if not definition.runtime_record_scalar_bindings | default(none) %}\n{% for field in definition.immutable_call_record_bindings | default(()) %}\n    {{ field.name }} = tl.load(_flagmega_call_offsets + {{ field.column }})\n{% endfor %}\n{% endif %}\n{% if definition.owner_matched_join | default(none) %}'
    if template.count(prologue) != 1:
        raise CodegenError('Snapshot consumer prologue requires the complete accepted join template')
    template = template.replace(prologue, snapshot)
    join_body=Path(__file__).with_name('join-body.py.jinja').read_text()

    def render(self,name,context):
        if name!='module.py.jinja':
            return original(self,name,context)
        loader=self.environment.loader
        self.environment.loader=ChoiceLoader([DictLoader({TEMPLATE:template, 'join-body.py.jinja':join_body}),loader])
        try:
            return original(self,name,context)
        finally:
            self.environment.loader=loader
    render._persistent_join_renderer=True
    TritonTemplateRegistry.render=render
