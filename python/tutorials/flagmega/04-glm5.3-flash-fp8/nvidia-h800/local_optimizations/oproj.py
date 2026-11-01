"""Fuse the frozen O-proj/TP4 reduction/mHC region by tile ownership.

The declared numerical calls remain the reference. The replacement owns their
complete computation and external writes. Ready/consumed protect the declared
persistent payload. Intermediate boxing writes are internal to this region and
must be elided: their scratch alias overlaps other CTAs' projection input.
The terminal local grid barrier protects the following scratch reuse.
"""
import ast,hashlib,json,copy
from pathlib import Path
from jinja2 import ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError

OUT=Path(__file__).resolve().parent
KEY='kernels/partitioned_linear/platforms/nvidia/sm90/wgmma_local.py.jinja'
CALLGRAPH='entrypoints/call_graph.py.jinja'
FUNCTIONS={'decode_linear_attention_dense','decode_linear_attention_sparse.prefix','decode_mla_attention_sparse.prefix'}

def arguments(text):
    return [ast.unparse(a) for a in ast.parse('f('+text+')').body[0].value.args]

def validate_function_pool_abi(abi, call_count):
    """Validate the normal address contract used by the owner replacement."""
    from triton.flagmega.codegen.triton.function_pool_abi import function_pool_aliases
    width=abi.get('record_width')
    fields=abi.get('bindings',())
    calls=abi.get('calls',())
    if (abi.get('schema')!='flagmega.function-pool-abi/v2' or type(width) is not int or width<=0
            or len(calls)!=call_count or len({field['formal'] for field in fields})!=len(fields)):
        raise CodegenError('O-proj requires its normal complete function-pool call domain')
    pools={pool['storage']:pool['parameter'] for pool in abi.get('pools',())}
    if pools!={storage:'_flagmega_call_pool_'+storage for storage in ('rdata','workspace')}:
        raise CodegenError('O-proj requires its typed rdata/workspace pools')
    columns={field['record_column'] for field in fields if 'record_column' in field}
    if columns!=set(range(width)):
        raise CodegenError('O-proj function-pool record columns must cover the normal complete record')
    start=calls[0]['record_offset']
    if any(type(call['record_offset']) is not int or call['record_offset']!=start+index*width
           for index,call in enumerate(calls)):
        raise CodegenError('O-proj function-pool records must retain the complete contiguous call domain')
    frame=next((field for field in fields if field['formal']=='workspace'),None)
    if (frame is None or frame['storage']!='workspace' or frame['scalar_dtype']!='uint8'
            or frame.get('value_kind','pointer')!='pointer'):
        raise CodegenError('O-proj workspace frame must retain its normal byte-pointer ABI')
    # This checks every affine proof against its record width, offsets and
    # complete caller domain; the normal renderer uses the same named aliases.
    function_pool_aliases(abi)

def describe(module,previous):
    plan=json.loads((OUT/'plan.json').read_text())
    if module.semantic_hash!=plan['semantic_hash']:
        raise CodegenError('O-proj override requires its frozen verified complete module')
    d=previous(module)
    resource,=[r for r in d['persistent_window_resources']['resources'] if r['name']=='ordered_oproj']
    actors=[k['symbol'] for k in d['kernels'] if k.get('entry_function') in FUNCTIONS]
    if actors!=resource['programs'] or len(actors)!=45 or resource['nbytes']!=9216:
        raise CodegenError('O-proj channel ownership differs from its verified ordered DAG')
    geometry=dict(payload=resource['offset'],ready=resource['offset']+8192,consumed=resource['offset']+8704)
    definitions=list(d['device_functions'])
    for p in d['programs']:definitions.extend(p.get('device_functions',()))
    definitions=list({id(v):v for v in definitions}.values())
    receipt={}
    for definition in definitions:
        fn=definition['function']
        if fn not in FUNCTIONS:continue
        events=definition['schedule']['consumer_events']
        indices=[i for i,e in enumerate(events) if e.get('family')=='partitioned_linear' and e.get('variant')=='wgmma_local' and 'o_proj_weight_apply' in e.get('call','')]
        if len(indices)!=1:raise CodegenError('O-proj producer selection changed')
        i=indices[0];region=events[i:i+6]
        expected=[('tir.kernel_call','partitioned_linear'),('barrier','device'),('tir.kernel_call','distributed_boxing'),('barrier','grid'),('tir.kernel_call','mhc_post'),('barrier','grid')]
        if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=expected:
            raise CodegenError(f'O-proj region effect boundaries changed: {fn}: {[(e["kind"],e.get("family",e.get("scope"))) for e in region]}')
        op,post=region[0],region[4]
        op_args=arguments(op['arguments']);post_args=arguments(post['arguments'])
        if len(op_args)!=5 or len(post_args) not in (4,5) or post_args[0]!=op_args[0]:
            raise CodegenError('O-proj/mHC workspace signature changed')
        call=next(c for c in d['render_calls'] if c['symbol']==op['symbol'])
        expected_splits=2 if fn=='decode_mla_attention_sparse.prefix' else 1
        if call['tasks']!=64 or call['rows']!=4096 or call['accumulation_splits']!=expected_splits:
            raise CodegenError('O-proj computation must retain the frozen ordered WGMMA contract')
        post_call=next(c for c in d['render_calls'] if c['symbol']==post['symbol'])
        if post_call['hc']!=4 or not post_call['value_first_fma'] or post_call['residual_input_type']!='tl.bfloat16':
            raise CodegenError('mHC numerical contract changed')
        abi=definition['function_pool_abi']
        validate_function_pool_abi(abi,sum(k.get('entry_function')==fn for k in d['kernels']))
        data=copy.deepcopy(post_call)
        data.update(symbol=op['symbol'],signature=call['signature']+', _oproj_window, _oproj_residual',**geometry)
        # These expressions use the original helper formals, so substitute only
        # residual and result with the new owner-owned arguments. Workspace
        # coeff/post expressions stay verbatim from the ordinary descriptor.
        data['residual']='_oproj_residual'
        if len(post_args)==5:
            data['post']='_oproj_post'
            data['comb']='_oproj_comb'
            data['signature']+=', _oproj_post, _oproj_comb'
        else:
            data['result']='_oproj_result'
            data['signature']+=', _oproj_result'
        call['signature']=data['signature']
        call['owner_matched_oproj']=data
        op['arguments']+=', _flagmega_call_pool_workspace, '+', '.join(post_args[1:-1])
        # Kernel-level algorithm owns its ordinary local-grid completion once.
        events[i:i+5]=[op]
        receipt[fn]=dict(numerical_region=region,geometry=geometry,intermediate_boxing_elided=True,
                        alias_reason='Boxing destination overlaps projection source; owner fusion keeps its value in registers')
    if set(receipt)!=FUNCTIONS:raise CodegenError('All O-proj families must be replaced')
    d['implementation_override']['kind']='ordered-rms-qk64-and-oproj-tile-publication/v1'
    d['implementation_override']['inputs'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'oproj.py',OUT/'oproj-head.py.jinja',OUT/'oproj-tail.py.jinja']})
    d['owner_matched_oproj']=receipt
    return d

def transform(source):
    anchor='    for _fm_task in tl.range(block_index, {{ call.tasks }}, tl.num_programs(0)):'
    if source.count(anchor)!=1:raise CodegenError('Projection tile loop changed')
    source=source.replace(anchor,anchor+'\n{% if call.owner_matched_oproj | default(none) %}\n{% include "oproj-head.py.jinja" %}\n{% endif %}')
    tail='        tl.store({{ call.result }} + _fm_n, _fm_total.to(tl.bfloat16), _fm_n < {{ call.rows }})'
    if source.count(tail)!=1:raise CodegenError('Projection rounding/storage boundary changed')
    return source.replace(tail,tail+'\n{% if call.owner_matched_oproj | default(none) %}\n{% include "oproj-tail.py.jinja" %}\n{% endif %}')

def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous=tir_package.describe_tir_package
    def wrapped(module):return describe(module,previous)
    tir_package.describe_tir_package=wrapped
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,wrapped,spec.render)
    original=TritonTemplateRegistry.render
    # The accepted split implementation replaces this key in an inner renderer.
    # Extend that authoritative source so the outer overlay cannot be shadowed.
    current=original
    found=False
    while callable(current):
        cells=dict(zip(current.__code__.co_freevars,current.__closure__ or ()))
        cell=cells.get('source')
        if cell is not None and isinstance(cell.cell_contents,str) and 'call.accumulation_splits' in cell.cell_contents and 'for _fm_task' in cell.cell_contents:
            cell.cell_contents=transform(cell.cell_contents)
            found=True
            break
        parents=[cells[name].cell_contents for name in
                 ('original','previous_render','original_render')
                 if name in cells and callable(cells[name].cell_contents)]
        if len(parents)>1:
            raise CodegenError('Ambiguous parent renderer chain')
        current=parents[0] if parents else None
    if not found:raise CodegenError('Accepted ordered split template override not found')
    def render(self,name,context):
        if name!='module.py.jinja':return original(self,name,context)
        loader=self.environment.loader
        overlays={n:(OUT/n).read_text() for n in ['oproj-head.py.jinja','oproj-tail.py.jinja']}
        self.environment.loader=ChoiceLoader([DictLoader(overlays),loader])
        try:return original(self,name,context)
        finally:self.environment.loader=loader
    TritonTemplateRegistry.render=render
