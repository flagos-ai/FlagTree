"""Normalize ready KV partials on idle QA CTAs, retain the QB completion grid."""
import ast, copy, hashlib
from pathlib import Path
from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

X = Path(__file__).resolve().parent
KEY = 'kernels/partitioned_linear/platforms/nvidia/sm90/wgmma_local.py.jinja'

def arguments(text):
    return [ast.unparse(v) for v in ast.parse('f('+text+')').body[0].value.args]

def describe(module, previous):
    d = previous(module)
    fs = list(d['device_functions'])
    for p in d['programs']:
        fs.extend(p.get('device_functions',()))
    fs = list({id(f):f for f in fs}.values())
    f, = [f for f in fs if f['function']=='decode_mla_attention_sparse.prefix']
    resources = [r for r in d['persistent_window_resources']['resources']
                 if r['name']=='qkv_concurrent_normalized']
    if len(resources)!=1 or resources[0]['nbytes']!=1024:
        raise CodegenError('KV early normalization requires planned1024-byte normalized storage')
    resource = resources[0]
    actors = [k['symbol'] for k in d['kernels']
              if k.get('entry_function')=='decode_mla_attention_sparse.prefix']
    if actors!=resource['programs'] or len(actors)!=11:
        raise CodegenError('KV normalized storage requires all11 ordered actors')
    normalized = (f'(_qkv_window.to(tl.pointer_type(tl.uint8)) + {resource["offset"]})'
                  '.to(tl.pointer_type(tl.bfloat16))')
    es = f['schedule']['consumer_events']
    calls = {c['symbol']:c for c in d['render_calls']}
    qi, = [i for i,e in enumerate(es) if e.get('symbol') in calls and calls[e['symbol']].get('qa_norm_fused')]
    q = calls[es[qi]['symbol']]
    ki, = [i for i,e in enumerate(es) if 'self_attn_kv_a_proj_with_mqa' in e.get('symbol','')]
    region = copy.deepcopy(es[ki:ki+5])
    expect = [('tir.kernel_call','partitioned_linear'),('barrier','grid'),
              ('tir.kernel_call','norm_stats'),('barrier','block'),('tir.kernel_call','norm_apply')]
    if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=expect or ki!=qi+3:
        raise CodegenError('Original QB/KV complete region changed')
    k = calls[region[0]['symbol'].removesuffix('__merge')]
    stats, norm = calls[region[2]['symbol']], calls[region[4]['symbol']]
    if not (q.get('qkv_concurrent') and k.get('qkv_partial_pointer') and q['qa_norm_fused']['writer_active']=='(shard_coord1 == 0)'
            and q['qa_norm_fused']['local_capacity']==96 and k['rows']==512 and k['parts']==1
            and k['accumulation_splits']==2 and stats['reduction_capacity']==512 and stats['tile']==512
            and stats['norm_stats_reduction_layout']=='contiguous_8' and stats['norm_stats_reduction_algorithm']=='tree'
            and norm['local_capacity']==32 and norm['tile']==32 and norm['normalization_size']==512
            and not norm['use_mean'] and not norm['round_before_scale'] and norm['input_conversion_type'] is None
            and norm['output_type']==norm['input_type']=='tl.bfloat16'):
        raise CodegenError('Original QA/KV arithmetic or complete ownership changed')
    if stats['source']!=norm['source'] or stats['result']!=norm['stats']:
        raise CodegenError('Original KV intermediate binding changed')
    na = arguments(norm['signature'])
    actual = dict(zip(na,arguments(region[4]['arguments']),strict=True))
    substitutions = {norm['scale']:'_kv_norm_weight','rdata':'_kv_norm_rdata','block_local_data':'_kv_local'}
    def subst(text):
        for old,new in substitutions.items():
            text=text.replace(old,new)
        return text
    data = {key:subst(norm[key]) if isinstance(norm[key],str) else norm[key] for key in
            ['source','stats','scale','bias','result','active','writer_active','source_offset',
             'stats_square_sum_offset','scale_offset','bias_offset','result_offset','epsilon','input_type',
             'output_type','tile','local_capacity','normalization_size']}
    # Q-A merge still reads its partials while these CTAs normalize KV.
    # The old raw-KV range can alias those partials after sequential allocation;
    # publish early results in their declared, disjoint persistent resource.
    data.update(stats_output=stats['result'].replace('block_local_data','_kv_local'),
                stats_offset=stats['result_offsets'][0].replace('norm_outer_index','0'),
                source=normalized, result=normalized, result_offset=norm['source_offset'],
                signature=k['signature']+', _kv_norm_weight, _kv_norm_rdata, _kv_local')
    data['partial_pointer']=k['qkv_partial_pointer']
    k['kv_idle_normalization']=data
    k['signature']=data['signature']
    extra = [actual[norm['scale']],actual['rdata'],actual['block_local_data']]
    es[ki]['arguments']+=', '+', '.join(extra)
    q['signature']+=', _early_kv_norm_weight, _early_kv_norm_rdata, _early_kv_local'
    es[qi]['arguments']+=', '+', '.join(extra)
    qa = arguments(q['signature'])
    kv_actual = arguments(es[ki]['arguments'])
    qa_actual = arguments(es[qi]['arguments'])
    mapping = dict(zip(qa_actual,qa,strict=True))
    early = []
    for v in kv_actual:
        if v in mapping:
            early.append(mapping[v])
        else:
            if v==kv_actual[1]:early.append('_qkv_kv_weight')
            elif v==kv_actual[4]:early.append('_qkv_kv_descriptor')
            else:raise CodegenError('KV early actual not carried by original QA signature: '+v)
    q['kv_idle_normalization']=dict(symbol=k['symbol'], arguments=', '.join(early))
    q['qkv_concurrent']['kv_arguments']+=', _early_kv_norm_weight, _early_kv_norm_rdata, _early_kv_local'
    # Retain the original post-QB grid. DSA writes may reuse QB output ranges.
    # The original later block completion still protects private cache inputs.
    es[ki]['symbol']=k['symbol']+'__copy_normalized'
    es[ki:ki+5]=[es[ki],copy.deepcopy(region[1])]
    if es[ki+2]['scope']!='block':raise CodegenError('Original KV cache block completion changed')
    d['kv_idle_normalization']=dict(function=f['function'], original_region=region,
        replacement_region=copy.deepcopy(es[ki:ki+3]), q_formals=q['signature'],
        early_kv_arguments=early, original_stats=stats, original_norm=norm,
        qa_owners=list(range(16)),kv_owners=list(range(16,32)),
        kv_global_result=(resource['offset'],resource['offset']+resource['nbytes']),
        normalized_private_result=(0,64), normalized_window_resource=resource,
        original_all_math=True, original_qb_completion_grid_retained=True,
        original_cache_block_completion_retained=True, new_window_bytes=resource['nbytes'])
    d['implementation_override']['inputs'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [X/'kv_before_qb.py',X/'kv-norm-body.py.jinja']})
    return d

def transform(source):
    old='    {{ call.symbol }}__merge({{ call.signature }})'
    if source.count(old)!=1 or 'call.qkv_concurrent' not in source:
        raise CodegenError('Accepted concurrent QKV wrapper changed')
    new='''{% if call.kv_idle_normalization | default(none) %}
    if tl.program_id(0)<16:
        {{ call.symbol }}__merge({{ call.signature }})
    elif tl.program_id(0)<32:
        {{ call.kv_idle_normalization.symbol }}__early_norm({{ call.kv_idle_normalization.arguments }})
{% else %}
'''+old+'\n{% endif %}'
    source=source.replace(old,new)
    tail='\n{% endfor %}'
    if tail not in source:raise CodegenError('Original WGMMA template tail changed')
    helper='''
{% if call.kv_idle_normalization | default(none) and call.rows==512 %}
@triton.jit(noinline=True)
def {{ call.symbol }}__early_norm({{ call.signature }}):
{{ entry.mesh_coordinates() }}
    shard_coord2=tl.program_id(0)-16
{% set _kv_original_call=call %}
{% set call=call | copy_kv_idle %}
{% include "kv-norm-body.py.jinja" %}
{% set call=_kv_original_call %}
{% endif %}
'''
    position=source.rindex(tail)
    source=source[:position]+helper+source[position:]
    custom='''
@triton.jit(noinline=True)
def {{ call.symbol }}__copy_normalized({{ call.signature }}):
{{ entry.mesh_coordinates() }}
    _kv_copy_offsets=tl.arange(0,32)
    _kv_copy_owned=shard_coord2*32+_kv_copy_offsets
    _kv_copy_value=tl.load({{ call.kv_idle_normalization.source }}+_kv_copy_owned)
    tl.store(_kv_local.to(tl.pointer_type(tl.bfloat16))+_kv_copy_offsets,_kv_copy_value)
'''
    marker='\n{% endfor %}'
    position=source.rindex(marker)
    source=source[:position]+'\n{% if call.kv_idle_normalization | default(none) and call.rows==512 %}\n'+custom+'\n{% endif %}'+source[position:]
    return source

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
    def render(self,name,context):
        if name!='module.py.jinja':return original(self,name,context)
        loader=self.environment.loader
        self.environment.filters['copy_kv_idle']=lambda c:{**c,**c['kv_idle_normalization'], 'kv_norm_fused':c['kv_idle_normalization']}
        self.environment.loader=ChoiceLoader([DictLoader({
            'kv-norm-body.py.jinja':(X/'kv-norm-body.py.jinja').read_text()}),loader])
        try:return original(self,name,context)
        finally:self.environment.loader=loader
    TritonTemplateRegistry.render=render
