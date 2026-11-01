"""Complete owner-preserving mHC collapse/RMS/norm-apply physical fusion."""
import ast,copy,hashlib,json
from pathlib import Path
from jinja2 import ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/mhc_pre/cooperative_stream_fma.py.jinja'
FUNCTIONS={'decode_linear_attention_dense','decode_linear_attention_sparse.prefix','decode_mla_attention_sparse.prefix'}

def args(text):return [ast.unparse(a) for a in ast.parse('f('+text+')').body[0].value.args]

def describe(module,previous):
    d=previous(module)
    definitions=list(d['device_functions'])
    for p in d['programs']:definitions.extend(p.get('device_functions',()))
    definitions=list({id(v):v for v in definitions}.values())
    calls={c['symbol']:c for c in d['render_calls']};receipts=[]
    resource,=[r for r in d['persistent_window_resources']['resources'] if r['name']=='mhc_projection_partials']
    if resource['nbytes']!=1024:raise CodegenError('mHC window capacity changed')
    actors=[k['symbol'] for k in d['kernels'] if k.get('entry_function') in FUNCTIONS]
    if actors!=resource['programs']:raise CodegenError('mHC window actor order changed')
    for definition in definitions:
        if definition['function'] not in FUNCTIONS:continue
        events=definition['schedule']['consumer_events'];indices=[i for i,e in enumerate(events) if e.get('family')=='mhc_pre']
        for i in reversed(indices):
            region=events[i:i+5]
            original_region=copy.deepcopy(region)
            expected=[('tir.kernel_call','mhc_pre'),('barrier','grid'),('tir.kernel_call','norm_stats'),('barrier','block'),('tir.kernel_call','norm_apply')]
            if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=expected:
                raise CodegenError('Complete mHC/RMS region changed')
            pre,stats,norm=[calls[region[k]['symbol']] for k in (0,2,4)]
            if not (pre['hc']==4 and pre['hidden']==4096 and pre['projection_splits']==8 and pre['independent_collapse_pre'] and pre['collapse_sequential_fma'] and pre['collapse_input_type']=='tl.bfloat16' and pre['collapsed_type']=='tl.float32'):
                raise CodegenError('mHC arithmetic differs from the validated candidate')
            if stats['norm_stats_reduction_algorithm']!='fma_1024' or stats['reduction_capacity']!=4096:
                raise CodegenError('Ordered RMS contract differs')
            if norm['local_capacity']!=32 or norm['normalization_size']!=4096 or norm['use_mean'] or norm['round_before_scale'] or norm['input_conversion_type']!='tl.bfloat16':
                raise CodegenError('Norm-apply rounding or owner differs')
            if stats['source']!=pre['collapsed'] or norm['source']!=pre['collapsed'] or stats['result']!=norm['stats']:
                raise CodegenError('Region intermediate ownership changed')
            pa=args(pre['signature']);na=args(norm['signature']);newargs=pa[:-1]+['_mhc_norm_weight','_mhc_norm_rdata','_mhc_norm_stats','_mhc_window']
            # All five matched norm weights are the only non-pool formal.
            if norm['scale'] not in na:raise CodegenError('Norm weight is not its original formal')
            replacements={norm['scale']:'_mhc_norm_weight','rdata':'_mhc_norm_rdata','block_local_data':'_mhc_norm_stats'}
            result=norm['result']
            if result in na and result not in pa:
                replacements[result]='_mhc_norm_result';newargs.append('_mhc_norm_result')
            newargs.append(pa[-1])
            def subst(text):
                for old,new in replacements.items():text=text.replace(old,new)
                return text
            original_partials=pre['projection_partials']
            pre['projection_partials']=f'(_mhc_window.to(tl.pointer_type(tl.float32)) + {resource["offset"]//4})'
            fused=dict(streams=pre['streams'],base=pre['base'],scale=pre['scale'],partials=pre['projection_partials'],
                collapsed=pre['collapsed'],stats=subst(stats['result']),weight=subst(norm['scale']),bias=subst(norm['bias']),result=subst(result))
            pre['signature']=', '.join(newargs);pre['owner_fused_rms']=fused
            actual=args(region[0]['arguments']);normactual=dict(zip(na,args(region[4]['arguments']),strict=True))
            additional=[normactual[norm['scale']],normactual['rdata'],normactual['block_local_data'],'_flagmega_call_pool_workspace']
            if '_mhc_norm_result' in newargs:additional.append(normactual[result])
            region[0]['arguments']=', '.join(actual[:-1]+additional+actual[-1:])
            # The physical region keeps project completion and mix, but has
            # no dependence on another CTA's internal collapsed result.
            events[i:i+5]=[region[0]]
            receipts.append(dict(function=definition['function'],numerical_region=original_region,implementation=fused,
                intermediate_collapsed_elided=True,statistics_dtype='float32',norm_input_conversion='bfloat16',
                intermediate_statistic_consumed_only_in_registers=True,
                normalized_owner_mapping=dict(tile_width=32,cta0='mix-only',ctas1_to127='tile=cta-1',cta127_extra_tile=127,grid_ctas=128),
                output_publication='existing-next-grid-barrier-or-whole-program-completion',
                original_partials=original_partials,projection_window=resource))
    if len(receipts)!=5:raise CodegenError('All five supported complete regions must match')
    # The exact region matcher above proves norm_stats has one regional
    # consumer and its statistic is a block-local unexported intermediate.
    # Changed normalized owners need global completion, not a block fence.
    for receipt in receipts:
        definition=next(v for v in definitions if v['function']==receipt['function'])
        events=definition['schedule']['consumer_events']
        symbol=receipt['numerical_region'][0]['symbol']
        index=next(i for i,e in enumerate(events) if e.get('symbol')==symbol)
        after=events[index+1:]
        if after and (after[0]['kind'],after[0].get('scope'))!=('barrier','grid'):
            raise CodegenError('Rotated normalized producer lacks grid completion')
        if not after and receipt['implementation']['result']!='_mhc_norm_result':
            raise CodegenError('End-of-program region is not the exported normalized result')
        receipt['normalized_completion']=('grid-barrier' if after else 'kernel-completion')
    d['owner_fused_mhc_rms']=receipts
    d['implementation_override']['kind']='singlewarp-mhc-overlap-rotated-rms/v1'
    d['implementation_override']['inputs'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'fusion.py',OUT/'fused-body.py.jinja')})
    return d

def transform(source):
    wrapper='''    {{ call.symbol }}__collapse({{ call.signature }})'''
    if source.count(wrapper)!=1:raise CodegenError('Independent mHC wrapper changed')
    replacement='''{% if call.owner_fused_rms | default(none) %}
    if tl.program_id(0) != 0:
        _flagmega_mhc_rms_fused_body({{ call.owner_fused_rms.streams }}, {{ call.owner_fused_rms.base }}, {{ call.owner_fused_rms.scale }}, {{ call.owner_fused_rms.partials }}, {{ call.owner_fused_rms.collapsed }}, {{ call.owner_fused_rms.stats }}, {{ call.owner_fused_rms.weight }}, {{ call.owner_fused_rms.bias }}, {{ call.owner_fused_rms.result }})
{% else %}
'''+wrapper+'''\n{% endif %}'''
    header='''{% if render_calls | selectattr("owner_fused_rms", "defined") | list %}
{% include "fused-body.py.jinja" %}
{% endif %}
'''
    return header+source.replace(wrapper,replacement)

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
        source=self.environment.loader.get_source(self.environment,KEY)[0]
        loader=self.environment.loader
        overlays={KEY:transform(source),'fused-body.py.jinja':(OUT/'fused-body.py.jinja').read_text()}
        self.environment.loader=ChoiceLoader([DictLoader(overlays),loader])
        try:return original(self,name,context)
        finally:self.environment.loader=loader
    TritonTemplateRegistry.render=render
