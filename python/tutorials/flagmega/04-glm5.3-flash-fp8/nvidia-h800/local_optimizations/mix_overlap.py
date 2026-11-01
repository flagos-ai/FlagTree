"""Defer mHC post/comb into independent KDA projection CTA101; original two grids remain."""
from pathlib import Path
import ast,copy,hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
MHC='kernels/mhc_pre/cooperative_stream_fma.py.jinja'
ENTRY='entrypoints/call_graph.py.jinja'
def describe(module,previous):
 d=previous(module);definitions=list(d['device_functions'])
 for p in d['programs']:definitions.extend(p.get('device_functions',()))
 definitions=list({id(v):v for v in definitions}.values());calls={c['symbol']:c for c in d['render_calls']};receipts=[]
 for definition in definitions:
  if definition['function']!='decode_linear_attention_sparse.prefix':continue
  events=definition['schedule']['consumer_events'];region=events[:4]
  if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=[('tir.kernel_call','mhc_pre'),('barrier','grid'),('tir.kernel_call','kda_input_projection'),('barrier','grid')]:raise CodegenError('Original KDA start region differs')
  m,p=calls[region[0]['symbol']],calls[region[2]['symbol']]
  if not m.get('owner_fused_rms') or not(m['hc']==4 and m['hidden']==4096 and m['projection_splits']==8 and m['sinkhorn_iterations']==20):raise CodegenError('Original mHC arithmetic contract differs')
  if not(p['variant']=='wgmma_local' and p['tasks']==101 and p['block_k']==256 and p['reduction']==4096):raise CodegenError('KDA original101 task/K256 contract differs')
  m['mhc_deferred_mix_cta']=101
  region[2]['overlapped_mhc_mix']=dict(symbol=m['symbol']+'__mix',arguments=region[0]['arguments'],cta=101)
  receipts.append(dict(function=definition['function'],original_region=copy.deepcopy(region),mix_cta=101,original_projection_tasks=101,original_two_grids_retained=True,mix_source='persistent_mhc_projection_partials',outputs_not_consumed_until_original_later_mhc=True))
 if len(receipts)!=1:raise CodegenError('Expected one KDA prefix start region')
 d['owner_mhc_kda_mix_overlap']=receipts;d['implementation_override']['inputs']['mix_overlap.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return d
def mhc_transform(source):
 call='    {{ call.symbol }}__mix({{ call.signature }})';assert source.count(call)==1
 source=source.replace(call,'{% if not call.mhc_deferred_mix_cta | default(none) %}\n'+call+'\n{% endif %}')
 start=source.index('def {{ call.symbol }}__mix(');pos=source.index('{{ cta_coordinates() }}',start);end=pos+len('{{ cta_coordinates() }}')
 source=source[:end]+'\n{% if call.mhc_deferred_mix_cta | default(none) %}\n    _fm_cta = _fm_cta - {{ call.mhc_deferred_mix_cta }}\n{% endif %}'+source[end:]
 return source
def entry_transform(source):
 old="""{% if event.participation_active | default("True") != "True" %}    if {{ event.participation_active }}:
        {{ event.symbol }}({{ event.arguments }})
{% else %}    {{ event.symbol }}({{ event.arguments }})
{% endif %}"""
 # There are function_call and tir.kernel_call branches; modify only the latter.
 start=source.index('    # tir.call {{ event.call }} -> {{ event.family }}');pos=source.index(old,start)
 new="""{% if event.overlapped_mhc_mix | default(none) %}
    if tl.program_id(0) == {{ event.overlapped_mhc_mix.cta }}:
        {{ event.overlapped_mhc_mix.symbol }}({{ event.overlapped_mhc_mix.arguments }})
    elif tl.program_id(0) < {{ event.overlapped_mhc_mix.cta }}:
        {{ event.symbol }}({{ event.arguments }})
{% else %}
"""+old+'\n{% endif %}'
 return source[:pos]+new+source[pos+len(old):]
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def wrapped(module):return describe(module,previous)
 tir_package.describe_tir_package=wrapped
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,wrapped,spec.render)
 original=Environment.get_template
 def selected(self,name,*args,**kw):
  for key,transform,marker in [(MHC,mhc_transform,'mhc_deferred_mix_cta'),(ENTRY,entry_transform,'overlapped_mhc_mix')]:
   if name in ['module.py.jinja',key]:
    source=self.loader.get_source(self,key)[0]
    if marker not in source:self.loader=ChoiceLoader([DictLoader({key:transform(source)}),self.loader])
  return original(self,name,*args,**kw)
 Environment.get_template=selected
