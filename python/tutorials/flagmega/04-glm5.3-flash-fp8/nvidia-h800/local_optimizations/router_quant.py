"""Private logits extend router lifetime while disjoint owners quantize input."""
from pathlib import Path
import ast,copy,hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
ROUTER='kernels/staged_routing/local.py.jinja'
PROJECT='kernels/dense_matmul/platforms/nvidia/packed_k_major_threaded_fma.py.jinja'
UP='kernels/sparse_experts_gate_up/platforms/nvidia/sm90/complete_k_wgmma.py.jinja'
def args(s):return [ast.unparse(n) for n in ast.parse('f('+s+')').body[0].value.args]
def describe(module,previous):
 d=previous(module);definitions=list(d['device_functions'])
 for p in d['programs']:definitions.extend(p.get('device_functions',()))
 definitions=list({id(v):v for v in definitions}.values());calls={c['symbol']:c for c in d['render_calls']}
 resource,=[r for r in d['persistent_window_resources']['resources'] if r['name']=='router_private_logits'];assert resource['nbytes']==1280
 receipts=[]
 for definition in definitions:
  if definition['function'] not in {'decode_mla_attention_sparse.routed_up','decode_linear_attention_sparse.routed_up'}:continue
  events=definition['schedule']['consumer_events'];indices=[i for i,e in enumerate(events) if e.get('family')=='staged_routing']
  if len(indices)!=1:raise CodegenError('Exactly one original router required')
  i=indices[0];region=events[i:i+4];original=copy.deepcopy(region)
  if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=[('tir.kernel_call','staged_routing'),('tir.kernel_call','sparse_experts_dispatch'),('barrier','grid'),('tir.kernel_call','sparse_experts_gate_up')]:raise CodegenError('Original router/dispatch/quantization schedule differs')
  pe=events[i-2];assert pe['family']=='dense_matmul' and events[i-1]['kind']=='barrier'
  projection,router,dispatch,up=calls[pe['symbol']],calls[region[0]['symbol']],calls[region[1]['symbol']],calls[region[3]['symbol']]
  if projection['variant']!='packed_k_major_threaded_fma' or router['capacity']!=288 or router['rows']!=1 or router['tile']!=512 or router['k']!=8:raise CodegenError('Original288→top8 router differs')
  if not (dispatch['capacity']==256 and dispatch['tile']==128 and up['scalar_k']==4096 and up['quantization_tasks']==256 and up['variant']=='complete_k_wgmma'):raise CodegenError('Original routed input quantization differs')
  pointer=f'(_router_window.to(tl.pointer_type(tl.float32))+{resource["offset"]//4})'
  projection['signature']+=', _router_window';pe['arguments']+=', _flagmega_call_pool_workspace';projection['result']=pointer
  actual=args(region[1]['arguments']);ra=args(router['signature'])
  if args(region[3]['arguments'])[0]!=actual[1] or ra[0]!='workspace':raise CodegenError('Original shared workspace/source ABI differs')
  router['signature']+=', _router_source, _router_window';region[0]['arguments']+=', '+actual[0]+', _flagmega_call_pool_workspace';router['source']=pointer
  router['router_quant_overlap']=dict(quant_symbol=up['symbol']+'__quantize',arguments=', '.join([ra[0],'0','0','0','0','0','0','0',ra[-1],'_router_source']))
  up['signature']+=', _dispatch_source';region[3]['arguments']+=', '+actual[0]
  up['router_quant_overlap']=True;up['original_dispatch_pointer']=up['peer_input'];up['original_dispatch_offset']=up['peer_input_offset'];up['peer_input']='_dispatch_source';up['peer_input_offset']='_fm_token*4096+_fm_k'
  events[i:i+4]=[region[0],region[3]]
  receipts.append(dict(function=definition['function'],original_region=original,private_logits=resource,canonical_router=0,quant_owners=127,original_top8_tanh_and_rounding=True,original_all_dispatch_bytes_preserved=True,original_project_and_completion=True))
 if len(receipts)!=2:raise CodegenError('Both full routed regions required')
 d['owner_overlap_router_quant']=receipts;d['implementation_override']['inputs']['router_quant.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return d
def router_transform(source):
 marker='def {{ call.symbol }}({{ call.signature }}):'
 if source.count(marker)!=1:raise CodegenError('Original staged router body differs')
 source=source.replace(marker,'def {{ call.symbol }}{% if call.router_quant_overlap | default(none) %}__body{% endif %}({{ call.signature }}):')
 tail='\n\n{% endif %}\n{% endfor %}'
 if source.count(tail)!=1:raise CodegenError('Original staged router template tail differs')
 wrapper='''
{% if call.router_quant_overlap | default(none) %}
@triton.jit
def {{ call.symbol }}({{ call.signature }}):
    if tl.program_id(0)==0:
        {{ call.symbol }}__body({{ call.signature }})
    else:
        {{ call.router_quant_overlap.quant_symbol }}({{ call.router_quant_overlap.arguments }})
{% endif %}
'''
 return source.replace(tail,wrapper+tail)
def up_transform(source):
 anchor='for _fm_task in tl.range(tl.program_id(0), {{ call.quantization_tasks }}, tl.num_programs(0)):'
 if source.count(anchor)!=1:raise CodegenError('Original256 quantization tasks differ')
 source=source.replace(anchor,'for _fm_task in tl.range({% if call.router_quant_overlap | default(false) %}tl.program_id(0)-1{% else %}tl.program_id(0){% endif %}, {{ call.quantization_tasks }}, {% if call.router_quant_overlap | default(false) %}tl.num_programs(0)-1{% else %}tl.num_programs(0){% endif %}):')
 anchor='        _fm_source = tl.load({{ call.peer_input }} + ({{ call.peer_input_offset }}), _fm_active_slot, other=0.).to(tl.float32)'
 source=source.replace(anchor,anchor+'''\n{% if call.router_quant_overlap | default(false) %}
        tl.store({{ call.original_dispatch_pointer }}+({{ call.original_dispatch_offset }}),_fm_source.to(tl.bfloat16),_fm_active_slot)
{% endif %}
''')
 phase='{{ entry.phase_schedule(call, ("quantize", "project")) }}'
 if source.count(phase)!=1:raise CodegenError('Original quantization/project completion differs')
 replacement='''{% if call.router_quant_overlap | default(false) %}
@triton.jit
def {{ call.symbol }}({{ call.signature }}):
{{ entry.stage_barrier() }}
    {{ call.symbol }}__project({{ call.signature }})
{% else %}
'''+phase+'\n{% endif %}'
 return source.replace(phase,replacement)
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def wrapped(module):return describe(module,previous)
 tir_package.describe_tir_package=wrapped
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,wrapped,spec.render)
 original=Environment.get_template
 def selected(self,name,*a,**kw):
  if name in {'module.py.jinja',ROUTER,UP,PROJECT}:
   for key,transform in [(ROUTER,router_transform),(UP,up_transform)]:
    source=self.loader.get_source(self,key)[0]
    if 'router_quant_overlap' not in source:self.loader=ChoiceLoader([DictLoader({key:transform(source)}),self.loader])
  return original(self,name,*a,**kw)
 Environment.get_template=selected
