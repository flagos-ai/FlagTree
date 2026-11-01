
import ast,copy,hashlib,json
from pathlib import Path
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/partitioned_linear/platforms/nvidia/sm90/wgmma_local.py.jinja'
def arguments(text):return [ast.unparse(a) for a in ast.parse('f('+text+')').body[0].value.args]
def describe(module,previous):
 d=previous(module)
 definitions=list(d['device_functions'])
 for p in d['programs']:definitions.extend(p.get('device_functions',()))
 definitions=list({id(v):v for v in definitions}.values())
 calls={c['symbol']:c for c in d['render_calls']};receipts=[]
 for definition in definitions:
  if definition['function']!='decode_mla_attention_sparse.prefix':continue
  events=definition['schedule']['consumer_events']
  indices=[i for i,e in enumerate(events) if e.get('family')=='partitioned_linear' and 'self_attn_q_a_proj' in e.get('symbol','')]
  assert len(indices)==1
  i=indices[0];region=events[i:i+5];original=copy.deepcopy(region)
  expected=[('tir.kernel_call','partitioned_linear'),('barrier','grid'),('tir.kernel_call','norm_stats'),('barrier','grid'),('tir.kernel_call','norm_apply')]
  if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=expected:raise CodegenError('Q_A complete projection/RMS/norm region changed')
  projection,stats,norm=[calls[region[k]['symbol']] for k in (0,2,4)]
  if not (projection['rows']==1536 and projection['parts']==1 and projection['accumulation_splits']==2 and projection['accumulation_k']==2048 and projection['variant']=='wgmma_local'):raise CodegenError('Q_A projection contract changed')
  if not (stats['reduction_capacity']==1536 and stats['tile']==2048 and stats['outer_capacity']==1 and stats['norm_stats_reduction_layout']=='contiguous_8' and stats['norm_stats_reduction_algorithm']=='tree' and not stats['use_mean']):raise CodegenError('Q_A stats original tree changed')
  if not (norm['local_capacity']==96 and norm['tile']==128 and norm['normalization_size']==1536 and not norm['use_mean'] and not norm['round_before_scale'] and norm['input_type']=='tl.bfloat16' and norm['output_type']=='tl.bfloat16' and norm['input_conversion_type'] is None):raise CodegenError('Q_A norm original arithmetic or ownership changed')
  if stats['source']!=norm['source'] or stats['result']!=norm['stats']:raise CodegenError('Q_A intermediate binding changed')
  pa=arguments(projection['signature']);na=arguments(norm['signature'])
  if norm['scale'] not in na:raise CodegenError('Q_A norm weight not a formal')
  replacements={norm['scale']:'_qa_norm_weight','rdata':'_qa_norm_rdata','block_local_data':'_qa_norm_stats'}
  def subst(text):
   for a,b in replacements.items():text=text.replace(a,b)
   return text
  fused={k:subst(norm[k]) if isinstance(norm[k],str) else norm[k] for k in ['source','stats','scale','bias','result','active','writer_active','source_offset','stats_square_sum_offset','scale_offset','bias_offset','result_offset','epsilon','input_type','output_type','tile','local_capacity','normalization_size']}
  fused.update(stats_output=subst(stats['result']),stats_offset=stats['result_offsets'][0].replace('norm_outer_index','0'))
  projection['qa_norm_fused']=fused;projection['signature']+=', _qa_norm_weight, _qa_norm_rdata, _qa_norm_stats'
  actual=dict(zip(na,arguments(region[4]['arguments']),strict=True))
  region[0]['arguments']+=', '+', '.join(actual[k] for k in [norm['scale'],'rdata','block_local_data'])
  events[i:i+5]=[region[0]]
  # Existing next grid completes normalized global result before Q_B consumes it.
  if (events[i+1]['kind'],events[i+1].get('scope'))!=('barrier','grid'):raise CodegenError('Q_A normalized result missing global completion')
  receipts.append(dict(function=definition['function'],original_region=original,projection_symbol=projection['symbol'],original_stats=stats,original_norm=norm,implementation=fused,exact_bf16_rounding=True,exact_contiguous8_square_tree=True,normalized_completion='existing-next-grid',stats_intermediate='block-local',global_partials_completion='existing-internal-projection-grid'))
 if len(receipts)!=1:raise CodegenError('Expected exactly one Q_A region')
 d['owner_fused_qa_norm']=receipts
 d['implementation_override']['inputs']['qa_norm.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 d['implementation_override']['inputs']['qa-norm-body.py.jinja']=hashlib.sha256((OUT/'qa-norm-body.py.jinja').read_bytes()).hexdigest()
 return d
def transform(source):
 start=source.index('    for _fm_task in tl.range(block_index,{{ call.tasks }},tl.num_programs(0)):',source.index('def {{ call.symbol }}__merge'))
 end=source.index('{{ entry.phase_schedule',start)
 old=source[start:end]
 return source[:start]+'{% if call.qa_norm_fused | default(none) %}\n{% include "qa-norm-body.py.jinja" %}\n{% else %}\n'+old+'{% endif %}\n'+source[end:]
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def wrapped(module):return describe(module,previous)
 tir_package.describe_tir_package=wrapped
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,wrapped,spec.render)
 original=Environment.get_template
 def selected(self,name,*args,**kwargs):
  if name in {'module.py.jinja',KEY}:
   source=self.loader.get_source(self,KEY)[0]
   if 'qa_norm_fused' not in source and 'parallel_tasks' in source:
    self.loader=ChoiceLoader([DictLoader({KEY:transform(source),'qa-norm-body.py.jinja':(OUT/'qa-norm-body.py.jinja').read_text()}),self.loader])
  return original(self,name,*args,**kwargs)
 Environment.get_template=selected
