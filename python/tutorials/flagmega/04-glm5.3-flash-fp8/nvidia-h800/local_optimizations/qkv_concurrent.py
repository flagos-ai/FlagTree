"""Normal complete-region scheduling with explicitly planned disjoint partials."""
from pathlib import Path
import ast,copy,hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/partitioned_linear/platforms/nvidia/sm90/wgmma_local.py.jinja'
def args(text):return [ast.unparse(n) for n in ast.parse('f('+text+')').body[0].value.args]
def describe(module,previous):
 d=previous(module);definitions=list(d['device_functions'])
 for p in d['programs']:definitions.extend(p.get('device_functions',()))
 definitions=list({id(v):v for v in definitions}.values());calls={c['symbol']:c for c in d['render_calls']}
 resources=[r for r in d['persistent_window_resources']['resources'] if r['name']=='qkv_concurrent_partials']
 if len(resources)!=1 or resources[0]['nbytes']!=4096:raise CodegenError('QKV requires planned4096-byte partial window')
 resource=resources[0];actors=[k['symbol'] for k in d['kernels'] if k.get('entry_function')=='decode_mla_attention_sparse.prefix']
 if actors!=resource['programs'] or len(actors)!=11:raise CodegenError('QKV window lacks original11 ordered actors')
 receipts=[]
 for definition in definitions:
  if definition['function']!='decode_mla_attention_sparse.prefix':continue
  events=definition['schedule']['consumer_events']
  qi=[i for i,e in enumerate(events) if e.get('family')=='partitioned_linear' and 'self_attn_q_a_proj' in e.get('symbol','')]
  ki=[i for i,e in enumerate(events) if e.get('family')=='partitioned_linear' and 'self_attn_kv_a_proj' in e.get('symbol','')]
  if len(qi)!=1 or len(ki)!=1:raise CodegenError('QKV original complete schedule changed')
  qindex,kindex=qi[0],ki[0];qevent,kevent=events[qindex],events[kindex];q,k=calls[qevent['symbol']],calls[kevent['symbol']]
  original=copy.deepcopy(events[qindex:kindex+1]);qargs=args(q['signature']);kargs=args(k['signature'])
  if not (q['rows']==1536 and k['rows']==512 and all(c['parts']==1 and c['accumulation_splits']==2 and c['accumulation_k']==2048 and c['block_k']==512 and c['variant']=='wgmma_local' for c in [q,k])):
   raise CodegenError('QKV full original WGMMA contract differs')
  if not (kindex==qindex+3 and events[qindex+1]['kind']=='barrier' and events[qindex+1]['scope']=='grid' and events[qindex+2]['family']=='dense_matmul'):
   raise CodegenError('QKV consumers between original projections changed')
  if q['source_k_origin']!=k['source_k_origin']:raise CodegenError('QKV projections do not share original input descriptor')
  if not q.get('qa_norm_fused') or k.get('kv_norm_fused'):raise CodegenError('QKV expected accepted Q_A fusion and original KV merge')
  qactual=args(qevent['arguments']);kactual=args(kevent['arguments'])
  if qactual[0]!=kactual[0] or qactual[3]!=kactual[3]:raise CodegenError('QKV actual source/descriptor differs')
  if len(kargs)!=5 or len(qargs)!=8:raise CodegenError('QKV external projection signature changed')
  from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer,_pointer
  original_pointer=_pointer(_buffer(k,'workspaces','projection_partials'))
  k['qkv_partial_pointer']=f'(_qkv_window.to(tl.pointer_type(tl.float32)) + {resource["offset"]//4})'
  k['signature']+=', _qkv_window';kevent['arguments']+=', _flagmega_call_pool_workspace'
  k['concurrent_task_base']=48
  k['concurrent_partial_window']=resource
  q['signature']+=', _qkv_kv_weight, _qkv_kv_descriptor, _qkv_window'
  qevent['arguments']+=', '+', '.join([kactual[1],kactual[4],'_flagmega_call_pool_workspace'])
  kvformals=kargs+['_qkv_window'];kvvalues=[qargs[0],'_qkv_kv_weight',qargs[2],qargs[3],'_qkv_kv_descriptor','_qkv_window']
  q['qkv_concurrent']=dict(kv_symbol=k['symbol'],kv_arguments=', '.join(kvvalues),cta_q='0<=pid<48',cta_kv='48<=pid<64',window=resource)
  # Both original partials complete at Q_A's internal grid. Its norm/Q_B
  # preserve order; only the later KV wrapper becomes its original merge.
  kevent['symbol']+='__merge'
  receipts.append(dict(function=definition['function'],original_region=original,original_kv_pointer=original_pointer,window=resource,original_q_partial_tasks=48,original_kv_partial_tasks=16,source_readonly=True,all_original_dot_order=True,completion='Q_A internal original grid',reuse_completion='All11 fullprefix invocations ordered by existing program DAG'))
 if len(receipts)!=1:raise CodegenError('Exactly one full QKV region expected')
 d['owner_concurrent_qkv']=receipts
 d['implementation_override']['inputs']['qkv_concurrent.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 return d

def transform(source):
 anchor='    for _fm_task in tl.range(block_index, {{ call.parallel_tasks }}, tl.num_programs(0)):'
 if source.count(anchor)!=1:raise CodegenError('Original parallel WGMMA task loop changed')
 replacement='    for _fm_task in tl.range({% if call.concurrent_task_base | default(none) %}block_index-{{ call.concurrent_task_base }}{% else %}block_index{% endif %}, {{ call.parallel_tasks }}, tl.num_programs(0)):'
 source=source.replace(anchor,replacement)
 wrapper='{{ entry.phase_schedule(call, ("partials", "merge")) }}'
 if source.count(wrapper)!=1:raise CodegenError('Original partial publication changed')
 new='''{% if call.qkv_concurrent | default(none) %}
@triton.jit
def {{ call.symbol }}({{ call.signature }}):
    if tl.program_id(0)<48:
        {{ call.symbol }}__partials({{ call.signature }})
    elif tl.program_id(0)<64:
        {{ call.qkv_concurrent.kv_symbol }}__partials({{ call.qkv_concurrent.kv_arguments }})
{% if call.qkv_mhc_mix | default(none) %}
    elif tl.program_id(0)=={{ call.qkv_mhc_mix.cta }}:
        {{ call.qkv_mhc_mix.symbol }}({{ call.qkv_mhc_mix.arguments }})
{% endif %}
{{ entry.stage_barrier() }}
    {{ call.symbol }}__merge({{ call.signature }})
{% else %}
'''+wrapper+'\n{% endif %}'
 source=source.replace(wrapper,new)
 marker='{% if call.family == \"partitioned_linear\" and call.variant == \"wgmma_local\" and call.parameters.get(\"parallel_accumulation_splits\", false) %}'
 if source.count(marker)!=1:raise CodegenError('Parallel implementation loop changed')
 return source.replace(marker,marker+'\n{% if call.qkv_partial_pointer | default(none) %}{% set _original_qkv_partial_pointer=call.partial_pointer %}{% set _unused_qkv_update=call.update(partial_pointer=call.qkv_partial_pointer) %}{% endif %}')
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def wrapped(module):return describe(module,previous)
 tir_package.describe_tir_package=wrapped
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,wrapped,spec.render)
 original=Environment.get_template
 def selected(self,name,*a,**kw):
  if name in {'module.py.jinja',KEY}:
   source=self.loader.get_source(self,KEY)[0]
   if 'qkv_concurrent' not in source and 'parallel_tasks' in source:
    # Apply the already active Q_A merge transformation before inserting
    # the new wrapper, so its original merge-range replacement cannot
    # consume the wrapper condition.
    if 'qa_norm_fused' not in source:
     from local_optimizations.qa_norm import transform as qa_transform
     source=qa_transform(source)
    from local_optimizations.kv_before_qb import transform as kv_idle_transform
    self.loader=ChoiceLoader([DictLoader({KEY:kv_idle_transform(transform(source)),'qa-norm-body.py.jinja':(OUT/'qa-norm-body.py.jinja').read_text()}),self.loader])
  return original(self,name,*a,**kw)
 Environment.get_template=selected
