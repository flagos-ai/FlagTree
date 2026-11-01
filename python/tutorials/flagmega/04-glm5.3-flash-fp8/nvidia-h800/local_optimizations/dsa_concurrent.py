"""Early independent DSA projections with explicit deferred result ownership."""
from pathlib import Path
import ast,copy,hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/dense_matmul/platforms/nvidia/sm90/packed_k_major_mma_prefetch.py.jinja'
def args(s):return [ast.unparse(n) for n in ast.parse('f('+s+')').body[0].value.args]
def describe(module,previous):
 d=previous(module);calls={c['symbol']:c for c in d['render_calls']}
 resources=[r for r in d['persistent_window_resources']['resources'] if r['name']=='dsa_deferred_results']
 if len(resources)!=1 or resources[0]['nbytes']!=768:raise CodegenError('DSA requires planned768B deferred results')
 resource=resources[0]
 definitions=list(d['device_functions'])
 for p in d['programs']:definitions.extend(p.get('device_functions',()))
 definitions=list({id(v):v for v in definitions}.values());receipts=[]
 for definition in definitions:
  if definition['function']!='decode_mla_attention_sparse.prefix':continue
  events=definition['schedule']['consumer_events']
  selected=[(i,e,calls[e['symbol']]) for i,e in enumerate(events) if e.get('symbol') in calls and calls[e['symbol']]['variant']=='packed_k_major_mma_prefetch']
  if len(selected)!=4:raise CodegenError('Exactly four original DSA prefetch events required')
  qi,qe,q=selected[0];others=selected[1:]
  if any(c['block_k']!=128 or c['prefetch_k']!=2048 or c['prefetch_n']!=8 or c['local_n_capacity']!=8 or c['tile_n']!=16 for _,_,c in selected):raise CodegenError('Original accepted K128 DSA contract differs')
  if q['local_k_capacity']!=1536 or any(c['local_k_capacity']!=4096 for _,_,c in others):raise CodegenError('Original DSA input shapes differ')
  original=copy.deepcopy(events[qi:others[-1][0]+1])
  qa=args(q['signature']);actual=args(qe['arguments']);q['signature']+=', _dsa_key, _dsa_gate, _dsa_weights, _dsa_window'
  qe['arguments']+=', '+', '.join([args(e['arguments'])[1] for _,e,_ in others]+['_flagmega_call_pool_workspace'])
  q['dsa_concurrent']=dict(window=resource,others=[],original=original)
  q['local_n_capacity']=16;q['prefetch_n']=16
  for key in ['result_active','prefetch_result_active']:
   text=ast.unparse(ast.parse(q[key],mode='eval'));q[key]=text.replace('dense_local_n_offsets // 8 < 1','dense_local_n_offsets // 8 < 2').replace('dense_bulk_n_offsets // 8 < 1','dense_bulk_n_offsets // 8 < 2')
  n='dense_bulk_n_offsets[:, None]'
  if n not in q['prefetch_weight_offset']:raise CodegenError('Original Q physical offset changed')
  q['prefetch_weight_offset']=q['prefetch_weight_offset'].replace(n,'(dense_bulk_n_offsets[:, None] % 8)')+' + (dense_bulk_n_offsets[:, None] // 8) * 12288'
  # The typed result pointer includes allocation base and owner displacement.
  # Remapped coordinates update that pointer; append only each CTA local lane.
  q['result_offset']='dense_local_n_offsets'
  for slot,(i,e,c) in enumerate(others):
   if ['dsa_indexer_key','dsa_indexer_gate','dsa_indexer_weights'][slot] not in c['symbol']:raise CodegenError('Original DSA projection order changed')
   if len(args(c['signature']))!=3:raise CodegenError('Original DSA pointer ABI differs')
   c['dsa_deferred']=dict(window=resource,slot=slot,result=c['result'],result_offset=c['result_offset'],result_active=c['result_active'])
   c['signature']+=', _dsa_window';e['arguments']+=', _flagmega_call_pool_workspace';e['symbol']+='__publish'
   q['dsa_concurrent']['others'].append(dict(symbol=c['symbol'],arguments=', '.join([qa[0],['_dsa_key','_dsa_gate','_dsa_weights'][slot],qa[2],'_dsa_window'])))
  receipts.append(dict(function=definition['function'],original_region=original,window=resource,owners=dict(q=64,key=16,gate=16,weights=1),original_all_inner_k128_math=True,deferred_original_publication=True))
 if len(receipts)!=1:raise CodegenError('Exactly one full prefix DSA region required')
 d['owner_concurrent_dsa']=receipts;d['implementation_override']['inputs']['dsa_concurrent.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return d
def transform(source):
 marker='def {{ call.symbol }}({{ call.signature }}):'
 if source.count(marker)!=1:raise CodegenError('Original DSA prefetch helper changed')
 source=source.replace(marker,'def {{ call.symbol }}{% if call.dsa_concurrent | default(none) or call.dsa_deferred | default(none) %}__body{% endif %}({{ call.signature }}):')
 anchor='{{ entry.mesh_coordinates() }}'
 if source.count(anchor)!=1:raise CodegenError('Original DSA mesh coordinates changed')
 source=source.replace(anchor,anchor+'''\n{% if call.dsa_concurrent | default(none) %}
    shard_coord1=tl.program_id(0)//8
    shard_coord2=tl.program_id(0)%8*2
    shard_index=(shard_coord0*8+shard_coord1)*16+shard_coord2
    block_index=shard_coord1*16+shard_coord2
{% elif call.dsa_deferred | default(none) %}
    shard_coord1=tl.full((),{{ call.dsa_deferred.slot }},tl.int64)
    shard_coord2={% if call.dsa_deferred.slot==0 %}tl.program_id(0)-64{% elif call.dsa_deferred.slot==1 %}tl.program_id(0)-80{% else %}tl.full((),0,tl.int64){% endif %}
    shard_index=(shard_coord0*8+shard_coord1)*16+shard_coord2
    block_index=shard_coord1*16+shard_coord2
{% endif %}
''')
 store='tl.store({{ call.result }} + ({{ call.result_offset }}), dense_result.to(tl.bfloat16), mask=dense_n_mask)'
 if source.count(store)!=1:raise CodegenError('Original DSA result store changed')
 pointer='{% if call.dsa_deferred | default(none) %}_dsa_window.to(tl.pointer_type(tl.bfloat16))+{{ call.dsa_deferred.window.offset//2 }}+{{ call.dsa_deferred.slot*128 }}+{% if call.dsa_deferred.slot<2 %}shard_coord2*8+{% else %}shard_coord0*8+{% endif %}dense_local_n_offsets{% else %}{{ call.result }} + ({{ call.result_offset }}){% endif %}'
 source=source.replace(store,'tl.store('+pointer+', dense_result.to(tl.bfloat16), mask=dense_n_mask)')
 tail='\n{% endif %}\n{% endfor %}'
 if source.count(tail)!=1:raise CodegenError('Original DSA template tail changed')
 wrappers='''
{% if call.dsa_concurrent | default(none) %}
@triton.jit
def {{ call.symbol }}({{ call.signature }}):
    pid=tl.program_id(0)
    if pid<64:
        {{ call.symbol }}__body({{ call.signature }})
    elif pid<80:
        {{ call.dsa_concurrent.others[0].symbol }}__body({{ call.dsa_concurrent.others[0].arguments }})
    elif pid<96:
        {{ call.dsa_concurrent.others[1].symbol }}__body({{ call.dsa_concurrent.others[1].arguments }})
    elif pid==96:
        {{ call.dsa_concurrent.others[2].symbol }}__body({{ call.dsa_concurrent.others[2].arguments }})
{% elif call.dsa_deferred | default(none) %}
@triton.jit
def {{ call.symbol }}__publish({{ call.signature }}):
{{ entry.mesh_coordinates() }}
    dense_local_n_offsets=tl.arange(0,16)
    dense_n_mask=(dense_local_n_offsets<8)&({{ call.dsa_deferred.result_active }})
    result=tl.load(_dsa_window.to(tl.pointer_type(tl.bfloat16))+{{ call.dsa_deferred.window.offset//2 }}+{{ call.dsa_deferred.slot*128 }}+{% if call.dsa_deferred.slot<2 %}shard_coord2*8+{% else %}shard_coord0*8+{% endif %}dense_local_n_offsets,mask=dense_n_mask,other=0.)
    tl.store({{ call.dsa_deferred.result }}+({{ call.dsa_deferred.result_offset }}),result,mask=dense_n_mask)
{% endif %}
'''
 source=source.replace(tail,wrappers+tail)
 # trim_blocks consumes the newline after inline conditionals. Preserve an
 # explicit Python line break before following statements and arguments.
 source=source.replace('{% endif %}\n    shard_index','{% endif %}{{ "\\n" }}\n    shard_index')
 return source
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
   if 'dsa_deferred' not in source:self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
  return original(self,name,*a,**kw)
 Environment.get_template=selected
