"""Complete normal KDA region: independent recurrent/G_B on disjoint CTAs."""
from pathlib import Path
import ast,copy,hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
RK='kernels/kda_gated_recurrent_core/local.py.jinja'
GK='kernels/dense_matmul/platforms/nvidia/sm90/packed_k_major_mma.py.jinja'
def args(text):return [ast.unparse(n) for n in ast.parse('f('+text+')').body[0].value.args]
def describe(module,previous):
 d=previous(module);definitions=list(d['device_functions'])
 for p in d['programs']:definitions.extend(p.get('device_functions',()))
 definitions=list({id(v):v for v in definitions}.values());calls={c['symbol']:c for c in d['render_calls']};receipts=[]
 for definition in definitions:
  if definition['function'] not in {'decode_linear_attention_dense','decode_linear_attention_sparse.prefix'}:continue
  events=definition['schedule']['consumer_events'];indices=[i for i,e in enumerate(events) if e.get('family')=='kda_gated_recurrent_core']
  if len(indices)!=1:raise CodegenError('Expected exactly one original recurrent region')
  i=indices[0];region=events[i:i+3];original=copy.deepcopy(region)
  if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=[('tir.kernel_call','kda_gated_recurrent_core'),('tir.kernel_call','dense_matmul'),('barrier','grid')]:
   raise CodegenError('Original recurrent/G_B completion schedule changed: '+repr(region))
  rec,gb=calls[region[0]['symbol']],calls[region[1]['symbol']]
  if 'self_attn_g_b_proj' not in gb['symbol'] or gb['variant']!='packed_k_major_mma' or gb['local_n_capacity']!=16 or gb['local_k_capacity']!=128 or gb['tile_n']!=16 or gb['block_k']!=32:raise CodegenError('Original full G_B projection contract differs')
  from local_optimizations.kda_work import value_group
  group=value_group(rec,d)
  if group is None or tuple(group['axes'])!=(1,) or group['count']!=8 or group['index']!='(0 * 8 + shard_coord1)' or rec['value_chunks']!=4 or rec['value_dim']!=128 or rec['key_dim']!=128:raise CodegenError('Requires original four V32 chunk owners')
  ga=args(gb['signature']);actual=args(region[1]['arguments']);ra=args(rec['signature'])
  if len(ga)!=3 or len(ra)!=7 or ga[0]!='workspace' or ga[2]!='device_dptr':raise CodegenError('Original pointer ABI changed')
  rec['signature']+=', _kda_gb_weight';region[0]['arguments']+=', '+actual[1]
  rec['concurrent_gb']=dict(symbol=gb['symbol'],arguments=', '.join([ga[0],'_kda_gb_weight',ga[2]]),original=original)
  gb['concurrent_logical_task']=True;gb['signature']+=', _kda_gb_task';gb['local_n_capacity']=32;gb['tile_n']=32;gb['result_active']=ast.unparse(ast.parse(gb['result_active'],mode='eval')).replace('dense_local_n_offsets // 8 < 2','dense_local_n_offsets // 8 < 4')
  original_n='dense_local_n_offsets[:, None]'
  if original_n not in gb['weight_offset']:raise CodegenError('Original G_B packed column index changed')
  gb['weight_offset']=gb['weight_offset'].replace(original_n,'(dense_local_n_offsets[:, None] % 16)')+' + (dense_local_n_offsets[:, None] // 16) * 2048'
  # New wrapper retains the subsequent grid and completes both writes.
  # No original grid existed between recurrent and G_B.
  events[i:i+3]=[region[0],region[2]]
  receipts.append(dict(function=definition['function'],original_region=original,recurrence=copy.deepcopy(rec),projection=copy.deepcopy(gb),logical_tasks=128,gb_physical_ctas=64,recurrent_ctas=64,completion='Existing subsequent grid completes both original output windows',numerical_order='All original per-output operations and dtype conversions retained'))
 if len(receipts)!=2:raise CodegenError('Both dense and sparse original KDA regions required')
 d['owner_concurrent_recurrent_gb']=receipts;d['implementation_override']['inputs']['recurrent_gb.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 return d

def recurrent(source):
 anchor='    if {{ call.request_prefill }}:\n        {{ call.symbol }}__prefill({{ call.signature }})\n    else:\n        {{ call.symbol }}__decode({{ call.signature }})'
 if source.count(anchor)!=1:raise CodegenError('Original recurrent phase dispatch changed')
 replacement='{% if call.concurrent_gb | default(none) %}\n    if tl.program_id(0)<64:\n        if {{ call.request_prefill }}:\n            {{ call.symbol }}__prefill({{ call.signature }})\n        else:\n            {{ call.symbol }}__decode({{ call.signature }})\n    else:\n        {{ call.concurrent_gb.symbol }}({{ call.concurrent_gb.arguments }}, tl.program_id(0)-64)\n{% else %}\n'+anchor+'\n{% endif %}'
 return source.replace(anchor,replacement)
def projection(source):
 anchor='{{ entry.mesh_coordinates() }}'
 if source.count(anchor)!=1:raise CodegenError('Original G_B mesh coordinates changed')
 return source.replace(anchor,anchor+'\n{% if call.concurrent_logical_task | default(false) %}\n    shard_coord1 = _kda_gb_task // 8\n    shard_coord2 = _kda_gb_task % 8 * 2\n    shard_index = (shard_coord0 * 8 + shard_coord1) * 16 + shard_coord2\n    block_index = shard_coord1 * 16 + shard_coord2\n{% endif %}\n')
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def wrapped(module):return describe(module,previous)
 tir_package.describe_tir_package=wrapped
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,wrapped,spec.render)
 original=Environment.get_template
 def selected(self,name,*a,**kw):
  if name in {'module.py.jinja',RK,GK}:
   for key,transform in [(RK,recurrent),(GK,projection)]:
    source=self.loader.get_source(self,key)[0]
    if 'concurrent_gb' not in source and 'concurrent_logical_task' not in source:self.loader=ChoiceLoader([DictLoader({key:transform(source)}),self.loader])
  return original(self,name,*a,**kw)
 Environment.get_template=selected
