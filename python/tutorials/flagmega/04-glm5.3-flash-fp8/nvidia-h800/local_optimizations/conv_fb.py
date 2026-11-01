"""Normal independent convolution/F_B region with original ordered math."""
from pathlib import Path
import ast,copy,hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
CK='kernels/kda_convolution/local.py.jinja'
FK='kernels/dense_matmul/platforms/nvidia/sm90/packed_k_major_mma.py.jinja'
def args(text):return [ast.unparse(n) for n in ast.parse('f('+text+')').body[0].value.args]
def describe(module,previous):
 d=previous(module);definitions=list(d['device_functions'])
 for p in d['programs']:definitions.extend(p.get('device_functions',()))
 definitions=list({id(v):v for v in definitions}.values());calls={c['symbol']:c for c in d['render_calls']};receipts=[]
 for definition in definitions:
  if definition['function'] not in {'decode_linear_attention_dense','decode_linear_attention_sparse.prefix'}:continue
  events=definition['schedule']['consumer_events'];indices=[i for i,e in enumerate(events) if e.get('family')=='kda_convolution']
  if len(indices)!=1:raise CodegenError('Expected one original convolution region')
  i=indices[0];region=events[i:i+3];original=copy.deepcopy(region)
  if [(e['kind'],e.get('family',e.get('scope'))) for e in region]!=[('tir.kernel_call','kda_convolution'),('tir.kernel_call','dense_matmul'),('barrier','grid')]:raise CodegenError('Original conv/F_B singlegrid schedule changed')
  conv,fb=calls[region[0]['symbol']],calls[region[1]['symbol']]
  if 'self_attn_f_b_proj' not in fb['symbol'] or fb['variant']!='packed_k_major_mma' or fb['local_n_capacity']!=16 or fb['local_k_capacity']!=128 or fb['tile_n']!=16 or fb['block_k']!=32:raise CodegenError('Original full F_B projection contract differs')
  if conv['capacity']!=128 or conv['tile']!=128 or conv['kernel']!=4 or len(conv['parts'])!=3:raise CodegenError('Original full convolution contract differs')
  if any(p['result_writer_active']!='(shard_coord1 == 0)' for p in conv['parts']) or conv['writer_active']!='(shard_coord1 == 0)':raise CodegenError('Original convolution canonical writer owner differs')
  fa=args(fb['signature']);actual=args(region[1]['arguments']);ca=args(conv['signature'])
  if len(fa)!=3 or len(ca)!=5 or fa[0]!='workspace' or fa[2]!='device_dptr':raise CodegenError('Original pointer ABI changed')
  conv['signature']+=', _kda_fb_weight';region[0]['arguments']+=', '+actual[1]
  conv['concurrent_fb']=dict(symbol=fb['symbol'],arguments=', '.join([fa[0],'_kda_fb_weight',fa[2]]),original=original)
  fb['concurrent_fb_task']=True;fb['signature']+=', _kda_fb_task';fb['local_n_capacity']=32;fb['tile_n']=32
  fb['result_active']=ast.unparse(ast.parse(fb['result_active'],mode='eval')).replace('dense_local_n_offsets // 8 < 2','dense_local_n_offsets // 8 < 4')
  original_n='dense_local_n_offsets[:, None]'
  if original_n not in fb['weight_offset']:raise CodegenError('Original F_B packed column index changed')
  fb['weight_offset']=fb['weight_offset'].replace(original_n,'(dense_local_n_offsets[:, None] % 16)')+' + (dense_local_n_offsets[:, None] // 16) * 2048'
  events[i:i+3]=[region[0],region[2]]
  receipts.append(dict(function=definition['function'],original_region=original,conv_ctas=16,fb_ctas=64,original_single_grid=True,all_per_output_operations_retained=True))
 if len(receipts)!=2:raise CodegenError('Both dense and sparse conv/F_B regions required')
 d['owner_concurrent_conv_fb']=receipts;d['implementation_override']['inputs']['conv_fb.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return d

def convolution(source):
 marker='def {{ call.symbol }}({{ call.signature }}):'
 if source.count(marker)!=1:raise CodegenError('Original convolution helper changed')
 source=source.replace(marker,'def {{ call.symbol }}{% if call.concurrent_fb | default(none) %}__body{% endif %}({{ call.signature }}):')
 anchor='\n\n{% endif %}\n{% endfor %}'
 if source.count(anchor)!=1:raise CodegenError('Original convolution region tail changed')
 wrapper='\n{% if call.concurrent_fb | default(none) %}\n@triton.jit\ndef {{ call.symbol }}({{ call.signature }}):\n    if tl.program_id(0)<16:\n        {{ call.symbol }}__body({{ call.signature }})\n    elif tl.program_id(0)<80:\n        {{ call.concurrent_fb.symbol }}({{ call.concurrent_fb.arguments }}, tl.program_id(0)-16)\n{% endif %}\n'
 return source.replace(anchor,wrapper+anchor)
def projection(source):
 anchor='{{ entry.mesh_coordinates() }}'
 if source.count(anchor)!=1:raise CodegenError('Original F_B coordinates changed')
 return source.replace(anchor,anchor+'\n{% if call.concurrent_fb_task | default(false) %}\n    shard_coord1 = _kda_fb_task // 8\n    shard_coord2 = _kda_fb_task % 8 * 2\n    shard_index = (shard_coord0 * 8 + shard_coord1) * 16 + shard_coord2\n    block_index = shard_coord1 * 16 + shard_coord2\n{% endif %}\n')
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def wrapped(module):return describe(module,previous)
 tir_package.describe_tir_package=wrapped
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,wrapped,spec.render)
 original=Environment.get_template
 def selected(self,name,*a,**kw):
  if name in {'module.py.jinja',CK,FK}:
   for key,transform in [(CK,convolution),(FK,projection)]:
    source=self.loader.get_source(self,key)[0]
    if 'concurrent_fb' not in source:self.loader=ChoiceLoader([DictLoader({key:transform(source)}),self.loader])
  return original(self,name,*a,**kw)
 Environment.get_template=selected
