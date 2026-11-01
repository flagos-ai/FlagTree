"""Complete one original 32-column Q_B owner in one MMA fragment."""
from pathlib import Path
import hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/dense_matmul/platforms/nvidia/sm90/packed_k_major_mma_accumulator.py.jinja'
def transform(source):
 anchor='{% set load_warps_n = [num_warps, call.tile_n // 4] | min %}'
 if source.count(anchor)!=1:raise CodegenError('Original accumulator weight-layout decision differs')
 return source.replace(anchor,'{% if call.qb_n32_original_load_warps_n | default(none) %}\n{% set load_warps_n = call.qb_n32_original_load_warps_n %}\n{% else %}\n'+anchor+'\n{% endif %}')
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def describe(module):
  d=previous(module)
  calls=[c for c in d['render_calls'] if 'q_b_proj' in c['call']]
  if len(calls)!=1:raise CodegenError('Exactly one shared full Q_B definition required')
  c=calls[0]
  expected=dict(family='dense_matmul',variant='packed_k_major_mma_accumulator',local_m_capacity=1,local_n_capacity=32,local_k_capacity=1536,tile_n=16,block_k=128,accumulation_splits=1,partition_k=1536,coalesced_weight_load=True,output_type='tl.bfloat16')
  bad={k:(c.get(k),v) for k,v in expected.items() if c.get(k)!=v}
  if bad:raise CodegenError('Original complete Q_B contract differs: '+repr(bad))
  c['tile_n']=32;c['qb_n32_original_load_warps_n']=4
  d['mla_qb_complete_n32']=dict(symbol=c['symbol'],tile_n=32,original_tile_n=16,weight_load_layout=[[1,2],[4,8],[4,2],[1,0]],original_complete_ordered_k1536=True,original128owners_and4096outputs=True)
  d['implementation_override']['inputs']['qb_n32.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  return d
 tir_package.describe_tir_package=describe
 spec=renderer.renderer_registry._specs['bufferized-tir']
 renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
 original=Environment.get_template
 def selected(self,name,*a,**kw):
  if name in {'module.py.jinja',KEY}:
   source=self.loader.get_source(self,KEY)[0]
   if 'qb_n32_original_load_warps_n' not in source:self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
  return original(self,name,*a,**kw)
 Environment.get_template=selected
