"""Use original head projection K order in K128 fragments with accepted memory layouts."""
from pathlib import Path
import hashlib
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 def describe(module):
  d=previous(module);calls=[c for c in d['render_calls'] if c['family']=='mla_projection']
  if len(calls)!=2 or any(c['variant']!='mma' or not c['head_sharded'] or c['heads']!=16 or c['reduction_tile']!=64 or c['tile_n']!=64 or c['quantized'] for c in calls):raise CodegenError('Original BF16 16head M64/K64 MLA contract differs')
  for c in calls:c['reduction_tile']=128
  d['mla_head_k128']=dict(original_k_tile=64,new_k_tile=128,original_per_output_k_order=True,accepted_memory_layouts=True)
  d['implementation_override']['inputs']['head_k128.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return d
 tir_package.describe_tir_package=describe
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
