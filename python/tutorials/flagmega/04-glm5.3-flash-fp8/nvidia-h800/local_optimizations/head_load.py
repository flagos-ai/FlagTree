"""Coalesced loads for the existing complete chip-owned MLA head projections."""
from pathlib import Path
import hashlib
from jinja2 import ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/mla_projection/platforms/nvidia/sm90/mma.py.jinja'
def transform(source):
 anchor='{{ entry.kernel_decorator(call) }}'
 assert source.count(anchor)==1
 layouts="{{ call.symbol }}__memory_layout = tl.constexpr(tle.gpu.BlockEncoding({% if call.direction == 'absorb_query' %}[1,8],[4,8],[2,4],[1,0]{% else %}[8,1],[8,4],[4,2],[0,1]{% endif %}))\n"
 source=source.replace(anchor,layouts+anchor)
 load='_fm_mla_w = tl.load({{ call.weight }} + ({{ call.weight_offset }}), mask=_fm_mla_mask, other=0.)'
 if source.count(load)!=1:raise CodegenError('Head-sharded MLA weight pointer changed')
 return source.replace(load,'_fm_mla_w = tle.gpu.set_layout(tl.load(tle.gpu.set_layout({{ call.weight }} + ({{ call.weight_offset }}), {{ call.symbol }}__memory_layout), mask=_fm_mla_mask, other=0.), {{ call.symbol }}__memory_layout)')
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
 previous=tir_package.describe_tir_package
 def describe(module):
  d=previous(module)
  calls=[c for c in d['render_calls'] if c['family']=='mla_projection']
  if len(calls)!=2 or any(c['variant']!='mma' or not c['head_sharded'] or c['heads']!=16 or c['reduction_tile']!=64 or c['tile_n']!=64 or c['quantized'] for c in calls):raise CodegenError('Only existing BF16 head-owned M64/K64 MLA projections supported')
  d['implementation_override']['inputs']['head_load.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  d['mla_head_weight_memory_layout']={'absorb_query':[[1,8],[4,8],[2,4],[1,0]],'expand_value':[[8,1],[8,4],[4,2],[0,1]]}
  return d
 tir_package.describe_tir_package=describe
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
 original=TritonTemplateRegistry.prepare
 def prepare(self,name,context):
  if name in {'module.py.jinja',KEY}:
   source=self.environment.loader.get_source(self.environment,KEY)[0]
   if '__memory_layout' not in source:
    self.environment.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.environment.loader])
  return original(self,name,context)
 TritonTemplateRegistry.prepare=prepare
