"""Give the original packed logits load its physical memory layout."""
from pathlib import Path
import hashlib
from jinja2 import Environment, ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/dense_matmul/platforms/nvidia/sm90/packed_k_major_mma.py.jinja'
def transform(source):
    anchor='{{ entry.kernel_decorator(call) }}'
    if source.count(anchor)!=1:raise CodegenError('Original packed MMA declaration changed')
    source=source.replace(anchor,'{% if call.logits_coalesced_load | default(false) %}\n{{ call.symbol }}__physical_load_layout=tl.constexpr(tle.gpu.BlockEncoding([1,8],[8,4],[4,2],[1,0]))\n{% endif %}\n'+anchor)
    start=source.index('                dense_weight_value = tl.load(')
    end=source.index('                dense_activation_matrix =',start)
    original=source[start:end]
    load='                dense_weight_value = tle.gpu.set_layout(tl.load(tle.gpu.set_layout({{ call.weight }} + ({{ call.weight_offset }}), {{ call.symbol }}__physical_load_layout), mask=dense_n_mask[:,None] & dense_k_mask[None,:], other=0.), {{ call.symbol }}__physical_load_layout)\n'
    return source[:start]+'{% if call.logits_coalesced_load | default(false) %}\n'+load+'{% else %}\n'+original+'{% endif %}\n'+source[end:]
def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    previous=tir_package.describe_tir_package
    def describe(module):
        d=previous(module)
        calls=[c for c in d['render_calls'] if c['call']=='logits.bf16.vectorized.compute']
        if len(calls)!=1:raise CodegenError('Exactly one original full logits definition required')
        c=calls[0]
        if not (c['family']=='dense_matmul' and c['variant']=='packed_k_major_mma' and c['local_n_capacity']==304 and c['local_k_capacity']==4096 and c['local_m_capacity']==1 and c['tile_n']==64 and c['block_k']==128 and c['signature']=='workspace, rdata, device_dptr'):
            raise CodegenError('Original full logits computation/ABI differs')
        c['logits_coalesced_load']=True
        d['logits_physical_load_layout']=[[1,8],[8,4],[4,2],[1,0]]
        d['implementation_override']['inputs']['logits_load.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
    original=Environment.get_template
    def selected(self,name,*a,**kw):
        if name in {'module.py.jinja',KEY}:
            source=self.loader.get_source(self,KEY)[0]
            if 'logits_coalesced_load' not in source:self.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),self.loader])
        return original(self,name,*a,**kw)
    Environment.get_template=selected
