"""Keep the ordered per-lane RMS merge in registers after the FMA tiles."""
import hashlib
from pathlib import Path
from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

KEY="kernels/norm_stats/platforms/nvidia/sm90/ordered_rms.py.jinja"


def transform(source):
    old="""        norm_lane_sum = tl.zeros((64,), tl.float32)
        for norm_ordinal in tl.static_range(16):
            norm_element = (norm_ordinal % 2) * 8 + norm_ordinal // 2
            norm_scalar = tl.sum(tl.gather(norm_squares, tl.full((64, 1), norm_element, tl.int32), 1), 1)
            norm_lane_sum = norm_lane_sum + norm_scalar"""
    if source.count(old)!=1:
        raise CodegenError("Ordered RMS merge source changed; register extraction requires review")
    lines=[]
    columns={}
    def split(name, width, indices):
        if width==1:
            columns[indices[0]]=name
            return
        even, odd = name+"_even", name+"_odd"
        lines.append(f"        {even}, {odd} = tl.split(tl.reshape({name}, (64, {width//2}, 2)))")
        split(even,width//2,indices[::2]);split(odd,width//2,indices[1::2])
    split("norm_squares",16,list(range(16)))
    lines.append("        norm_lane_sum = tl.zeros((64,), tl.float32)")
    for ordinal in range(16):
        index=(ordinal%2)*8+ordinal//2
        lines.append(f"        norm_lane_sum = norm_lane_sum + tl.reshape({columns[index]}, (64,))")
    return source.replace(old,"\n".join(lines))


def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous=tir_package.describe_tir_package
    def describe(module):
        descriptor=previous(module)
        descriptor['implementation_override']['kind']='ordered-rms-register-and-coalesced-qk64/v1'
        descriptor['implementation_override']['inputs']['norm_layout.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        descriptor['ordered_rms_merge']='register-split-v1'
        return descriptor
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
    original=TritonTemplateRegistry.render
    def render(self,name,context):
        if name!='module.py.jinja':return original(self,name,context)
        source=self.environment.loader.get_source(self.environment,KEY)[0]
        loader=self.environment.loader
        self.environment.loader=ChoiceLoader([DictLoader({KEY:transform(source)}),loader])
        try:return original(self,name,context)
        finally:self.environment.loader=loader
    TritonTemplateRegistry.render=render
