"""Distribute every ordered RMS tile over all threads once, then restore lanes."""
import hashlib
from pathlib import Path
from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

KEY = 'kernels/norm_stats/platforms/nvidia/sm90/ordered_rms.py.jinja'

def transform(source):
    old = '{{ call.symbol }}__matrix_layout = tl.constexpr(tle.gpu.BlockEncoding([1, 16], [32, 1], [2, {{ num_warps // 2 }}], [0, 1]))'
    new = '{{ call.symbol }}__matrix_layout = tl.constexpr(tle.gpu.BlockEncoding([1, 4], [32, 1], [2, {{ num_warps // 2 }}], [0, 1]))'
    merge = '        norm_lane_sum = tl.zeros((64,), tl.float32)'
    if source.count(old) != 1 or source.count(merge) != 1:
        raise CodegenError('Ordered RMS source changed; distinct thread ownership needs review')
    restoration = '        norm_squares = tle.gpu.set_layout(norm_squares, tle.gpu.BlockEncoding([1, 16], [32, 1], [2, {{ num_warps // 2 }}], [0, 1]))\n'
    return source.replace(old, new).replace(merge, restoration + merge)

def activate():
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous = tir_package.describe_tir_package
    def describe(module):
        d = previous(module)
        d['implementation_override']['inputs']['norm_distinct_owner.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        d['ordered_rms_distinct_thread_ownership'] = dict(
            fma_encoding=[1, 4], merge_encoding=[1, 16],
            arithmetic_order_unchanged=True, physical_buffer_and_effects_unchanged=True)
        return d
    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = PackageRendererSpec(
        spec.name, spec.kind, spec.matches, describe, spec.render)
    original = TritonTemplateRegistry.render
    def render(self, name, context):
        if name != 'module.py.jinja':
            return original(self, name, context)
        source = self.environment.loader.get_source(self.environment, KEY)[0]
        loader = self.environment.loader
        self.environment.loader = ChoiceLoader([DictLoader({KEY: transform(source)}), loader])
        try:
            return original(self, name, context)
        finally:
            self.environment.loader = loader
    TritonTemplateRegistry.render = render
