"""Unroll the full mHC recurrence while preserving its RN32/FTZ divisions."""
import hashlib
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader, Environment


KEY = 'kernels/mhc_pre/cooperative_stream_fma.py.jinja'
BEFORE = 'for _fm_iteration in tl.range(0, {{ call.sinkhorn_iterations - 1 }}):'
AFTER = ('for _fm_iteration in tl.range(0, {{ call.sinkhorn_iterations - 1 }}, '
         'loop_unroll_factor={{ call.sinkhorn_iterations - 1 }}):')


def transform(source):
    if source.count(BEFORE) != 1:
        raise ValueError('Expected the original complete mHC recurrence')
    from .sinkhorn_normal_division import transform_template
    return transform_template(source.replace(BEFORE, AFTER))


def activate():
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec

    previous_describe = tir_package.describe_tir_package

    def describe(module):
        package = previous_describe(module)
        package['implementation_override']['inputs']['sinkhorn_unroll.py'] = hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()
        from .sinkhorn_normal_division import describe as describe_normal_division
        from .sigmoid_unit_reciprocal import describe as describe_sigmoid
        return describe_sigmoid(describe_normal_division(package))

    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = PackageRendererSpec(
        spec.name, spec.kind, spec.matches, describe, spec.render)
    previous_template = Environment.get_template

    def selected(self, name, *args, **kwargs):
        if name in ('module.py.jinja', KEY):
            source = self.loader.get_source(self, KEY)[0]
            if AFTER not in source:
                from .sigmoid_unit_reciprocal import transform_template as transform_sigmoid
                self.loader = ChoiceLoader([DictLoader({KEY: transform_sigmoid(transform(source))}), self.loader])
        return previous_template(self, name, *args, **kwargs)

    Environment.get_template = selected
