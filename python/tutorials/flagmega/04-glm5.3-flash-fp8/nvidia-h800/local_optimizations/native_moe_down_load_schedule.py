"""Keep Native MoE activation and weight memory layouts independent."""
from dataclasses import replace
import hashlib
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

SCHEDULE = 'warp_contiguous_activation_and_k_contiguous_weight'
KEY = 'kernels/partitioned_sparse_experts_down/platforms/nvidia/sm90/wgmma.py.jinja'
_active = False


def implementation_model(model):
    """Declare both memory boundaries before physical variant selection."""
    return replace(model, implementations=tuple(
        replace(value, parameters={**value.parameters, 'native_moe_down_load_schedule': SCHEDULE})
        if value.id == 'tir.partitioned_sparse_experts_down.wgmma' else value
        for value in model.implementations))


def _template(source):
    declaration = '{{ entry.kernel_decorator(call) }}'
    activation = '                        _fm_input = tl.load({{ call.pointers.activations }} + ({{ call.activation_offset }}), mask=_fm_k_mask, other=0.).to(tl.float32)'
    weight = '                        _fm_weight = tl.load({{ call.pointers.down_weight }} + ({{ call.weight_offset }}), mask=_fm_n_mask[:, None] & _fm_k_mask[None, :], other=0.)'
    if any(source.count(value) != 1 for value in (declaration, activation, weight)):
        raise CodegenError('Native MoE down memory boundary source contract changed')
    layouts = '''{% if call.native_moe_down_loads | default(false) %}
{{ call.symbol }}__input_layout = tl.constexpr(tle.gpu.SlicedEncoding(0, tle.gpu.BlockEncoding([1, 4], [1, 32], [8, 1], [1, 0])))
{{ call.symbol }}__weight_layout = tl.constexpr(tle.gpu.BlockEncoding([1, 16], [4, 8], [8, 1], [1, 0]))
{% endif %}
'''
    source = source.replace(declaration, layouts + declaration)
    replacements = (
        (activation, '                        _fm_input = tle.gpu.set_layout(tl.load(tle.gpu.set_layout({{ call.pointers.activations }} + ({{ call.activation_offset }}), {{ call.symbol }}__input_layout), mask=_fm_k_mask, other=0.), {{ call.symbol }}__input_layout).to(tl.float32)'),
        (weight, '                        _fm_weight = tle.gpu.set_layout(tl.load(tle.gpu.set_layout({{ call.pointers.down_weight }} + ({{ call.weight_offset }}), {{ call.symbol }}__weight_layout), mask=_fm_n_mask[:, None] & _fm_k_mask[None, :], other=0.), {{ call.symbol }}__weight_layout)'),
    )
    for original, updated in replacements:
        source = source.replace(original, '{% if call.native_moe_down_loads | default(false) %}\n'
            + updated + '\n{% else %}\n' + original + '\n{% endif %}')
    return source


def activate():
    global _active
    if _active:
        return
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous_describe = tir_package.describe_tir_package

    def describe(module):
        descriptor = previous_describe(module)
        selected = []
        for call in descriptor['render_calls']:
            if call['parameters'].get('native_moe_down_load_schedule') != SCHEDULE:
                continue
            if (not str(module.metadata.get('numerical_contract', '')).startswith('glm5-vllm-0.31-tp4/')
                    or call['family'] != 'partitioned_sparse_experts_down' or call['variant'] != 'wgmma'
                    or call['parameters'].get('compute_num_warps') != 8
                    or (call['partitions'], call['tokens'], call['routes'], call['scalar_n'],
                        call['scalar_k'], call['output_tile_n']) != (1, 1, 1, 256, 512, 256)):
                raise CodegenError('Native MoE down load schedule requires its complete local FP8 contract')
            call['native_moe_down_loads'] = True
            selected.append(dict(call=call['call'], local_k=512, local_n=256, compute_num_warps=8))
        if selected:
            descriptor['native_moe_down_load_schedule'] = dict(schedule=SCHEDULE, selected=selected,
                arithmetic_addresses_and_owner_mapping_unchanged=True, no_extra_workspaces_or_barriers=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return descriptor

    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = replace(spec, describe=describe)
    previous_prepare = TritonTemplateRegistry.prepare

    def prepare(self, name, context):
        result = previous_prepare(self, name, context)
        if name == 'module.py.jinja' and any(call.get('native_moe_down_loads')
                                            for call in context.get('render_calls', ())):
            if context['num_warps'] != 8:
                raise CodegenError('Native MoE down load schedule requires its eight-warps program')
            loader = self.environment.loader
            source = loader.get_source(self.environment, KEY)[0]
            self.environment.loader = ChoiceLoader([DictLoader({KEY: _template(source)}), loader])
        return result

    TritonTemplateRegistry.prepare = prepare
    previous_render = TritonTemplateRegistry.render

    def render(self, name, context):
        if name != 'module.py.jinja' or not any(call.get('native_moe_down_loads')
                                                for call in context.get('render_calls', ())):
            return previous_render(self, name, context)
        loader = self.environment.loader
        try:
            return previous_render(self, name, context)
        finally:
            self.environment.loader = loader

    TritonTemplateRegistry.render = render
    _active = True
