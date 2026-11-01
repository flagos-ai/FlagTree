"""Retain the native GLM chunk-dot order independently of program warp count."""
from dataclasses import replace
import hashlib
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

PROFILE = 'glm5-vllm-0.31-tp4/'
KEY = 'kernels/kda_gated_recurrent_core/local.py.jinja'
_active = False

# This is the already validated native chunk contract: eight virtual groups
# each accumulate sixteen K terms; the group merge uses distances 4, 2, 1.
# The virtual groups are independent of the selected physical warp count.
_HELPER = '''{% if arithmetic == "chunk_bf16" and call.native_ordered_chunk_dot | default(false) %}
@triton.jit
def {{ call.symbol }}__ordered_chunk_dot(state, factors, columns):
    virtual = tl.arange(0, 8)
    value = tl.load(state + virtual[:, None] * 128 + columns[None, :]).to(tl.bfloat16).to(tl.float32)
    coefficient = tl.gather(factors, virtual, 0)
    partial = value * coefficient[:, None]
    for step in tl.static_range(1, 16):
        index = virtual + step * 8
        value = tl.load(state + index[:, None] * 128 + columns[None, :]).to(tl.bfloat16).to(tl.float32)
        coefficient = tl.gather(factors, index, 0)
        partial = libdevice.fma_rn(value, coefficient[:, None], partial)
    for distance in tl.static_range(4, 0, -1):
        if distance == 4 or distance == 2 or distance == 1:
            partner = tl.gather(partial, tl.broadcast_to((virtual ^ distance)[:, None], (8, 32)), 0)
            partial = libdevice.add_rn(partial, partner)
    return tl.gather(partial, tl.full((1, 32), 0, tl.int32), 0).reshape((32,))
{% endif %}
'''

def _template(source):
    anchor = '{{ entry.kernel_decorator(call) }}\ndef {{ symbol }}('
    if source.count(anchor) != 1:
        raise CodegenError('Native chunk dot requires the declared KDA phase function')
    source = source.replace(anchor, _HELPER + anchor)
    for name, factor in (('_fm_prediction', '_fm_kd'), ('_fm_previous', '_fm_qd')):
        old = f'{name} = tl.sum(_fm_h * {factor}[:, None], 0).to(tl.bfloat16).to(tl.float32)'
        if source.count(old) != 1:
            raise CodegenError(f'Native chunk dot expression changed: {name}')
        new = (f'{name} = {{{{ call.symbol }}}}__ordered_chunk_dot('
               '{{ call.state }} + ({{ call.state_head_offset }}), '
               f'{factor}, _fm_offsets_v).to(tl.bfloat16).to(tl.float32)')
        source = source.replace(old, '{% if call.native_ordered_chunk_dot | default(false) %}\n'
                                '            ' + new + '\n{% else %}\n'
                                '            ' + old + '\n{% endif %}')
    return source

def activate():
    """Install the task-owned precision contract for the explicit native profile."""
    global _active
    if _active:
        return
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

    original_describe = tir_package.describe_tir_package
    def describe(module):
        result = original_describe(module)
        if not str(module.metadata.get('numerical_contract', '')).startswith(PROFILE):
            return result
        calls = [c for c in result['render_calls'] if c['family'] == 'kda_gated_recurrent_core']
        for call in calls:
            if (call['key_dim'], call['value_dim']) != (128, 128):
                raise CodegenError('Native GLM chunk-dot contract requires K128/V128')
            call['native_ordered_chunk_dot'] = True
        if calls:
            result['native_chunk_dot_precision'] = dict(
                virtual_groups=8, terms_per_group=16, group_tree=[4, 2, 1],
                local_accumulation='first multiply then fifteen fma.rn in K order',
                physical_warp_count_independent=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result

    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = replace(spec, describe=describe)
    original_render = TritonTemplateRegistry.render
    def render(self, name, context):
        calls = context.get('render_calls', ())
        if not any(c.get('native_ordered_chunk_dot', False) for c in calls):
            return original_render(self, name, context)
        loader = self.environment.loader
        source = loader.get_source(self.environment, KEY)[0]
        self.environment.loader = ChoiceLoader([DictLoader({KEY: _template(source)}), loader])
        try:
            return original_render(self, name, context)
        finally:
            self.environment.loader = loader
    TritonTemplateRegistry.render = render
    _active = True
