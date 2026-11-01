"""Native latent MLA: complete scores, contiguous-K loads, independent output columns."""
from dataclasses import replace
import hashlib
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

from .dsa_work import head_group
from .qk64 import transform as coalesced_fragments

SCHEDULE = 'complete_heads_contiguous_k_output_columns'
KEY = 'kernels/dsa_paged_attention/platforms/nvidia/sm90/online_mma.py.jinja'
_active = False


def implementation_model(model, *, compact_pipeline=False):
    """Declare the work mapping before selecting and planning physical kernels."""
    implementations = []
    for implementation in model.implementations:
        if implementation.id == 'tir.dsa_paged_attention.decode':
            implementation = replace(implementation, parameters={**implementation.parameters,
                'latent_work_schedule': SCHEDULE, 'latent_qk_tile': 256 if compact_pipeline else 128,
                'latent_pv_tile': 64 if compact_pipeline else 32, 'latent_column_tile': 64,
                **({'latent_compact_pipeline': True} if compact_pipeline else {})})
        implementations.append(implementation)
    return replace(model, implementations=tuple(implementations))


def _template(source, *, compact_pipeline=False):
    source = coalesced_fragments(source)
    dimension = '        _fm_dsa_dim = tl.arange(0, {{ call.padded_head_dim }})'
    accumulator = 'tl.zeros((16, {{ call.padded_head_dim }}), tl.float32)'
    if source.count(dimension) != 1 or source.count(accumulator) != 1:
        raise CodegenError('Native latent independent-column source contract changed')
    source = source.replace(dimension,
        '        _fm_dsa_dim = ({{ call.native_latent_schedule.group.index }}) * '
        '{{ call.native_latent_schedule.column_tile }} + tl.arange(0, {{ call.native_latent_schedule.column_tile }})')
    source = source.replace(accumulator, 'tl.zeros((16, {{ call.native_latent_schedule.column_tile }}), tl.float32)')
    if compact_pipeline:
        from .native_latent_compact_pipeline import transform
        source = transform(source)
    return source


def activate():
    """Compose only explicitly selected Native schedules through normal templates."""
    global _active
    if _active:
        return
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous_describe = tir_package.describe_tir_package

    def describe(module):
        descriptor = previous_describe(module)
        selected = []
        attention = [call for call in descriptor['render_calls'] if call['family'] == 'dsa_paged_attention']
        for call in attention:
            parameters = call['parameters']
            if parameters.get('latent_work_schedule') != SCHEDULE:
                continue
            if (not str(module.metadata.get('numerical_contract', '')).startswith('glm5-vllm-0.31-tp4/')
                    or call['variant'] != 'decode' or call['decode_schedule'] != 'bf16_mma'
                    or not call['head_sharded'] or call['num_query_heads'] != 16
                    or call['head_dim'] != 512 or call['padded_head_dim'] != 512
                    or call['probability_rounding'] != 'online_bf16' or call['index_traversal'] != 'compact'
                    or parameters.get('compute_num_warps') != 8
                    or (parameters.get('latent_qk_tile'), parameters.get('latent_pv_tile'),
                        parameters.get('latent_column_tile'), parameters.get('latent_compact_pipeline', False))
                        not in ((128, 32, 64, False), (256, 64, 64, True))):
                raise CodegenError('Native latent schedule requires its complete head/fragment/rounding contract')
            group = head_group(call, descriptor)
            if group is None or group['count'] < call['head_dim'] // parameters['latent_column_tile']:
                raise CodegenError('Native latent output columns require proven complete chip-owner replicas')
            owners = call['head_dim'] // parameters['latent_column_tile']
            schedule = dict(group=group, column_tile=parameters['latent_column_tile'], owners=owners,
                            compact_pipeline=parameters.get('latent_compact_pipeline', False),
                            original_writer_active=call['writer_active'])
            call['native_latent_schedule'] = schedule
            call['writer_active'] = f"(({group['index']}) < {owners})"
            selected.append(dict(call=call['call'], **schedule))
        if selected:
            if len(selected) != 1 or len(attention) != 1:
                raise CodegenError('Native latent schedule currently requires one complete shared implementation')
            descriptor['native_latent_attention_schedule'] = dict(schedule=SCHEDULE, selected=selected,
                qk_tile=parameters['latent_qk_tile'], pv_tile=parameters['latent_pv_tile'], compute_num_warps=8,
                complete_head_scores_and_original_online64_rounding=True,
                output_columns_disjoint_complete=True, no_extra_workspace_or_barriers=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return descriptor

    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = replace(spec, describe=describe)
    previous_prepare = TritonTemplateRegistry.prepare

    def prepare(self, name, context):
        result = previous_prepare(self, name, context)
        selected = [call for call in context.get('render_calls', ()) if call.get('native_latent_schedule')]
        if name == 'module.py.jinja' and selected:
            if len(selected) != 1 or context['num_warps'] != 8:
                raise CodegenError('Native latent schedule requires its actual eight-warps program')
            loader = self.environment.loader
            source = loader.get_source(self.environment, KEY)[0]
            self.environment.loader = ChoiceLoader([DictLoader({KEY: _template(source,
                compact_pipeline=selected[0]['native_latent_schedule']['compact_pipeline'])}), loader])
        return result

    TritonTemplateRegistry.prepare = prepare
    previous_render = TritonTemplateRegistry.render

    def render(self, name, context):
        if name != 'module.py.jinja' or not any(call.get('native_latent_schedule')
                                                for call in context.get('render_calls', ())):
            return previous_render(self, name, context)
        loader = self.environment.loader
        try:
            return previous_render(self, name, context)
        finally:
            self.environment.loader = loader

    TritonTemplateRegistry.render = render
    _active = True
