"""Keep full 16-head scores and original online order, partition only PV output columns.

The original elected CTA owns all 512 independent result columns. Four CTAs
each retain the original complete score/max/sum/rounding and produce a disjoint
128-column interval. No shared partial reduction, persistent window or ABI changes.
"""
from pathlib import Path
import hashlib
from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

KEY = "kernels/dsa_paged_attention/platforms/nvidia/sm90/online_mma.py.jinja"

def transform(source):
    old_dim = "        _fm_dsa_dim = tl.arange(0, {{ call.padded_head_dim }})"
    new_dim = "        _fm_dsa_dim = block_index * {{ call.output_column_width }} + tl.arange(0, {{ call.output_column_width }})"
    old_acc = "tl.zeros((16, {{ call.padded_head_dim }}), tl.float32)"
    new_acc = "tl.zeros((16, {{ call.output_column_width }}), tl.float32)"
    if source.count(old_dim) != 1 or source.count(old_acc) != 1:
        raise CodegenError("Original independent-column accumulator changed")
    return source.replace(old_dim, new_dim).replace(old_acc, new_acc)

def probability_selected(call):
    from .native_latent_probability_model import SCHEDULE
    parameters=call['parameters']
    if parameters.get('latent_work_schedule')!=SCHEDULE:return False
    if ((call['head_dim'],call['padded_head_dim'],call['num_query_heads'],call['head_sharded'],call['index_traversal'],call['compact_topk_capacity'],call['probability_rounding'])!=(512,512,16,True,'compact',4096,'online_bf16')
            or parameters['compute_num_warps']!=8 or parameters['latent_column_tile']!=64
            or parameters['latent_long_column_tile']!=32 or not parameters['latent_compact_pipeline']):
        raise CodegenError('Declared probability output columns require complete original math and W8 short64/long32 geometry')
    return True

def activate():
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous = tir_package.describe_tir_package
    def describe(module):
        d = previous(module)
        calls = [c for c in d['render_calls'] if c['family'] == 'dsa_paged_attention'
                 and c['variant'] == 'decode' and c['decode_schedule'] == 'bf16_mma']
        if len(calls) != 1:
            raise CodegenError("Requires one original shared latent implementation")
        c = calls[0]
        if probability_selected(c):
            d['declared_probability_output_column_replacement']=dict(short_owners=8,short_width=64,long_owners=16,long_width=32,
                legacy_owners=4,legacy_width=128,complete_online64_and_BF16_math=True,selected_by_declared_probability_implementation=True)
            d['implementation_override']['inputs']['dsa_column_owners.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            return d
        if (c['head_dim'], c['padded_head_dim'], c['num_query_heads'],
            c['head_sharded'], c['index_traversal'], c['compact_topk_capacity'],
            c['writer_active']) != (512, 512, 16, True, 'compact', 4096,
                                   '(shard_coord1 == 0) & (shard_coord2 == 0)'):
            raise CodegenError("Original latent coordinate/owner contract changed")
        c['writer_active'] = 'block_index < 4'
        c['output_column_width'] = 128
        d['dsa_output_column_owners'] = dict(owners=4, width=128,
            original_full_16_head_qk_per_owner=True,
            original_online_tile64_and_bf16_boundaries=True,
            output_columns_disjoint_complete=True, persistent_window_unchanged=True)
        d['implementation_override']['inputs']['dsa_column_owners.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = PackageRendererSpec(spec.name, spec.kind, spec.matches, describe, spec.render)
    original = TritonTemplateRegistry.render
    def render(self, name, context):
        if name != 'module.py.jinja':
            return original(self, name, context)
        calls = [c for c in context['render_calls'] if c['family'] == 'dsa_paged_attention'
                 and c['variant'] == 'decode' and c['decode_schedule'] == 'bf16_mma']
        if not calls:
            return original(self, name, context)
        if len(calls) != 1:
            raise CodegenError("Requires one selected latent implementation per namespace")
        c = calls[0]
        if probability_selected(c):
            schedule=c.get('probability_schedule')
            if (not schedule or schedule['column_tile']!=64 or schedule['long_column_tile']!=32
                    or schedule['group']['count']!=128 or schedule['internal_grid_barriers']!=3
                    or schedule['metadata_payload_bytes']!=8320):
                raise CodegenError('Probability output-column selection requires normally propagated workspace and publishing contracts')
            return original(self,name,context)
        if c.get('output_column_width') != 128 or c['writer_active'] != 'block_index < 4':
            raise CodegenError("Column-owner descriptor was not normally propagated")
        loader = self.environment.loader
        source = loader.get_source(self.environment, KEY)[0]
        self.environment.loader = ChoiceLoader([DictLoader({KEY: transform(source)}), loader])
        try:
            return original(self, name, context)
        finally:
            self.environment.loader = loader
    TritonTemplateRegistry.render = render

