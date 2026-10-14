"""Contract-compatible IQ4 kernel replacements for RTX 5060 Ti decode."""

from pathlib import Path
from dataclasses import replace

from jinja2 import ChoiceLoader, FileSystemLoader
from triton.flagmega.codegen.triton import tir_package
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

ROOT = Path(__file__).parent / 'templates'
CHUNK = 1024
MODE = 'paired'
WARPS = 4
GUARD = 128
GDN_FAST_HEAD = False
GDN_UNIFORM_NORM = False
GDN_CACHE_HEADS = False
Q8_GROUP_TILE = 8
Q8_NIBBLE_PAIR = False
Q8_VECTOR_QUANT = False


class IQ4Registry(TritonTemplateRegistry):

    def __init__(self):
        super().__init__()
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(ROOT)), self.environment.loader])

    def resolve(self, spec):
        for path in self.candidates(spec):
            if (ROOT / path).is_file():
                return path
        return super().resolve(spec)

    def render(self, template_name, context):
        return super().render(
            template_name, {
                **context, 'iq4_chunk': CHUNK, 'iq4_mode': MODE, 'iq4_warps': WARPS, 'refinement_guard': GUARD,
                'gdn_fast_head': GDN_FAST_HEAD, 'gdn_uniform_norm': GDN_UNIFORM_NORM, 'gdn_cache_heads':
                GDN_CACHE_HEADS, 'q8_group_tile': Q8_GROUP_TILE, 'q8_vector_quant': Q8_VECTOR_QUANT, 'q8_nibble_pair':
                Q8_NIBBLE_PAIR
            })


def install(*, chunk=1024, tile_n=16, warps=4, mode='paired', grid=32, guard=128, gdn_value_tile=4, gdn_fast_head=False,
            pc_segments=False, q8_group_tile=8, pc_dual_activation=False, pc_dual_all=False, balanced_rows=False,
            pc_dense_dual=False, pc_global_activation=False, pc_packed_activation=False, pc_down_chunk=1024,
            pc_precision_scope='all', pc_nibble_pair=False, gdn_linear_owners=False, gdn_uniform_norm=False,
            gdn_cache_heads=False, q8_vector_quant=False):
    global CHUNK, MODE, WARPS, GUARD, GDN_FAST_HEAD, Q8_GROUP_TILE, Q8_NIBBLE_PAIR, GDN_UNIFORM_NORM, GDN_CACHE_HEADS
    global Q8_VECTOR_QUANT
    Q8_VECTOR_QUANT = q8_vector_quant
    if chunk not in (256, 512, 1024, 2048, 4096):
        raise ValueError('IQ4 chunk must be a supported power of two')
    CHUNK = chunk
    MODE = mode
    WARPS = warps
    GUARD = guard
    GDN_FAST_HEAD = gdn_fast_head
    GDN_UNIFORM_NORM = gdn_uniform_norm
    GDN_CACHE_HEADS = gdn_cache_heads
    if gdn_cache_heads and not gdn_fast_head:
        raise ValueError('paired head cache requires the native IQ4 fast head helper')
    if gdn_cache_heads:
        from .gdn_heads import configure as configure_gdn_heads
        configure_gdn_heads()
    if q8_group_tile not in (8, 16, 32, 64, 128, 256):
        raise ValueError('Q8 group tile must be a supported power of two')
    Q8_GROUP_TILE = q8_group_tile
    Q8_NIBBLE_PAIR = pc_nibble_pair
    tir_package.TritonTemplateRegistry = IQ4Registry
    from triton.flagmega.targets import get_target, NvidiaSm120Target
    target = get_target('nvidia-sm120')
    if mode in ('pc_tma', 'pc_tma_inline'):
        if warps != 8 or grid not in (32, 36):
            raise ValueError('native IQ4 p/c TMA requires 8 compute warps and a resident grid')
    if grid != 32:
        from triton.flagmega.ir import Placement
        if grid != 36:
            raise ValueError('supported persistent grids are 32 and 36 CTAs')
        configured = NvidiaSm120Target(options=replace(target.options, placements=(Placement((6, 6), 'yx', 'bb'), )))
        target.options = configured.options
        target.distribution_policy = configured.distribution_policy
    if mode in ('pc_tma', 'pc_tma_inline'):
        from .pc_catalog import configure
        if pc_dense_dual and not pc_dual_all:
            raise ValueError('dense dual buffers require --pc-dual-all')
        if pc_global_activation and not (pc_dual_all and pc_dense_dual):
            raise ValueError('global activation scratch requires dual-all and dense-dual')
        if pc_packed_activation and not pc_global_activation:
            raise ValueError('packed activation requires global activation scratch')
        if pc_down_chunk not in (1024, 2048, 4096):
            raise ValueError('invalid down projection K chunk')
        if pc_precision_scope not in ('all', 'ffn', 'glu', 'down', 'attention'):
            raise ValueError('invalid activation precision scope')
        if pc_global_activation:
            from .global_activation import configure as configure_global_activation
            configure_global_activation()
        if (pc_segments or pc_dual_activation or pc_dual_all) and mode == 'pc_tma_inline':
            raise ValueError('segmented candidates currently use noinline leaves')
        configure(target, leaf_inline=mode == 'pc_tma_inline', segmented=pc_segments,
                  dual_activation=pc_dual_activation, dual_all=pc_dual_all, dense_dual=pc_dense_dual,
                  global_activation=pc_global_activation, packed_activation=pc_packed_activation,
                  down_chunk=pc_down_chunk, precision_scope=pc_precision_scope, nibble_pair=pc_nibble_pair)
    if gdn_value_tile not in (4, 8, 16, 32, 64):
        raise ValueError('GDN value tile must be a supported power of two')
    if gdn_value_tile != 4:
        model = target.triton_implementation_model
        entries = tuple(
            replace(value, parameters={**value.parameters, 'tile_state': (
                128, gdn_value_tile)}) if value.id == 'tir.gdn_recurrent.persistent' else value
            for value in model.implementations)
        target.triton_implementation_model = replace(model, implementations=entries)
    if balanced_rows:
        from .balanced_rows import configure as configure_balanced_rows
        configure_balanced_rows(target, gdn_linear_owners=gdn_linear_owners)
    elif gdn_linear_owners:
        raise ValueError('linear GDN ownership requires --balanced-rows')
    if tile_n != 16 or warps != 4:
        model = target.triton_implementation_model
        entries = tuple(
            replace(value, parameters={**value.parameters, 'tile_n': tile_n, 'compute_num_warps': warps}) if value.id in
            ('tir.block_fp8.simt', 'tir.matmul_glu.simt') else value for value in model.implementations)
        target.triton_implementation_model = replace(model, implementations=entries)
