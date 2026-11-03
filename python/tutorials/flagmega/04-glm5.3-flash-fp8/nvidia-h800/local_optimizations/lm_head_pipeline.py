"""Keep the full BF16 vocabulary while grouping original K128 updates."""
import hashlib
from pathlib import Path

from triton.flagmega.errors import CodegenError

CALL = 'logits.bf16.vectorized.compute'
SCHEDULE = 'complete_vocab_K128_unroll2_four_warps'


def select_definition(module, definitions):
    from dataclasses import replace
    node = module.node_map[CALL]
    callee = node.attrs.get('callee')
    if node.op != 'tir.call' or not callee:
        raise CodegenError('Full logits must retain its typed kernel call')
    uses = [n.id for n in module.nodes if n.op == 'tir.call' and n.attrs.get('callee') == callee]
    if uses != [CALL]:
        raise CodegenError('Full logits kernel definition must not be shared with another call')
    matches = [d for d in definitions if d.name == callee]
    if len(matches) != 1:
        raise CodegenError('Exactly one full logits kernel definition required')
    definition = matches[0]
    selection = definition.dispatch.microkernel
    if (selection.family, selection.variant) != ('dense_matmul', 'packed_k_major_mma') or (
            selection.parameters.get('compute_num_warps'), selection.parameters.get('block_k'),
            selection.parameters.get('tile_n')) != (8,128,64):
        raise CodegenError('Full logits must retain its original K128/N64 kernel selection')
    revised = replace(definition, dispatch=replace(definition.dispatch,
        microkernel=replace(selection, parameters={**selection.parameters, 'compute_num_warps':4})))
    return [revised if d is definition else d for d in definitions], callee


def select_programs(requests):
    selected, result = [], []
    for request in requests:
        if CALL not in request['calls']:
            result.append(request)
            continue
        if tuple(request['calls']) != (CALL,) or request['num_warps'] != 8:
            raise CodegenError('Full logits requires its separate original eight-warp invocation')
        selected.append(request['symbol'])
        result.append({**request, 'num_warps': 4})
    if len(selected) != 1:
        raise CodegenError('Exactly one full logits invocation required')
    return tuple(result), selected


def transform(source):
    layout = 'tle.gpu.BlockEncoding([1,8],[8,4],[4,2],[1,0])'
    loop = 'for dense_local_k_start in tl.range(0, {{ call.local_k_capacity }}, {{ call.block_k }}):'
    if source.count(layout) != 1 or source.count(loop) != 1:
        raise CodegenError('Full packed logits memory or K-loop boundary changed')
    source = source.replace(layout,
        'tle.gpu.BlockEncoding([1,8],[8,4],[4,{% if call.logits_four_warp_pipeline | default(false) %}1{% else %}2{% endif %}],[1,0])')
    return source.replace(loop,
        'for dense_local_k_start in tl.range(0, {{ call.local_k_capacity }}, {{ call.block_k }}{% if call.logits_four_warp_pipeline | default(false) %}, loop_unroll_factor=2{% endif %}):')


def describe(module, previous):
    descriptor = previous(module)
    calls = [c for c in descriptor['render_calls'] if c['call'] == CALL]
    if len(calls) != 1:
        raise CodegenError('Exactly one full logits definition required')
    call = calls[0]
    if (call['family'] != 'dense_matmul' or call['variant'] != 'packed_k_major_mma'
            or (call['local_m_capacity'], call['local_n_capacity'], call['local_k_capacity'],
                call['tile_n'], call['block_k']) != (1, 304, 4096, 64, 128)
            or call['signature'] != 'workspace, rdata, device_dptr'
            or not call.get('logits_coalesced_load')):
        raise CodegenError('Full logits arithmetic and memory contract changed')
    call['logits_four_warp_pipeline'] = True
    descriptor['full_logits_pipeline'] = dict(schedule=SCHEDULE, num_warps=4,
        logical_vocab=154880, local_valid_outputs=38720, physical_slots=38912,
        K=4096, block_k=128, loop_unroll_factor=2, original_native_K16_MMA_order=True,
        all_masks_addresses_and_rounding_unchanged=True)
    descriptor['implementation_override']['inputs']['lm_head_pipeline.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return descriptor


def activate():
    from dataclasses import replace
    from triton.flagmega.codegen.triton import renderer, tir_package
    previous = tir_package.describe_tir_package
    def selected(module):
        return describe(module, previous)
    tir_package.describe_tir_package = selected
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = replace(spec, describe=selected)
