"""Assign complete DSA pool scores to proven replicas of one chip owner."""
from dataclasses import replace
import hashlib
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

from .replica_work import replica_group

KEY = 'kernels/dsa_indexer/_function.py.jinja'
SCHEDULE = 'parallel_complete_pools'
_active = False


def implementation_model(model):
    """Declare the score-to-selection grid phase before physical planning."""
    selected = [v for v in model.implementations if v.family == 'dsa_indexer' and v.variant == 'decode']
    if not selected or any(v.facts.get('internal_grid_barriers') != 2 for v in selected):
        raise CodegenError('Parallel pools require the existing synchronized write/select DSA implementation')
    return replace(model, implementations=tuple(
        replace(value, parameters={**value.parameters, 'pool_score_schedule': SCHEDULE},
                facts={**value.facts, 'internal_grid_barriers': 3})
        if value in selected else value for value in model.implementations))


def _template(source):
    begin = source.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__select(')
    end = source.index('\n\n\n@triton.jit(noinline=True)\ndef {{ call.symbol }}__advance', begin)
    original = source[begin:end]
    scoring = original.index('    for pool_index in tl.range(0, pool_count):')
    output = original.index('    output_offsets =', scoring)
    loop = original[scoring:output]
    last = '        pool_scores = tl.where(pool_ids == pool_index, tl.where(complete, score, -float("inf")), pool_scores)\n'
    if loop.count(last) != 1:
        raise CodegenError('Complete DSA pool score expression changed')
    score = original[:scoring].replace('__select(', '__pool_scores(')
    score = score.replace('    pool_ids = tl.arange(0, {{ call.pool_tile }})\n', '')
    score = score.replace('    pool_scores = tl.full(({{ call.pool_tile }},), -float("inf"), tl.float32)\n', '')
    body = loop[loop.index('\n') + 1:].replace(last,
        '        tl.store({{ call.result }}.to(tl.pointer_type(tl.float32)) + pool_index, '
        'tl.where(complete, score, -float("inf")))\n')
    score += ('    for pool_index in tl.range(({{ call.pool_score_group.index }}), '
              '{{ call.pool_tile }}, {{ call.pool_score_group.count }}):\n'
              '        if pool_index < pool_count:\n'
              + ''.join('    ' + line if line.strip() else line for line in body.splitlines(keepends=True))
              + '        else:\n'
              '            tl.store({{ call.result }}.to(tl.pointer_type(tl.float32)) + pool_index, -float("inf"))\n')
    select = original[:scoring]
    prefix_end = select.index('    pool_offsets =')
    select = select[:prefix_end] + (
        '    pool_offsets = tl.arange(0, {{ call.pool }})\n'
        '    pool_scores = tl.load({{ call.result }}.to(tl.pointer_type(tl.float32)) + pool_ids)\n') + original[output:]
    selection = select.index('    for selection in tl.range(')
    select = select[:selection] + '''    # Float order and low pool ID preserve the original deterministic ties.
    # The original maxnum reduction ignores NaN pools while selecting numbers.
    # Map those pools to the existing invalid sentinel before forming sort keys.
    pool_scores = tl.where(pool_scores == pool_scores, pool_scores, -float("inf"))
    score_bits = tl.where(pool_scores == 0., 0., pool_scores).to(tl.uint32, bitcast=True)
    score_order = tl.where((score_bits & 0x80000000) != 0, ~score_bits, score_bits ^ 0x80000000)
    sort_keys = (score_order.to(tl.uint64) << 32) | ({{ call.pool_tile - 1 }} - pool_ids).to(tl.uint64)
    sorted_keys = tl.sort(sort_keys, descending=True)
    ordered_pool = {{ call.pool_tile - 1 }} - (sorted_keys & 0xffffffff).to(tl.int32)
    valid_score = ((sorted_keys >> 32).to(tl.uint32) > 0x007fffff)
    selected_value = first + ordered_pool[:, None] * {{ call.pool }} + pool_offsets[None, :]
    output_position = pool_ids[:, None] * {{ call.pool }} + pool_offsets[None, :]
    tl.store({{ call.result }} + output_position, tl.where(valid_score[:, None], selected_value, -1),
             mask=output_position < {{ call.index_topk }})
'''
    source = source[:begin] + '{% if call.pool_score_group | default(none) %}\n' + score + '\n\n' + select + '\n{% else %}\n' + original + '\n{% endif %}' + source[end:]
    old = '    {{ call.symbol }}__select({{ call.signature }})\n'
    if source.count(old) != 1:
        raise CodegenError('DSA write/select/advance wrapper changed')
    source = source.replace(old, '''{% if call.pool_score_group | default(none) %}
{{ entry.mesh_coordinates() }}
    {{ call.symbol }}__pool_scores({{ call.signature }})
{{ entry.stage_barrier() }}
    if ({{ call.pool_score_group.index }}) == 0:
        {{ call.symbol }}__select({{ call.signature }})
{% else %}
''' + old + '{% endif %}\n')
    return source


def activate():
    """Render only the explicit typed parallel schedule under the native profile."""
    global _active
    if _active:
        return
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous = tir_package.describe_tir_package

    def describe(module):
        result = previous(module)
        selected = []
        for call in result['render_calls']:
            if call['family'] != 'dsa_indexer' or call['parameters'].get('pool_score_schedule') != SCHEDULE:
                continue
            if (not str(module.metadata.get('numerical_contract', '')).startswith('glm5-vllm-0.31-tp4/')
                    or call['variant'] != 'decode' or call['internal_grid_barriers'] != 3
                    or call['pool_tile'] > call['output_width'] or call['value_type'] != 'tl.bfloat16'):
                raise CodegenError('Parallel DSA pools require the explicit native contract and complete output scratch')
            group = replica_group(call, result, expressions=tuple(call[k] for k in
                ('q', 'weights', 'state', 'sequence_lengths', 'compress_ape', 'result', 'layer_id')))
            if group is None:
                raise CodegenError('Parallel DSA pools require proven complete chip-owner replicas')
            call['pool_score_group'] = group
            selected.append(dict(call=call['call'], group=group, score_capacity=call['pool_tile']))
        if selected:
            result['native_dsa_pool_score_schedule'] = dict(schedule=SCHEDULE, selected=selected,
                score_math='complete original pool reductions; independent pools assigned to replicas',
                scratch='output prefix, float32 scores before destructive int32 selection',
                grid_phases=3, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result

    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = replace(spec, describe=describe)
    previous_render = TritonTemplateRegistry.render

    def render(self, name, context):
        if not any(c.get('pool_score_group') for c in context.get('render_calls', ())):
            return previous_render(self, name, context)
        loader = self.environment.loader
        source = loader.get_source(self.environment, KEY)[0]
        self.environment.loader = ChoiceLoader([DictLoader({KEY: _template(source)}), loader])
        try:
            return previous_render(self, name, context)
        finally:
            self.environment.loader = loader
    TritonTemplateRegistry.render = render
    _active = True
