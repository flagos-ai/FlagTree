"""Parallel probability production with the original ordered BF16 PV recurrence."""
from dataclasses import replace
import hashlib
from pathlib import Path
import re

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _static_shape
from triton.flagmega.errors import CodegenError
from .dsa_work import head_group
from .native_latent_attention_schedule import _template, KEY
from .native_latent_probability_model import SCHEDULE, SCORE_SHAPE, METADATA_SHAPE, METADATA_CHIP_STRIDE

_active = False


def shifted(source, delta):
    lines = []
    for line in source.splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith(('{%', '{#')):
            if delta < 0:
                if not line.startswith(' ' * (-delta)):
                    raise CodegenError('Probability template indentation changed')
                line = line[-delta:]
            else:
                line = ' ' * delta + line
        lines.append(line)
    return ''.join(lines)


def expression(source, name, *, contains=None):
    values = [line.strip().split(' = ', 1)[1] for line in source.splitlines()
              if line.strip().startswith(name + ' = ') and (contains is None or contains in line)]
    if len(values) != 1:
        raise CodegenError('Original online probability expression changed: ' + name)
    return values[0]


def replace_expression(source, name, value, *, contains=None):
    lines = []
    count = 0
    for line in source.splitlines(keepends=True):
        if line.strip().startswith(name + ' = ') and (contains is None or contains in line):
            count += 1
            line = line[:len(line) - len(line.lstrip())] + name + ' = ' + value + '\n'
        lines.append(line)
    if count != 1:
        raise CodegenError('Original ordered consumer expression changed: ' + name)
    return ''.join(lines)


def probability_template(source):
    source = _template(source, compact_pipeline=True).replace('call.native_latent_schedule', 'call.probability_schedule')
    guard = '    if {{ call.writer_active }}:\n'
    if source.count(guard) != 1:
        raise CodegenError('Probability template requires one original complete owner')
    header, body = source.split(guard)
    anchor = '        _fm_dsa_tokens = tl.arange(0, 64)\n'
    if body.count(anchor) != 1:
        raise CodegenError('Original compact MLA math changed')
    compact, math = body.split(anchor)
    math = anchor + math
    tile_start = '                _fm_dsa_slot = _fm_dsa_start + _fm_dsa_tokens\n'
    score_start = '                _fm_dsa_score = tle.gpu.set_layout('
    online_start = '                _fm_dsa_updated_max = tl.maximum('
    if any(math.count(v) != 1 for v in (tile_start, score_start, online_start)):
        raise CodegenError('Probability producer/consumer arithmetic boundary changed')
    lo, score, hi = (math.index(v) for v in (tile_start, score_start, online_start))
    producer = math[lo:hi].replace('_fm_dsa_start', '_fm_dsa_producer_start')
    score_pointer = '{{ call.probability_schedule.score_pointer }}'
    meta_pointer = '{{ call.probability_schedule.metadata_pointer }}'
    row = '{{ call.symbol }}__row_layout'
    mma = '{{ call.symbol }}__mma_layout'
    producer_score_load = ('tle.gpu.set_layout(tl.load(' + score_pointer +
                  ' + _fm_dsa_start // 64 * 1024 + _fm_dsa_head[:, None] * 64 + _fm_dsa_tokens[None, :]), ' + mma + ')')
    p_load_layout = '{{ call.symbol }}__p_load_layout'
    score_load = ('tle.gpu.set_layout(tl.load(tle.gpu.set_layout(' + score_pointer +
                  ' + _fm_dsa_start // 64 * 1024 + _fm_dsa_head[:, None] * 64 + _fm_dsa_tokens[None, :], ' + p_load_layout + ')), ' + p_load_layout + ')')
    max_load = ('tle.gpu.set_layout(tl.load(' + meta_pointer +
                ' + _fm_dsa_start // 64 * 16 + _fm_dsa_head), ' + row + ')')
    previous_load = ('tle.gpu.set_layout(tl.load(' + meta_pointer +
                     ' + 1024 + _fm_dsa_start // 64 * 32 + _fm_dsa_head * 2), ' + row + ')')
    sum_load = ('tle.gpu.set_layout(tl.load(' + meta_pointer +
                ' + 1024 + _fm_dsa_start // 64 * 32 + _fm_dsa_head * 2 + 1), ' + row + ')')
    consumer = math[:score] + '                _fm_dsa_score = ' + score_load + '\n' + math[hi:]
    consumer = replace_expression(consumer, '_fm_dsa_updated_max', max_load)
    consumer = replace_expression(consumer, '_fm_dsa_previous', previous_load)
    consumer = replace_expression(consumer, '_fm_dsa_p', '_fm_dsa_score')
    sum_expression = expression(math, '_fm_dsa_sum', contains='tl.sum(_fm_dsa_p, 1)')
    if sum_expression.count('tl.sum(_fm_dsa_p, 1)') != 1:
        raise CodegenError('Original online probability sum changed')
    consumer = replace_expression(consumer, '_fm_dsa_sum', sum_expression.replace('tl.sum(_fm_dsa_p, 1)', sum_load),
                                  contains='tl.sum(_fm_dsa_p, 1)')
    if consumer.count('call.probability_schedule.column_tile') != 3:
        raise CodegenError('Long PV column geometry changed')
    consumer = consumer.replace('call.probability_schedule.column_tile', 'call.probability_schedule.long_column_tile')
    pv_load_layout = '{{ call.symbol }}__pv_load_layout'
    original_v = expression(consumer, '_fm_dsa_v')
    if not original_v.startswith('tl.load(') or original_v.count(', _fm_dsa_pv_valid') != 1:
        raise CodegenError('Ordered long PV V load boundary changed')
    pointer, suffix = original_v[len('tl.load('):].split(', _fm_dsa_pv_valid', 1)
    v_load = ('tle.gpu.set_layout(tl.load(tle.gpu.set_layout(' + pointer + ', ' + pv_load_layout +
              '), _fm_dsa_pv_valid' + suffix + ', ' + pv_load_layout + ')')
    consumer = replace_expression(consumer, '_fm_dsa_v', v_load)
    pv_address_layout = '{{ call.symbol }}__pv_address_layout'
    for target, old_layout in (('_fm_dsa_pv_token', '{{ call.symbol }}__column_layout'),
                               ('_fm_dsa_pv_base', row), ('_fm_dsa_pv_valid', row)):
        ordered_expression = expression(consumer, target)
        suffix = ', ' + old_layout + ')'
        if not ordered_expression.startswith('tle.gpu.set_layout(') or not ordered_expression.endswith(suffix):
            raise CodegenError('Ordered PV gather layout boundary changed: ' + target)
        consumer = replace_expression(consumer, target,
                                      ordered_expression[:-len(suffix)] + ', ' + pv_address_layout + ')')
    if consumer.count('num_stages=2') != 1:
        raise CodegenError('Ordered long PV outer pipeline changed')
    consumer = consumer.replace('num_stages=2', 'num_stages=5')
    marker = '@triton.jit(noinline=True)'
    if header.count(marker) != 1:
        raise CodegenError('Probability template requires one original callable')
    declarations = '''{{ call.symbol }}__pv_load_layout = tl.constexpr(tle.gpu.BlockEncoding([1, 8], [8, 4], [8, 1], [1, 0]))
{{ call.symbol }}__p_load_layout = tl.constexpr(tle.gpu.BlockEncoding([1, 4], [4, 8], [4, 2], [1, 0]))
{{ call.symbol }}__pv_address_layout = tl.constexpr(tle.gpu.SlicedEncoding(1, {{ call.symbol }}__pv_load_layout))
{{ call.symbol }}__score_mesh = tl.constexpr(tle.device_mesh(dict(block=[('block_y', 8), ('block_x', 16)])))
@triton.jit
def {{ call.symbol }}__prefix_maximum(a, b):
    return tl.maximum(a, b)

'''
    header = header.replace(marker, declarations + marker)
    parallel = '''    if _fm_dsa_count >= {{ call.probability_schedule.threshold }}:
        _fm_dsa_tokens = tl.arange(0, 64)
        _fm_dsa_qk_k = tl.arange(0, 256)
        _fm_dsa_head = tl.arange(0, 16)
        _fm_dsa_producer_start = ({{ call.probability_schedule.group.index }}) * 64
        if _fm_dsa_producer_start < _fm_dsa_count:
''' + shifted(producer, -4) + '''            tl.store(''' + score_pointer + ''' + _fm_dsa_producer_start // 64 * 1024 + _fm_dsa_head[:, None] * 64 + _fm_dsa_tokens[None, :], _fm_dsa_score)
            tl.store(''' + meta_pointer + ''' + _fm_dsa_producer_start // 64 * 16 + _fm_dsa_head, tl.max(_fm_dsa_score, 1))
        tle.distributed_barrier({{ call.symbol }}__score_mesh)
        if tl.program_id(0) == 0:
            _fm_prefix_tiles = tl.arange(0, 64)
            _fm_prefix_heads = tl.arange(0, 16)
            _fm_tile_maxima = tl.load(''' + meta_pointer + ''' + _fm_prefix_tiles[:, None] * 16 + _fm_prefix_heads[None, :], mask=_fm_prefix_tiles[:, None] * 64 < _fm_dsa_count, other=-float('inf'))
            _fm_prefix_maxima = tl.associative_scan(_fm_tile_maxima, 0, {{ call.symbol }}__prefix_maximum)
            tl.store(''' + meta_pointer + ''' + _fm_prefix_tiles[:, None] * 16 + _fm_prefix_heads[None, :], _fm_prefix_maxima)
        tle.distributed_barrier({{ call.symbol }}__score_mesh)
        if _fm_dsa_producer_start < _fm_dsa_count:
            _fm_dsa_start = _fm_dsa_producer_start
            _fm_dsa_score = ''' + producer_score_load + '''
            _fm_dsa_updated_max = ''' + max_load + '''
            _fm_dsa_max = tle.gpu.set_layout(tl.load(''' + meta_pointer + ''' + (_fm_dsa_start // 64 - 1) * 16 + _fm_dsa_head, mask=_fm_dsa_start > 0, other=-float('inf')), ''' + row + ''')
            _fm_dsa_previous = ''' + expression(math, '_fm_dsa_previous') + '''
            _fm_dsa_p = ''' + expression(math, '_fm_dsa_p') + '''
            _fm_dsa_rounded_p = ''' + expression(math, '_fm_dsa_rounded_p') + '''
            tl.store(''' + score_pointer + ''' + _fm_dsa_start // 64 * 1024 + _fm_dsa_head[:, None] * 64 + _fm_dsa_tokens[None, :], _fm_dsa_rounded_p.to(tl.float32))
            tl.store(''' + meta_pointer + ''' + 1024 + _fm_dsa_start // 64 * 32 + _fm_dsa_head * 2, _fm_dsa_previous)
            tl.store(''' + meta_pointer + ''' + 1024 + _fm_dsa_start // 64 * 32 + _fm_dsa_head * 2 + 1, tl.sum(_fm_dsa_p, 1))
        tle.distributed_barrier({{ call.symbol }}__score_mesh)
        if ({{ call.probability_schedule.group.index }}) < 16:
''' + shifted(consumer, 4) + '''    else:
        if ({{ call.probability_schedule.group.index }}) < 8:
''' + shifted(math, 4)
    return header + shifted(compact, -4) + parallel


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
            if call['parameters'].get('latent_work_schedule') != SCHEDULE:
                continue
            p = call['parameters']
            if (call['family'] != 'dsa_paged_attention' or call['variant'] != 'decode'
                    or call['decode_schedule'] != 'bf16_mma' or not call['head_sharded']
                    or call['num_query_heads'] != 16 or call['head_dim'] != 512
                    or call['probability_rounding'] != 'online_bf16' or call['index_traversal'] != 'compact'
                    or p['compute_num_warps'] != 8 or call['internal_grid_barriers'] != 3
                    or p['latent_long_column_tile'] != 32
                    or tuple(descriptor['mesh_hierarchy']) != (4, 8, 16)
                    or ''.join(descriptor['mesh_levels']) != 'cbb'
                    or tuple(descriptor['grid']) != (128, 1, 1)
                    or not str(module.metadata.get('numerical_contract', '')).startswith('glm5-vllm-0.31-tp4/')):
                raise CodegenError('Probability scheduling requires its complete chip/head/rounding/sync contract')
            group = head_group(call, descriptor)
            if group is None or group['count'] != 128 or group['axes'] != [1, 2]:
                raise CodegenError('Probability producer needs all proven chip replicas')
            score = _buffer(call, 'workspaces', 'latent_qk_scores')
            meta = _buffer(call, 'workspaces', 'latent_probability_metadata')
            pointers = []
            for actual, shape, strides in ((score, SCORE_SHAPE, (33792, 1024, 64, 1)),
                                            (meta, METADATA_SHAPE, (METADATA_CHIP_STRIDE, 16, 1))):
                abi = actual['abi']
                if (_static_shape(abi, 'logical_shape') != shape or abi['scalar_dtype'] != 'float32'
                        or tuple(abi['scalar_storage_strides']) != strides
                        or abi.get('memory_sharing_scope') != 'chip'):
                    raise CodegenError('Probability workspaces require their typed chip-shared FP32 geometry')
                pointer = _pointer(actual)
                if re.search(r'block_index|program_id|shard_coord[12]|block_scope', pointer):
                    raise CodegenError('Probability workspace bases must be invariant among chip replicas')
                pointers.append(pointer)
            schedule = dict(group=group, column_tile=64, long_column_tile=32,
                threshold=p['latent_parallel_score_threshold'],
                score_pointer=f'({pointers[0]} + shard_coord0 * 33792)',
                metadata_pointer=f'({pointers[1]} + shard_coord0 * {METADATA_CHIP_STRIDE})',
                normal_score_workspace_binding=score, normal_metadata_workspace_binding=meta,
                internal_grid_barriers=3, metadata_payload_bytes=8320,
                metadata_allocated_bytes_per_chip=METADATA_CHIP_STRIDE * 4,
                original_ordered_PV=True, original_short_math=True)
            call['probability_schedule'] = schedule
            call['writer_active'] = f"(({group['index']}) < 8)"
            selected.append(dict(call=call['call'], **schedule))
        if selected:
            if len(selected) != 1:
                raise CodegenError('Expected one shared complete probability implementation')
            descriptor['native_latent_parallel_probability'] = dict(schedule=SCHEDULE, selected=selected,
                normal_typed_workspaces=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return descriptor

    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = replace(spec, describe=describe)
    previous_prepare = TritonTemplateRegistry.prepare
    previous_render = TritonTemplateRegistry.render
    source = probability_template((Path('/root/repos/flagtree/python/triton/flagmega/codegen/triton') / KEY).read_text())

    def prepare(self, name, context):
        result = previous_prepare(self, name, context)
        if name == 'module.py.jinja' and any(call.get('probability_schedule') for call in context.get('render_calls', ())):
            if context['num_warps'] != 8:
                raise CodegenError('Probability whole program requires its selected eight warps')
            self.environment.loader = ChoiceLoader([DictLoader({KEY: source}), self.environment.loader])
        return result

    def render(self, name, context):
        if name != 'module.py.jinja' or not any(call.get('probability_schedule') for call in context.get('render_calls', ())):
            return previous_render(self, name, context)
        loader = self.environment.loader
        try:
            return previous_render(self, name, context)
        finally:
            self.environment.loader = loader

    TritonTemplateRegistry.prepare = prepare
    TritonTemplateRegistry.render = render
    _active = True
