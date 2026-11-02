"""Publish integer visibility once before the existing score grid phase."""
from triton.flagmega.errors import CodegenError

SCHEDULE = 'single_writer_first_and_count_tail_scratch_v1'

SCAN = '''    positions = tl.arange(0, {{ call.state_tile }})
    flags = tl.load(
        {{ call.state }} + layer * {{ call.state_layer_stride }}
        + positions * {{ call.state_token_stride }} + 2 * {{ call.index_dim }} * {{ call.state_value_stride }},
        mask=(positions < {{ call.state_capacity }}) & (positions <= base), other=0,
    ) != 0
    first = tl.min(tl.where(flags, positions, {{ call.state_capacity }}), axis=0)
    visible_count = tl.sum(flags.to(tl.int32), axis=0)
'''
WRITE = '        tl.store({{ call.state }} + state_base + 2 * {{ call.index_dim }} * {{ call.state_value_stride }}, 1.0)\n'
LOAD = '''    first = tl.load({{ call.result }} + {{ call.index_topk }})
    visible_count = tl.load({{ call.result }} + {{ call.index_topk + 1 }})
'''


def helpers(source):
    if source.count(WRITE) != 1 or source.count(SCAN) < 3:
        raise CodegenError('Visibility publication requires the original write and complete score/select scans')
    candidate = SCAN.replace('(positions <= base)', '(positions <= base) & (positions != base)')
    candidate = candidate.replace('    first =', '    flags = tl.where(positions == base, True, flags)\n    first =', 1)
    candidate += '''    tl.store({{ call.result }} + {{ call.index_topk }}, first)
    tl.store({{ call.result }} + {{ call.index_topk + 1 }}, visible_count)
'''
    # First change every existing scan, then append the new sole-writer scan.
    source = source.replace(SCAN, '{% if call.visibility_publish | default(false) %}\n'+LOAD+'{% else %}\n'+SCAN+'{% endif %}\n')
    return source.replace(WRITE, WRITE+'{% if call.visibility_publish | default(false) %}\n'+candidate+'{% endif %}\n')


def configure(call):
    value=call['parameters'].get('visibility_publish_schedule')
    if value is None:
        return
    if (value != SCHEDULE or call['internal_grid_barriers'] != 3
            or call['pool_tile'] >= call['index_topk']
            or call['output_width'] < call['index_topk']+2
            or call['value_type'] != 'tl.bfloat16'):
        raise CodegenError('Visibility metadata needs its typed disjoint score/tail lifetime contract')
    call['visibility_publish'] = True
