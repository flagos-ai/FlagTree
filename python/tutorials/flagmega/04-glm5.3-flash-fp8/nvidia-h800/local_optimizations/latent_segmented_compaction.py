"""Bound the stable scan to 2051 slots without assumptions about valid order."""
from triton.flagmega.errors import CodegenError

SCHEDULE = 'stable_prefix2048_group4_tail3_v1'


def transform(source):
    begin = '        _fm_dsa_original_slot = tl.arange(0, {{ call.compact_topk_capacity }})\n'
    end = '        tl.debug_barrier()\n'
    if source.count(begin) != 1 or source.count(end) != 1:
        raise CodegenError('Stable compact scan boundaries changed')
    start = source.index(begin)
    stop = source.index(end, start) + len(end)
    old = source[start:stop]
    if old.count('tl.cumsum(') != 1 or old.count('tl.store(') != 1:
        raise CodegenError('Original stable compact scan changed')
    block = '''        _fm_dsa_original_slot = tl.arange(0, 2048)
        _fm_dsa_original_token = tl.load({{ call.topk }} + _fm_dsa_original_slot * {{ call.topk_stride }}).to(tl.int32)
        _fm_dsa_original_valid = (_fm_dsa_original_token >= 0) & (_fm_dsa_original_token <= _fm_dsa_position)
        _fm_dsa_group_valid = _fm_dsa_original_valid.reshape((512, 4)).to(tl.int32)
        _fm_dsa_lane_rank = tl.cumsum(_fm_dsa_group_valid, 1) - 1
        _fm_dsa_group_count = tl.sum(_fm_dsa_group_valid, 1)
        _fm_dsa_group_end = tl.cumsum(_fm_dsa_group_count, 0)
        _fm_dsa_prefix_rank = (_fm_dsa_group_end[:, None] - _fm_dsa_group_count[:, None] + _fm_dsa_lane_rank).reshape((2048,))
        _fm_dsa_prefix_count = tl.sum(_fm_dsa_group_count, 0)
        _fm_dsa_prefix_write = tle.gpu.local_ptr(_fm_dsa_compact, (tl.maximum(_fm_dsa_prefix_rank, 0),), [2048])
        tl.store(_fm_dsa_prefix_write, _fm_dsa_original_token, _fm_dsa_original_valid)
        _fm_dsa_tail_lane = tl.arange(0, 4)
        _fm_dsa_tail_token = tl.load({{ call.topk }} + (2048 + _fm_dsa_tail_lane) * {{ call.topk_stride }}, _fm_dsa_tail_lane < 3, other=-1).to(tl.int32)
        _fm_dsa_tail_valid = (_fm_dsa_tail_lane < 3) & (_fm_dsa_tail_token >= 0) & (_fm_dsa_tail_token <= _fm_dsa_position)
        _fm_dsa_tail_rank = _fm_dsa_prefix_count + tl.cumsum(_fm_dsa_tail_valid.to(tl.int32), 0) - 1
        _fm_dsa_tail_write = tle.gpu.local_ptr(_fm_dsa_compact, (tl.maximum(_fm_dsa_tail_rank, 0),), [4])
        tl.store(_fm_dsa_tail_write, _fm_dsa_tail_token, _fm_dsa_tail_valid)
        _fm_dsa_count = _fm_dsa_prefix_count + tl.sum(_fm_dsa_tail_valid.to(tl.int32), 0)
        tl.debug_barrier()
'''
    return source[:start] + block + source[stop:]


def validate(call):
    if (call.get('parameters', {}).get('latent_compaction_schedule') != SCHEDULE
            or call.get('topk_width') != 2051 or call.get('compact_topk_capacity') != 4096
            or call.get('topk_stride') != 1 or call.get('index_traversal') != 'compact'):
        raise CodegenError('Segmented compaction requires its complete 2051-slot ABI')
