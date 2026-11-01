MERGE_HELPER = "@triton.jit(noinline=True)\ndef {{ call.symbol }}__sort_merge(SortResult, First, VisibleCount, Base):\n    prefix_ids = tl.arange(0, 512)\n    tail_ids = 512 + tl.arange(0, 16)\n    pool_offsets = tl.arange(0, 4)\n    prefix_scores = tl.load(SortResult.to(tl.pointer_type(tl.float32)) + prefix_ids)\n    tail_scores = tl.load(SortResult.to(tl.pointer_type(tl.float32)) + tail_ids)\n    prefix_scores = tl.where(prefix_scores == prefix_scores, prefix_scores, -float('inf'))\n    tail_scores = tl.where(tail_scores == tail_scores, tail_scores, -float('inf'))\n    prefix_bits = tl.where(prefix_scores == 0.0, 0.0, prefix_scores).to(tl.uint32, bitcast=True)\n    tail_bits = tl.where(tail_scores == 0.0, 0.0, tail_scores).to(tl.uint32, bitcast=True)\n    prefix_order = tl.where(prefix_bits & 2147483648 != 0, ~prefix_bits, prefix_bits ^ 2147483648)\n    tail_order = tl.where(tail_bits & 2147483648 != 0, ~tail_bits, tail_bits ^ 2147483648)\n    prefix_keys = prefix_order.to(tl.uint64) << 32 | (1023 - prefix_ids).to(tl.uint64)\n    tail_keys = tail_order.to(tl.uint64) << 32 | (1023 - tail_ids).to(tl.uint64)\n    prefix_keys = tl.sort(prefix_keys, descending=True)\n    tail_greater = tail_keys[:, None] > prefix_keys[None, :]\n    prefix_rank = prefix_ids + tl.sum(tail_greater.to(tl.int32), axis=0)\n    tail_rank = 512 - tl.sum(tail_greater.to(tl.int32), axis=1) + tl.sum((tail_keys[None, :] > tail_keys[:, None]).to(tl.int32), axis=1)\n    output_offsets = tl.arange(0, 4096)\n    tail_count = VisibleCount % 4\n    tail_offset = output_offsets - 2048\n    tail_value = First + VisibleCount - tail_count + tail_offset\n    tail_valid = (tail_offset >= 0) & (tail_offset < tail_count) & (tail_value < Base + 1)\n    tl.store(SortResult + output_offsets, tl.where(tail_valid, tail_value, -1), output_offsets < 2051)\n    prefix_pool = 1023 - (prefix_keys & 4294967295).to(tl.int32)\n    prefix_valid = (prefix_keys >> 32).to(tl.uint32) > 8388607\n    tail_pool = 1023 - (tail_keys & 4294967295).to(tl.int32)\n    tail_valid_score = (tail_keys >> 32).to(tl.uint32) > 8388607\n    tl.store(SortResult + prefix_rank[:, None] * 4 + pool_offsets[None, :], tl.where(prefix_valid[:, None], First + prefix_pool[:, None] * 4 + pool_offsets[None, :], -1), prefix_rank[:, None] < 512)\n    tl.store(SortResult + tail_rank[:, None] * 4 + pool_offsets[None, :], tl.where(tail_valid_score[:, None], First + tail_pool[:, None] * 4 + pool_offsets[None, :], -1), tail_rank[:, None] < 512)\n"

"""Bound the complete-pool sort while preserving score and tie semantics."""

def template(parallel_source):
    anchor='@triton.jit(noinline=True)\ndef {{ call.symbol }}__select('
    start=parallel_source.index(anchor)
    end=parallel_source.index('\n{% else %}\n'+anchor,start)
    old=parallel_source[start:end]
    prefix=old[:old.index('    pool_ids =')]
    tail=old[old.index('    output_offsets ='):old.index('    # Float order')]
    sort=old[old.index('    # Float order'):]
    tail=tail.replace('{{ call.result }}','SortResult').replace('first','First').replace('visible_count','VisibleCount').replace('base','Base')
    sort=sort.replace('{{ call.result }}','SortResult').replace('first','First')
    helper='''@triton.jit(noinline=True)
def {{ call.symbol }}__sort_tile(SortResult, First, VisibleCount, Base, SortTile: tl.constexpr):
    pool_ids = tl.arange(0, SortTile)
    pool_offsets = tl.arange(0, {{ call.pool }})
    # Capture scores before the same scratch prefix becomes int selections.
    pool_scores = tl.load(SortResult.to(tl.pointer_type(tl.float32)) + pool_ids)
'''+tail+sort
    helper += '\n' + MERGE_HELPER + '\n'
    select=prefix+'''{% if call.pool_sort_tiles | length == 1 %}
    {{ call.symbol }}__sort_tile({{ call.result }}, first, visible_count, base, {{ call.pool_sort_tiles[0] }})
{% else %}
{% for tile in call.pool_sort_tiles %}
{% if loop.last %}
    elif pool_count <= 528:
        {{ call.symbol }}__sort_merge({{ call.result }}, first, visible_count, base)
    else:
{% else %}
    {{ 'if' if loop.first else 'elif' }} pool_count <= {{ tile }}:
{% endif %}
        {{ call.symbol }}__sort_tile({{ call.result }}, first, visible_count, base, {{ tile }})
{% endfor %}
{% endif %}
'''
    return parallel_source[:start]+'{% if call.pool_sort_tiles | default(none) %}\n'+helper+'\n\n'+select+'\n{% else %}\n'+old+'\n{% endif %}'+parallel_source[end:]
