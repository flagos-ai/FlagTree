"""Rank complete pools before one barrier permits destructive publication.

The original 1024 FP32 scores occupy result slots 0..1023. Each CTA owns
8 pool IDs and captures its complete candidate population before the barrier.
For 513..528 pools, the population is the 512 prefix plus all 16 tail IDs.
The original score phase fills all higher IDs with -inf; low-ID ties ensure
those padded keys cannot enter the top 512. The total key
(score order, inverse pool ID) makes ranks unique, so each top-512 pool owns
four disjoint output slots. For 257..511 pools, the full 512-key population
includes the original invalid padding; low-ID ties put every omitted key
after those 512 keys. Invalid ranks publish -1. CTA0 owns the three
trailing positions. No workspace or score read survives publication.
One warp computes each original pool owner's comparison count. This avoids
duplicating the eight owners across warps while preserving the complete keys.
"""
import hashlib
from pathlib import Path
SCHEDULE = 'rank_complete_pool_sort_v3'
SHORT_SORT_DISPATCH = """    if pool_count <= 16:
        {{ call.symbol }}__sort_tile({{ call.result }}, first, visible_count, base, 16)
    elif pool_count <= 64:
        {{ call.symbol }}__sort_tile({{ call.result }}, first, visible_count, base, 64)
    elif pool_count <= 256:
        {{ call.symbol }}__sort_tile({{ call.result }}, first, visible_count, base, 256)
    else:
        {{ call.symbol }}__sort_tile({{ call.result }}, first, visible_count, base, 512)
"""
HELPER = """@triton.jit(noinline=True)
def {{ call.symbol }}__parallel_select(SortResult, First, VisibleCount, Base, device_dptr, Tile: tl.constexpr):
    pid = tl.program_id(0)
    own_ids = pid * 8 + tl.arange(0, 8)
    all_ids = tl.arange(0, 512 if Tile == 528 else Tile)
    all_scores = tl.load(SortResult.to(tl.pointer_type(tl.float32)) + all_ids)
    own_scores = tl.load(SortResult.to(tl.pointer_type(tl.float32)) + own_ids, own_ids < Tile, other=-float('inf'))
    all_scores = tl.where(all_scores == all_scores, all_scores, -float('inf'))
    own_scores = tl.where(own_scores == own_scores, own_scores, -float('inf'))
    all_bits = tl.where(all_scores == 0.0, 0.0, all_scores).to(tl.uint32, bitcast=True)
    own_bits = tl.where(own_scores == 0.0, 0.0, own_scores).to(tl.uint32, bitcast=True)
    all_order = tl.where(all_bits & 2147483648 != 0, ~all_bits, all_bits ^ 2147483648)
    own_order = tl.where(own_bits & 2147483648 != 0, ~own_bits, own_bits ^ 2147483648)
    all_keys = all_order.to(tl.uint64) << 32 | (1023 - all_ids).to(tl.uint64)
    own_keys = own_order.to(tl.uint64) << 32 | (1023 - own_ids).to(tl.uint64)
    rank = tl.sum(tle.gpu.set_layout((all_keys[None, :] > own_keys[:, None]).to(tl.int32), tle.gpu.BlockEncoding([1, 1], [1, 32], [8, 1], [1, 0])), axis=1)
    if Tile == 528:
        tail_ids = 512 + tl.arange(0, 16)
        tail_scores = tl.load(SortResult.to(tl.pointer_type(tl.float32)) + tail_ids)
        tail_scores = tl.where(tail_scores == tail_scores, tail_scores, -float('inf'))
        tail_bits = tl.where(tail_scores == 0.0, 0.0, tail_scores).to(tl.uint32, bitcast=True)
        tail_order = tl.where(tail_bits & 2147483648 != 0, ~tail_bits, tail_bits ^ 2147483648)
        tail_keys = tail_order.to(tl.uint64) << 32 | (1023 - tail_ids).to(tl.uint64)
        rank += tl.sum(tle.gpu.set_layout((tail_keys[None, :] > own_keys[:, None]).to(tl.int32), tle.gpu.BlockEncoding([1, 1], [1, 32], [8, 1], [1, 0])), axis=1)
{{ entry.stage_barrier() }}
    positions = rank[:, None] * 4 + tl.arange(0, 4)[None, :]
    values = tl.where((own_order > 8388607)[:, None], First + own_ids[:, None] * 4 + tl.arange(0, 4)[None, :], -1)
    tl.store(SortResult + positions, values, (own_ids < Tile)[:, None] & (rank < 512)[:, None])
    if pid == 0:
        tail = tl.arange(0, 4)
        tail_count = VisibleCount % 4
        tail_value = First + VisibleCount - tail_count + tail
        tail_valid = (tail < tail_count) & (tail_value < Base + 1)
        tl.store(SortResult + 2048 + tail, tl.where(tail_valid, tail_value, -1), tail < 3)
"""

def helpers(source):
    marker = '@triton.jit\ndef {{ call.symbol }}({{ call.signature }}):'
    if source.count(marker) != 1:
        raise ValueError('Complete DSA wrapper changed')
    return source.replace(marker, '{% if call.parallel_dsa_sort | default(false) %}\n'+HELPER+'{% endif %}\n'+marker)

def selection():
    return '{% if call.parallel_dsa_sort | default(false) %}\n    _sort_base = tl.load({{ call.sequence_lengths }}).to(tl.int32)\n    _sort_first = tl.load({{ call.result }} + {{ call.index_topk }})\n    _sort_visible = tl.load({{ call.result }} + {{ call.index_topk + 1 }})\n    _sort_count = tl.maximum((_sort_base + 1 - _sort_first + 3) // 4, 0)\n    if _sort_count >= 257:\n        if _sort_count <= 512:\n            {{ call.symbol }}__parallel_select({{ call.result }}, _sort_first, _sort_visible, _sort_base, device_dptr, 512)\n        elif _sort_count <= 528:\n            {{ call.symbol }}__parallel_select({{ call.result }}, _sort_first, _sort_visible, _sort_base, device_dptr, 528)\n        else:\n            {{ call.symbol }}__parallel_select({{ call.result }}, _sort_first, _sort_visible, _sort_base, device_dptr, 1024)\n    else:\n        if tl.program_id(0) == 0:\n            {{ call.symbol }}__select({{ call.signature }})\n{% else %}\n    if tl.program_id(0) == 0:\n        {{ call.symbol }}__select({{ call.signature }})\n{% endif %}\n'

def source_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
