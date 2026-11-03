"""Batch complete pools while preserving the original FP32 reduction trees.

The supported helper has 32 heads, 128 dimensions and eight warps. Its
original head sum first folds register offset 16, then lane offset 1, then
warp offsets 8, 4, 2. Explicit pair reductions preserve that order when eight
independent pools occupy different warp groups. Every one of the 1024 score
slots is initialized, including incomplete and unused pools.
"""
from pathlib import Path

SCHEDULE = 'complete_pool_batch8_streamed_heads_ordered_v3'
BATCH = 8
MIN_BASE = 512

def helpers(source):
    begin = source.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__pool_scores(')
    loop = source.index('    # Score each visible pool once.', begin)
    prefix = source[begin:loop]
    original_q = '    q_offsets = tl.arange(0, {{ call.q_width }})\n    head_offsets = tl.arange(0, {{ call.index_heads }})\n    q = tl.load({{ call.q }} + q_offsets * {{ call.q_stride }}).to(tl.float32)\n    weight_value = tl.load({{ call.weights }} + head_offsets * {{ call.weights_stride }}).to(tl.float32)\n'
    if prefix.count(original_q) != 1:
        raise ValueError('Original full-head query and weights changed')
    half_q = '    heads = tl.arange(0, 16)\n'
    prefix = prefix.replace(original_q, half_q)
    body = (Path(__file__).parent/'dsa-pool-score-batch.py.jinja').read_text()
    parts = []
    for suffix, count in (('__batched', 'count'), ('__batched_long96', 'long_count')):
        head = prefix.replace('__pool_scores(', '__pool_scores'+suffix+'(')
        parts.append(head + body.replace('{{ score_ctas }}', '{{ call.joint_dsa_pool_scores.'+count+' }}'))
    marker = '@triton.jit\ndef {{ call.symbol }}({{ call.signature }}):'
    if source.count(marker) != 1:
        raise ValueError('Complete DSA wrapper changed')
    return source.replace(marker, '{% if call.batch_dsa_pool_scores | default(false) %}\n'+'\n'.join(parts)+'{% endif %}\n'+marker)
