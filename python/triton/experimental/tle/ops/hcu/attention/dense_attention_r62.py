"""Public FA plan with explicit self-written fallback for padded NHD rows.

Contiguous FA30 inputs use the exact R58 optimized branches. Common packed-QKV
views with contiguous head/dimension axes use the stride-aware generic kernel;
its speed is not represented by the contiguous FA30 performance table.
"""
from .dense_attention_r58 import DenseFlashAttentionPlan as OptimizedPlan
from .generic_dense_fa_r17 import GenericDenseFlashAttentionPlan
IMPLEMENTATION_MODULES=('triton.experimental.tle.ops.hcu.attention.dense_attention_r58', 'triton.experimental.tle.ops.hcu.attention.dense_attention_r41', 'triton.experimental.tle.ops.hcu.attention.dense_attention_tail_r58', 'triton.experimental.tle.ops.hcu.attention.dense_attention_r43', 'triton.experimental.tle.ops.hcu.attention.panel32_compiled_tail_r40', 'triton.experimental.tle.ops.hcu.attention.panel32_gqa_r22', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r22', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r43', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r58', 'triton.experimental.tle.ops.hcu.attention.generic_dense_fa_r17')
class DenseFlashAttentionPlan:
    def __new__(cls,q,k,v):
        if any(x.ndim!=4 or not x.is_contiguous()for x in(q,k,v)):
            return GenericDenseFlashAttentionPlan(q,k,v)
        return OptimizedPlan(q,k,v)
def dense_attention(q,k,v):return DenseFlashAttentionPlan(q,k,v).full()
