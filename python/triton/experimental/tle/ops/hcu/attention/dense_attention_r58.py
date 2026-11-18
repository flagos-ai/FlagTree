"""Self-written dense noncausal FA; shape dispatch uses only metadata.

BF16/FP16, D multiples of8 in[8,256], MHA/GQA. All routes are own kernels;
no vendor FlashAttention call is made by this module. Specialized configurations
are candidates, not a claim that each shape already beats native.
"""
from .dense_attention_r41 import DenseFlashAttentionPlan as DefaultPlan
from .dense_attention_tail_r58 import DenseFlashAttentionPlan as TailPlan
from .dense_attention_r43 import DenseFlashAttentionPlan as LargeDPlan
from .generic_dense_fa_r17 import GenericDenseFlashAttentionPlan
IMPLEMENTATION_MODULES=('triton.experimental.tle.ops.hcu.attention.dense_attention_r41', 'triton.experimental.tle.ops.hcu.attention.dense_attention_tail_r58', 'triton.experimental.tle.ops.hcu.attention.dense_attention_r43', 'triton.experimental.tle.ops.hcu.attention.panel32_compiled_tail_r40', 'triton.experimental.tle.ops.hcu.attention.panel32_gqa_r22', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r22', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r43', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r58', 'triton.experimental.tle.ops.hcu.attention.generic_dense_fa_r17')
class DenseFlashAttentionPlan:
    def __new__(cls,q,k,v):
        d=q.shape[-1];nk=k.shape[1]
        if d not in(64,96,128,192,256):return GenericDenseFlashAttentionPlan(q,k,v)
        if d==96 and k.shape[2]>=2:return TailPlan(q,k,v)
        if d==256 and nk>=4096:return LargeDPlan(q,k,v)
        return DefaultPlan(q,k,v)
def dense_attention(q,k,v):return DenseFlashAttentionPlan(q,k,v).full()
