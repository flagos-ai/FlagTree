"""Self-written dense FA; metadata dispatch for qualified gfx938 candidates.

Requires C85 over C67. No vendor FA call. Row-strided inputs retain the existing
stride-aware fallback. New MMAC/WAR scheduling is enabled only on the prefill
families with measured gains; short, split-KV and other-D routes are preserved.
"""
from .dense_attention_r62 import DenseFlashAttentionPlan as Previous
from .panel_gqa_r80 import PrefillPlan
IMPLEMENTATION_MODULES=('triton.experimental.tle.ops.hcu.attention.dense_attention_r62', 'triton.experimental.tle.ops.hcu.attention.dense_attention_r58', 'triton.experimental.tle.ops.hcu.attention.dense_attention_r41', 'triton.experimental.tle.ops.hcu.attention.dense_attention_tail_r58', 'triton.experimental.tle.ops.hcu.attention.dense_attention_r43', 'triton.experimental.tle.ops.hcu.attention.panel32_compiled_tail_r40', 'triton.experimental.tle.ops.hcu.attention.panel32_gqa_r22', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r22', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r43', 'triton.experimental.tle.ops.hcu.attention.panel_dense_r58', 'triton.experimental.tle.ops.hcu.attention.generic_dense_fa_r17', 'triton.experimental.tle.ops.hcu.attention.panel_gqa_r80')
class DenseFlashAttentionPlan:
    def __new__(cls,q,k,v):
        if any(x.ndim!=4 or not x.is_contiguous()for x in(q,k,v)):return Previous(q,k,v)
        d=q.shape[-1];nq=q.shape[1];nk=k.shape[1]
        if d==128 and nq>512 and q.shape[2]!=k.shape[2]:p=PrefillPlan(q,k,v,static_split=True,outvec=8)
        else:p=Previous(q,k,v)
        if (d==128 and nq>512 and nk>=1024)or(d==256 and nk>=4096):
            p.config.update(hcu_mmac_priority='high',hcu_panel_war_after_mma=24 if d==128 else 8)
        return p

def dense_attention(q,k,v):return DenseFlashAttentionPlan(q,k,v).full()
