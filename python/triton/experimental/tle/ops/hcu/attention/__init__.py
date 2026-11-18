"""Self-written dense FA and INT8/FP8 Sage; no vendor FA fallback."""
from .dense_attention_r87 import DenseFlashAttentionPlan
from .sage_general_r83 import RawSagePlan as SageAttentionPlan
__all__ = ["DenseFlashAttentionPlan", "SageAttentionPlan"]
