# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.ir.ops.nn.embedding import Embedding
from triton.flagmega.ir.ops.nn.softmax import Softmax
from triton.flagmega.ir.ops.nn.l2_normalization import L2Normalization
from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates
from triton.flagmega.ir.ops.nn.delta_rule_coefficients import DeltaRuleCoefficients
from triton.flagmega.ir.ops.nn.delta_rule_log_prefix import DeltaRuleLogPrefix
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate
from triton.flagmega.ir.ops.nn.dense_matmul_glu import DenseMatMulGlu
from triton.flagmega.ir.ops.nn.gated_delta_net import GatedDeltaNet
from triton.flagmega.ir.ops.nn.gdn_convolution import GatedDeltaNetConvolution
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
from triton.flagmega.ir.ops.nn.gdn_state_slice import GatedDeltaNetStateSlice
from triton.flagmega.ir.ops.nn.greedy_sample import GreedySample
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.matmul_glu import MatMulGlu
from triton.flagmega.ir.ops.nn.packed_matmul_glu import PackedMatMulGlu
from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import PackedDenseMatMulGlu
from triton.flagmega.ir.ops.nn.packed_qwen3_paged_attention import PackedQwen3PagedAttention
from triton.flagmega.ir.ops.nn.rms_norm import RMSNorm
from triton.flagmega.ir.ops.nn.qkv_parallel_linear import QKVParallelLinear
from triton.flagmega.ir.ops.nn.qkv_rope_with_cache import QKVRoPEWithCache
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import Qwen3PagedAttention
from triton.flagmega.ir.ops.nn.vectorized_rms_norm import VectorizedRMSNorm
from triton.flagmega.ir.ops.nn.bind_norm_stats import BindNormStats
from triton.flagmega.ir.ops.nn.layer_norm import LayerNorm
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.ops.nn.paged_attention import PagedAttention
from triton.flagmega.ir.ops.nn.rope import RoPE
from triton.flagmega.ir.ops.nn.rotary_embedding import RotaryEmbedding
from triton.flagmega.ir.ops.nn.update_paged_attention_kv_cache import (
    UpdatePagedAttentionKVCache,
)

__all__ = [
    "DeltaRuleGates",
    "L2Normalization",
    "DeltaRuleBlockUpdate",
    "DeltaRuleLogPrefix",
    "DeltaRuleCoefficients",
    "Softmax",
    "Embedding",
    "DenseMatMulGlu",
    "GatedDeltaNet",
    "GatedDeltaNetConvolution",
    "GatedDeltaNetRecurrentCore",
    "GatedDeltaNetStateSlice",
    "GreedySample",
    "SparseExperts",
    "SparseExpertsGateUp",
    "SparseExpertsDown",
    "MatMulGlu",
    "PackedMatMulGlu",
    "PackedDenseMatMulGlu",
    "PackedQwen3PagedAttention",
    "PagedAttention",
    "RoPE",
    "RotaryEmbedding",
    "UpdatePagedAttentionKVCache",
    "BindNormStats",
    "LayerNorm",
    "NormApply",
    "NormStats",
    "QKVParallelLinear",
    "QKVRoPEWithCache",
    "RMSNorm",
    "Qwen3PagedAttention",
    "VectorizedRMSNorm",
]
