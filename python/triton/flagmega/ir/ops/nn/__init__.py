# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.ir.ops.nn.bind_norm_stats import BindNormStats
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate
from triton.flagmega.ir.ops.nn.delta_rule_coefficients import DeltaRuleCoefficients
from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates
from triton.flagmega.ir.ops.nn.delta_rule_log_prefix import DeltaRuleLogPrefix
from triton.flagmega.ir.ops.nn.dense_matmul_glu import DenseMatMulGlu
from triton.flagmega.ir.ops.nn._dsa_indexer_state import DsaIndexerState, DsaIndexerStateConfig
from triton.flagmega.ir.ops.nn.dsa_indexer import DsaIndexer
from triton.flagmega.ir.ops.nn.dsa_paged_attention import DsaPagedAttention
from triton.flagmega.ir.ops.nn.embedding import Embedding
from triton.flagmega.ir.ops.nn.gated_delta_net import GatedDeltaNet
from triton.flagmega.ir.ops.nn.gated_norm_apply import GatedNormApply
from triton.flagmega.ir.ops.nn.gdn_convolution import GatedDeltaNetConvolution
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
from triton.flagmega.ir.ops.nn.gdn_state_slice import GatedDeltaNetStateSlice
from triton.flagmega.ir.ops.nn.greedy_sample import GreedySample
from triton.flagmega.ir.ops.nn.kda import KdaConvolution, KdaRecurrentCore
from triton.flagmega.ir.ops.nn.kda_gated_recurrent_core import KdaGatedRecurrentCore
from triton.flagmega.ir.ops.nn.kda_gated_step import KdaGatedStep
from triton.flagmega.ir.ops.nn.l2_normalization import L2Normalization
from triton.flagmega.ir.ops.nn.layer_norm import LayerNorm
from triton.flagmega.ir.ops.nn.matmul_glu import MatMulGlu
from triton.flagmega.ir.ops.nn.mhc_post import MhcPost
from triton.flagmega.ir.ops.nn.mhc_pre import MhcPre
from triton.flagmega.ir.ops.nn.mla_projection import MlaProjection
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import PackedDenseMatMulGlu
from triton.flagmega.ir.ops.nn.packed_matmul_glu import PackedMatMulGlu
from triton.flagmega.ir.ops.nn.packed_qwen3_paged_attention import PackedQwen3PagedAttention
from triton.flagmega.ir.ops.nn.paged_attention import PagedAttention
from triton.flagmega.ir.ops.nn.partitioned_sparse_experts_down import PartitionedSparseExpertsDown
from triton.flagmega.ir.ops.nn.qkv_parallel_linear import QKVParallelLinear
from triton.flagmega.ir.ops.nn.qkv_rope_with_cache import QKVRoPEWithCache
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import Qwen3PagedAttention
from triton.flagmega.ir.ops.nn.requantize_block_scaled_fp8 import RequantizeBlockScaledFP8
from triton.flagmega.ir.ops.nn.rms_norm import RMSNorm
from triton.flagmega.ir.ops.nn.rope import RoPE
from triton.flagmega.ir.ops.nn.rotary_embedding import RotaryEmbedding
from triton.flagmega.ir.ops.nn.softmax import Softmax
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsCombine, SparseExpertsWeightedSum
from triton.flagmega.ir.ops.nn.sparse_experts_dispatch import SparseExpertsDispatch
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.staged_routing import StagedRouting
from triton.flagmega.ir.ops.nn.update_paged_attention_kv_cache import (
    UpdatePagedAttentionKVCache,
)
from triton.flagmega.ir.ops.nn.update_paged_attention_kv_cache_pair import UpdatePagedAttentionKVCachePair
from triton.flagmega.ir.ops.nn.vectorized_rms_norm import VectorizedRMSNorm

__all__ = [
    "DeltaRuleGates",
    "L2Normalization",
    "DeltaRuleBlockUpdate",
    "DeltaRuleLogPrefix",
    "DeltaRuleCoefficients",
    "Softmax", "KdaConvolution", "KdaRecurrentCore", "KdaGatedRecurrentCore", "KdaGatedStep", "StagedRouting",
    "MhcPre", "MhcPost",
    "MlaProjection",
    "RequantizeBlockScaledFP8",
    "Embedding",
    "DenseMatMulGlu",
    "DsaIndexer",
    "DsaIndexerState",
    "DsaIndexerStateConfig",
    "DsaPagedAttention",
    "GatedDeltaNet",
    "GatedDeltaNetConvolution",
    "GatedDeltaNetRecurrentCore",
    "GatedDeltaNetStateSlice",
    "GreedySample",
    "SparseExperts",
    "SparseExpertsGateUp",
    "SparseExpertsDown",
    "PartitionedSparseExpertsDown",
    "SparseExpertsDispatch",
    "SparseExpertsCombine",
    "SparseExpertsWeightedSum",
    "MatMulGlu",
    "PackedMatMulGlu",
    "PackedDenseMatMulGlu",
    "PackedQwen3PagedAttention",
    "PagedAttention",
    "RoPE",
    "RotaryEmbedding",
    "UpdatePagedAttentionKVCache",
    "UpdatePagedAttentionKVCachePair",
    "BindNormStats",
    "LayerNorm",
    "GatedNormApply",
    "NormApply",
    "NormStats",
    "QKVParallelLinear",
    "QKVRoPEWithCache",
    "RMSNorm",
    "Qwen3PagedAttention",
    "VectorizedRMSNorm",
]
