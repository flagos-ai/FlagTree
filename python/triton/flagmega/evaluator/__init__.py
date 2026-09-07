# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reference evaluator namespace."""

from triton.flagmega.evaluator.context import EvaluationContext, EvaluationFrame
from triton.flagmega.evaluator.cost import CostCoverage, CostCoverageEntry, inspect_cost_coverage
from triton.flagmega.evaluator.result import EvaluationResult
from triton.flagmega.evaluator.support import (
    EvaluationHandlerGap,
    EvaluationSupport,
    inspect_evaluation_support,
)
from triton.flagmega.evaluator.torch_backend import (
    CheckpointWeightResolver,
    DictWeightResolver,
    TorchEvaluator,
    iter_materialized_constant_assets,
    materialize_constant_recipe,
    materialize_constant_assets,
)
from triton.flagmega.evaluator.numpy_backend import (
    NumpyMaterializationContext,
    iter_numpy_materialized_constant_assets,
)
from triton.flagmega.ir.ops.math._block_scaled import block_scaled_linear, dynamic_block_quant_dequant
from triton.flagmega.ir.ops.nn._gdn_state import (
    GatedDeltaNetState,
    GatedDeltaNetStateConfig,
    GatedDeltaNetStateDimKind,
    GatedDeltaNetStateKind,
    create_gdn_state,
    gdn_state_config,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionState,
    PagedAttentionStateConfig,
    create_paged_attention_state,
)
from triton.flagmega.ir.ops.nn.embedding import embedding
from triton.flagmega.ir.ops.nn.gated_delta_net import gated_delta_net
from triton.flagmega.ir.ops.nn.gdn_convolution import gated_delta_net_convolution
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import gated_delta_net_recurrent_core
from triton.flagmega.ir.ops.nn.rms_norm import rms_norm
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import qwen3_paged_attention

__all__ = [
    "CheckpointWeightResolver",
    "CostCoverage",
    "CostCoverageEntry",
    "DictWeightResolver",
    "EvaluationContext",
    "EvaluationFrame",
    "EvaluationHandlerGap",
    "EvaluationResult",
    "EvaluationSupport",
    "GatedDeltaNetState",
    "GatedDeltaNetStateConfig",
    "GatedDeltaNetStateDimKind",
    "GatedDeltaNetStateKind",
    "PagedAttentionState",
    "PagedAttentionStateConfig",
    "TorchEvaluator",
    "NumpyMaterializationContext",
    "block_scaled_linear",
    "create_gdn_state",
    "create_paged_attention_state",
    "dynamic_block_quant_dequant",
    "embedding",
    "gated_delta_net",
    "gated_delta_net_convolution",
    "gated_delta_net_recurrent_core",
    "gdn_state_config",
    "inspect_evaluation_support",
    "inspect_cost_coverage",
    "iter_materialized_constant_assets",
    "iter_numpy_materialized_constant_assets",
    "materialize_constant_recipe",
    "materialize_constant_assets",
    "rms_norm",
    "qwen3_paged_attention",
]
