# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega pass management namespace."""

from triton.flagmega.passes.auto_vectorize import AutoVectorizePass
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.passes.packed_qkv_combine import (
    fold_materialized_packed_qkv_parallel_linear_combine,
    lower_packed_qkv_parallel_linear_combine,
)
from triton.flagmega.passes.analysis import AnalysisCacheKey, AnalysisManager, AnalysisProvider
from triton.flagmega.passes.context import RunPassContext, current_pass_context
from triton.flagmega.passes.function_pass import (
    FunctionPass,
    FunctionTraversalOrder,
    FunctionalFunctionPass,
)
from triton.flagmega.passes.prim_function_pass import (
    FunctionalPrimFunctionPass,
    PrimFuncPass,
    PrimFunctionPass,
)
from triton.flagmega.passes.constants import (
    ConstantCSEPass,
    ConstnessAnalysis,
    ConstnessResult,
    FreezeConstantIslandsPass,
    constant_phase,
    freeze_constant_islands,
    require_constants_open,
)
from triton.flagmega.passes.manager import FunctionalPass, ModulePass, PassExecution, PassManager, PassResult
from triton.flagmega.passes.pipeline import (
    PIPELINE_GROUPS,
    PipelineGroup,
    PipelinePass,
    PipelinePassRegistry,
    TargetPipelineExtension,
    expand_pipeline_passes,
)
from triton.flagmega.passes.rewriter import (
    DataflowPass,
    EGraphConstructPass,
    EGraphExtractPass,
    EGraphRulesPass,
)
from triton.flagmega.passes.gated_delta_net import decompose_gated_delta_net

__all__ = [
    "AutoVectorizePass",
    "AutoDistributedPass",
    "fold_materialized_packed_qkv_parallel_linear_combine",
    "lower_packed_qkv_parallel_linear_combine",
    "AnalysisCacheKey",
    "AnalysisManager",
    "AnalysisProvider",
    "ConstantCSEPass",
    "ConstnessAnalysis",
    "ConstnessResult",
    "FreezeConstantIslandsPass",
    "FunctionalPass",
    "FunctionalFunctionPass",
    "FunctionalPrimFunctionPass",
    "FunctionPass",
    "FunctionTraversalOrder",
    "PrimFuncPass",
    "PrimFunctionPass",
    "DataflowPass",
    "EGraphConstructPass",
    "EGraphExtractPass",
    "EGraphRulesPass",
    "ModulePass",
    "PassExecution",
    "PassManager",
    "PassResult",
    "RunPassContext",
    "PIPELINE_GROUPS",
    "PipelineGroup",
    "PipelinePass",
    "PipelinePassRegistry",
    "TargetPipelineExtension",
    "expand_pipeline_passes",
    "constant_phase",
    "decompose_gated_delta_net",
    "freeze_constant_islands",
    "require_constants_open",
    "current_pass_context",
]
