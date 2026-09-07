# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TIR graph analyses and transformations."""

from .projection_residual_norm import (
    ProjectionResidualNormMatch,
    find_projection_residual_norm_matches,
)
from .projection_logits_argmax import (
    ProjectionLogitsArgmaxMatch,
    find_projection_logits_argmax_matches,
)
from .norm_consumer import NormConsumerMatch, find_norm_consumer_matches
from .materialize_kernel_functions import (
    MaterializeKernelPrimFunctionsPass,
    materialize_kernel_prim_functions,
)
from .materialize_kernel_definitions import (
    MaterializeKernelDefinitionsPass, materialize_kernel_definitions,
)
from .bind_prim_function_buffers import BindPrimFunctionBuffersPass, bind_prim_function_buffers
from .canonicalize_packed_qkv_weights import (
    CanonicalizePackedQKVWeightsPass,
    canonicalize_packed_qkv_weights,
)
from .fuse_gather_reduce_qkv_rope_with_cache import (
    FuseGatherReduceQKVRoPEWithCachePass,
    fuse_gather_reduce_qkv_rope_with_cache,
)
from .fuse_gather_reduce_norm_apply import (
    FuseGatherReduceNormApplyPass,
    fuse_gather_reduce_norm_apply,
)
from .fuse_gather_reduce_add_norm_apply import (
    FuseGatherReduceAddNormApplyPass,
    fuse_gather_reduce_add_norm_apply,
)
from .specialize_prim_function_layouts import (
    specialize_prim_functions_for_buffer_layouts,
)
from .lower_transfer_pipeline_regions import lower_transfer_pipeline_regions
from .lower_tuple_boxing import lower_tuple_boxing
from .materialize_execution_functions import materialize_execution_functions
from .materialize_memory_synchronization import (
    materialize_memory_synchronization,
    memory_synchronization_from_execution_functions,
)
from .plan_function_memory import (
    FUNCTION_MEMORY_PLACEMENT_SCHEMA,
    MEMORY_SPACE_METADATA,
    plan_function_memory,
)

__all__ = [
    "MaterializeKernelDefinitionsPass", "materialize_kernel_definitions",
    "ProjectionResidualNormMatch",
    "ProjectionLogitsArgmaxMatch",
    "NormConsumerMatch",
    "MaterializeKernelPrimFunctionsPass",
    "BindPrimFunctionBuffersPass",
    "CanonicalizePackedQKVWeightsPass",
    "FuseGatherReduceQKVRoPEWithCachePass",
    "FuseGatherReduceAddNormApplyPass",
    "FuseGatherReduceNormApplyPass",
    "canonicalize_packed_qkv_weights",
    "find_norm_consumer_matches",
    "find_projection_logits_argmax_matches",
    "find_projection_residual_norm_matches",
    "fuse_gather_reduce_qkv_rope_with_cache",
    "fuse_gather_reduce_add_norm_apply",
    "fuse_gather_reduce_norm_apply",
    "materialize_kernel_prim_functions",
    "materialize_execution_functions",
    "materialize_memory_synchronization",
    "memory_synchronization_from_execution_functions",
    "FUNCTION_MEMORY_PLACEMENT_SCHEMA",
    "MEMORY_SPACE_METADATA",
    "plan_function_memory",
    "bind_prim_function_buffers",
    "specialize_prim_functions_for_buffer_layouts",
    "lower_transfer_pipeline_regions",
    "lower_tuple_boxing",
]
