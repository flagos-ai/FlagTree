# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Function-boundary transformations shared by middle-end stages."""

from .graph import (
    callee_first_functions,
    function_nodes,
    reusable_function_node_ids,
    static_function_invocation_counts,
    static_node_invocation_counts,
)
from .form_matmul_norm_stats_combine import (
    FormMatMulNormStatsCombinePass,
    form_matmul_norm_stats_combine,
)
from .function_boundary_layout import propagate_function_boundary_layouts
from .post_boundary_pack import post_function_boundary_pack_propagation
from .remove_unused import remove_unused_functions
from .distributed_boundary_layout import (
    propagate_post_auto_distributed_function_boundary_layouts,
)
from .lift_parameter_transforms import lift_parameter_constant_transforms
from .lift_constant_expressions import lift_constant_parameter_expressions
from .hoist_call_invariants import hoist_call_invariant_expressions
from .thread_norm_stats import thread_norm_stats_across_function_boundaries
from .sink_norm_stats_boxing import sink_norm_stats_boxing_across_function_boundaries

__all__ = [
    "callee_first_functions",
    "FormMatMulNormStatsCombinePass",
    "form_matmul_norm_stats_combine",
    "function_nodes",
    "propagate_function_boundary_layouts",
    "post_function_boundary_pack_propagation",
    "remove_unused_functions",
    "propagate_post_auto_distributed_function_boundary_layouts",
    "reusable_function_node_ids",
    "static_function_invocation_counts",
    "static_node_invocation_counts",
    "lift_parameter_constant_transforms",
    "lift_constant_parameter_expressions",
    "hoist_call_invariant_expressions",
    "thread_norm_stats_across_function_boundaries",
    "sink_norm_stats_boxing_across_function_boundaries",
]
