# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Post-function-boundary equality propagation for vector layouts."""

from triton.flagmega.ir import DistributedType, IRModule, Node, TensorType
from triton.flagmega.ir.types import VectorType
from triton.flagmega.passes.manager import PassManager
from triton.flagmega.passes.rewriter import EGraphRulesPass
from triton.flagmega.rules.neutral import (
    post_boundary_auto_packing_neutral_rules,
)
from triton.flagmega.rules.ntt.vectorize import VectorizeRuleRegistry


def post_function_boundary_pack_propagation(module: IRModule, target) -> IRModule:
    """Saturate target pack propagation rules after function ABI rewriting.

    The categorical extraction objective is deliberately not a hardware cost
    model: it prefers fewer representation boundaries and typed-vector compute
    while keeping every equality alternative in the EGraph dump.  An agent can
    inspect/modify the extracted Python checkpoint; standalone compilation has
    this deterministic default.
    """

    registry = VectorizeRuleRegistry()
    target.register_pack_propagation_rules(registry)
    rules = (
        *registry.propagation_rules,
        *post_boundary_auto_packing_neutral_rules(),
    )
    return PassManager("PostFunctionBoundaryPackPropagation").add(
        EGraphRulesPass(
            "PostFunctionBoundaryPackPropagation",
            rules,
            cost=_representation_cost,
            cost_model="representation-boundary-preference/v2",
            max_iterations=128,
        )
    ).run(module).module


def _representation_cost(node: Node, _module: IRModule) -> float:
    if node.op in {"builtin.tuple", "builtin.get_item"}:
        # A direct SSA edge is the canonical representative of
        # GetItem(Tuple(...)); make the neutral equality deterministic.
        return 1.0
    if node.op.startswith("builtin."):
        return 0.0
    if node.op == "tensors.bitcast":
        # A Bitcast is a zero-copy storage view.  In particular, nncase's
        # UnpackToBitcast equality must win over a typed-vector boundary when
        # both represent the same final-axis bytes.
        return 0.0
    if node.op in {
        "tensors.pack",
        "tensors.unpack",
        "tensors.permute",
        "tensors.reshape",
    }:
        return 4.0
    tensor_type = (
        node.type.tensor if isinstance(node.type, DistributedType) else node.type)
    vector = (
        isinstance(tensor_type, TensorType)
        and isinstance(tensor_type.dtype, VectorType)
    )
    return 1.0 if vector else 2.0


__all__ = ["post_function_boundary_pack_propagation"]
