# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Post-function-boundary equality propagation for vector layouts."""

from triton.flagmega.ir import DistributedType, IRModule, Node, TensorType
from triton.flagmega.ir.types import VectorType
from triton.flagmega.diagnostics import DumpScope
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.manager import FunctionalPass, PassManager
from triton.flagmega.passes.functions.function_boundary_layout import propagate_function_boundary_layouts
from triton.flagmega.passes.rewriter import DataflowPass, EGraphRulesPass
from triton.flagmega.rules.neutral import post_boundary_auto_packing_neutral_rules
from triton.flagmega.rules.ntt.vectorize import VectorizeRuleRegistry
from triton.flagmega.rules.ntt.vectorize.propagation.embedding import embedding_producer_rule
from triton.flagmega.rules.ntt.vectorize.propagation.rotary_embedding import rotary_embedding_producer_rule


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
    current = module
    for iteration in range(16):
        dumper = DumpScope.current()
        if iteration:
            dumper = dumper.create_sub_dumper(f"Iterations/{iteration:02d}")
        result = _propagation_round(current, rules, dumper)
        if result.semantic_hash == current.semantic_hash:
            return result
        current = result
    raise IRVerificationError("Post-boundary pack/layout propagation did not reach a fixed point.", stage=module.stage)


def _propagation_round(module, rules, dumper):
    return PassManager("PostFunctionBoundaryPackPropagation", dumper=dumper).add(
        EGraphRulesPass(
            "PostFunctionBoundaryPackPropagation",
            rules,
            cost=_representation_cost,
            cost_model="representation-boundary-preference/v2",
            max_iterations=128,
        )
    ).add(
        # Local rules may expose a new parameter Pack (e.g. a RoPE table).
        # Reconcile the reusable ABI before pushing caller demands further.
        FunctionalPass("PropagateExposedFunctionLayouts", propagate_function_boundary_layouts)
    ).add(
        # Rebuild shared/stateful producers once, at their original position.
        # This is a dataflow transformation: an effectful cos/sin generator
        # must not become a duplicated helper in an egraph alternative.
        DataflowPass("PackResultProducers", (embedding_producer_rule(), rotary_embedding_producer_rule()))
    ).add(
        EGraphRulesPass("FoldProducerPacking", rules, cost=_representation_cost,
                        cost_model="representation-boundary-preference/v2", max_iterations=128)
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
