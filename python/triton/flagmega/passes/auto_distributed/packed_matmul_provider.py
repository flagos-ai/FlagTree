# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PackedMatMul distribution relations derived from its operand layouts."""

from itertools import combinations, product
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import (
    DistributedType, IRType, Node, NoneType, SBP, SBPSplit, VectorType,
    is_distributable, scale_split_units,
)
from triton.flagmega.ir.distributed_inference import broadcast_ir_type, tensor_of
from triton.flagmega.ir.ops.core import tensor_nbytes
from triton.flagmega.ir.ops.ntt.packed_matmul import PackedMatMul
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate, DistributedCandidateContext,
    DistributedCandidateProviderBase,
)


class PackedMatMulCandidateProvider(DistributedCandidateProviderBase):
    """nncase PackedMatMulDistributedCandidates: align K, preserve M and N.

    The Python search keeps logical originators and creates requested layouts
    on edges. Include those target leaf requests here as well as the layouts
    propagated from producers; reshard planning proves each selected edge.
    """

    op_names = frozenset({PackedMatMul.op_name})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(
        self, context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        if len(context.available_input_types) != len(PackedMatMul.input_parameters):
            return ()
        lhs_types = _operand_layouts(context, PackedMatMul.lhs.index)
        rhs_types = _operand_layouts(context, PackedMatMul.rhs.index)
        scales = tuple(dict.fromkeys(
            broadcast_ir_type(value, context.placement)
            for value in context.available_input_types[PackedMatMul.scale.index]
        ))
        has_addend = any(
            not isinstance(value, NoneType)
            for value in context.available_input_types[PackedMatMul.addend.index]
        )
        results: list[DistributedCandidate] = []
        seen: set[tuple[IRType, ...]] = set()
        for lhs, rhs, scale in product(lhs_types, rhs_types, scales):
            aligned_rhs = align_rhs_reduction_policy(lhs, rhs)
            if aligned_rhs is None:
                continue
            inputs = (lhs, aligned_rhs, scale, NoneType())
            try:
                output = PackedMatMul.infer_type(_typed_inputs(inputs), node.attrs)
                if has_addend:
                    inputs = (lhs, aligned_rhs, scale, output)
                    output = PackedMatMul.infer_type(_typed_inputs(inputs), node.attrs)
            except IRSchemaError:
                continue
            if inputs in seen:
                continue
            seen.add(inputs)
            factors = PackedMatMul.cost_factors(_typed_inputs(inputs), node.attrs, output)
            results.append(DistributedCandidate(
                f"distribution.{node.id}.packed_matmul",
                output, inputs,
                (
                    context.operation_cost_model.get_latency(factors, output)
                    if factors is not None
                    else min(tensor_nbytes(output) or 1_000_000, 2_000_000_000)
                ),
                "packed-matmul-operand-sbp",
                objective_kind="analytic" if factors is not None else "heuristic",
                objective_model=(
                    context.operation_cost_model.identity if factors is not None
                    else "flagmega.packed-matmul-output-bytes/v1"
                ),
                objective_evidence=(
                    "producer-candidate-layout", "packed-rhs-n-policy",
                    "lhs-aligned-reduction-policy",
                    "operation-owned-distributed-type-inference",
                    "op-definition-cost-factors",
                ),
            ))
        return tuple(results)


def align_rhs_reduction_policy(
    lhs: DistributedType, rhs: DistributedType,
) -> DistributedType | None:
    """Pack the LHS logical K policy into RHS units without changing RHS N."""

    vector = rhs.tensor.dtype
    if (
        lhs.placement != rhs.placement
        or lhs.partial is not None or rhs.partial is not None
        or lhs.tensor.rank != 2 or rhs.tensor.rank != 2
        or not isinstance(vector, VectorType) or len(vector.lanes) != 3
    ):
        return None
    logical_k = lhs.axis_policies[1]
    k_policy = (
        scale_split_units(logical_k, 1, prod(vector.lanes[1:]))
        if isinstance(logical_k, SBPSplit) else logical_k
    )
    if k_policy is None:
        return None
    policies = (k_policy, rhs.axis_policies[1])
    if not is_distributable(rhs.tensor, policies, rhs.placement):
        return None
    return DistributedType(rhs.tensor, policies, rhs.placement)


def _operand_layouts(
    context: DistributedCandidateContext, input_index: int,
) -> tuple[DistributedType, ...]:
    tensor = tensor_of(
        context.module.node_map[context.source_call.inputs[input_index]].type
    )
    values = [
        value for value in context.available_input_types[input_index]
        if isinstance(value, DistributedType) and value.tensor == tensor
        and value.placement == context.placement and value.partial is None
    ]
    by_axis = []
    for axis, dimension in enumerate(tensor.shape):
        policies = [SBP.broadcast()]
        for count in range(1, context.placement.rank + 1):
            for axes in combinations(range(context.placement.rank), count):
                owners = prod(context.placement.hierarchy[a] for a in axes)
                if dimension.is_fixed and dimension.fixed_value < owners:
                    continue
                policies.extend(context.split_candidates(tensor, axis, axes))
        by_axis.append(tuple(dict.fromkeys(policies)))
    for policies in product(*by_axis):
        if is_distributable(tensor, policies, context.placement):
            values.append(DistributedType(tensor, policies, context.placement))
    return tuple(dict.fromkeys(values))


def _typed_inputs(types: tuple[IRType, ...]) -> tuple[Node, ...]:
    return tuple(
        Node(f"<packed_input_{i}>", "builtin.var", (), value)
        for i, value in enumerate(types)
    )
