# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase sparse expert stage relations over arbitrary target-owned meshes.

Experts/routes remain replicated because selection is dynamic. GateUp can
partition tokens and intermediate features; Down can independently partition
tokens, intermediate reduction and output features. Op inference is the final
authority, including explicit per-route rounding that forbids split-K.
"""

from itertools import combinations, product

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, Node, SBP
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.core import tensor_nbytes
from triton.flagmega.ir.ops.nn._sparse_experts import lanes, role_axes, scale_policy
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidate, DistributedCandidateProviderBase


def _axis_policies(context, tensor, axis, sources=()):
    policies = [SBP.broadcast()]
    for count in range(1, context.placement.rank + 1):
        for axes in combinations(range(context.placement.rank), count):
            policies.extend(context.split_candidates(tensor, axis, axes))
    for parameter, source_axis, numerator, denominator in sources:
        for value in context.available_input_types[parameter.input_index]:
            if isinstance(value, DistributedType) and value.placement == context.placement and value.partial is None:
                try:
                    policies.append(scale_policy(value.axis_policies[source_axis], numerator, denominator))
                except IRSchemaError:
                    continue
    return tuple(dict.fromkeys(policies))


def _candidate(context, definition, tensors, policies):
    inputs = tuple(
        DistributedType(tensors[parameter.name],
                        tuple(policies.get(parameter.name, (SBP.broadcast(), ) *
                                           tensors[parameter.name].rank)), context.placement)
        for parameter in definition.input_parameters)
    typed_inputs = tuple(
        Node(parameter.name, "builtin.var", (), value) for parameter, value in zip(definition.input_parameters, inputs))
    output = definition.infer_type(typed_inputs, context.source_call.attrs)
    factors = definition.cost_factors(typed_inputs, context.source_call.attrs, output)
    return DistributedCandidate(
        f"distribution.{context.source_call.id}.{definition.functional_name}",
        output,
        inputs,
        context.operation_cost_model.get_latency(factors, output) if factors is not None else min(
            tensor_nbytes(output) or 1_000_000, 2_000_000_000),
        f"{definition.functional_name}-operand-sbp",
        objective_kind="analytic" if factors is not None else "heuristic",
        objective_model=context.operation_cost_model.identity
        if factors is not None else "sparse-experts-output-bytes/v1",
        objective_evidence=("nncase-sparse-experts-stage-relations", "target-owned-split-policies",
                            "vector-scalar-split-units", "op-owned-rounding-legality", "selected-expert-cost"),
    )


def _source_tensors(context, definition):
    return {
        parameter.name: tensor_of(context.module.node_map[parameter.read(context.source_call.inputs)].type)
        for parameter in definition.input_parameters
    }


class SparseExpertsGateUpCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({SparseExpertsGateUp.op_name})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context):
        definition = SparseExpertsGateUp
        tensors = _source_tensors(context, definition)
        output = tensor_of(context.source_call.type)
        token_policies = _axis_policies(context, output, 0,
                                        ((definition.q, 0, 1, 1), (definition.router_expert_ids, 0, 1, 1)))
        intermediate_policies = _axis_policies(context, output, 2, ((definition.gate_weight, 1, 1, lanes(output.dtype)),
                                                                    (definition.up_weight, 1, 1, lanes(output.dtype))))
        broadcast = SBP.broadcast()
        results = []
        for token, intermediate in product(token_policies, intermediate_policies):
            try:
                role_axes(token, intermediate)
                scalar_intermediate = scale_policy(intermediate, lanes(output.dtype), 1)
                results.append(
                    _candidate(
                        context, definition, tensors, {
                            "q": (token, broadcast),
                            "router_expert_ids": (token, broadcast),
                            "gate_weight": (broadcast, scalar_intermediate, broadcast),
                            "up_weight": (broadcast, scalar_intermediate, broadcast),
                        }))
            except IRSchemaError:
                continue
        return tuple(results)


class SparseExpertsDownCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({SparseExpertsDown.op_name})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context):
        definition = SparseExpertsDown
        tensors = _source_tensors(context, definition)
        output, activation = tensor_of(context.source_call.type), tensors["activations"]
        token_policies = _axis_policies(context, output, 0,
                                        ((definition.activations, 0, 1, 1), (definition.router_expert_ids, 0, 1, 1)))
        intermediate_policies = _axis_policies(context, activation, 2,
                                               ((definition.activations, 2, 1, 1),
                                                (definition.down_weight, 2, 1, lanes(activation.dtype))))
        output_policies = _axis_policies(context, output, 1, ((definition.down_weight, 1, 1, lanes(output.dtype)), ))
        broadcast = SBP.broadcast()
        results = []
        for token, intermediate, output_policy in product(token_policies, intermediate_policies, output_policies):
            try:
                role_axes(token, intermediate, output_policy)
                scalar_intermediate = scale_policy(intermediate, lanes(activation.dtype), 1)
                scalar_output = scale_policy(output_policy, lanes(output.dtype), 1)
                results.append(
                    _candidate(
                        context, definition, tensors, {
                            "activations": (token, broadcast, intermediate),
                            "router_expert_ids": (token, broadcast),
                            "router_expert_weights": (token, broadcast),
                            "down_weight": (broadcast, scalar_output, scalar_intermediate),
                        }))
            except IRSchemaError:
                continue
        return tuple(results)


__all__ = ["SparseExpertsGateUpCandidateProvider", "SparseExpertsDownCandidateProvider"]
