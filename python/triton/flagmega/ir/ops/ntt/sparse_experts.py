# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local fusions of independently distributed expert stages."""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import Node, DistributedType
from triton.flagmega.ir.ops.core import OpDefinition, attribute_parameter, input_parameter, op_definition
from triton.flagmega.ir.ops.nn._sparse_experts import EXPERT_SCALE, ROUTER_IDS, floating_tensor, python_dtype_call
from triton.flagmega.ir.ops.nn.sparse_experts_dispatch import SparseExpertsDispatch
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp, evaluate_gate_up
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown, evaluate_down
from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsWeightedSum, combine_type


def stage(definition, inputs, attrs):
    prepared = definition.prepare(inputs, attrs)
    return Node("stage", definition.op_name, tuple(n.id for n in inputs), prepared.result_type,
                prepared.effect, prepared.attrs)


@op_definition("ntt.dispatched_experts_gate_up", namespace="ntt", functional_name="dispatched_experts_gate_up",
               display_name="NTT.DispatchedExpertsGateUp")
class DispatchedExpertsGateUp(OpDefinition):
    supports_broadcast_lifting = False
    q = input_parameter(floating_tensor(2, packed=True))
    router_expert_ids = input_parameter(ROUTER_IDS)
    gate_input_scale = input_parameter(EXPERT_SCALE)
    gate_weight = input_parameter(floating_tensor(3))
    gate_proj_scale = input_parameter(EXPERT_SCALE)
    up_input_scale = input_parameter(EXPERT_SCALE)
    up_weight = input_parameter(floating_tensor(3))
    up_proj_scale = input_parameter(EXPERT_SCALE)
    output_dtype = attribute_parameter(default=None)
    round_projections = attribute_parameter(default=False)
    round_activation = attribute_parameter(default=False)

    @classmethod
    def normalize_attrs(cls, attrs):
        return SparseExpertsGateUp.normalize_attrs(attrs)

    @classmethod
    def python_call(cls, node):
        return python_dtype_call(super().python_call(node), "output_dtype")

    @classmethod
    def gate_inputs(cls, inputs):
        return (stage(SparseExpertsDispatch, inputs[:2], {}), *inputs[1:])

    @classmethod
    def infer_type(cls, inputs, attrs):
        return SparseExpertsGateUp.prepare(cls.gate_inputs(inputs), attrs).result_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {p.name: p.read(arguments) for p in cls.input_parameters}
        values["dispatched"] = values["q"].unsqueeze(1).expand(-1, values["router_expert_ids"].shape[1], *values["q"].shape[1:])
        inputs = tuple(Node(n, "builtin.var", (), context.types[n]) for n in node.inputs)
        return evaluate_gate_up(values, cls.gate_inputs(inputs)[0].type, node.type, node.attrs, context)

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        return SparseExpertsGateUp.cost_factors(cls.gate_inputs(inputs), attrs, return_type)


@op_definition("ntt.sparse_experts_down_combine", namespace="ntt", functional_name="sparse_experts_down_combine",
               display_name="NTT.SparseExpertsDownCombine")
class SparseExpertsDownCombine(OpDefinition):
    supports_broadcast_lifting = False
    activations = input_parameter(floating_tensor(3, packed=True))
    router_expert_ids = input_parameter(ROUTER_IDS)
    down_input_scale = input_parameter(EXPERT_SCALE)
    down_weight = input_parameter(floating_tensor(3))
    down_proj_scale = input_parameter(EXPERT_SCALE)
    router_expert_weights = input_parameter(floating_tensor(2))
    round_projection = attribute_parameter(default=False)
    output_dtype = attribute_parameter(default=None)
    round_weighted_output = attribute_parameter(default=False)
    cast_output = attribute_parameter(default=False)

    @classmethod
    def python_call(cls, node):
        return python_dtype_call(super().python_call(node), "output_dtype")

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = super().normalize_attrs(attrs)
        normalized = SparseExpertsWeightedSum.normalize_attrs({k: attrs[k] for k in ("output_dtype", "round_weighted_output")})
        if type(attrs["cast_output"]) is not bool:
            raise IRSchemaError("SparseExpertsDownCombine cast_output must be boolean")
        return {**attrs, **normalized, **SparseExpertsDown.normalize_attrs({"round_projection": attrs["round_projection"]})}

    @classmethod
    def combine_inputs(cls, inputs, attrs):
        return (stage(SparseExpertsDown, inputs[:5], {"round_projection": attrs["round_projection"]}), inputs[5])

    @classmethod
    def infer_type(cls, inputs, attrs):
        values = cls.combine_inputs(inputs, attrs)
        result = combine_type(values, attrs, local=True)
        if attrs["cast_output"]:
            if isinstance(result, DistributedType) and result.partial is not None:
                raise IRSchemaError("SparseExpertsDownCombine cannot cast before owner reductions")
            return combine_type(values, attrs)
        return result

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {p.name: p.read(arguments) for p in cls.input_parameters}
        inputs = tuple(Node(n, "builtin.var", (), context.types[n]) for n in node.inputs)
        down, _ = cls.combine_inputs(inputs, node.attrs)
        projections = evaluate_down(values, inputs[0].type, down.type, down.attrs, context)
        from triton.flagmega.ir.ops.nn.sparse_experts_combine import evaluate_combine
        return evaluate_combine(projections, values["router_expert_weights"], down.type, node.type, node.attrs, context)

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        down, _ = cls.combine_inputs(inputs, attrs)
        factors = SparseExpertsDown.cost_factors(inputs[:5], down.attrs, down.type)
        combine = SparseExpertsWeightedSum.cost_factors((down, inputs[5]), attrs, return_type)
        if factors is None or combine is None:
            return None
        from triton.flagmega.ir.distributed_type import local_tensor_type
        from triton.flagmega.ir.ops.core import tensor_nbytes
        coeff = local_tensor_type(inputs[5].type) if isinstance(inputs[5].type, DistributedType) else inputs[5].type
        return replace(factors, elementwise_operations=factors.elementwise_operations + combine.elementwise_operations,
                       block_local_memory_load_bytes=factors.block_local_memory_load_bytes + tensor_nbytes(coeff),
                       block_local_memory_store_bytes=combine.block_local_memory_store_bytes)
