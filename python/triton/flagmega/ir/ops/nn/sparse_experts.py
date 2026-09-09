# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sparse SwiGLU experts, decomposable into nncase's GateUp/Down stages."""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import Node
from triton.flagmega.ir.ops.core import OpCost, OpCostFactors, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn._sparse_experts import (
    EXPERT_SCALE,
    ROUTER_IDS,
    floating_tensor,
    normalize_numerics,
    python_dtype_call,
)
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp, evaluate_gate_up
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown, evaluate_down


@op_definition("nn.sparse_experts", namespace="nn", functional_name="sparse_experts", display_name="NN.SparseExperts")
class SparseExperts(OpDefinition):
    supports_broadcast_lifting = False
    q = input_parameter(floating_tensor(2, packed=True))
    router_expert_ids = input_parameter(ROUTER_IDS)
    router_expert_weights = input_parameter(floating_tensor(2))
    gate_input_scale = input_parameter(EXPERT_SCALE)
    gate_weight = input_parameter(floating_tensor(3))
    gate_proj_scale = input_parameter(EXPERT_SCALE)
    down_input_scale = input_parameter(EXPERT_SCALE)
    down_weight = input_parameter(floating_tensor(3))
    down_proj_scale = input_parameter(EXPERT_SCALE)
    up_input_scale = input_parameter(EXPERT_SCALE)
    up_weight = input_parameter(floating_tensor(3))
    up_proj_scale = input_parameter(EXPERT_SCALE)
    output_dtype = attribute_parameter(default=None)
    intermediate_dtype = attribute_parameter(default=None)
    round_projections = attribute_parameter(default=False)
    round_activation = attribute_parameter(default=False)
    round_down_projection = attribute_parameter(default=False)
    round_weighted_output = attribute_parameter(default=False)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = normalize_numerics(super().normalize_attrs(attributes), "round_projections", "round_activation",
                                   "round_down_projection", "round_weighted_output")
        attrs["intermediate_dtype"] = normalize_numerics({"output_dtype": attrs["intermediate_dtype"]})["output_dtype"]
        return attrs

    @classmethod
    def python_call(cls, node):
        return python_dtype_call(super().python_call(node), "output_dtype", "intermediate_dtype")

    @classmethod
    def stage_calls(cls, inputs, attrs, *, name="sparse_experts", metadata=None):
        """One source of truth for stage binding in inference and rewriting."""
        operands = {parameter.name: parameter.read(inputs) for parameter in cls.input_parameters}
        gate_inputs = tuple(operands[parameter.name] for parameter in SparseExpertsGateUp.input_parameters)
        gate = SparseExpertsGateUp.prepare(
            gate_inputs, {
                "output_dtype": attrs["intermediate_dtype"],
                "round_projections": attrs["round_projections"],
                "round_activation": attrs["round_activation"],
            })
        gate_node = Node(f"{name}.gate_up", SparseExpertsGateUp.op_name, tuple(value.id for value in gate.inputs),
                         gate.result_type, gate.effect, gate.attrs, metadata or {})
        operands["activations"] = gate_node
        down_inputs = tuple(operands[parameter.name] for parameter in SparseExpertsDown.input_parameters)
        down = SparseExpertsDown.prepare(
            down_inputs, {
                "output_dtype":
                tensor_of(cls.q.type_of(inputs)).dtype if attrs["output_dtype"] is None else attrs["output_dtype"],
                "round_projection":
                attrs["round_down_projection"],
                "round_weighted_output":
                attrs["round_weighted_output"],
            })
        down_node = Node(name, SparseExpertsDown.op_name, tuple(value.id for value in down.inputs), down.result_type,
                         down.effect, down.attrs, metadata or {})
        return gate_node, down_node

    @classmethod
    def infer_type(cls, inputs, attrs):
        return cls.stage_calls(inputs, attrs)[1].type

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {parameter.name: parameter.read(arguments) for parameter in cls.input_parameters}
        operands = tuple(Node(name, "builtin.var", (), context.types[name]) for name in node.inputs)
        gate, down = cls.stage_calls(operands, node.attrs, name=node.id)
        values["activations"] = evaluate_gate_up(values, context.types[cls.q.read(node.inputs)], gate.type, gate.attrs,
                                                 context)
        return evaluate_down(values, gate.type, down.type, down.attrs, context)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)),
                      notes=("sparse-experts", "decompose-gate-up-down"))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        gate, down = cls.stage_calls(inputs, attrs)
        nodes = {value.id: value for value in (*inputs, gate)}
        factors = tuple(
            definition.cost_factors(tuple(nodes[name]
                                          for name in call.inputs), call.attrs, call.type)
            for definition, call in ((SparseExpertsGateUp, gate), (SparseExpertsDown, down)))
        if any(value is None for value in factors):
            return None
        return OpCostFactors(
            **{name: sum(getattr(value, name)
                         for value in factors)
               for name in OpCostFactors.__dataclass_fields__})


__all__ = ["SparseExperts"]
