# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected expert gate/up projections and SwiGLU, following nncase stages."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn._sparse_experts import (
    EXPERT_SCALE,
    ROUTER_IDS,
    check_routes,
    distributed_inputs,
    element_type,
    floating_tensor,
    lanes,
    normalize_numerics,
    output_tensor,
    pack_result,
    python_dtype_call,
    require_policies,
    require_shape,
    role_axes,
    scale_policy,
    scaled_projection,
    static_elements,
    unpack_value,
    validate_expert_ids,
)


@op_definition("nn.sparse_experts_gate_up", namespace="nn", functional_name="sparse_experts_gate_up",
               display_name="NN.SparseExpertsGateUp")
class SparseExpertsGateUp(OpDefinition):
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
    def normalize_attrs(cls, attributes):
        return normalize_numerics(super().normalize_attrs(attributes), "round_projections", "round_activation")

    @classmethod
    def python_call(cls, node):
        return python_dtype_call(super().python_call(node), "output_dtype")

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = {parameter.name: parameter.type_of(inputs) for parameter in cls.input_parameters}
        tensors = {name: tensor_of(value) for name, value in types.items()}
        q, ids, gate, up = (tensors[name] for name in ("q", "router_expert_ids", "gate_weight", "up_weight"))
        if gate != up or gate.dtype != element_type(q.dtype):
            raise IRSchemaError("SparseExpertsGateUp gate/up weights and activation element dtypes must match.")
        experts, intermediate, hidden = gate.shape
        if q.shape[1] * lanes(q.dtype) != hidden:
            raise IRSchemaError("SparseExpertsGateUp activation hidden extent does not match weights.")
        check_routes(ids, q.shape[0], experts)
        scale_names = ("gate_input_scale", "gate_proj_scale", "up_input_scale", "up_proj_scale")
        for name in scale_names:
            require_shape(tensors[name], (experts, 1), name)
        output = output_tensor(q, (q.shape[0], ids.shape[1], intermediate), attrs)
        placement = distributed_inputs(types)
        if placement is None:
            return output
        broadcast = SBP.broadcast()
        token = types["q"].axis_policies[0]
        scalar_intermediate = types["gate_weight"].axis_policies[1]
        intermediate_policy = scale_policy(scalar_intermediate, 1, lanes(output.dtype))
        require_policies(
            types, {
                "q": (token, broadcast),
                "router_expert_ids": (token, broadcast),
                "gate_weight": (broadcast, scalar_intermediate, broadcast),
                "up_weight": (broadcast, scalar_intermediate, broadcast),
                **{name: (broadcast, broadcast)
                   for name in scale_names},
            })
        role_axes(token, intermediate_policy)
        return DistributedType(output, (token, broadcast, intermediate_policy), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {parameter.name: parameter.read(arguments) for parameter in cls.input_parameters}
        return evaluate_gate_up(values, context.types[cls.q.read(node.inputs)], node.type, node.attrs, context)

    @classmethod
    def cost(cls, node):
        output = tensor_of(node.type)
        routes = static_elements(output.shape[:2])
        # K is an operand dimension and cannot be recovered from Node's input
        # ids alone. cost_factors has operand types and provides the full model.
        return OpCost(bytes_written=tensor_nbytes(output), notes=("selected-experts-gate-up", f"routes={routes}"))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        from triton.flagmega.ir.ops.nn.sparse_experts_down import sparse_stage_cost_factors

        return sparse_stage_cost_factors(cls, inputs, return_type, gate_up=True)


def evaluate_gate_up(values, q_type, output_type, attrs, context):
    q = unpack_value(values["q"], q_type)
    ids = values["router_expert_ids"]
    gate_weight, up_weight = values["gate_weight"], values["up_weight"]
    validate_expert_ids(ids, gate_weight.shape[0])
    result = q.new_empty((*ids.shape, gate_weight.shape[1]), dtype=context.torch.float32)
    dtype = context.torch_dtype(element_type(tensor_of(output_type).dtype))
    for token in range(q.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            gate = scaled_projection(q[token], gate_weight[expert], values["gate_input_scale"][expert],
                                     values["gate_proj_scale"][expert])
            up = scaled_projection(q[token], up_weight[expert], values["up_input_scale"][expert],
                                   values["up_proj_scale"][expert])
            if attrs["round_projections"]:
                gate, up = gate.to(dtype).float(), up.to(dtype).float()
            activated = context.torch.nn.functional.silu(gate)
            if attrs["round_activation"]:
                activated = activated.to(dtype).float()
            result[token, route] = activated * up
    return pack_result(result, output_type, context)


__all__ = ["SparseExpertsGateUp"]
