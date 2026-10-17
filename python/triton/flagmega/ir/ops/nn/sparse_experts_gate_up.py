# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected expert gate/up projections and SwiGLU, following nncase stages."""

from math import isfinite

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import SBP, DistributedType, DType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._sparse_experts import (
    EXPERT_SCALE,
    ROUTER_IDS,
    check_routes,
    distributed_inputs,
    element_type,
    expert_weight_tensor,
    floating_tensor,
    lanes,
    normalize_numerics,
    normalize_sparse_quantization,
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
    verify_sparse_quantization,
)


@op_definition("nn.sparse_experts_gate_up", namespace="nn", functional_name="sparse_experts_gate_up",
               display_name="NN.SparseExpertsGateUp")
class SparseExpertsGateUp(OpDefinition):
    supports_broadcast_lifting = False
    dispatched = input_parameter(floating_tensor(3, packed=True))
    router_expert_ids = input_parameter(ROUTER_IDS)
    gate_input_scale = input_parameter(EXPERT_SCALE)
    gate_weight = input_parameter(expert_weight_tensor())
    gate_proj_scale = input_parameter(EXPERT_SCALE)
    up_input_scale = input_parameter(EXPERT_SCALE)
    up_weight = input_parameter(expert_weight_tensor())
    up_proj_scale = input_parameter(EXPERT_SCALE)
    output_dtype = attribute_parameter(default=None)
    round_projections = attribute_parameter(default=False)
    round_activation = attribute_parameter(default=False)
    gate_clamp_max = attribute_parameter(default=None)
    up_clamp = attribute_parameter(default=None)
    weight_block_n = attribute_parameter(default=128)
    weight_block_k = attribute_parameter(default=128)
    activation_quantization = attribute_parameter(default="dynamic_fp8")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = normalize_numerics(super().normalize_attrs(attributes), "round_projections", "round_activation")
        for name in ("gate_clamp_max", "up_clamp"):
            value = attrs[name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not isfinite(value) or not value > 0):
                raise IRSchemaError(f"SparseExpertsGateUp {name} must be a finite positive number or None.")
        return normalize_sparse_quantization(attrs)

    @classmethod
    def python_call(cls, node):
        return python_dtype_call(super().python_call(node), "output_dtype")

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = {parameter.name: parameter.type_of(inputs) for parameter in cls.input_parameters}
        tensors = {name: tensor_of(value) for name, value in types.items()}
        q, ids, gate, up = (tensors[name] for name in ("dispatched", "router_expert_ids", "gate_weight", "up_weight"))
        for name, weight in (("gate", gate), ("up", up)):
            verify_sparse_quantization(q, weight, tensors[f"{name}_proj_scale"], attrs)
        if gate != up or gate.dtype not in {element_type(q.dtype), DType.FLOAT8_E4M3FN}:
            raise IRSchemaError("SparseExpertsGateUp gate/up weights must match each other and be BF16/FP8.")
        experts, intermediate, hidden = gate.shape
        if q.shape[2] * lanes(q.dtype) != hidden:
            raise IRSchemaError("SparseExpertsGateUp activation hidden extent does not match weights.")
        check_routes(ids, q.shape[0], experts)
        require_shape(ids, q.shape[:2], "router_expert_ids")
        scale_names = ("gate_input_scale", "gate_proj_scale", "up_input_scale", "up_proj_scale")
        block_n = int(attrs["weight_block_n"])
        block_k = int(attrs["weight_block_k"])
        for name in scale_names:
            scale = tensors[name]
            if scale.rank == 2:
                require_shape(scale, (experts, 1), name)
            else:
                matrix = gate if name in ("gate_proj_scale", "up_proj_scale") else q
                expected = (experts, (matrix.shape[1].fixed_value + block_n - 1) // block_n,
                            (matrix.shape[2].fixed_value + block_k - 1) // block_k)
                require_shape(scale, expected, name)
        output = output_tensor(q, (q.shape[0], ids.shape[1], intermediate), attrs)
        placement = distributed_inputs(types)
        if placement is None:
            return output
        broadcast = SBP.broadcast()
        token, route, _ = types["dispatched"].axis_policies
        scalar_intermediate = types["gate_weight"].axis_policies[1]
        intermediate_policy = scale_policy(scalar_intermediate, 1, lanes(output.dtype))
        require_policies(
            types, {
                "dispatched": (token, route, broadcast),
                "router_expert_ids": (token, route),
                "gate_weight": (broadcast, scalar_intermediate, broadcast),
                "up_weight": (broadcast, scalar_intermediate, broadcast),
                **{name: (broadcast,) * tensor_of(types[name]).rank
                   for name in scale_names},
            })
        role_axes(token, route, intermediate_policy)
        return DistributedType(output, (token, route, intermediate_policy), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {parameter.name: parameter.read(arguments) for parameter in cls.input_parameters}
        return evaluate_gate_up(values, context.types[cls.dispatched.read(node.inputs)], node.type, node.attrs, context)

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
    q = unpack_value(values["dispatched"], q_type)
    ids = values["router_expert_ids"]
    gate_weight, up_weight = values["gate_weight"], values["up_weight"]
    validate_expert_ids(ids, gate_weight.shape[0])
    result = q.new_empty((*ids.shape, gate_weight.shape[1]), dtype=context.torch.float32)
    dtype = context.torch_dtype(element_type(tensor_of(output_type).dtype))
    for token in range(q.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            gate = scaled_projection(q[token, route], gate_weight[expert], values["gate_input_scale"][expert],
                                     values["gate_proj_scale"][expert],
                                     block_n=int(attrs["weight_block_n"]), block_k=int(attrs["weight_block_k"]),
                                     activation_quantization=attrs.get("activation_quantization", "dynamic_fp8"))
            up = scaled_projection(q[token, route], up_weight[expert], values["up_input_scale"][expert],
                                   values["up_proj_scale"][expert],
                                   block_n=int(attrs["weight_block_n"]), block_k=int(attrs["weight_block_k"]),
                                   activation_quantization=attrs.get("activation_quantization", "dynamic_fp8"))
            if attrs["round_projections"]:
                gate, up = gate.to(dtype).float(), up.to(dtype).float()
            if attrs["gate_clamp_max"] is not None:
                gate = context.torch.clamp(gate, max=attrs["gate_clamp_max"])
            if attrs["up_clamp"] is not None:
                up = context.torch.clamp(up, min=-attrs["up_clamp"], max=attrs["up_clamp"])
            activated = context.torch.nn.functional.silu(gate)
            if attrs["round_activation"]:
                activated = activated.to(dtype).float()
            result[token, route] = activated * up
    return pack_result(result, output_type, context)


__all__ = ["SparseExpertsGateUp"]
