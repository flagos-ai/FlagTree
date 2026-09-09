# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected expert down projection and ordered router-weighted reduction."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.model import DistributedType, SBP, SBPPartial
from triton.flagmega.ir.ops.core import OpCost, OpCostFactors, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
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


@op_definition("nn.sparse_experts_down", namespace="nn", functional_name="sparse_experts_down",
               display_name="NN.SparseExpertsDown")
class SparseExpertsDown(OpDefinition):
    supports_broadcast_lifting = False
    activations = input_parameter(floating_tensor(3, packed=True))
    router_expert_ids = input_parameter(ROUTER_IDS)
    router_expert_weights = input_parameter(floating_tensor(2))
    down_input_scale = input_parameter(EXPERT_SCALE)
    down_weight = input_parameter(floating_tensor(3))
    down_proj_scale = input_parameter(EXPERT_SCALE)
    output_dtype = attribute_parameter(default=None)
    round_projection = attribute_parameter(default=False)
    round_weighted_output = attribute_parameter(default=False)

    @classmethod
    def normalize_attrs(cls, attributes):
        return normalize_numerics(super().normalize_attrs(attributes), "round_projection", "round_weighted_output")

    @classmethod
    def python_call(cls, node):
        return python_dtype_call(super().python_call(node), "output_dtype")

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = {parameter.name: parameter.type_of(inputs) for parameter in cls.input_parameters}
        tensors = {name: tensor_of(value) for name, value in types.items()}
        activation, ids, weights, down = (tensors[name] for name in ("activations", "router_expert_ids",
                                                                     "router_expert_weights", "down_weight"))
        experts, hidden, intermediate = down.shape
        if down.dtype != element_type(activation.dtype):
            raise IRSchemaError("SparseExpertsDown weights and activation element dtypes must match.")
        if activation.shape[2] * lanes(activation.dtype) != intermediate:
            raise IRSchemaError("SparseExpertsDown activation intermediate extent does not match weights.")
        check_routes(ids, activation.shape[0], experts)
        require_shape(ids, activation.shape[:2], "router_expert_ids")
        require_shape(weights, ids.shape, "router_expert_weights")
        for name in ("down_input_scale", "down_proj_scale"):
            require_shape(tensors[name], (experts, 1), name)
        output = output_tensor(activation, (activation.shape[0], hidden), attrs)
        placement = distributed_inputs(types)
        if placement is None:
            return output
        broadcast = SBP.broadcast()
        token, _, intermediate_policy = types["activations"].axis_policies
        scalar_intermediate = scale_policy(intermediate_policy, lanes(activation.dtype), 1)
        scalar_output = types["down_weight"].axis_policies[1]
        output_policy = scale_policy(scalar_output, 1, lanes(output.dtype))
        require_policies(
            types, {
                "activations": (token, broadcast, intermediate_policy),
                "router_expert_ids": (token, broadcast),
                "router_expert_weights": (token, broadcast),
                "down_weight": (broadcast, scalar_output, scalar_intermediate),
                "down_input_scale": (broadcast, broadcast),
                "down_proj_scale": (broadcast, broadcast),
            })
        _, reduction_axes, _ = role_axes(token, intermediate_policy, output_policy)
        if reduction_axes and (attrs["round_projection"] or attrs["round_weighted_output"]):
            raise IRSchemaError("SparseExpertsDown cannot move per-route rounding across a split-K reduction.")
        return DistributedType(output, (token, output_policy), placement,
                               partial=SBPPartial(reduction_axes) if reduction_axes else None)

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {parameter.name: parameter.read(arguments) for parameter in cls.input_parameters}
        return evaluate_down(values, context.types[cls.activations.read(node.inputs)], node.type, node.attrs, context)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)), notes=("selected-experts-down", ))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        return sparse_stage_cost_factors(cls, inputs, return_type, gate_up=False)


def evaluate_down(values, activation_type, output_type, attrs, context):
    activation = unpack_value(values["activations"], activation_type)
    ids, weights = values["router_expert_ids"], values["router_expert_weights"]
    down = values["down_weight"]
    validate_expert_ids(ids, down.shape[0])
    dtype = context.torch_dtype(element_type(tensor_of(output_type).dtype))
    output = activation.new_zeros((activation.shape[0], down.shape[1]), dtype=context.torch.float32)
    # Route order is explicit: the reference MoE sum promotes each stored route
    # to FP32 and adds in top-k order, then casts once to the result dtype.
    for token in range(activation.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            projection = scaled_projection(activation[token, route], down[expert], values["down_input_scale"][expert],
                                           values["down_proj_scale"][expert])
            if attrs["round_projection"]:
                projection = projection.to(dtype).float()
            weighted = projection * weights[token, route].float()
            if attrs["round_weighted_output"]:
                weighted = weighted.to(dtype).float()
            output[token] += weighted
    return pack_result(output, output_type, context)


def sparse_stage_cost_factors(definition, inputs, return_type, *, gate_up):
    """Count selected-expert traffic, never charge a read of the full bank."""

    def local(value):
        return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)

    types = {parameter.name: local(parameter.type_of(inputs)) for parameter in definition.input_parameters}
    output = local(return_type)
    matrix = types["gate_weight" if gate_up else "down_weight"]
    routes = static_elements(types["router_expert_ids"].shape)
    matrix_elements = static_elements(matrix.shape[1:])
    output_elements = static_elements(output.shape)
    if routes is None or matrix_elements is None or output_elements is None:
        return None
    projections = 2 if gate_up else 1
    selected_bytes = projections * routes * matrix_elements * matrix.dtype.itemsize
    small_bytes = sum(
        tensor_nbytes(value) or 0
        for name, value in types.items()
        if not name.endswith("_weight") and not name.endswith("_scale"))
    scale_bytes = projections * routes * 2 * 4
    return OpCostFactors(
        simt_fma_operations=projections * routes * matrix_elements,
        elementwise_operations=output_elements * lanes(output.dtype) *
        (5 if gate_up else types["router_expert_ids"].shape[1].fixed_value),
        chip_global_memory_load_bytes=selected_bytes + small_bytes + scale_bytes,
        chip_global_memory_store_bytes=tensor_nbytes(output) or 0,
    )


__all__ = ["SparseExpertsDown"]
