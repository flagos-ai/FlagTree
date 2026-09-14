# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected expert down projections, retaining every route in FP32."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.local_shard import aggregate_active_elements
from triton.flagmega.ir.model import DistributedType, SBP, SBPPartial, DType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpCostFactors, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn._sparse_experts import (
    EXPERT_SCALE,
    ROUTER_IDS,
    check_routes,
    distributed_inputs,
    element_type,
    floating_tensor,
    lanes,
    pack_result,
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
    down_input_scale = input_parameter(EXPERT_SCALE)
    down_weight = input_parameter(floating_tensor(3))
    down_proj_scale = input_parameter(EXPERT_SCALE)
    round_projection = attribute_parameter(default=False)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if type(attrs["round_projection"]) is not bool:
            raise IRSchemaError("SparseExpertsDown round_projection must be boolean")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = {parameter.name: parameter.type_of(inputs) for parameter in cls.input_parameters}
        tensors = {name: tensor_of(value) for name, value in types.items()}
        activation, ids, down = (tensors[name] for name in ("activations", "router_expert_ids", "down_weight"))
        experts, hidden, intermediate = down.shape
        if down.dtype != element_type(activation.dtype):
            raise IRSchemaError("SparseExpertsDown weights and activation element dtypes must match.")
        if activation.shape[2] * lanes(activation.dtype) != intermediate:
            raise IRSchemaError("SparseExpertsDown activation intermediate extent does not match weights.")
        check_routes(ids, activation.shape[0], experts)
        require_shape(ids, activation.shape[:2], "router_expert_ids")
        for name in ("down_input_scale", "down_proj_scale"):
            require_shape(tensors[name], (experts, 1), name)
        output = tensor_type(DType.FLOAT32, (activation.shape[0], ids.shape[1], hidden))
        placement = distributed_inputs(types)
        if placement is None:
            return output
        broadcast = SBP.broadcast()
        token, route, intermediate_policy = types["activations"].axis_policies
        scalar_intermediate = scale_policy(intermediate_policy, lanes(activation.dtype), 1)
        scalar_output = types["down_weight"].axis_policies[1]
        output_policy = scale_policy(scalar_output, 1, lanes(output.dtype))
        require_policies(
            types, {
                "activations": (token, route, intermediate_policy),
                "router_expert_ids": (token, route),
                "down_weight": (broadcast, scalar_output, scalar_intermediate),
                "down_input_scale": (broadcast, broadcast),
                "down_proj_scale": (broadcast, broadcast),
            })
        _, _, reduction_axes, _ = role_axes(token, route, intermediate_policy, output_policy)
        if reduction_axes and attrs["round_projection"]:
            raise IRSchemaError("SparseExpertsDown cannot move per-route rounding across a split-K reduction.")
        return DistributedType(output, (token, route, output_policy), placement,
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
    ids = values["router_expert_ids"]
    down = values["down_weight"]
    validate_expert_ids(ids, down.shape[0])
    dtype = context.torch_dtype(element_type(tensor_of(activation_type).dtype))
    output = activation.new_empty((*ids.shape, down.shape[1]), dtype=context.torch.float32)
    for token in range(activation.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            projection = scaled_projection(activation[token, route], down[expert], values["down_input_scale"][expert],
                                           values["down_proj_scale"][expert])
            if attrs["round_projection"]:
                projection = projection.to(dtype).float()
            output[token, route] = projection
    return pack_result(output, output_type, context)


def sparse_stage_cost_factors(definition, inputs, return_type, *, gate_up):
    """Count selected-expert traffic, never charge a read of the full bank."""

    def local(value):
        return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)

    operand_types = {parameter.name: parameter.type_of(inputs) for parameter in definition.input_parameters}
    types = {name: local(value) for name, value in operand_types.items()}
    output = local(return_type)
    matrix = types["gate_weight" if gate_up else "down_weight"]
    routes = static_elements(types["router_expert_ids"].shape)
    matrix_elements = static_elements(matrix.shape[1:])
    output_elements = static_elements(output.shape)
    if routes is None or matrix_elements is None or output_elements is None:
        return None
    projections = 2 if gate_up else 1
    ids_type = operand_types["router_expert_ids"]
    matrix_type = operand_types["gate_weight" if gate_up else "down_weight"]
    if isinstance(ids_type, DistributedType):
        # Each selected matrix is visited once per active token/route and N/K
        # owner. The expert-bank axis is indexed, not scanned or partitioned.
        domain = DistributedType(
            tensor_type(matrix.dtype, (*ids_type.tensor.shape, *matrix_type.tensor.shape[1:])),
            (*ids_type.axis_policies, *matrix_type.axis_policies[1:]), ids_type.placement)
        selected_elements = aggregate_active_elements(domain)
        scale_routes = aggregate_active_elements(ids_type)
    else:
        selected_elements = routes * matrix_elements
        scale_routes = routes
    if selected_elements is None or scale_routes is None:
        return None
    selected_bytes = projections * selected_elements * matrix.dtype.itemsize
    small_bytes = sum(
        tensor_nbytes(value) or 0
        for name, value in types.items()
        if not name.endswith("_weight") and not name.endswith("_scale"))
    scale_bytes = projections * scale_routes * 2 * 4
    return OpCostFactors(
        simt_fma_operations=projections * routes * matrix_elements,
        elementwise_operations=output_elements * lanes(output.dtype) *
        (5 if gate_up else 1),
        chip_aggregate_memory_load_bytes=selected_bytes + scale_bytes,
        block_local_memory_load_bytes=small_bytes,
        block_local_memory_store_bytes=tensor_nbytes(output) or 0,
    )


__all__ = ["SparseExpertsDown"]
