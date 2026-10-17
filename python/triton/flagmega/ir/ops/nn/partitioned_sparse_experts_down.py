# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected block-FP8 projections with a material reduction-partition axis."""

from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.local_shard import aggregate_active_elements
from triton.flagmega.ir.model import SBP, DistributedType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.math._block_scaled import normalize_output_lanes
from triton.flagmega.ir.ops.nn._sparse_experts import (
    EXPERT_SCALE,
    ROUTER_IDS,
    check_routes,
    distributed_inputs,
    element_type,
    floating_tensor,
    lanes,
    normalize_sparse_quantization,
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
from triton.flagmega.ir.type_pattern import TypePattern, has_rank, is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition("nn.partitioned_sparse_experts_down", namespace="nn",
               functional_name="partitioned_sparse_experts_down", display_name="NN.PartitionedSparseExpertsDown")
class PartitionedSparseExpertsDown(OpDefinition):
    """Project each [T,R,P,K] partition into FP32 [P,T,R,H].

    The partition is an ordinary tensor axis, never an implicit Partial sum.
    Consumers can round route weighting and the local sum before reducing P.
    Scalar FP8 weights stay [E,P,H,K]; output_lanes packs only final H.
    """

    supports_broadcast_lifting = False
    activations = input_parameter(floating_tensor(4, packed=True))
    router_expert_ids = input_parameter(ROUTER_IDS)
    down_input_scale = input_parameter(EXPERT_SCALE)
    down_weight = input_parameter(is_tensor() & has_rank(4) & TypePattern(
        lambda value: tensor_of(value).dtype == DType.FLOAT8_E4M3FN, "scalar FP8 partitioned expert weights"))
    down_proj_scale = input_parameter(is_tensor() & has_rank(4) & TypePattern(
        lambda value: tensor_of(value).dtype == DType.FLOAT32, "F32 partitioned block scales"))
    weight_block_n = attribute_parameter(default=128)
    weight_block_k = attribute_parameter(default=128)
    activation_quantization = attribute_parameter(default="dynamic_fp8")
    round_projection = attribute_parameter(default=False)
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = normalize_sparse_quantization(super().normalize_attrs(attributes))
        if type(attrs["round_projection"]) is not bool:
            raise IRSchemaError("PartitionedSparseExpertsDown round_projection must be boolean.")
        attrs["output_lanes"] = normalize_output_lanes(attrs["output_lanes"], "PartitionedSparseExpertsDown")
        return attrs

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items() if key != "output_lanes" or value}

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = {parameter.name: parameter.type_of(inputs) for parameter in cls.input_parameters}
        tensors = {name: tensor_of(value) for name, value in types.items()}
        activation, weight = tensors["activations"], tensors["down_weight"]
        experts, partitions, hidden, reduction = weight.shape
        if any(not value.is_fixed or value.fixed_value <= 0 for value in (partitions, hidden, reduction)):
            raise IRSchemaError("PartitionedSparseExpertsDown requires fixed positive P/H/K extents.")
        if activation.shape[2] != partitions or activation.shape[3] * lanes(activation.dtype) != reduction:
            raise IRSchemaError("PartitionedSparseExpertsDown activation P/K extents must match the weight.")
        if attrs.get("activation_quantization") == "dynamic_fp8_ue8m0" and element_type(activation.dtype) != DType.BFLOAT16:
            raise IRSchemaError("PartitionedSparseExpertsDown UE8M0 requires BF16 activations.")
        check_routes(tensors["router_expert_ids"], activation.shape[0], experts)
        require_shape(tensors["router_expert_ids"], activation.shape[:2], "router_expert_ids")
        require_shape(tensors["down_input_scale"], (experts, 1), "down_input_scale")
        bn, bk = attrs["weight_block_n"], attrs["weight_block_k"]
        require_shape(tensors["down_proj_scale"], (experts, partitions, (hidden.fixed_value + bn - 1) // bn,
                                                  (reduction.fixed_value + bk - 1) // bk), "down_proj_scale")
        width = prod(attrs.get("output_lanes", ()))
        if hidden.fixed_value % width:
            raise IRSchemaError("PartitionedSparseExpertsDown output lanes must divide H.")
        dtype = VectorType(DType.FLOAT32, attrs["output_lanes"]) if width != 1 or attrs.get("output_lanes") else DType.FLOAT32
        output = tensor_type(dtype, (partitions, *activation.shape[:2], hidden.fixed_value // width))
        placement = distributed_inputs(types)
        if placement is None:
            return output
        b = SBP.broadcast()
        token, route, partition, k = types["activations"].axis_policies
        if k != b or types["down_weight"].axis_policies[3] != b:
            raise IRSchemaError("PartitionedSparseExpertsDown requires complete K inside each partition; split P instead.")
        scalar_hidden = types["down_weight"].axis_policies[2]
        output_hidden = scale_policy(scalar_hidden, 1, width)
        role_axes(partition, token, route, output_hidden)
        require_policies(types, {
            "activations": (token, route, partition, b), "router_expert_ids": (token, route),
            "down_weight": (b, partition, scalar_hidden, b), "down_input_scale": (b, b),
            "down_proj_scale": (b, b, b, b),
        })
        return DistributedType(output, (partition, token, route, output_hidden), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = {parameter.name: parameter.read(arguments) for parameter in cls.input_parameters}
        activation = unpack_value(values["activations"], context.types[node.inputs[0]])
        ids, weights = values["router_expert_ids"], values["down_weight"]
        validate_expert_ids(ids, weights.shape[0])
        p, t, r, h = weights.shape[1], activation.shape[0], activation.shape[1], weights.shape[2]
        result = activation.new_empty((p, t, r, h), dtype=context.torch.float32)
        dtype = context.torch_dtype(element_type(tensor_of(context.types[node.inputs[0]]).dtype))
        for partition in range(p):
            for token in range(t):
                for route in range(r):
                    expert = int(ids[token, route])
                    projected = scaled_projection(activation[token, route, partition], weights[expert, partition],
                        values["down_input_scale"][expert], values["down_proj_scale"][expert, partition],
                        block_n=node.attrs["weight_block_n"], block_k=node.attrs["weight_block_k"],
                        activation_quantization=node.attrs.get("activation_quantization", "dynamic_fp8"))
                    if node.attrs["round_projection"]:
                        projected = projected.to(dtype).float()
                    result[partition, token, route] = projected
        return pack_result(result, node.type, context)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)), notes=("partitioned-selected-expert-projection",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        kinds = {parameter.name: parameter.type_of(inputs) for parameter in cls.input_parameters}
        weight = tensor_of(kinds["down_weight"])
        ids = tensor_of(kinds["router_expert_ids"])
        domain = tensor_type(weight.dtype, (weight.shape[1], *ids.shape, *weight.shape[2:]))
        if isinstance(return_type, DistributedType):
            hidden = kinds["down_weight"].axis_policies[2]
            domain = DistributedType(domain, (*return_type.axis_policies[:3], hidden, SBP.broadcast()), return_type.placement)
            selected = aggregate_active_elements(domain)
        else:
            selected = static_elements(domain.shape)
        if selected is None:
            return None
        local_output = local_tensor_type(return_type) if isinstance(return_type, DistributedType) else return_type
        return OpCostFactors(simt_fma_operations=selected,
            chip_aggregate_memory_load_bytes=selected * weight.dtype.itemsize,
            block_local_memory_store_bytes=tensor_nbytes(local_output) or 0)


__all__ = ["PartitionedSparseExpertsDown"]
