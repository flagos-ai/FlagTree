# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Route-weighted combination, with explicit local and owner-reduction stages."""

from dataclasses import replace
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DType, DistributedType, SBP, VectorType, tensor_type
from triton.flagmega.ir.distributed_type import ReduceOp
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.ops.core import OpDefinition, OpCost, OpCostFactors, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn._sparse_experts import (
    element_type, floating_tensor, lanes, normalize_numerics, pack_result, python_dtype_call,
    require_shape, role_axes, static_elements, unpack_value,
)
from triton.flagmega.ir.types import data_type


def combine_type(inputs, attrs, *, local=False):
    values, weights = (node.type for node in inputs)
    value, weight = tensor_of(values), tensor_of(weights)
    require_shape(weight, value.shape[:2], "router_expert_weights")
    dtype = value.dtype if attrs["output_dtype"] is None else data_type(attrs["output_dtype"])
    if local:
        dtype = VectorType(DType.FLOAT32, dtype.lanes) if isinstance(dtype, VectorType) else DType.FLOAT32
    hidden = value.shape[2] * lanes(value.dtype)
    from triton.flagmega.ir.dim_expr import try_div_exactly
    extent = try_div_exactly(hidden, lanes(dtype))
    if extent is None:
        raise IRSchemaError("SparseExpertsCombine output lanes must divide hidden size")
    output = tensor_type(dtype, (value.shape[0], extent))
    if not isinstance(values, DistributedType):
        if isinstance(weights, DistributedType):
            raise IRSchemaError("SparseExpertsCombine operands must share a placement")
        return output
    if (not isinstance(weights, DistributedType) or weights.placement != values.placement
            or weights.partial is not None or values.exclusive is not None or weights.exclusive is not None
            or weights.axis_policies != values.axis_policies[:2]):
        raise IRSchemaError("SparseExpertsCombine coefficients must follow token/route owners")
    if values.partial is not None and (values.partial.reduce_op is not ReduceOp.SUM
                                       or element_type(value.dtype) is not DType.FLOAT32
                                       or attrs["round_weighted_output"]):
        raise IRSchemaError("SparseExpertsCombine needs materialized input before per-route rounding")
    token, route, hidden_policy = values.axis_policies
    _, route_axes, _ = role_axes(token, route, hidden_policy)
    from triton.flagmega.ir.ops.nn._sparse_experts import scale_policy
    hidden_policy = scale_policy(hidden_policy, lanes(value.dtype), lanes(dtype))
    axes = tuple(sorted(set(route_axes) | set(values.partial.axes if values.partial else ())))
    return DistributedType(output, (token, hidden_policy), values.placement,
                           partial=SBP.partial(axes) if local and axes else None)


@op_definition("nn.sparse_experts_combine", namespace="nn", functional_name="sparse_experts_combine",
               display_name="NN.SparseExpertsCombine")
class SparseExpertsCombine(OpDefinition):
    projections = input_parameter(floating_tensor(3, packed=True))
    router_expert_weights = input_parameter(floating_tensor(2))
    output_dtype = attribute_parameter(default=None)
    round_weighted_output = attribute_parameter(default=False)
    supports_broadcast_lifting = False

    @classmethod
    def normalize_attrs(cls, attrs):
        return normalize_numerics(super().normalize_attrs(attrs), "round_weighted_output")

    @classmethod
    def python_call(cls, node):
        return python_dtype_call(super().python_call(node), "output_dtype")

    @classmethod
    def infer_type(cls, inputs, attrs):
        return combine_type(inputs, attrs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        return evaluate_combine(arguments[0], arguments[1], context.types[node.inputs[0]], node.type, node.attrs, context)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("route-weighted-combine",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        local_type = combine_type(inputs, attrs, local=True)
        factors = SparseExpertsWeightedSum.cost_factors(inputs, attrs, local_type)
        if factors is None:
            return None
        local = local_tensor_type(local_type) if isinstance(local_type, DistributedType) else local_type
        result = local_tensor_type(return_type) if isinstance(return_type, DistributedType) else return_type
        fan = prod(local_type.placement.hierarchy[axis] for axis in local_type.partial.axes) if isinstance(local_type, DistributedType) and local_type.partial else 1
        size = tensor_nbytes(local)
        cast = local.dtype != result.dtype
        return replace(factors,
                       elementwise_operations=factors.elementwise_operations + static_elements(local.shape) * lanes(local.dtype) * (fan - 1),
                       chip_global_memory_load_bytes=size * fan if fan > 1 else 0,
                       chip_global_memory_store_bytes=size if fan > 1 else 0,
                       block_local_memory_load_bytes=factors.block_local_memory_load_bytes + (size if cast else 0),
                       block_local_memory_store_bytes=factors.block_local_memory_store_bytes + (tensor_nbytes(result) if cast else 0),
                       grid_synchronizations=int(fan > 1))


@op_definition("nn.sparse_experts_weighted_sum", namespace="nn", functional_name="sparse_experts_weighted_sum",
               display_name="NN.SparseExpertsWeightedSum")
class SparseExpertsWeightedSum(SparseExpertsCombine):
    """Only this owner's route slots; K/route owner sums remain Partial."""

    projections = input_parameter(floating_tensor(3, packed=True))
    router_expert_weights = input_parameter(floating_tensor(2))
    output_dtype = attribute_parameter(default=None)
    round_weighted_output = attribute_parameter(default=False)

    @classmethod
    def infer_type(cls, inputs, attrs):
        return combine_type(inputs, attrs, local=True)

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        def local(value):
            return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)
        values, weights = (local(node.type) for node in inputs)
        output = local(return_type)
        elements = static_elements(values.shape)
        if elements is None or tensor_nbytes(output) is None:
            return None
        return OpCostFactors(elementwise_operations=2 * elements * lanes(values.dtype),
                             block_local_memory_load_bytes=tensor_nbytes(values) + tensor_nbytes(weights),
                             block_local_memory_store_bytes=tensor_nbytes(output))


def evaluate_combine(projections, coefficients, input_type, output_type, attrs, context):
    value = unpack_value(projections, input_type).float()
    dtype = element_type(tensor_of(input_type).dtype if attrs["output_dtype"] is None else data_type(attrs["output_dtype"]))
    result = value.new_zeros((value.shape[0], value.shape[2]))
    for route in range(value.shape[1]):
        weighted = value[:, route] * coefficients[:, route, None].float()
        if attrs["round_weighted_output"]:
            weighted = weighted.to(context.torch_dtype(dtype)).float()
        result += weighted
    return pack_result(result, output_type, context)
