# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared shape/packing contracts for nncase-style sparse expert stages.

Expert weights remain scalar [expert, N, K]. Vector lanes on activations and
results belong to their last axis; router and per-expert scale axes are scalar.
Dimensions come from operands instead of redundant hidden/chunk-size attributes.
"""

from dataclasses import replace
from math import prod

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.dim_expr import try_div_exactly
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBPSplit, scale_split_units
from triton.flagmega.ir.model import DistributedType, SBP, tensor_type
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import has_rank, is_tensor, TypePattern
from triton.flagmega.ir.types import DType, VectorType, data_type, data_type_to_data


def element_type(dtype):
    return dtype.elem_type if isinstance(dtype, VectorType) else dtype


def lanes(dtype):
    return dtype.lane_count if isinstance(dtype, VectorType) else 1


def floating_tensor(rank, *, packed=False):
    return is_tensor() & has_rank(rank) & TypePattern(
        lambda value: (element_type(tensor_of(value).dtype) in {DType.BFLOAT16, DType.FLOAT32} and
                       (packed or not isinstance(tensor_of(value).dtype, VectorType))),
        "BF16/FP32 tensor" + (" (optional last-axis vector lanes)" if packed else " (scalar elements)"),
    )


ROUTER_IDS = is_tensor() & has_rank(2) & TypePattern(lambda value: tensor_of(value).dtype in {DType.INT32, DType.INT64},
                                                     "INT32/INT64 expert ids")
EXPERT_SCALE = is_tensor() & has_rank(2) & TypePattern(lambda value: tensor_of(value).dtype == DType.FLOAT32,
                                                       "FP32 expert scales")


def normalize_numerics(attrs, *rounding_names):
    attrs = dict(attrs)
    if attrs["output_dtype"] is not None:
        dtype = data_type(attrs["output_dtype"])
        if element_type(dtype) not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("SparseExperts output_dtype must be BF16/FP32, optionally vector packed.")
        attrs["output_dtype"] = data_type_to_data(dtype)
    for name in rounding_names:
        if not isinstance(attrs[name], bool):
            raise IRSchemaError(f"SparseExperts {name} must be boolean.")
    return attrs


def python_dtype_call(call, *names):
    keywords = dict(call.keywords)
    for name in names:
        if keywords.get(name) is not None:
            keywords[name] = data_type(keywords[name])
    return replace(call, keywords=keywords)


def output_tensor(source, shape, attrs):
    dtype = source.dtype if attrs["output_dtype"] is None else data_type(attrs["output_dtype"])
    if element_type(dtype) != element_type(source.dtype):
        raise IRSchemaError("SparseExperts output and activation element dtypes must match.")
    shape = list(shape)
    extent = try_div_exactly(shape[-1], lanes(dtype))
    if extent is None:
        raise IRSchemaError("SparseExperts output extent must be divisible by its vector lane count.")
    shape[-1] = extent
    return tensor_type(dtype, shape)


def require_shape(tensor, shape, name):
    if tensor.shape != tensor_type(tensor.dtype, shape).shape:
        raise IRSchemaError(f"SparseExperts {name} has incompatible shape; expected {tuple(shape)}.")


def check_routes(ids, tokens, experts):
    require_shape(ids, (tokens, ids.shape[1]), "router_expert_ids")
    if ids.shape[1].is_fixed and ids.shape[1].fixed_value <= 0:
        raise IRSchemaError("SparseExperts requires at least one route.")
    if experts.is_fixed and experts.fixed_value <= 0:
        raise IRSchemaError("SparseExperts requires at least one expert.")
    if ids.shape[1].is_fixed and experts.is_fixed and ids.shape[1].fixed_value > experts.fixed_value:
        raise IRSchemaError("SparseExperts route count cannot exceed expert count.")


def distributed_inputs(types):
    placement = placement_of(*types.values())
    if placement is not None:
        if not all(isinstance(value, DistributedType) for value in types.values()):
            raise IRSchemaError("Distributed SparseExperts requires every operand on one placement.")
        if any(value.partial is not None for value in types.values()):
            raise IRSchemaError("SparseExperts requires materialized inputs, not partials.")
    return placement


def scale_policy(policy, numerator, denominator):
    if policy == SBP.broadcast():
        return policy
    scaled = scale_split_units(policy, numerator, denominator) if isinstance(policy, SBPSplit) else None
    if scaled is None:
        raise IRSchemaError("SparseExperts cannot scale split units to the vector lane count.")
    return scaled


def role_axes(*policies):
    groups = tuple(tuple(policy.hierarchy_axes) if isinstance(policy, SBPSplit) else () for policy in policies)
    if len(set(axis for group in groups for axis in group)) != sum(map(len, groups)):
        raise IRSchemaError("SparseExperts token, intermediate and output splits must use disjoint mesh axes.")
    return groups


def require_policies(types, expected):
    for name, policies in expected.items():
        if types[name].axis_policies != tuple(policies):
            raise IRSchemaError(f"SparseExperts {name} has incompatible axis policies; expected {policies}.")


def unpack_value(value, value_type):
    tensor = tensor_of(value_type)
    if isinstance(tensor.dtype, VectorType):
        return unpack_physical(value, tensor.rank, tensor.dtype.lanes, (tensor.rank - 1, ) * len(tensor.dtype.lanes))
    return value


def pack_result(value, output_type, context):
    tensor = tensor_of(output_type)
    value = value.to(context.torch_dtype(element_type(tensor.dtype)))
    if isinstance(tensor.dtype, VectorType):
        return pack_physical(value, tensor.rank, tensor.dtype.lanes, (tensor.rank - 1, ) * len(tensor.dtype.lanes))
    return value


def validate_expert_ids(ids, count):
    if ids.numel() and bool(((ids < 0) | (ids >= count)).any()):
        raise EvaluationError(f"SparseExperts expert ids must be in [0, {count}).")


def scaled_projection(value, weight, input_scale, projection_scale):
    # Match nncase's explicit input-scale / projection-scale arithmetic. BF16
    # import uses unit scales; this is not a quantizing FP8 matmul contract.
    return ((value.float() / input_scale) @ weight.float().T) * input_scale * projection_scale


def static_elements(shape):
    return None if any(not dim.is_fixed for dim in shape) else prod(dim.fixed_value for dim in shape)
