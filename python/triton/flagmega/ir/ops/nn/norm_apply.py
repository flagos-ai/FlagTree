# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Apply LayerNorm/RMSNorm from explicit additive statistics."""

from __future__ import annotations

from dataclasses import replace
from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPPartial
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._norm import (
    is_float_dtype,
    norm_apply_value,
    normalize_axis,
    repack_default_vector,
    stats_tensor_type,
    unpack_default_vector,
)
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "nn.norm_apply",
    namespace="nn",
    functional_name="norm_apply",
    display_name="NN.NormApply",
)
class NormApply(OpDefinition):
    value = input_parameter(is_tensor(), name="input")
    stats = input_parameter(is_tensor())
    scale = input_parameter(is_tensor())
    bias = input_parameter(is_tensor())
    axis = attribute_parameter()
    epsilon = attribute_parameter()
    use_mean = attribute_parameter()
    round_before_scale = attribute_parameter(default=False)
    inplace_input_parameters = (value,)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("NormApply axis must be an integer.")
        epsilon = float(attrs["epsilon"])
        if epsilon <= 0:
            raise IRSchemaError("NormApply epsilon must be positive.")
        if not isinstance(attrs["round_before_scale"], bool):
            raise IRSchemaError("NormApply round_before_scale must be boolean.")
        return {"axis": axis, "epsilon": epsilon, "use_mean": bool(attrs["use_mean"]),
                "round_before_scale": attrs["round_before_scale"]}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        stats_type = cls.stats.type_of(inputs)
        scale_type = cls.scale.type_of(inputs)
        bias_type = cls.bias.type_of(inputs)
        value = tensor_of(value_type)
        stats = tensor_of(stats_type)
        scale = tensor_of(scale_type)
        bias = tensor_of(bias_type)
        axis = normalize_axis(int(attrs["axis"]), value.rank)
        if not all(is_float_dtype(item.dtype) for item in (value, scale, bias)):
            raise IRSchemaError("NormApply input, scale and bias must be floating point.")
        expected_stats = stats_tensor_type(value, axis, bool(attrs["use_mean"]))
        if stats != expected_stats:
            raise IRSchemaError(
                f"NormApply stats type {stats!r} does not match expected {expected_stats!r}.")
        _require_suffix_parameter(value, scale, axis, "scale")
        _require_suffix_parameter(value, bias, axis, "bias")
        distributed = tuple(
            isinstance(item, DistributedType)
            for item in (value_type, stats_type, scale_type, bias_type)
        )
        if not any(distributed):
            return value
        if not all(distributed):
            raise IRSchemaError("Distributed NormApply requires all arguments to be distributed tensors.")
        assert isinstance(value_type, DistributedType)
        assert isinstance(stats_type, DistributedType)
        assert isinstance(scale_type, DistributedType)
        assert isinstance(bias_type, DistributedType)
        if len({item.placement for item in (value_type, stats_type, scale_type, bias_type)}) != 1:
            raise IRSchemaError("NormApply argument placements must be equal.")
        if any(
            item.partial is not None
            or any(isinstance(policy, SBPPartial) for policy in item.axis_policies)
            for item in (value_type, stats_type, scale_type, bias_type)
        ):
            raise IRSchemaError("NormApply requires non-partial input, stats, scale and bias.")
        if len(stats_type.axis_policies) != value.rank + 1 or not isinstance(
            stats_type.axis_policies[0], SBPBroadCast
        ):
            raise IRSchemaError("NormApply stats policies must match NormStats output rank.")
        for index, input_policy in enumerate(value_type.axis_policies):
            stats_policy = stats_type.axis_policies[index + 1]
            if index < axis:
                if input_policy != stats_policy:
                    raise IRSchemaError(
                        f"NormApply stats policy on outer axis {index} must match the input.")
            elif not isinstance(stats_policy, SBPBroadCast):
                raise IRSchemaError(
                    f"NormApply stats policy on normalized axis {index} must be broadcast.")
        parameter_rank = value.rank - axis
        if scale.rank != parameter_rank or bias.rank != parameter_rank:
            raise IRSchemaError("Distributed NormApply parameters must have normalized-suffix rank.")
        for index in range(parameter_rank):
            input_policy = value_type.axis_policies[axis + index]
            if scale_type.axis_policies[index] != input_policy:
                raise IRSchemaError(f"NormApply scale policy {index} must match the input suffix.")
            if bias_type.axis_policies[index] != input_policy:
                raise IRSchemaError(f"NormApply bias policy {index} must match the input suffix.")
        return replace(value_type, partial=None)

    @classmethod
    def evaluate(cls, node, arguments, context):
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        scale_type = tensor_of(context.types[cls.scale.read(node.inputs)])
        bias_type = tensor_of(context.types[cls.bias.read(node.inputs)])
        value = unpack_default_vector(cls.value.read(arguments), input_type)
        scale = unpack_default_vector(cls.scale.read(arguments), scale_type)
        bias = unpack_default_vector(cls.bias.read(arguments), bias_type)
        output = norm_apply_value(
            value,
            cls.stats.read(arguments),
            scale,
            bias,
            axis=int(node.attrs["axis"]),
            epsilon=float(node.attrs["epsilon"]),
            use_mean=bool(node.attrs["use_mean"]),
            round_before_scale=bool(node.attrs.get("round_before_scale", False)),
        )
        return repack_default_vector(output, input_type)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = tensor_of(node.type)
        elements = tensor_elements(output)
        return OpCost(
            flops=None if elements is None else elements * (7 if node.attrs["use_mean"] else 5),
            bytes_read=None,
            bytes_written=tensor_nbytes(output),
            notes=("normalization-apply",),
        )

    @classmethod
    def cost_factors(
        cls,
        inputs: Sequence[Node],
        attrs: Mapping[str, object],
        return_type: IRType,
    ) -> OpCostFactors | None:
        tensors = tuple(
            _local_cost_tensor(value)
            for value in (
                cls.value.type_of(inputs),
                cls.stats.type_of(inputs),
                cls.scale.type_of(inputs),
                cls.bias.type_of(inputs),
                return_type,
            )
        )
        if any(
            not dimension.is_fixed
            for tensor in tensors
            for dimension in tensor.shape
        ):
            return None
        input_tensor, stats_tensor, scale_tensor, bias_tensor, output_tensor = tensors
        input_elements = prod(dimension.fixed_value for dimension in input_tensor.shape)
        return OpCostFactors(
            cpu_cycles=input_elements * (7 if attrs["use_mean"] else 5),
            block_local_memory_load_bytes=sum(
                _fixed_tensor_nbytes(tensor)
                for tensor in (input_tensor, stats_tensor, scale_tensor, bias_tensor)
            ),
            block_local_memory_store_bytes=_fixed_tensor_nbytes(output_tensor),
        )


def _require_suffix_parameter(value: TensorType, parameter: TensorType, axis: int, name: str) -> None:
    suffix = value.shape[axis:]
    if parameter.rank != len(suffix):
        raise IRSchemaError(f"NormApply {name} rank must equal the normalized suffix rank.")
    for expected, actual in zip(suffix, parameter.shape):
        if actual != expected and not (actual.is_fixed and actual.fixed_value == 1):
            raise IRSchemaError(f"NormApply {name} is not broadcastable to the normalized suffix.")


def _local_cost_tensor(value: IRType) -> TensorType:
    from triton.flagmega.ir.distributed_type import local_tensor_type

    return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)


def _fixed_tensor_nbytes(value: TensorType) -> int:
    return prod(dimension.fixed_value for dimension in value.shape) * value.dtype.itemsize


__all__ = ["NormApply"]
