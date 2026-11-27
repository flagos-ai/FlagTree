# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RMSNorm from explicit statistics, gated by an activated second operand.

This is the reference ``FusedRMSNormGated`` contract (KDA's ``o_norm``): the
normalized value, the affine parameters and the gate activation all evaluate in
FP32, and the only rounding boundary is the declared ``output_dtype``.  Writing
it as ``norm_apply`` plus ``cast``/``sigmoid``/``mul``/``cast`` cannot express
that, because each materialized node publishes its own dtype: the gate's
``sigmoid`` would round to the gate's own dtype before the multiply.
"""

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
    gate_activation_value,
    is_float_dtype,
    norm_apply_wide,
    normalize_axis,
    stats_tensor_type,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType


_ACTIVATIONS = frozenset({"sigmoid", "silu", "swish"})


@op_definition(
    "nn.gated_norm_apply",
    namespace="nn",
    functional_name="gated_norm_apply",
    display_name="NN.GatedNormApply",
)
class GatedNormApply(OpDefinition):
    """``norm_apply(value) * activation(gate)`` with one FP32 interior."""

    const_evaluable = True
    value = input_parameter(is_tensor(), name="input")
    stats = input_parameter(is_tensor())
    scale = input_parameter(is_tensor())
    bias = input_parameter(is_tensor())
    gate = input_parameter(is_tensor())
    axis = attribute_parameter()
    epsilon = attribute_parameter()
    use_mean = attribute_parameter()
    activation = attribute_parameter(default="sigmoid")
    round_before_scale = attribute_parameter(default=False)
    output_dtype = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("GatedNormApply axis must be an integer.")
        epsilon = float(attrs["epsilon"])
        if epsilon <= 0:
            raise IRSchemaError("GatedNormApply epsilon must be positive.")
        if attrs["activation"] not in _ACTIVATIONS:
            raise IRSchemaError("GatedNormApply activation must be sigmoid, silu or swish.")
        if not isinstance(attrs["round_before_scale"], bool):
            raise IRSchemaError("GatedNormApply round_before_scale must be boolean.")
        output_dtype = attrs["output_dtype"]
        if output_dtype is not None:
            try:
                output_dtype = DType(output_dtype)
            except (ValueError, TypeError) as error:
                raise IRSchemaError(
                    "GatedNormApply output_dtype must be bfloat16 or float32.") from error
            if output_dtype not in {DType.BFLOAT16, DType.FLOAT32}:
                raise IRSchemaError("GatedNormApply output_dtype must be bfloat16 or float32.")
        return {**({"output_dtype": output_dtype.value} if output_dtype is not None else {}),
                "axis": axis, "epsilon": epsilon, "use_mean": bool(attrs["use_mean"]),
                "activation": str(attrs["activation"]),
                "round_before_scale": attrs["round_before_scale"]}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        stats_type = cls.stats.type_of(inputs)
        scale_type = cls.scale.type_of(inputs)
        bias_type = cls.bias.type_of(inputs)
        gate_type = cls.gate.type_of(inputs)
        value = tensor_of(value_type)
        stats = tensor_of(stats_type)
        scale = tensor_of(scale_type)
        bias = tensor_of(bias_type)
        gate = tensor_of(gate_type)
        axis = normalize_axis(int(attrs["axis"]), value.rank)
        if not all(is_float_dtype(item.dtype) for item in (value, scale, bias, gate)):
            raise IRSchemaError(
                "GatedNormApply input, scale, bias and gate must be floating point.")
        if isinstance(value.dtype, VectorType) or isinstance(gate.dtype, VectorType):
            raise IRSchemaError("GatedNormApply does not support vector elements.")
        expected_stats = stats_tensor_type(value, axis, bool(attrs["use_mean"]))
        if stats != expected_stats:
            raise IRSchemaError(
                f"GatedNormApply stats type {stats!r} does not match expected {expected_stats!r}.")
        _require_suffix_parameter(value, scale, axis, "scale")
        _require_suffix_parameter(value, bias, axis, "bias")
        # The gate is elementwise over the whole value, not over the normalized
        # suffix: the reference gates every (token, head, channel) separately.
        if gate.shape != value.shape:
            raise IRSchemaError("GatedNormApply gate shape must equal the input shape.")
        output = value
        if attrs.get("output_dtype") is not None:
            output = replace(value, dtype=DType(attrs["output_dtype"]))
        distributed = tuple(
            isinstance(item, DistributedType)
            for item in (value_type, stats_type, scale_type, bias_type, gate_type)
        )
        if not any(distributed):
            return output
        if not all(distributed):
            raise IRSchemaError(
                "Distributed GatedNormApply requires all arguments to be distributed tensors.")
        assert isinstance(value_type, DistributedType)
        assert isinstance(stats_type, DistributedType)
        assert isinstance(scale_type, DistributedType)
        assert isinstance(bias_type, DistributedType)
        assert isinstance(gate_type, DistributedType)
        operands = (value_type, stats_type, scale_type, bias_type, gate_type)
        if len({item.placement for item in operands}) != 1:
            raise IRSchemaError("GatedNormApply argument placements must be equal.")
        if any(
            item.partial is not None
            or any(isinstance(policy, SBPPartial) for policy in item.axis_policies)
            for item in operands
        ):
            raise IRSchemaError("GatedNormApply requires non-partial operands.")
        if len(stats_type.axis_policies) != value.rank + 1 or not isinstance(
            stats_type.axis_policies[0], SBPBroadCast
        ):
            raise IRSchemaError("GatedNormApply stats policies must match NormStats output rank.")
        for index, input_policy in enumerate(value_type.axis_policies):
            stats_policy = stats_type.axis_policies[index + 1]
            if index < axis:
                if input_policy != stats_policy:
                    raise IRSchemaError(
                        f"GatedNormApply stats policy on outer axis {index} must match the input.")
            elif not isinstance(stats_policy, SBPBroadCast):
                raise IRSchemaError(
                    f"GatedNormApply stats policy on normalized axis {index} must be broadcast.")
        if gate_type.axis_policies != value_type.axis_policies:
            raise IRSchemaError("GatedNormApply gate policies must match the input; insert Boxing.")
        parameter_rank = value.rank - axis
        if scale.rank != parameter_rank or bias.rank != parameter_rank:
            raise IRSchemaError(
                "Distributed GatedNormApply parameters must have normalized-suffix rank.")
        for index in range(parameter_rank):
            input_policy = value_type.axis_policies[axis + index]
            if scale_type.axis_policies[index] != input_policy:
                raise IRSchemaError(
                    f"GatedNormApply scale policy {index} must match the input suffix.")
            if bias_type.axis_policies[index] != input_policy:
                raise IRSchemaError(
                    f"GatedNormApply bias policy {index} must match the input suffix.")
        return replace(value_type, tensor=output, partial=None)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        normalized = norm_apply_wide(
            value,
            cls.stats.read(arguments),
            cls.scale.read(arguments),
            cls.bias.read(arguments),
            axis=int(node.attrs["axis"]),
            epsilon=float(node.attrs["epsilon"]),
            use_mean=bool(node.attrs["use_mean"]),
            round_before_scale=bool(node.attrs.get("round_before_scale", False)),
        )
        gated = normalized * gate_activation_value(
            cls.gate.read(arguments), str(node.attrs.get("activation", "sigmoid")))
        return gated.to(dtype=context.torch_dtype(tensor_of(node.type).dtype))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = tensor_of(node.type)
        elements = tensor_elements(output)
        return OpCost(
            flops=None if elements is None else elements * (10 if node.attrs["use_mean"] else 8),
            bytes_read=None,
            bytes_written=tensor_nbytes(output),
            notes=("gated-normalization-apply",),
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
                cls.gate.type_of(inputs),
                return_type,
            )
        )
        if any(not dimension.is_fixed for tensor in tensors for dimension in tensor.shape):
            return None
        input_tensor = tensors[0]
        input_elements = prod(dimension.fixed_value for dimension in input_tensor.shape)
        return OpCostFactors(
            cpu_cycles=input_elements * (10 if attrs["use_mean"] else 8),
            block_local_memory_load_bytes=sum(
                _fixed_tensor_nbytes(tensor) for tensor in tensors[:-1]
            ),
            block_local_memory_store_bytes=_fixed_tensor_nbytes(tensors[-1]),
        )


def _require_suffix_parameter(value: TensorType, parameter: TensorType, axis: int, name: str) -> None:
    suffix = value.shape[axis:]
    if parameter.rank != len(suffix):
        raise IRSchemaError(f"GatedNormApply {name} rank must equal the normalized suffix rank.")
    for expected, actual in zip(suffix, parameter.shape):
        if actual != expected and not (actual.is_fixed and actual.fixed_value == 1):
            raise IRSchemaError(
                f"GatedNormApply {name} is not broadcastable to the normalized suffix.")


def _local_cost_tensor(value: IRType) -> TensorType:
    from triton.flagmega.ir.distributed_type import local_tensor_type

    return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)


def _fixed_tensor_nbytes(value: TensorType) -> int:
    return prod(dimension.fixed_value for dimension in value.shape) * value.dtype.itemsize


__all__ = ["GatedNormApply"]
