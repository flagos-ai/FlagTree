# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent fused Q/K/V linear projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DType, IRType, Node, NoneType, TensorType, TupleType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.type_pattern import has_rank, is_none, is_tensor


_OPTIONAL_TENSOR = is_tensor() | is_none()


@op_definition(
    "nn.qkv_parallel_linear",
    namespace="nn",
    functional_name="qkv_parallel_linear",
    display_name="NN.QKVParallelLinear",
)
class QKVParallelLinear(OpDefinition):
    """nncase-compatible QKV projection with logical ``[K,N]`` weights."""

    input = input_parameter(is_tensor() & has_rank(2))
    q_weight = input_parameter(is_tensor() & has_rank(2))
    k_weight = input_parameter(is_tensor() & has_rank(2))
    v_weight = input_parameter(is_tensor() & has_rank(2))
    q_bias = input_parameter(_OPTIONAL_TENSOR)
    k_bias = input_parameter(_OPTIONAL_TENSOR)
    v_bias = input_parameter(_OPTIONAL_TENSOR)
    q_input_scale = input_parameter(_OPTIONAL_TENSOR)
    k_input_scale = input_parameter(_OPTIONAL_TENSOR)
    v_input_scale = input_parameter(_OPTIONAL_TENSOR)
    q_weight_scale = input_parameter(_OPTIONAL_TENSOR)
    k_weight_scale = input_parameter(_OPTIONAL_TENSOR)
    v_weight_scale = input_parameter(_OPTIONAL_TENSOR)
    num_heads = attribute_parameter()
    num_kv_heads = attribute_parameter()
    output_data_type = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        for name in ("num_heads", "num_kv_heads"):
            value = attrs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IRSchemaError(f"QKVParallelLinear {name} must be positive.")
        output_type = DType(attrs["output_data_type"])
        return {
            "num_heads": int(attrs["num_heads"]),
            "num_kv_heads": int(attrs["num_kv_heads"]),
            "output_data_type": output_type,
        }

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "num_heads": attrs["num_heads"],
            "num_kv_heads": attrs["num_kv_heads"],
            "output_data_type": DType(attrs["output_data_type"]).value,
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = tensor_of(cls.input.type_of(inputs))
        weights = tuple(tensor_of(parameter.type_of(inputs)) for parameter in (
            cls.q_weight, cls.k_weight, cls.v_weight,
        ))
        if not isinstance(value.dtype, DType):
            raise IRSchemaError("QKVParallelLinear input must have a scalar dtype.")
        if any(weight.shape[0] != value.shape[1] for weight in weights):
            raise IRSchemaError("QKVParallelLinear logical [K,N] weights must match input K.")
        scales = tuple(parameter.type_of(inputs) for parameter in (
            cls.q_input_scale, cls.k_input_scale, cls.v_input_scale,
            cls.q_weight_scale, cls.k_weight_scale, cls.v_weight_scale,
        ))
        scale_presence = tuple(not isinstance(scale, NoneType) for scale in scales)
        if any(scale_presence) and not all(scale_presence):
            raise IRSchemaError("QKVParallelLinear requires either no scales or all six scales.")
        if not any(scale_presence):
            if any(weight.dtype != value.dtype for weight in weights):
                raise IRSchemaError("Unscaled QKVParallelLinear weights must match input dtype.")
        elif any(weight.dtype != DType.FLOAT8_E4M3FN for weight in weights):
            raise IRSchemaError("Scaled QKVParallelLinear requires float8_e4m3fn weights.")

        output_dtype = DType(attrs["output_data_type"])
        outputs = tuple(
            tensor_type(output_dtype, (value.shape[0], weight.shape[1]))
            for weight in weights
        )
        _check_heads(outputs, int(attrs["num_heads"]), int(attrs["num_kv_heads"]))
        for name, bias_parameter, output in zip(
            ("q", "k", "v"),
            (cls.q_bias, cls.k_bias, cls.v_bias),
            outputs,
        ):
            bias = bias_parameter.type_of(inputs)
            if isinstance(bias, NoneType):
                continue
            bias_tensor = tensor_of(bias)
            if bias_tensor.rank != 1 or bias_tensor.shape[0] != output.shape[-1]:
                raise IRSchemaError(
                    f"QKVParallelLinear {name} bias must match its output N dimension.")
        return TupleType(outputs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        output_dtype = context.torch_dtype(DType(node.attrs["output_data_type"]))
        return tuple(
            _project(
                cls.input.read(arguments),
                weight.read(arguments),
                bias.read(arguments),
                input_scale.read(arguments),
                weight_scale.read(arguments),
                output_dtype=output_dtype,
                torch=context.torch,
            )
            for weight, bias, input_scale, weight_scale in zip(
                (cls.q_weight, cls.k_weight, cls.v_weight),
                (cls.q_bias, cls.k_bias, cls.v_bias),
                (cls.q_input_scale, cls.k_input_scale, cls.v_input_scale),
                (cls.q_weight_scale, cls.k_weight_scale, cls.v_weight_scale),
            )
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("three-linear-projections",))


def _project(value, weight, bias, input_scale, weight_scale, *, output_dtype, torch):
    del input_scale
    effective_weight = weight
    if weight_scale is not None:
        effective_weight = weight.float() * weight_scale.float()
    result = value @ effective_weight.to(dtype=value.dtype)
    if bias is not None:
        result = result + bias.to(dtype=result.dtype)
    return result.to(dtype=output_dtype)


def _check_heads(outputs: tuple[TensorType, ...], num_heads: int, num_kv_heads: int) -> None:
    if outputs[0].shape[-1].is_fixed and outputs[0].shape[-1].fixed_value % num_heads:
        raise IRSchemaError("Q output dimension must be divisible by num_heads.")
    for output in outputs[1:]:
        if output.shape[-1].is_fixed and output.shape[-1].fixed_value % num_kv_heads:
            raise IRSchemaError("K/V output dimension must be divisible by num_kv_heads.")


__all__ = ["QKVParallelLinear"]
