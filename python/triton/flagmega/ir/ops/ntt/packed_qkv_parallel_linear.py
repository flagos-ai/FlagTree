# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Q/K/V projection over three independent K-major VectorType weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import (
    SBP,
    SBPPartial,
    SBPSplit,
    scale_split_units,
)
from triton.flagmega.ir.model import (
    DType,
    DistributedType,
    IRType,
    Node,
    NoneType,
    TensorType,
    TupleType,
    tensor_type,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn.qkv_parallel_linear import _check_heads, _project
from triton.flagmega.ir.type_pattern import has_rank, is_none, is_tensor
from triton.flagmega.ir.types import VectorType


_OPTIONAL_TENSOR = is_tensor() | is_none()


@op_definition(
    "ntt.packed_qkv_parallel_linear",
    namespace="ntt",
    functional_name="packed_qkv_parallel_linear",
    display_name="NTT.PackedQKVParallelLinear",
)
class PackedQKVParallelLinear(OpDefinition):
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
    rhs_layout = attribute_parameter(default="k_major")

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        for name in ("num_heads", "num_kv_heads"):
            value = attrs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IRSchemaError(f"PackedQKVParallelLinear {name} must be positive.")
        if attrs["rhs_layout"] != "k_major":
            raise IRSchemaError("PackedQKVParallelLinear currently requires K-major RHS.")
        return {
            "num_heads": int(attrs["num_heads"]),
            "num_kv_heads": int(attrs["num_kv_heads"]),
            "output_data_type": DType(attrs["output_data_type"]),
            "rhs_layout": "k_major",
        }

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "num_heads": attrs["num_heads"],
            "num_kv_heads": attrs["num_kv_heads"],
            "output_data_type": DType(attrs["output_data_type"]).value,
            "rhs_layout": attrs["rhs_layout"],
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = tensor_of(cls.input.type_of(inputs))
        weights = tuple(tensor_of(parameter.type_of(inputs)) for parameter in (
            cls.q_weight, cls.k_weight, cls.v_weight,
        ))
        lane_types = tuple(weight.dtype for weight in weights)
        if any(not isinstance(dtype, VectorType) or len(dtype.lanes) != 3 for dtype in lane_types):
            raise IRSchemaError(
                "PackedQKVParallelLinear K-major weights require VectorType(N,KPack,KVector).")
        first = lane_types[0]
        assert isinstance(first, VectorType)
        if any(dtype != first for dtype in lane_types):
            raise IRSchemaError("PackedQKVParallelLinear weights must use one VectorType.")
        n_vector, k_pack, k_vector = first.lanes
        if any(weight.shape[0] * (k_pack * k_vector) != value.shape[1] for weight in weights):
            raise IRSchemaError("PackedQKVParallelLinear packed K does not match input K.")
        output_dtype = DType(attrs["output_data_type"])
        outputs = tuple(
            tensor_type(
                VectorType(output_dtype, (n_vector,)),
                (value.shape[0], weight.shape[1]),
            )
            for weight in weights
        )
        logical_outputs = tuple(
            tensor_type(output_dtype, (value.shape[0], weight.shape[1] * n_vector))
            for weight in weights
        )
        _check_heads(logical_outputs, int(attrs["num_heads"]), int(attrs["num_kv_heads"]))
        for bias_parameter, output in zip(
            (cls.q_bias, cls.k_bias, cls.v_bias), outputs,
        ):
            bias = bias_parameter.type_of(inputs)
            if isinstance(bias, NoneType):
                continue
            if tensor_of(bias) != tensor_type(output.dtype, (output.shape[-1],)):
                raise IRSchemaError("PackedQKVParallelLinear bias must use packed N lanes.")
        scales = tuple(parameter.type_of(inputs) for parameter in (
            cls.q_input_scale, cls.k_input_scale, cls.v_input_scale,
            cls.q_weight_scale, cls.k_weight_scale, cls.v_weight_scale,
        ))
        presence = tuple(not isinstance(scale, NoneType) for scale in scales)
        if any(presence) and not all(presence):
            raise IRSchemaError(
                "PackedQKVParallelLinear requires either no scales or all six scales.")
        logical_result = TupleType(outputs)
        input_type = cls.input.type_of(inputs)
        weight_types = tuple(parameter.type_of(inputs) for parameter in (
            cls.q_weight, cls.k_weight, cls.v_weight,
        ))
        if not isinstance(input_type, DistributedType) and not any(
            isinstance(weight, DistributedType) for weight in weight_types
        ):
            return logical_result
        if not isinstance(input_type, DistributedType) or not all(
            isinstance(weight, DistributedType) for weight in weight_types
        ):
            raise IRSchemaError(
                "Distributed PackedQKVParallelLinear requires input and all weights distributed.")
        placement = input_type.placement
        if any(weight.placement != placement or weight.partial is not None for weight in weight_types):
            raise IRSchemaError("PackedQKVParallelLinear operands must share one placement without partial weights.")
        input_m = input_type.axis_policies[0]
        input_k = input_type.axis_policies[1]
        distributed_outputs: list[DistributedType] = []
        for index, (weight_type, output) in enumerate(zip(weight_types, outputs)):
            assert isinstance(weight_type, DistributedType)
            weight_k, weight_n = weight_type.axis_policies
            logical_weight_k = (
                scale_split_units(weight_k, k_pack * k_vector, 1)
                if isinstance(weight_k, SBPSplit)
                else weight_k
            )
            if input_k != logical_weight_k:
                raise IRSchemaError(
                    "PackedQKVParallelLinear input/weight reduction policies must match.")
            reduction_axes = (
                tuple(input_k.hierarchy_axes)
                if isinstance(input_k, SBPSplit)
                else ()
            )
            output_axes = (
                set(weight_n.hierarchy_axes)
                if isinstance(weight_n, SBPSplit)
                else set()
            )
            if output_axes.intersection(reduction_axes):
                raise IRSchemaError(
                    "PackedQKVParallelLinear output and reduction mesh axes must be disjoint.")
            if input_m != SBP.broadcast():
                raise IRSchemaError(
                    "PackedQKVParallelLinear currently requires broadcast batch/M.")
            partial = SBPPartial(reduction_axes) if reduction_axes else None
            distributed_outputs.append(DistributedType(
                output,
                (input_m, weight_n),
                placement,
                partial=partial,
            ))
            bias = (cls.q_bias, cls.k_bias, cls.v_bias)[index].type_of(inputs)
            if isinstance(bias, NoneType):
                continue
            if partial is not None:
                raise IRSchemaError(
                    "PackedQKVParallelLinear partial outputs require absent biases.")
            if (
                not isinstance(bias, DistributedType)
                or bias.placement != placement
                or bias.partial is not None
                or bias.axis_policies != (weight_n,)
            ):
                raise IRSchemaError(
                    "PackedQKVParallelLinear packed bias must follow output N policy.")
        return TupleType(tuple(distributed_outputs))

    @classmethod
    def evaluate(cls, node, arguments, context):
        packed_weights = tuple(parameter.read(arguments) for parameter in (
            cls.q_weight, cls.k_weight, cls.v_weight,
        ))
        weight_types = tuple(tensor_of(context.types[parameter.read(node.inputs)]) for parameter in (
            cls.q_weight, cls.k_weight, cls.v_weight,
        ))
        logical_weights = tuple(
            _unpack_k_major(weight, weight_type)
            for weight, weight_type in zip(packed_weights, weight_types)
        )
        output_dtype = context.torch_dtype(DType(node.attrs["output_data_type"]))
        logical = tuple(
            _project(
                cls.input.read(arguments),
                weight,
                bias.read(arguments),
                input_scale.read(arguments),
                weight_scale.read(arguments),
                output_dtype=output_dtype,
                torch=context.torch,
            )
            for weight, bias, input_scale, weight_scale in zip(
                logical_weights,
                (cls.q_bias, cls.k_bias, cls.v_bias),
                (cls.q_input_scale, cls.k_input_scale, cls.v_input_scale),
                (cls.q_weight_scale, cls.k_weight_scale, cls.v_weight_scale),
            )
        )
        n_vector = weight_types[0].dtype.lanes[0]
        return tuple(
            value.reshape(value.shape[0], value.shape[1] // n_vector, n_vector)
            for value in logical
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("three-packed-k-major-linear-projections",))


def _unpack_k_major(value, value_type: TensorType):
    assert isinstance(value_type.dtype, VectorType)
    n_vector, k_pack, k_vector = value_type.dtype.lanes
    k_groups, n_groups = value_type.shape
    if not k_groups.is_fixed or not n_groups.is_fixed:
        raise IRSchemaError("Packed QKV reference evaluation requires fixed weight shape.")
    return (
        value.permute(1, 2, 0, 3, 4)
        .reshape(n_groups.fixed_value * n_vector, k_groups.fixed_value * k_pack * k_vector)
        .transpose(0, 1)
        .contiguous()
    )


__all__ = ["PackedQKVParallelLinear"]
