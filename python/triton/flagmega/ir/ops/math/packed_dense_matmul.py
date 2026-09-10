# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense BF16 projection over an offline K-major physical weight."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import (
    all_broadcast,
    placement_of,
    split_policy,
    tensor_of,
)
from triton.flagmega.ir.distributed_type import SBPSplit, SplitStage, scale_split_units
from triton.flagmega.ir.model import (
    DistributedType,
    IRType,
    Node,
    SBP,
    SBPPartial,
    tensor_type,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.tensors._k_major import (
    parse_k_major_layout,
    unpack_k_major_weight,
)
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType
from triton.flagmega.ir.ops.math.matmul import matmul_value, normalize_output_data_type


@op_definition(
    "math.packed_dense_matmul",
    namespace="math",
    functional_name="packed_dense_matmul",
    display_name="Math.PackedDenseMatMul",
)
class PackedDenseMatMul(OpDefinition):
    """``lhs @ weight.T`` with a target-parameterized K-major weight."""

    lhs = input_parameter(is_tensor() & has_rank(2))
    weight = input_parameter(is_tensor())
    packed_layout = attribute_parameter(default="k_major_n8_k16")
    logical_n = attribute_parameter(default=None)
    output_data_type = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        n_lane, _, mesh_interleaved = parse_k_major_layout(attrs["packed_layout"])
        logical_n = attrs["logical_n"]
        if logical_n is not None and (
            isinstance(logical_n, bool) or not isinstance(logical_n, int)
            or logical_n <= 0 or logical_n % n_lane
        ):
            raise IRSchemaError(
                f"PackedDenseMatMul logical_n must be a positive multiple of {n_lane}."
            )
        if mesh_interleaved and logical_n is None:
            raise IRSchemaError("Mesh-interleaved PackedDenseMatMul requires logical_n.")
        return normalize_output_data_type(attrs)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        lhs_type = cls.lhs.type_of(inputs)
        weight_type = cls.weight.type_of(inputs)
        lhs = tensor_of(lhs_type)
        weight = tensor_of(weight_type)
        shape = tuple(dimension.fixed_value for dimension in weight.shape)
        layout = str(attrs["packed_layout"])
        n_lane, k_lane, mesh_interleaved = parse_k_major_layout(layout)
        expected_rank = 5 if mesh_interleaved else 4
        if (
            None in shape
            or len(shape) != expected_rank
            or _product(shape[-2:]) != n_lane * k_lane
        ):
            raise IRSchemaError(
                "PackedDenseMatMul weight shape does not match its physical layout.")
        logical_k = int(shape[0]) * k_lane
        logical_n = (
            int(attrs["logical_n"])
            if mesh_interleaved
            else int(shape[1]) * n_lane
        )
        if lhs.shape[1].fixed_value != logical_k or lhs.dtype != weight.dtype:
            raise IRSchemaError(
                "PackedDenseMatMul lhs dtype/K does not match the packed weight.")
        output = tensor_type(DType(attrs.get("output_data_type") or lhs.dtype), (lhs.shape[0], logical_n), layout=lhs.layout)
        placement = placement_of(lhs_type, weight_type)
        if placement is None:
            return output
        if not isinstance(lhs_type, DistributedType) or not isinstance(weight_type, DistributedType):
            raise IRSchemaError(
                "Distributed PackedDenseMatMul requires both operands to name a placement.")
        if all_broadcast(lhs_type) and all_broadcast(weight_type):
            return DistributedType(output, (SBP.broadcast(), SBP.broadcast()), placement)
        weight_n_axis = 2 if mesh_interleaved else 1
        weight_k_axis = 0
        weight_n_split = split_policy(weight_type, weight_n_axis)
        weight_k_split = split_policy(weight_type, weight_k_axis)
        lhs_k_split = split_policy(lhs_type, 1)

        def logical_n_policy():
            if weight_n_split is None:
                return SBP.broadcast()
            if mesh_interleaved:
                return SBPSplit(tuple(
                    SplitStage.block_cyclic(stage.hierarchy_axes, n_lane)
                    for stage in weight_n_split.stages
                ))
            scaled = scale_split_units(weight_n_split, n_lane, 1)
            if scaled is None:  # pragma: no cover - multiplication is exact.
                raise IRSchemaError(
                    "PackedDenseMatMul cannot scale its physical N split."
                )
            return scaled

        logical_weight_k_split = (
            None
            if weight_k_split is None
            else scale_split_units(weight_k_split, k_lane, 1)
        )

        if (
            not mesh_interleaved
            and lhs_type.partial is None
            and weight_type.partial is None
            and lhs_k_split is not None
            and lhs_k_split == logical_weight_k_split
            and lhs_type.axis_policies[0] == SBP.broadcast()
            and all(
                weight_type.axis_policies[axis] == SBP.broadcast()
                for axis in range(weight.rank)
                if axis not in {weight_k_axis, weight_n_axis}
            )
        ):
            reduction_axes = tuple(lhs_k_split.hierarchy_axes)
            output_n_policy = logical_n_policy()
            output_axes = (
                tuple(output_n_policy.hierarchy_axes)
                if isinstance(output_n_policy, SBPSplit)
                else ()
            )
            if set(reduction_axes) & set(output_axes):
                raise IRSchemaError(
                    "PackedDenseMatMul output and reduction splits must use disjoint "
                    "placement axes."
                )
            return DistributedType(
                output,
                (SBP.broadcast(), output_n_policy),
                placement,
                partial=SBPPartial(reduction_axes),
            )
        if (
            all_broadcast(lhs_type)
            and weight_n_split is not None
            and all(
                weight_type.axis_policies[axis] == SBP.broadcast()
                for axis in range(weight.rank)
                if axis != weight_n_axis
            )
        ):
            logical_n_split = logical_n_policy()
            return DistributedType(output, (SBP.broadcast(), logical_n_split), placement)
        raise IRSchemaError(
            "Distributed PackedDenseMatMul requires replicated/output-split operands "
            "or aligned physical-K reduction splits.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        lhs = cls.lhs.read(arguments)
        packed = cls.weight.read(arguments)
        weight = unpack_k_major_weight(
            packed,
            str(node.attrs["packed_layout"]),
            node.attrs.get("logical_n"),
        )
        return matmul_value(lhs, weight.transpose(-2, -1), node.attrs.get("output_data_type"), context)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(
            flops=None,
            bytes_read=None,
            bytes_written=tensor_nbytes(node.type),
            notes=("packed-k-major-dense-matmul",),
        )


def _product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


__all__ = ["PackedDenseMatMul"]
