# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""BF16 GLU over an offline K-major/N8/K16 physical weight layout."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import (
    all_broadcast,
    placement_of,
    split_policy,
    tensor_of,
)
from triton.flagmega.ir.model import DistributedType, IRType, Node, SBP, TensorType, tensor_type
from triton.flagmega.ir.distributed_type import scale_split_units
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
    unpack_k_major_n8_k16_weight,
)
from triton.flagmega.ir.type_pattern import has_rank, is_tensor


@op_definition(
    "nn.packed_dense_matmul_glu",
    namespace="nn",
    functional_name="packed_dense_matmul_glu",
    display_name="NN.PackedDenseMatMulGlu",
)
class PackedDenseMatMulGlu(OpDefinition):
    """Semantic GLU whose constant weights use a K-major physical layout."""

    value = input_parameter(is_tensor() & has_rank(2))
    gate_weight = input_parameter(is_tensor() & has_rank(4))
    up_weight = input_parameter(is_tensor() & has_rank(4))
    activation = attribute_parameter(default="silu")
    packed_layout = attribute_parameter(default="k_major_n8_k16")

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if attrs["activation"] != "silu":
            raise IRSchemaError("PackedDenseMatMulGlu currently supports only SiLU.")
        _, _, mesh_interleaved = parse_k_major_layout(attrs["packed_layout"])
        if mesh_interleaved:
            raise IRSchemaError("PackedDenseMatMulGlu does not support mesh-interleaved weights.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        gate_type = cls.gate_weight.type_of(inputs)
        up_type = cls.up_weight.type_of(inputs)
        value = tensor_of(value_type)
        gate = tensor_of(gate_type)
        up = tensor_of(up_type)
        if gate != up:
            raise IRSchemaError("PackedDenseMatMulGlu gate/up weight types must match.")
        shape = tuple(dimension.fixed_value for dimension in gate.shape)
        n_lane, k_lane, _ = parse_k_major_layout(attrs["packed_layout"])
        if (
            None in shape
            or len(shape) != 4
            or int(shape[2]) * int(shape[3]) != n_lane * k_lane
        ):
            raise IRSchemaError(
                "PackedDenseMatMulGlu weight payload does not match its K-major lanes.")
        logical_k = int(shape[0]) * k_lane
        logical_n = int(shape[1]) * n_lane
        if value.shape[1].fixed_value != logical_k or value.dtype != gate.dtype:
            raise IRSchemaError(
                "PackedDenseMatMulGlu value dtype/K does not match the packed weights.")
        output = tensor_type(value.dtype, (value.shape[0], logical_n))
        placement = placement_of(value_type, gate_type, up_type)
        if placement is None:
            return output
        if not all(isinstance(item, DistributedType) for item in (
            value_type, gate_type, up_type,
        )):
            raise IRSchemaError(
                "Distributed PackedDenseMatMulGlu requires all operands to name a placement.")
        if all_broadcast(value_type) and all_broadcast(gate_type) and all_broadcast(up_type):
            return DistributedType(
                output, (SBP.broadcast(), SBP.broadcast()), placement)
        gate_split = split_policy(gate_type, 1)
        up_split = split_policy(up_type, 1)
        if all_broadcast(value_type) and gate_split is not None and gate_split == up_split:
            other_axes = tuple(axis for axis in range(gate.rank) if axis != 1)
            if all(
                gate_type.axis_policies[axis] == SBP.broadcast()
                and up_type.axis_policies[axis] == SBP.broadcast()
                for axis in other_axes
            ):
                logical_split = scale_split_units(gate_split, n_lane, 1)
                if logical_split is None:  # pragma: no cover - multiplication is exact.
                    raise IRSchemaError(
                        "PackedDenseMatMulGlu cannot scale its physical N split."
                    )
                return DistributedType(
                    output, (SBP.broadcast(), logical_split), placement)
        raise IRSchemaError(
            "Distributed PackedDenseMatMulGlu requires replicated input and identical physical N splits.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        layout = str(node.attrs["packed_layout"])
        gate_weight = unpack_k_major_weight(cls.gate_weight.read(arguments), layout)
        up_weight = unpack_k_major_weight(cls.up_weight.read(arguments), layout)
        gate = context.torch.nn.functional.linear(value, gate_weight)
        up = context.torch.nn.functional.linear(value, up_weight)
        return context.torch.nn.functional.silu(gate) * up

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(
            flops=None,
            bytes_read=None,
            bytes_written=tensor_nbytes(node.type),
            notes=("packed-k-major-two-dense-matmuls+glu",),
        )


__all__ = ["PackedDenseMatMulGlu", "unpack_k_major_n8_k16_weight"]
