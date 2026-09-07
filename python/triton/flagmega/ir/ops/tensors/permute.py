# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Permute logical tensor axes."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("tensors.permute", namespace="tensors", functional_name="permute", display_name="Tensors.Permute")
class Permute(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    axes = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axes = attrs["axes"]
        if not isinstance(axes, (tuple, list)) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in axes
        ):
            raise IRSchemaError("F.tensors.permute axes must be an integer sequence.")
        return {"axes": tuple(axes)}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = cls.value.type_of(inputs)
        assert isinstance(value, TensorType)
        axes = tuple(int(axis) for axis in attrs["axes"])
        if sorted(axes) != list(range(value.rank)):
            raise IRSchemaError(f"F.tensors.permute axes must be a permutation of 0..{value.rank - 1}.")
        return tensor_type(value.dtype, tuple(value.shape[axis] for axis in axes), layout=value.layout)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        outer_axes = tuple(int(axis) for axis in node.attrs["axes"])
        # VectorType lanes are physical trailing dimensions, not logical
        # tensor axes named by this op.  Preserve them after permuting only the
        # outer tensor rank.
        lane_axes = tuple(range(len(outer_axes), value.ndim))
        return value.permute(*outer_axes, *lane_axes).contiguous()

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        value = cls.value.read(arguments)
        outer_axes = tuple(int(axis) for axis in node.attrs["axes"])
        lane_axes = tuple(range(len(outer_axes), value.ndim))
        return context.as_contiguous(value.transpose((*outer_axes, *lane_axes)))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(bytes_read=size, bytes_written=size, notes=("permute",))


__all__ = ["Permute"]
