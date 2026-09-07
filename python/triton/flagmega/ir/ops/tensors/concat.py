# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Concatenate tensors along one logical axis."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    op_definition,
    tensor_nbytes,
    variadic_input_parameter,
)
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("tensors.concat", namespace="tensors", functional_name="concat", display_name="Tensors.Concat")
class Concat(OpDefinition):
    """A normal tensor operation that is also legal in constant recipes."""

    const_evaluable = True
    numpy_materializable = True
    values = variadic_input_parameter(is_tensor())
    axis = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("F.tensors.concat axis must be an integer.")
        return {"axis": axis}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        values = cls.values.type_of(inputs)
        assert isinstance(values, tuple)
        if not values:
            raise IRSchemaError("F.tensors.concat requires at least one tensor.")
        tensors = tuple(value for value in values if isinstance(value, TensorType))
        if len(tensors) != len(values):
            raise IRSchemaError("F.tensors.concat accepts tensor inputs only.")
        reference = tensors[0]
        axis = int(attrs["axis"])
        axis = axis + reference.rank if axis < 0 else axis
        if axis < 0 or axis >= reference.rank:
            raise IRSchemaError(f"F.tensors.concat axis {attrs['axis']} is out of range for rank {reference.rank}.")
        extent = 0
        for value in tensors:
            if value.rank != reference.rank or value.dtype != reference.dtype or value.layout != reference.layout:
                raise IRSchemaError("F.tensors.concat inputs must have identical rank, dtype and layout.")
            for index, (lhs, rhs) in enumerate(zip(reference.shape, value.shape)):
                if index != axis and lhs != rhs:
                    raise IRSchemaError("F.tensors.concat non-concatenated dimensions must match.")
            if not value.shape[axis].is_fixed:
                raise IRSchemaError("F.tensors.concat currently requires a static concatenated dimension.")
            extent += value.shape[axis].fixed_value
        shape = list(reference.shape)
        shape[axis] = extent
        return tensor_type(reference.dtype, shape, layout=reference.layout)

    @classmethod
    def evaluate(cls, node, arguments, context):
        return context.torch.cat(tuple(arguments), dim=int(node.attrs["axis"])).contiguous()

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        return context.as_contiguous(
            context.numpy.concatenate(tuple(arguments), axis=int(node.attrs["axis"]))
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(bytes_read=size, bytes_written=size, notes=("concat",))


__all__ = ["Concat"]
