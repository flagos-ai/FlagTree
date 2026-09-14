# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A contiguous tensor interval borrowed from the same placement owners."""

from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_shape
from triton.flagmega.ir.model import DistributedType, SBP, TensorType, tensor_type
from triton.flagmega.ir.types import PointerType
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition
from triton.flagmega.ir.type_pattern import is_tensor


def dense_subspan_offset(source_shape, shape, offsets, itemsize=1):
    """Prove dense C-order contiguity and return the byte offset."""
    if len(shape) != len(source_shape) or len(offsets) != len(shape):
        raise IRSchemaError("BufferSubspan offsets/shape must match the source rank.")
    if any(start < 0 or size < 0 or start + size > extent
           for extent, size, start in zip(source_shape, shape, offsets)):
        raise IRSchemaError("BufferSubspan exceeds its source extent.")
    if not prod(shape):
        return 0
    source_stride = result_stride = 1
    offset = 0
    for extent, size, start in reversed(tuple(zip(source_shape, shape, offsets))):
        if size > 1 and source_stride != result_stride:
            raise IRSchemaError("BufferSubspan requires a contiguous interval, not strided rows.")
        offset += start * source_stride
        source_stride *= extent
        result_stride *= size
    return offset * itemsize


def _fixed_shape(shape):
    if any(not value.is_fixed for value in shape):
        raise IRSchemaError("BufferSubspan requires bounded static tensor extents.")
    return tuple(value.fixed_value for value in shape)


@op_definition("tir.buffer_subspan", namespace="tir", functional_name="buffer_subspan", display_name="T.BufferSubspan")
class BufferSubspan(OpDefinition):
    # Unlike a representation view, this does not preserve the entire source
    # byte payload. In particular it must not inherit the rdata streaming trait.
    value = input_parameter(is_tensor())
    offsets = attribute_parameter()
    shape = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        for key in ("offsets", "shape"):
            values = attrs[key]
            if not isinstance(values, (tuple, list)) or any(type(v) is not int or v < 0 for v in values):
                raise IRSchemaError(f"BufferSubspan {key} must be non-negative integer coordinates.")
        return {key: tuple(attrs[key]) for key in ("offsets", "shape")}

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        tensor = tensor_of(source)
        if not tensor.rank or isinstance(tensor.dtype, PointerType):
            raise IRSchemaError("BufferSubspan requires tensor storage, not a by-value scalar or pointer.")
        if tensor.layout.strides or tensor.layout.order:
            raise IRSchemaError("BufferSubspan requires dense row-major tensor layout.")
        shape, offsets = attrs["shape"], attrs["offsets"]
        extents = _fixed_shape(tensor.shape)
        dense_subspan_offset(extents, shape, offsets, tensor.dtype.itemsize)
        result = tensor_type(tensor.dtype, shape)
        if not isinstance(source, DistributedType):
            return result
        for extent, size, start, policy in zip(extents, shape, offsets, source.axis_policies):
            if (size != extent or start) and policy != SBP.broadcast():
                raise IRSchemaError("BufferSubspan cannot change a split axis or its ownership.")
        result = DistributedType(result, source.axis_policies, source.placement, source.partial, source.exclusive)
        dense_subspan_offset(_fixed_shape(local_shape(source)), _fixed_shape(local_shape(result)), offsets,
                             tensor.dtype.itemsize)
        return result

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        slices = tuple(slice(start, start + size) for start, size in zip(node.attrs["offsets"], node.attrs["shape"]))
        return value[slices]

    @classmethod
    def cost(cls, node):
        return OpCost(notes=("zero-copy-contiguous-subspan",))


def slice_subspan_attrs(node, module):
    """Return a proved subspan candidate, or retain normal Slice materialization."""
    from triton.flagmega.ir.ops.tensors.slice import Slice

    if node.op not in {"tensors.slice", "tensors.slice_to_shape"}:
        return None
    source = module.node_map[node.inputs[0]]
    tensor = tensor_of(source.type)
    result = tensor_of(node.type)
    if not isinstance(tensor, TensorType) or any(not d.is_fixed for d in result.shape):
        return None
    offsets = [0] * tensor.rank
    if node.op == "tensors.slice":
        for axis, indices in Slice.ranges(tensor, node.attrs):
            if len(indices) > 1 and indices.step != 1:
                return None
            offsets[axis] = indices.start if indices else 0
    attrs = {"offsets": tuple(offsets), "shape": _fixed_shape(result.shape)}
    try:
        inferred = BufferSubspan.infer_type((source,), attrs)
    except IRSchemaError:
        return None
    return attrs if inferred == node.type else None


__all__ = ["BufferSubspan", "dense_subspan_offset", "slice_subspan_attrs"]
