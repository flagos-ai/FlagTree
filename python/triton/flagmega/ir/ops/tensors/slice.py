# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Static-axis tensor slicing with Python/ONNX index semantics."""

import builtins

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("tensors.slice", namespace="tensors", functional_name="slice", display_name="Tensors.Slice")
class Slice(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    starts = attribute_parameter()
    ends = attribute_parameter()
    axes = attribute_parameter(default=None)
    steps = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        for name in ("starts", "ends"):
            items = attrs[name]
            if not isinstance(items, (tuple, list)) or any(
                    item is not None and (isinstance(item, bool) or not isinstance(item, int)) for item in items):
                raise IRSchemaError(f"Slice {name} must be a sequence of integers or None.")
            attrs[name] = tuple(items)
        count = len(attrs["starts"])
        if len(attrs["ends"]) != count:
            raise IRSchemaError("Slice starts/ends must have identical lengths.")
        for name, default in (("axes", tuple(range(count))), ("steps", (1, ) * count)):
            items = default if attrs[name] is None else attrs[name]
            if not isinstance(items, (tuple, list)) or len(items) != count or any(
                    isinstance(item, bool) or not isinstance(item, int) for item in items):
                raise IRSchemaError(f"Slice {name} must contain one integer per range.")
            attrs[name] = tuple(items)
        if any(step == 0 for step in attrs["steps"]):
            raise IRSchemaError("Slice steps must be nonzero.")
        return attrs

    @classmethod
    def ranges(cls, tensor, attrs):
        axes = tuple(axis + tensor.rank if axis < 0 else axis for axis in attrs["axes"])
        if len(set(axes)) != len(axes) or any(not 0 <= axis < tensor.rank for axis in axes):
            raise IRSchemaError("Slice axes must be distinct and within the tensor rank.")
        result = []
        for axis, start, end, step in zip(axes, attrs["starts"], attrs["ends"], attrs["steps"]):
            extent = tensor.shape[axis]
            if not extent.is_fixed:
                if start in (None, 0) and end is None and step == 1:
                    continue
                raise IRSchemaError(
                    "Slice requires a static extent on sliced axes; untouched dimensions may be dynamic.")
            result.append((axis, range(*builtins.slice(start, end, step).indices(extent.fixed_value))))
        return tuple(result)

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        value = tensor_of(source)
        shape = list(value.shape)
        ranges = cls.ranges(value, attrs)
        for axis, indices in ranges:
            shape[axis] = len(indices)
        output = tensor_type(value.dtype, shape)
        if not isinstance(source, DistributedType):
            return output
        for axis, indices in ranges:
            identity = indices == range(value.shape[axis].fixed_value)
            if not identity and source.axis_policies[axis] != SBP.broadcast():
                raise IRSchemaError("Slice on a split axis requires explicit resharding.")
        return DistributedType(output, source.axis_policies, source.placement, source.partial)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        for axis, indices in cls.ranges(tensor_of(context.types[cls.value.read(node.inputs)]), node.attrs):
            index = context.torch.tensor(list(indices), dtype=context.torch.int64, device=value.device)
            value = value.index_select(axis, index)
        return value.contiguous()

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        # Use native views before the final contiguous materialization: slicing
        # an expert bank must not iterate/copy each scalar through Python.
        value = cls.value.read(arguments)
        tensor = tensor_of(context.types[cls.value.read(node.inputs)])
        cls.ranges(tensor, node.attrs)
        slices = [builtins.slice(None)] * value.ndim
        for axis, start, end, step in zip(node.attrs["axes"], node.attrs["starts"], node.attrs["ends"],
                                          node.attrs["steps"]):
            slices[axis % tensor.rank] = builtins.slice(start, end, step)
        return context.as_contiguous(value[tuple(slices)])

    @classmethod
    def cost(cls, node):
        size = tensor_nbytes(node.type)
        return OpCost(bytes_read=size, bytes_written=size, notes=("static-axis-slice", ))
