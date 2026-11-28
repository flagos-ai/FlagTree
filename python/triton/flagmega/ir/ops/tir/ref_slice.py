# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Non-owning leading-axis slices of a reference's tensor fields."""

from collections.abc import Mapping

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.model import DType, RefType, TensorType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, ParameterKind, attribute_parameter, input_parameter, op_definition
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor


@op_definition("tir.ref_slice", namespace="tir", functional_name="ref_slice", display_name="T.RefSlice")
class RefSlice(OpDefinition):
    """Alias each field's [index:index+length] without reading field data.

    Runtime callers must satisfy 0 <= index <= extent-length. The field
    extents agree, so one bounded scalar defines every field's MemSpan.
    Nothing about a model, field names, or a state-update kernel is encoded.
    """
    value = input_parameter(is_ref(), parameter_kind=ParameterKind.ATTRIBUTE, memory_effect="none")
    index = input_parameter(is_tensor() & has_rank(0) & has_dtype(DType.INT32), parameter_kind=ParameterKind.ATTRIBUTE)
    length = attribute_parameter(default=1)

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = super().normalize_attrs(attrs)
        length = attrs["length"]
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise IRSchemaError("RefSlice length must be a positive integer.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        if not source.fields:
            raise IRSchemaError("RefSlice requires tensor fields, not an opaque reference.")
        length = attrs["length"]
        extent = None
        fields = []
        for name, field in source.fields:
            if not isinstance(field, TensorType) or field.rank == 0 or not field.shape[0].is_fixed:
                raise IRSchemaError("RefSlice fields require a static leading tensor extent.")
            size = field.shape[0].fixed_value
            if size < length or (extent is not None and size != extent):
                raise IRSchemaError("RefSlice fields require equal leading extents large enough for length.")
            extent = size
            fields.append((name, tensor_type(field.dtype, (length, *field.shape[1:]), layout=field.layout)))
        index = cls.index.read(inputs)
        if index.op in {"builtin.scalar_const", "tir.scalar_const"
                        } and not 0 <= index.attrs["value"] <= extent - length:
            raise IRSchemaError("RefSlice index is outside the reference extent.")
        return RefType(source.name, tuple(fields))

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        index = int(cls.index.read(arguments).item())
        length = int(node.attrs["length"])
        method = getattr(value, "__flagmega_ref_slice__", None)
        if method is not None:
            return method(index, length)
        if isinstance(value, Mapping):
            if any(not 0 <= index <= field.shape[0] - length for field in value.values()):
                raise EvaluationError("RefSlice index is outside the reference extent.")
            return {name: field[index:index + length] for name, field in value.items()}
        raise EvaluationError("Reference backing must implement __flagmega_ref_slice__ or expose a field mapping.")

    @classmethod
    def cost(cls, node):
        return OpCost(flops=0, bytes_read=0, bytes_written=0, notes=("non-owning-reference-view", ))
