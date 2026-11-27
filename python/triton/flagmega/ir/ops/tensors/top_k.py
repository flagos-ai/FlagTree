# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TopK values/indices with deterministic lower-index tie breaking."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, SBP, TupleType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("tensors.top_k", namespace="tensors", functional_name="top_k", display_name="Tensors.TopK")
class TopK(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    k = attribute_parameter()
    axis = attribute_parameter(default=-1)
    largest = attribute_parameter(default=True)
    sorted = attribute_parameter(default=True)
    index_dtype = attribute_parameter(default="int64")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if isinstance(attrs["k"], bool) or not isinstance(attrs["k"], int) or attrs["k"] < 0:
            raise IRSchemaError("TopK k must be a nonnegative integer.")
        if isinstance(attrs["axis"], bool) or not isinstance(attrs["axis"], int):
            raise IRSchemaError("TopK axis must be an integer.")
        if any(not isinstance(attrs[name], bool) for name in ("largest", "sorted")):
            raise IRSchemaError("TopK largest/sorted must be boolean.")
        if attrs["index_dtype"] not in {"int32", "int64"}:
            raise IRSchemaError("TopK index_dtype must be int32 or int64.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        value = tensor_of(source)
        if value.dtype not in {DType.BFLOAT16, DType.FLOAT32, DType.INT32, DType.INT64}:
            raise IRSchemaError("TopK requires scalar floating/integer elements.")
        axis = normalize_axis(attrs["axis"], value.rank)
        if not value.shape[axis].is_fixed or attrs["k"] > value.shape[axis].fixed_value:
            raise IRSchemaError("TopK k must not exceed the static selected-axis extent.")
        shape = list(value.shape)
        shape[axis] = attrs["k"]
        values = tensor_type(value.dtype, shape)
        indices = tensor_type(attrs["index_dtype"], shape)
        if isinstance(source, DistributedType):
            if source.partial is not None or source.axis_policies[axis] != SBP.broadcast():
                raise IRSchemaError("TopK requires a materialized broadcast selection axis.")
            values = DistributedType(values, source.axis_policies, source.placement)
            indices = DistributedType(indices, source.axis_policies, source.placement)
        return TupleType((values, indices))

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        axis = int(node.attrs["axis"])
        # A sorted result also satisfies sorted=False's unspecified ordering.
        # Stable sort implements ONNX/nncase's lower-index tie rule explicitly;
        # torch.topk alone does not guarantee it.
        order = context.torch.argsort(value, dim=axis, descending=node.attrs["largest"], stable=True)
        indices = order.narrow(axis, 0, node.attrs["k"])
        return value.gather(axis, indices), indices.to(context.torch_dtype(DType(node.attrs["index_dtype"])))

    @classmethod
    def cost(cls, node):
        sizes = tuple(tensor_nbytes(field) for field in node.type.fields)
        return OpCost(bytes_written=None if any(size is None for size in sizes) else sum(sizes),
                      notes=("top-k-stable-ties", ))
