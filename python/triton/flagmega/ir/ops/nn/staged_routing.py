# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Staged MoE routing: scoring -> bias-corrected stable TopK -> weight normalization.

Promoted from tutorial 02's ``local.staged_routing`` (Qwen softmax routing)
and generalized for the DeepSeek/GLM ``noaux_tc`` family: scoring is softmax
or sigmoid, selection runs on ``scores + bias`` (optional), the emitted
weights are the **unbiased** scores at the selected ids (optionally
normalized over the selected set and scaled by a constant), and ties resolve
to the lower expert id exactly like ``tensors.top_k``.
"""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import SBP, DistributedType, NoneType, TupleType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_none, is_tensor
from triton.flagmega.ir.types import DType

_OPTIONAL_TENSOR = is_tensor() | is_none()


@op_definition("nn.staged_routing", namespace="nn", functional_name="staged_routing",
               display_name="NN.StagedRouting")
class StagedRouting(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor())
    bias = input_parameter(_OPTIONAL_TENSOR)
    k = attribute_parameter()
    axis = attribute_parameter(default=-1)
    index_dtype = attribute_parameter(default="int64")
    scoring = attribute_parameter(default="softmax")
    normalize = attribute_parameter(default=True)
    scaling = attribute_parameter(default=1.0)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if attrs["scoring"] not in {"softmax", "sigmoid"}:
            raise IRSchemaError("StagedRouting scoring must be softmax or sigmoid.")
        if not isinstance(attrs["normalize"], bool):
            raise IRSchemaError("StagedRouting normalize must be boolean.")
        if (isinstance(attrs["scaling"], bool) or not isinstance(attrs["scaling"], (int, float))
                or not attrs["scaling"] > 0):
            raise IRSchemaError("StagedRouting scaling must be a positive number.")
        if isinstance(attrs["k"], bool) or not isinstance(attrs["k"], int) or attrs["k"] <= 0:
            raise IRSchemaError("StagedRouting k must be a positive integer.")
        if attrs["index_dtype"] not in {"int32", "int64"}:
            raise IRSchemaError("StagedRouting index_dtype must be int32 or int64.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        bias_source = cls.bias.type_of(inputs)
        tensor = tensor_of(source)
        if tensor.dtype != DType.FLOAT32:
            raise IRSchemaError("StagedRouting requires FP32 router logits.")
        axis = normalize_axis(attrs["axis"], tensor.rank)
        if axis != tensor.rank - 1 or not tensor.shape[axis].is_fixed:
            raise IRSchemaError("StagedRouting reduces the last materialized axis with a fixed extent.")
        if not 0 < attrs["k"] <= tensor.shape[axis].fixed_value:
            raise IRSchemaError("StagedRouting k must not exceed the router extent.")
        if isinstance(bias_source, NoneType):
            bias_source = None
        if bias_source is not None:
            bias = tensor_of(bias_source)
            if bias.dtype != DType.FLOAT32 or bias.rank != 1 or bias.shape[0] != tensor.shape[axis]:
                raise IRSchemaError("StagedRouting bias must be FP32 with one entry per router expert.")
            if isinstance(source, DistributedType) != isinstance(bias_source, DistributedType):
                raise IRSchemaError("StagedRouting logits and bias must use the same distribution contract.")
        weights = tensor_type(DType.FLOAT32, (*tensor.shape[:-1], attrs["k"]))
        indices = tensor_type(DType(attrs["index_dtype"]), (*tensor.shape[:-1], attrs["k"]))
        if isinstance(source, DistributedType):
            if source.partial is not None or any(policy != SBP.broadcast() for policy in source.axis_policies):
                raise IRSchemaError("StagedRouting requires broadcast router logits; insert explicit Boxing.")
            if bias_source is not None and (not isinstance(bias_source, DistributedType)
                                            or bias_source.placement != source.placement
                                            or any(policy != SBP.broadcast() for policy in bias_source.axis_policies)):
                raise IRSchemaError("StagedRouting bias must be broadcast like the logits.")
            result = TupleType((weights, indices))
            return TupleType(tuple(DistributedType(field, source.axis_policies, source.placement)
                                   for field in result.fields))
        return TupleType((weights, indices))

    @classmethod
    def evaluate(cls, node, arguments, context):
        logits = cls.value.read(arguments)
        bias = cls.bias.read(arguments)
        axis, k = int(node.attrs["axis"]), int(node.attrs["k"])
        scores = logits.float()
        if node.attrs["scoring"] == "softmax":
            scores = context.torch.softmax(scores, dim=axis)
        else:
            scores = context.torch.sigmoid(scores)
        choice = scores if bias is None else scores + bias.float()
        order = context.torch.argsort(choice, dim=axis, descending=True, stable=True).narrow(axis, 0, k)
        selected = scores.gather(axis, order)
        weights = selected
        if node.attrs["normalize"]:
            weights = weights / weights.sum(dim=axis, keepdim=True)
        weights = weights * float(node.attrs["scaling"])
        return weights, order.to(getattr(context.torch, node.attrs["index_dtype"]))

    @classmethod
    def cost(cls, node):
        sizes = [tensor_nbytes(field) for field in node.type.fields]
        return OpCost(bytes_written=None if None in sizes else sum(sizes), notes=("staged-routing", ))


__all__ = ["StagedRouting"]
