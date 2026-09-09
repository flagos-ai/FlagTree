# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Token embedding definition and its local compiler behaviors."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPPartial
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.embedding", namespace="nn", functional_name="embedding", display_name="NN.Embedding")
class Embedding(OpDefinition):
    indices = input_parameter(is_tensor() & (has_dtype(DType.INT32) | has_dtype(DType.INT64)))
    weight = input_parameter(is_tensor() & has_rank(2))
    padding_idx = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        padding_idx = attrs["padding_idx"]
        if padding_idx is not None and (isinstance(padding_idx, bool) or not isinstance(padding_idx, int)):
            raise IRSchemaError("F.nn.embedding padding_idx must be an integer or None.")
        return {"padding_idx": padding_idx}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        indices_type = cls.indices.type_of(inputs)
        weight_type = cls.weight.type_of(inputs)
        indices = tensor_of(indices_type)
        weight = tensor_of(weight_type)
        vocabulary = weight.shape[0]
        padding_idx = attrs["padding_idx"]
        if padding_idx is not None and vocabulary.is_fixed:
            normalized = padding_idx + vocabulary.fixed_value if padding_idx < 0 else padding_idx
            if normalized < 0 or normalized >= vocabulary.fixed_value:
                raise IRSchemaError(
                    f"F.nn.embedding padding_idx {padding_idx} is outside vocabulary size {vocabulary.fixed_value}.")
        output = tensor_type(weight.dtype, (*indices.shape, weight.shape[1]))
        distributed_inputs = tuple(
            isinstance(value, DistributedType)
            for value in (indices_type, weight_type)
        )
        if not any(distributed_inputs):
            return output
        if not all(distributed_inputs):
            raise IRSchemaError("F.nn.embedding distributed inputs must use one common placement.")
        assert isinstance(indices_type, DistributedType)
        assert isinstance(weight_type, DistributedType)
        if indices_type.placement != weight_type.placement:
            raise IRSchemaError("F.nn.embedding distributed inputs must use one common placement.")
        if indices_type.partial is not None or weight_type.partial is not None:
            raise IRSchemaError("F.nn.embedding distributed inputs must not be partial.")
        if any(
            isinstance(policy, SBPPartial)
            for value in (indices_type, weight_type)
            for policy in value.axis_policies
        ):
            raise IRSchemaError("F.nn.embedding distributed inputs must not be partial.")
        if not all(isinstance(policy, SBPBroadCast) for policy in indices_type.axis_policies):
            raise IRSchemaError("F.nn.embedding distributed indices must be broadcast.")
        if not isinstance(weight_type.axis_policies[0], SBPBroadCast):
            raise IRSchemaError("F.nn.embedding distributed vocabulary axis must be broadcast.")
        return DistributedType(
            output,
            (*indices_type.axis_policies, weight_type.axis_policies[1]),
            indices_type.placement,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        return embedding(
            cls.indices.read(arguments),
            cls.weight.read(arguments),
            padding_idx=cls.padding_idx.read(arguments, node.attrs),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output_bytes = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        elements = tensor_elements(node.type) if isinstance(node.type, TensorType) else None
        notes = ("padding-mask",) if node.attrs["padding_idx"] is not None else ()
        if elements is None:
            notes += ("dynamic-shape",)
        return OpCost(flops=0, bytes_read=output_bytes, bytes_written=output_bytes, notes=notes)


def embedding(indices, weight, *, padding_idx: int | None = None):
    """Gather embedding rows and apply nncase/HuggingFace padding semantics."""

    flat_indices = indices.to(dtype=_torch().int64).reshape(-1)
    # Vector element lanes are trailing physical dimensions of the table.
    # Gathering rows preserves them, just as it preserves scalar features.
    result = weight.index_select(0, flat_indices).reshape((*indices.shape, *weight.shape[1:]))
    if padding_idx is not None:
        normalized = padding_idx + weight.shape[0] if padding_idx < 0 else padding_idx
        mask = (indices == normalized).reshape((*indices.shape, *((1,) * (weight.ndim - 1))))
        result = result.masked_fill(mask, 0)
    return result


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("Embedding evaluation requires PyTorch.") from error
    return torch


__all__ = ["Embedding", "embedding"]
