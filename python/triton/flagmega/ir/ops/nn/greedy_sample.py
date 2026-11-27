# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Greedy token selection from a rank-2 logits tensor."""

from __future__ import annotations

from typing import Mapping, Sequence
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DistributedType, IRType, Node, tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPPartial, local_tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    OpDefinition,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType


@op_definition(
    "nn.greedy_sample",
    namespace="nn",
    functional_name="greedy_sample",
    display_name="NN.GreedySample",
)
class GreedySample(OpDefinition):
    logits = input_parameter(is_tensor() & has_rank(2))

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        del attrs
        value_type = cls.logits.type_of(inputs)
        logits = tensor_of(value_type)
        if logits.dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("F.nn.greedy_sample requires BF16 or FP32 logits.")
        if logits.shape[-1].is_fixed and logits.shape[-1].fixed_value == 0:
            raise IRSchemaError("F.nn.greedy_sample requires a non-empty vocabulary.")
        result = tensor_type(DType.INT32, logits.shape[:-1])
        if isinstance(value_type, DistributedType):
            if value_type.partial is not None or any(
                isinstance(policy, SBPPartial) for policy in value_type.axis_policies
            ):
                raise IRSchemaError("GreedySample requires materialized logits, not additive partials.")
            # This operation includes the collective argmax. A vocabulary
            # split is reduced/materialized; outer (batch) splits survive.
            return DistributedType(result, value_type.axis_policies[:-1], value_type.placement)
        return result

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        del attrs
        value = cls.logits.type_of(inputs)
        local = local_tensor_type(value) if isinstance(value, DistributedType) else value
        output = local_tensor_type(return_type) if isinstance(return_type, DistributedType) else return_type
        if any(not dimension.is_fixed for dimension in local.shape):
            return None
        elements = prod(dimension.fixed_value for dimension in local.shape)
        return OpCostFactors(
            cpu_cycles=elements * 2,
            block_local_memory_load_bytes=elements * local.dtype.itemsize,
            block_local_memory_store_bytes=tensor_nbytes(output) or 0,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        del node, context
        return cls.logits.read(arguments).argmax(dim=-1).to(dtype=_torch().int32)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(
            flops=None,
            bytes_read=None,
            bytes_written=tensor_nbytes(tensor_of(node.type)),
            notes=("argmax", "greedy-sampling"),
        )


def _torch():
    import torch

    return torch


__all__ = ["GreedySample"]
