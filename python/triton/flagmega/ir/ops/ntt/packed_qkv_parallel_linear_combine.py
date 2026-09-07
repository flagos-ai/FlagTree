# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Materialize the coupled optional split-K results of packed Q/K/V."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_type import ReduceOp, SBPPartial
from triton.flagmega.ir.model import DistributedType, IRType, Node, TupleType
from triton.flagmega.ir.ops.core import (
    CostKind,
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import (
    can_materialize_matmul_partial,
)
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition(
    "ntt.packed_qkv_parallel_linear_combine",
    namespace="ntt",
    functional_name="packed_qkv_parallel_linear_combine",
    display_name="NTT.PackedQKVParallelLinearCombine",
)
class PackedQKVParallelLinearCombine(OpDefinition):
    """A target-neutral collective boundary for a three-field Q/K/V tuple."""

    qkv = input_parameter(is_ir_type())
    output_type = attribute_parameter(positional=True)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["output_type"], IRType):
            raise IRSchemaError(
                "PackedQKVParallelLinearCombine output_type must be an IRType."
            )
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        input_type = cls.qkv.type_of(inputs)
        output_type = cls.output_type.read(inputs, attrs)
        if not can_materialize_packed_qkv(input_type, output_type):
            raise IRSchemaError(
                "PackedQKVParallelLinearCombine cannot materialize "
                f"{input_type!r} into {output_type!r}."
            )
        return output_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        del node, context
        return cls.qkv.read(arguments)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        if not isinstance(node.type, TupleType):
            return OpCost(notes=("invalid-packed-qkv-combine-type",))
        bytes_written = _sum_optional(tuple(
            tensor_nbytes(field.tensor if isinstance(field, DistributedType) else field)
            for field in node.type.fields
        ))
        return OpCost(
            bytes_read=bytes_written,
            bytes_written=bytes_written,
            communication_bytes=None,
            synchronizations=None,
            kind=CostKind.ANALYTIC,
            model="flagmega.packed-qkv-combine/v1",
            notes=("coupled-sum-partial-materialization",),
        )


def can_materialize_packed_qkv(input_type: IRType, output_type: IRType) -> bool:
    """Match nncase's three-field identity or coupled Sum-partial contract."""

    if not isinstance(input_type, TupleType) or not isinstance(output_type, TupleType):
        return False
    if len(input_type.fields) != 3 or len(output_type.fields) != 3:
        return False
    if input_type == output_type:
        return all(_is_materialized(field) for field in input_type.fields)
    input_fields = input_type.fields
    output_fields = output_type.fields
    if not all(isinstance(field, DistributedType) for field in input_fields):
        return False
    if not all(isinstance(field, DistributedType) for field in output_fields):
        return False
    distributed_inputs = tuple(input_fields)
    first_partial = distributed_inputs[0].partial
    if (
        not isinstance(first_partial, SBPPartial)
        or first_partial.reduce_op is not ReduceOp.SUM
        or any(field.partial != first_partial for field in distributed_inputs[1:])
    ):
        return False
    return all(
        can_materialize_matmul_partial(source, target)
        for source, target in zip(input_fields, output_fields)
    )


def _is_materialized(value_type: IRType) -> bool:
    return not isinstance(value_type, DistributedType) or (
        value_type.partial is None
        and all(not isinstance(policy, SBPPartial) for policy in value_type.axis_policies)
    )


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    return None if any(value is None for value in values) else sum(
        value for value in values if value is not None
    )


__all__ = ["PackedQKVParallelLinearCombine", "can_materialize_packed_qkv"]
