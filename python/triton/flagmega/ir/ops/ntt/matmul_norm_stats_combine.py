# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit matmul-partial materialization, residual add, and norm statistics."""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import ReduceOp, SBPBroadCast, SBPPartial, SBPSplit
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, TupleType
from triton.flagmega.ir.ops.core import (
    CostKind,
    OpCost,
    OpCostFactors,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._norm import (
    norm_stats_value,
    normalize_axis,
    unpack_default_vector,
)
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "ntt.matmul_norm_stats_combine",
    namespace="ntt",
    functional_name="matmul_norm_stats_combine",
    display_name="NTT.MatMulNormStatsCombine",
)
class MatMulNormStatsCombine(OpDefinition):
    """Materialize an additive matmul result and publish value plus statistics.

    The operation is deliberately target-neutral.  Its first operand may be a
    distributed Sum-partial matmul result; ``addend`` names the desired
    materialized value layout.  Keeping the two results in the graph prevents
    a backend from smuggling normalization statistics through a workspace.
    """

    input = input_parameter(is_tensor())
    addend = input_parameter(is_tensor())
    axis = attribute_parameter()
    use_mean = attribute_parameter()
    inplace_output_parameters = (addend, None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("MatMulNormStatsCombine axis must be an integer.")
        return {"axis": axis, "use_mean": bool(attrs["use_mean"])}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        input_type = cls.input.type_of(inputs)
        output_type = cls.addend.type_of(inputs)
        if tensor_of(input_type) != tensor_of(output_type):
            raise IRSchemaError(
                "MatMulNormStatsCombine input and addend must have the same logical tensor type."
            )
        if not can_materialize_matmul_partial(input_type, output_type):
            raise IRSchemaError(
                f"MatMulNormStatsCombine cannot materialize {input_type!r} into {output_type!r}."
            )
        axis = normalize_axis(int(attrs["axis"]), tensor_of(output_type).rank)
        stats_type = NormStats.infer_type(
            (_typed_node("<materialized_value>", output_type),),
            {"axis": axis, "use_mean": bool(attrs["use_mean"])},
        )
        return TupleType((output_type, stats_type))

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.input.read(arguments) + cls.addend.read(arguments)
        # Dense evaluation observes the logical (already materialized) value;
        # distributed collectives are represented and executed by TIR/runtime.
        value = value.to(dtype=cls.addend.read(arguments).dtype)
        value_type = tensor_of(context.types[cls.addend.read(node.inputs)])
        logical_value = unpack_default_vector(value, value_type)
        return (
            value,
            norm_stats_value(
                logical_value,
                axis=int(node.attrs["axis"]),
                use_mean=bool(node.attrs["use_mean"]),
            ),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        if not isinstance(node.type, TupleType) or len(node.type.fields) != 2:
            return OpCost(notes=("invalid-matmul-norm-stats-combine-type",))
        value = tensor_of(node.type.fields[0])
        stats = tensor_of(node.type.fields[1])
        elements = tensor_elements(value)
        return OpCost(
            flops=None if elements is None else elements * (4 if node.attrs["use_mean"] else 3),
            bytes_read=None if elements is None else 2 * tensor_nbytes(value),
            bytes_written=_sum_optional(tensor_nbytes(value), tensor_nbytes(stats)),
            # Cost(node) deliberately has no access to operand types.  Whether
            # this operation owns a collective is therefore unknown here and
            # is supplied by AutoDistribution/TIR candidate evidence.
            communication_bytes=None,
            synchronizations=None,
            kind=CostKind.ANALYTIC,
            model="flagmega.matmul-norm-stats-combine/v1",
            notes=("materialize-add-additive-normalization-statistics",),
        )

    @classmethod
    def cost_factors(
        cls,
        inputs: Sequence[Node],
        attrs: Mapping[str, object],
        return_type: IRType,
    ) -> OpCostFactors | None:
        """Return nncase-compatible factors for the fused materialization.

        The input is the matmul accumulator, while ``addend`` also names the
        materialized value layout.  A different accumulator/result layout
        means this operation owns the partial reduction and its grid barrier.
        Vector lanes deliberately participate in byte traffic but not in the
        outer-element CPU factor, matching PyNTT's cost evaluator.
        """

        if not isinstance(return_type, TupleType) or len(return_type.fields) != 2:
            return None
        input_type = cls.input.type_of(inputs)
        addend_type = cls.addend.type_of(inputs)
        value_type, stats_type = return_type.fields
        tensors = tuple(
            _local_cost_tensor(value)
            for value in (
                input_type,
                addend_type,
                value_type,
                stats_type,
            )
        )
        if any(
            not dimension.is_fixed
            for tensor in tensors
            for dimension in tensor.shape
        ):
            return None
        input_tensor, addend_tensor, value_tensor, stats_tensor = tensors
        value_elements = prod(
            dimension.fixed_value for dimension in value_tensor.shape
        )
        return OpCostFactors(
            cpu_cycles=value_elements * (4 if attrs["use_mean"] else 3),
            block_local_memory_load_bytes=sum(
                _fixed_tensor_nbytes(tensor)
                for tensor in (input_tensor, value_tensor, addend_tensor)
            ),
            block_local_memory_store_bytes=sum(
                _fixed_tensor_nbytes(tensor)
                for tensor in (value_tensor, stats_tensor)
            ),
            grid_synchronizations=int(input_type != value_type),
        )


def can_materialize_matmul_partial(input_type: IRType, output_type: IRType) -> bool:
    """Whether a Sum-partial value can become ``output_type`` in one combine.

    This is the Python port of nncase's
    ``PackedMatMulNormStatsCombineEvaluator.CanMaterialize``.  It compares
    placement hierarchy policies, not target names or mesh sizes.
    """

    if input_type == output_type:
        return _is_materialized(output_type)
    if not isinstance(input_type, DistributedType) or not isinstance(output_type, DistributedType):
        return False
    if (
        input_type.tensor != output_type.tensor
        or input_type.placement != output_type.placement
        or input_type.partial is None
        or input_type.partial.reduce_op is not ReduceOp.SUM
        or output_type.partial is not None
    ):
        return False
    input_hierarchy = _hierarchy_policies(input_type)
    output_hierarchy = _hierarchy_policies(output_type)
    if input_hierarchy is None or output_hierarchy is None:
        return False
    partial_axes = frozenset(input_type.partial.axes)
    for hierarchy_axis, (source, target) in enumerate(zip(input_hierarchy, output_hierarchy)):
        if hierarchy_axis in partial_axes:
            if source is not None:
                return False
            # A reduced axis may remain broadcast or become an output split.
            continue
        if source is None:
            if target is not None:
                return False
        elif target is not None and source != target:
            return False
        # A non-partial split may be gathered to broadcast.
    return True


def _is_materialized(value_type: IRType) -> bool:
    return not isinstance(value_type, DistributedType) or (
        value_type.partial is None
        and all(not isinstance(policy, SBPPartial) for policy in value_type.axis_policies)
    )


def _hierarchy_policies(value_type: DistributedType) -> tuple[int | None, ...] | None:
    result: list[int | None] = [None] * value_type.placement.rank
    for tensor_axis, policy in enumerate(value_type.axis_policies):
        if isinstance(policy, SBPBroadCast):
            continue
        if not isinstance(policy, SBPSplit):
            return None
        for hierarchy_axis in policy.hierarchy_axes:
            if result[hierarchy_axis] is not None:
                return None
            result[hierarchy_axis] = tensor_axis
    return tuple(result)


def _typed_node(node_id: str, value_type: IRType) -> Node:
    return Node(node_id, "builtin.var", (), value_type, attrs={"name": node_id})


def _local_cost_tensor(value: IRType) -> TensorType:
    from triton.flagmega.ir.distributed_type import local_tensor_type

    return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)


def _fixed_tensor_nbytes(value: TensorType) -> int:
    return prod(dimension.fixed_value for dimension in value.shape) * value.dtype.itemsize


def _sum_optional(lhs: int | None, rhs: int | None) -> int | None:
    return None if lhs is None or rhs is None else lhs + rhs


__all__ = ["MatMulNormStatsCombine", "can_materialize_matmul_partial"]
