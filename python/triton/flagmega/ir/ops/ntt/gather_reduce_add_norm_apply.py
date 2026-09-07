# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused partial materialization, residual add, and normalization apply."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import ReduceOp
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DistributedType, IRType, Node, TupleType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._norm import (
    norm_apply_value,
    norm_stats_value,
    normalize_axis,
    repack_default_vector,
    stats_tensor_type,
    unpack_default_vector,
)
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import (
    can_materialize_matmul_partial,
)
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "ntt.gather_reduce_add_norm_apply",
    namespace="ntt",
    functional_name="gather_reduce_add_norm_apply",
    display_name="NTT.GatherReduceAddNormApply",
)
class GatherReduceAddNormApply(OpDefinition):
    """Materialize a Sum-partial value and normalize the residual result.

    The first result is the rounded ``input + addend`` value and may reuse the
    addend buffer.  The second result is the normalized value.  Normalization
    statistics are intentionally not an SSA result: after the fusion proof
    they are private to one collective kernel and become an explicit TIR
    workspace during bufferization.
    """

    input = input_parameter(
        is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners()
    )
    # The collective's work domain may gather a source split or refine a
    # destination split. These accesses and both published results therefore
    # require chip-visible storage, not independent per-CTA copies.
    addend = input_parameter(is_tensor(), memory_effect=MemoryEffect.CHIP_READ)
    scale = input_parameter(is_tensor(), memory_effect=MemoryEffect.CHIP_READ)
    bias = input_parameter(is_tensor(), memory_effect=MemoryEffect.CHIP_READ)
    axis = attribute_parameter()
    epsilon = attribute_parameter()
    use_mean = attribute_parameter()
    round_before_scale = attribute_parameter(default=False)
    has_bias = attribute_parameter(default=True)
    inplace_output_parameters = (addend, None)
    result_memory_effects = (MemoryEffect.CHIP_WRITE, MemoryEffect.CHIP_WRITE)
    supports_broadcast_lifting = False

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("GatherReduceAddNormApply axis must be an integer.")
        epsilon = float(attrs["epsilon"])
        if epsilon <= 0:
            raise IRSchemaError("GatherReduceAddNormApply epsilon must be positive.")
        return {
            "axis": axis,
            "epsilon": epsilon,
            "use_mean": bool(attrs["use_mean"]),
            "round_before_scale": NormApply.normalize_attrs(_norm_attrs(attrs))["round_before_scale"],
            "has_bias": bool(attrs["has_bias"]),
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source_type = cls.input.type_of(inputs)
        value_type = cls.addend.type_of(inputs)
        if (
            not isinstance(source_type, DistributedType)
            or source_type.partial is None
            or source_type.partial.reduce_op is not ReduceOp.SUM
            or not source_type.partial.axes
        ):
            raise IRSchemaError(
                "GatherReduceAddNormApply requires a non-empty Sum-partial input."
            )
        if not can_materialize_matmul_partial(source_type, value_type):
            raise IRSchemaError(
                f"GatherReduceAddNormApply cannot materialize {source_type!r} "
                f"into {value_type!r}."
            )
        materialized = _typed_node("<materialized_value>", value_type)
        value_tensor = tensor_of(value_type)
        axis = normalize_axis(int(attrs["axis"]), value_tensor.rank)
        stats_tensor = stats_tensor_type(
            value_tensor, axis, bool(attrs["use_mean"])
        )
        if isinstance(value_type, DistributedType):
            stats_type: IRType = DistributedType(
                stats_tensor,
                (
                    SBPBroadCast(),
                    *value_type.axis_policies[:axis],
                    *(SBPBroadCast() for _ in range(value_tensor.rank - axis)),
                ),
                value_type.placement,
            )
        else:  # Kept explicit for a useful schema error if the contract grows.
            stats_type = stats_tensor
        normalized_type = NormApply.infer_type(
            (
                materialized,
                _typed_node("<private_stats>", stats_type),
                cls.scale.read(inputs),
                cls.bias.read(inputs),
            ),
            _norm_attrs(attrs),
        )
        return TupleType((value_type, normalized_type))

    @classmethod
    def evaluate(cls, node, arguments, context):
        input_type = tensor_of(context.types[cls.input.read(node.inputs)])
        value_type = tensor_of(context.types[cls.addend.read(node.inputs)])
        scale_type = tensor_of(context.types[cls.scale.read(node.inputs)])
        bias_type = tensor_of(context.types[cls.bias.read(node.inputs)])
        partial = unpack_default_vector(cls.input.read(arguments), input_type)
        addend = unpack_default_vector(cls.addend.read(arguments), value_type)
        scale = unpack_default_vector(cls.scale.read(arguments), scale_type)
        bias = unpack_default_vector(cls.bias.read(arguments), bias_type)
        if not bool(node.attrs["has_bias"]):
            try:
                bias = context.torch.zeros_like(bias)
            except AttributeError as error:
                raise EvaluationError(
                    "GatherReduceAddNormApply evaluation requires a torch-like context."
                ) from error
        value = (partial + addend).to(dtype=addend.dtype)
        stats = norm_stats_value(
            value,
            axis=int(node.attrs["axis"]),
            use_mean=bool(node.attrs["use_mean"]),
        )
        normalized = norm_apply_value(
            value,
            stats,
            scale,
            bias,
            axis=int(node.attrs["axis"]),
            epsilon=float(node.attrs["epsilon"]),
            use_mean=bool(node.attrs["use_mean"]),
            round_before_scale=bool(node.attrs.get("round_before_scale", False)),
        )
        return (
            repack_default_vector(value, value_type),
            repack_default_vector(normalized, value_type),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        if not isinstance(node.type, TupleType) or len(node.type.fields) != 2:
            return OpCost(notes=("invalid-gather-reduce-add-norm-apply-type",))
        value = tensor_of(node.type.fields[0])
        normalized = tensor_of(node.type.fields[1])
        elements = tensor_elements(value)
        return OpCost(
            flops=None if elements is None else elements * (11 if node.attrs["use_mean"] else 9),
            bytes_read=None,
            bytes_written=_sum_optional(
                tensor_nbytes(value), tensor_nbytes(normalized)
            ),
            # Operand types are not available through ``cost(node)``.  The
            # candidate provider records exact owner counts for target cost
            # and launch selection instead of inventing a free collective.
            communication_bytes=None,
            synchronizations=1,
            notes=("fused-gather-reduce-add-private-stats-norm-apply",),
        )


def _norm_attrs(attrs: Mapping[str, object]) -> dict[str, object]:
    return {
        "axis": int(attrs["axis"]),
        "epsilon": float(attrs["epsilon"]),
        "use_mean": bool(attrs["use_mean"]),
        "round_before_scale": attrs.get("round_before_scale", False),
    }


def _typed_node(node_id: str, value_type: IRType) -> Node:
    return Node(node_id, "builtin.var", (), value_type, attrs={"name": node_id})


def _sum_optional(lhs: int | None, rhs: int | None) -> int | None:
    return None if lhs is None or rhs is None else lhs + rhs


__all__ = ["GatherReduceAddNormApply"]
