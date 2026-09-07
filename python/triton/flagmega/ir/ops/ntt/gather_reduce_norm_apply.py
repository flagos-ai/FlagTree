# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused partial-statistics reduction and normalization apply."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import ReduceOp, SBPBroadCast, SBPPartial
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "ntt.gather_reduce_norm_apply",
    namespace="ntt",
    functional_name="gather_reduce_norm_apply",
    display_name="NTT.GatherReduceNormApply",
)
class GatherReduceNormApply(OpDefinition):
    """Consume compact Sum-partial statistics without materializing a copy.

    ``materialized_stats_type`` records the broadcast statistics value removed
    by fusion.  The input remains the compact per-owner partial value so alias
    analysis and bufferization retain the physical communication contract.
    """

    partial_stats = input_parameter(
        is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners()
    )
    value = input_parameter(is_tensor(), name="input")
    scale = input_parameter(is_tensor())
    bias = input_parameter(is_tensor())
    materialized_stats_type = attribute_parameter()
    axis = attribute_parameter()
    epsilon = attribute_parameter()
    use_mean = attribute_parameter()
    round_before_scale = attribute_parameter(default=False)
    has_bias = attribute_parameter(default=True)
    inplace_input_parameters = (value,)
    supports_broadcast_lifting = False

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        materialized = attrs["materialized_stats_type"]
        if not isinstance(materialized, DistributedType):
            raise IRSchemaError(
                "GatherReduceNormApply materialized_stats_type must be distributed."
            )
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("GatherReduceNormApply axis must be an integer.")
        epsilon = float(attrs["epsilon"])
        if epsilon <= 0:
            raise IRSchemaError("GatherReduceNormApply epsilon must be positive.")
        return {
            "materialized_stats_type": materialized,
            "axis": axis,
            "epsilon": epsilon,
            "use_mean": bool(attrs["use_mean"]),
            "round_before_scale": NormApply.normalize_attrs(_norm_attrs(attrs))["round_before_scale"],
            "has_bias": bool(attrs["has_bias"]),
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        partial = cls.partial_stats.type_of(inputs)
        materialized = attrs["materialized_stats_type"]
        if not isinstance(partial, DistributedType):
            raise IRSchemaError(
                "GatherReduceNormApply partial_stats must be distributed."
            )
        assert isinstance(materialized, DistributedType)
        if (
            partial.tensor != materialized.tensor
            or partial.placement != materialized.placement
            or partial.axis_policies != materialized.axis_policies
        ):
            raise IRSchemaError(
                "GatherReduceNormApply partial and materialized statistics must "
                "have identical tensor, placement, and SBP policies."
            )
        if (
            partial.partial is None
            or partial.partial.reduce_op is not ReduceOp.SUM
            or not partial.partial.axes
            or materialized.partial is not None
            or any(
                not isinstance(policy, SBPBroadCast)
                for policy in partial.axis_policies
            )
            or any(
                isinstance(policy, SBPPartial)
                for policy in materialized.axis_policies
            )
        ):
            raise IRSchemaError(
                "GatherReduceNormApply requires non-empty Sum-partial broadcast "
                "statistics reduced to a non-partial value."
            )
        stats = Node("<materialized_stats>", "builtin.var", (), materialized)
        return NormApply.infer_type(
            (cls.value.read(inputs), stats, cls.scale.read(inputs), cls.bias.read(inputs)),
            _norm_attrs(attrs),
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        partial_stats = cls.partial_stats.read(arguments)
        scale = cls.scale.read(arguments)
        bias = cls.bias.read(arguments)
        if not bool(node.attrs["has_bias"]):
            try:
                bias = context.torch.zeros_like(bias)
            except AttributeError as error:
                raise EvaluationError(
                    "GatherReduceNormApply evaluation requires a torch-like context."
                ) from error
        proxy = replace(
            node,
            op=NormApply.op_name,
            inputs=(node.inputs[1], node.inputs[0], node.inputs[2], node.inputs[3]),
            attrs=_norm_attrs(node.attrs),
        )
        # Reference evaluation stores distributed values as full logical
        # tensors.  The partial statistics argument therefore already denotes
        # the additive logical value; execution backends perform owner reads.
        return NormApply.evaluate(
            proxy, (value, partial_stats, scale, bias), context
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = tensor_of(node.type)
        elements = tensor_elements(output)
        output_bytes = tensor_nbytes(output)
        partial = node.attrs.get("materialized_stats_type")
        communication = (
            None
            if not isinstance(partial, DistributedType)
            else tensor_nbytes(partial.tensor)
            * partial.placement.size
        )
        return OpCost(
            flops=(
                None
                if elements is None
                else elements * (7 if node.attrs["use_mean"] else 5)
            ),
            bytes_read=None,
            bytes_written=output_bytes,
            communication_bytes=communication,
            synchronizations=0,
            notes=("fused-sum-partial-statistics-normalization-apply",),
        )


def _norm_attrs(attrs: Mapping[str, object]) -> dict[str, object]:
    return {
        "axis": int(attrs["axis"]),
        "epsilon": float(attrs["epsilon"]),
        "use_mean": bool(attrs["use_mean"]),
        "round_before_scale": attrs.get("round_before_scale", False),
    }


__all__ = ["GatherReduceNormApply"]
