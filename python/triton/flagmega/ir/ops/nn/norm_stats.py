# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Additive normalization statistics, aligned with nncase ``NN.NormStats``."""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBP, SBPBroadCast, SBPPartial, SBPSplit
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.ops.core import (
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
    stats_tensor_type,
    unpack_default_vector,
)
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "nn.norm_stats",
    namespace="nn",
    functional_name="norm_stats",
    display_name="NN.NormStats",
)
class NormStats(OpDefinition):
    value = input_parameter(is_tensor(), name="input")
    axis = attribute_parameter()
    use_mean = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("NormStats axis must be an integer.")
        return {"axis": axis, "use_mean": bool(attrs["use_mean"])}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        value = tensor_of(value_type)
        axis = normalize_axis(int(attrs["axis"]), value.rank)
        stats = stats_tensor_type(value, axis, bool(attrs["use_mean"]))
        if not isinstance(value_type, DistributedType):
            return stats
        if value_type.partial is not None or any(
            isinstance(policy, SBPPartial) for policy in value_type.axis_policies
        ):
            raise IRSchemaError("NormStats input must not be partial.")
        policies = [SBP.broadcast() for _ in range(value.rank + 1)]
        preserved: set[int] = set()
        reduced: set[int] = set()
        for index, policy in enumerate(value_type.axis_policies):
            if index < axis:
                policies[index + 1] = policy
                if isinstance(policy, SBPSplit):
                    preserved.update(policy.hierarchy_axes)
            elif isinstance(policy, SBPSplit):
                reduced.update(policy.hierarchy_axes)
            elif not isinstance(policy, SBPBroadCast):
                raise IRSchemaError(
                    f"NormStats does not support policy {policy} on normalized axis {index}.")
        if preserved.intersection(reduced):
            raise IRSchemaError("NormStats cannot preserve and reduce the same placement axis.")
        partial = None if not reduced else SBP.partial(tuple(sorted(reduced)))
        return DistributedType(stats, tuple(policies), value_type.placement, partial)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value_type = tensor_of(context.types[cls.value.read(node.inputs)])
        value = unpack_default_vector(cls.value.read(arguments), value_type)
        return norm_stats_value(
            value,
            axis=int(node.attrs["axis"]),
            use_mean=bool(node.attrs["use_mean"]),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = tensor_of(node.type)
        elements = tensor_elements(output)
        return OpCost(
            flops=None if elements is None else elements * (3 if node.attrs["use_mean"] else 2),
            bytes_read=None,
            bytes_written=tensor_nbytes(output),
            notes=("additive-normalization-statistics",),
        )

    @classmethod
    def cost_factors(
        cls,
        inputs: Sequence[Node],
        attrs: Mapping[str, object],
        return_type: IRType,
    ) -> OpCostFactors | None:
        input_tensor = _local_cost_tensor(cls.value.type_of(inputs))
        output_tensor = _local_cost_tensor(return_type)
        if any(
            not dimension.is_fixed
            for tensor in (input_tensor, output_tensor)
            for dimension in tensor.shape
        ):
            return None
        input_elements = prod(
            dimension.fixed_value for dimension in input_tensor.shape
        )
        return OpCostFactors(
            cpu_cycles=input_elements * (3 if attrs["use_mean"] else 2),
            block_local_memory_load_bytes=_fixed_tensor_nbytes(input_tensor),
            block_local_memory_store_bytes=_fixed_tensor_nbytes(output_tensor),
        )


def _local_cost_tensor(value: IRType):
    from triton.flagmega.ir.distributed_type import local_tensor_type

    return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)


def _fixed_tensor_nbytes(value) -> int:
    return prod(dimension.fixed_value for dimension in value.shape) * value.dtype.itemsize


__all__ = ["NormStats"]
