# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Additive normalization statistics, aligned with nncase ``NN.NormStats``."""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBP, SBPBroadCast, SBPPartial, SBPSplit
from triton.flagmega.ir.model import DistributedType, IRType, Node, VectorType
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
    reduction_group_size = attribute_parameter(default=1)
    reduction_order = attribute_parameter(default="unspecified")
    reduction_block_size = attribute_parameter(default=None)
    reduction_block_traversal = attribute_parameter(default="forward")
    reduction_tree_group_size = attribute_parameter(default=None)
    reduction_group_seed_index = attribute_parameter(default=0)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("NormStats axis must be an integer.")
        group = attrs["reduction_group_size"]
        if type(group) is not int or group <= 0 or group & (group - 1):
            raise IRSchemaError("NormStats reduction_group_size must be a positive power of two.")
        order = attrs["reduction_order"]
        if not isinstance(order, str) or order not in {"unspecified", "tree", "tile_fma", "group_fma"}:
            raise IRSchemaError("NormStats reduction_order must be unspecified, tree, tile_fma or group_fma.")
        seed = attrs["reduction_group_seed_index"]
        if type(seed) is not int or not 0 <= seed < group or (seed != 0 and order != "group_fma"):
            raise IRSchemaError("NormStats reduction_group_seed_index must name a group element, and nonzero seeds require group_fma.")
        block = attrs["reduction_block_size"]
        if block is not None and (type(block) is not int or block < group or block & (block - 1)):
            raise IRSchemaError("NormStats reduction_block_size must be a power of two at least as large as its group.")
        if (order == "tile_fma") != (block is not None):
            raise IRSchemaError("NormStats tile_fma requires an explicit reduction_block_size.")
        traversal = attrs["reduction_block_traversal"]
        if not isinstance(traversal, str) or traversal not in {"forward", "reverse"} or (traversal != "forward" and order != "tile_fma"):
            raise IRSchemaError("NormStats reduction_block_traversal must be forward, or reverse for tile_fma.")
        tree_group = attrs["reduction_tree_group_size"]
        if tree_group is not None and (type(tree_group) is not int or tree_group <= 0 or tree_group & (tree_group - 1)):
            raise IRSchemaError("NormStats reduction_tree_group_size must be a positive power of two.")
        if tree_group is not None and order == "unspecified":
            raise IRSchemaError("NormStats reduction_tree_group_size requires an explicit reduction_order.")
        return {"axis": axis, "use_mean": bool(attrs["use_mean"]),
                **({"reduction_block_size": block} if block is not None else {}),
                **({"reduction_block_traversal": traversal} if traversal != "forward" else {}),
                **({"reduction_tree_group_size": tree_group} if tree_group is not None else {}),
                **({"reduction_group_seed_index": seed} if seed != 0 else {}),
                **({"reduction_order": order} if order != "unspecified" else {}),
                **({"reduction_group_size": group} if group > 1 else {})}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        value = tensor_of(value_type)
        axis = normalize_axis(int(attrs["axis"]), value.rank)
        group = attrs.get("reduction_group_size", 1)
        if group > 1 or attrs.get("reduction_order", "unspecified") != "unspecified":
            suffix = value.shape[axis:]
            lanes = prod(value.dtype.lanes) if isinstance(value.dtype, VectorType) else 1
            if any(not dimension.is_fixed for dimension in suffix) or prod(
                    dimension.fixed_value for dimension in suffix) * lanes % group:
                raise IRSchemaError("NormStats reduction_group_size must divide the fixed scalar reduction capacity.")
            capacity = prod(d.fixed_value for d in suffix) * lanes
            tree_capacity = min(capacity, attrs.get("reduction_block_size") or capacity) // group
            tree_capacity = 1 << (tree_capacity - 1).bit_length()
            tree_group = attrs.get("reduction_tree_group_size")
            if tree_group is not None and (tree_group > tree_capacity or tree_capacity % tree_group):
                raise IRSchemaError("NormStats reduction_tree_group_size must divide the grouped reduction capacity.")
            if isinstance(value_type, DistributedType) and any(
                    not isinstance(policy, SBPBroadCast) for policy in value_type.axis_policies[axis:]):
                raise IRSchemaError("Ordered NormStats requires broadcast reduction axes.")
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
        return DistributedType(
            stats, tuple(policies), value_type.placement, partial,
            value_type.exclusive,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        value_type = tensor_of(context.types[cls.value.read(node.inputs)])
        value = unpack_default_vector(cls.value.read(arguments), value_type)
        group = node.attrs.get("reduction_group_size", 1)
        if group > 1 or node.attrs.get("reduction_order", "unspecified") != "unspecified":
            torch = context.torch
            axis = normalize_axis(int(node.attrs["axis"]), value.ndim)
            outer = value.shape[:axis]
            if node.attrs.get("reduction_order") == "tile_fma":
                from triton.flagmega.evaluator.cuda_norm_math import tiled_squares
                square, total = tiled_squares(value.float(), axis=axis, group=group,
                                               block=node.attrs["reduction_block_size"], torch=torch,
                                               traversal=node.attrs.get("reduction_block_traversal", "forward"))
                grouped = None
            elif node.attrs.get("reduction_order") == "group_fma":
                from triton.flagmega.evaluator.cuda_norm_math import grouped_squares
                square, total = grouped_squares(value.float(), axis=axis, group=group,
                                                seed=node.attrs.get("reduction_group_seed_index", 0), torch=torch)
            else:
                grouped = value.float().reshape(*outer, -1, group)
                square = torch.zeros_like(grouped[..., 0])
                total = torch.zeros_like(square)
                for column in range(group):
                    square = square + grouped[..., column].square()
                    if node.attrs["use_mean"]:
                        total = total + grouped[..., column]
            count = square.shape[-1]
            tile = 1 << (count - 1).bit_length()
            square = torch.nn.functional.pad(square, (0, tile - count))
            total = torch.nn.functional.pad(total, (0, tile - count))
            tree_group = node.attrs.get("reduction_tree_group_size")
            if tree_group is not None:
                square = square.reshape(*outer, -1, tree_group)
                total = total.reshape(*outer, -1, tree_group)
                local_lanes = torch.arange(tree_group, device=value.device)
                local_mask = tree_group // 2
                while local_mask:
                    square = square + square.index_select(-1, local_lanes ^ local_mask)
                    total = total + total.index_select(-1, local_lanes ^ local_mask)
                    local_mask //= 2
                square, total = square[..., 0], total[..., 0]
                tile //= tree_group
            lanes = torch.arange(tile, device=value.device)
            mask = tile // 2
            while mask:
                square = square + square.index_select(-1, lanes ^ mask)
                total = total + total.index_select(-1, lanes ^ mask)
                mask //= 2
            shape = (1, *outer, *((1,) * (value.ndim - axis)))
            square = square[..., :1].reshape(shape)
            return torch.cat((total[..., :1].reshape(shape), square), dim=0) if node.attrs["use_mean"] else square
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
        input_elements = tensor_elements(input_tensor)
        return OpCostFactors(
            elementwise_operations=input_elements * (3 if attrs["use_mean"] else 2),
            block_local_memory_load_bytes=_fixed_tensor_nbytes(input_tensor),
            block_local_memory_store_bytes=_fixed_tensor_nbytes(output_tensor),
        )


def _local_cost_tensor(value: IRType):
    from triton.flagmega.ir.distributed_type import local_tensor_type

    return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)


def _fixed_tensor_nbytes(value) -> int:
    return prod(dimension.fixed_value for dimension in value.shape) * value.dtype.itemsize


__all__ = ["NormStats"]
