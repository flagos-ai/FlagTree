# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compile-time owner-major materialization of an SBP distributed tensor."""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_type import (
    local_shape,
)
from triton.flagmega.ir.local_shard import local_shard_descriptor
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_distributed


@op_definition(
    "distributed.materialize_local_shards",
    namespace="distributed",
    functional_name="materialize_local_shards",
    display_name="Distributed.MaterializeLocalShards",
)
class MaterializeLocalShards(OpDefinition):
    """Convert a logical reference tensor to dense owner-major local shards.

    This is an offline readonly-data operation, not an executable reshard.  It
    is target-neutral: the complete staged SBP policy and placement are read
    from the input type, while the physical target later decides whether this
    owner-major asset is a supported kernel ABI.
    """

    const_evaluable = True
    numpy_materializable = True
    supports_broadcast_lifting = False
    value = input_parameter(is_distributed())

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        del attrs
        source = cls.value.type_of(inputs)
        assert isinstance(source, DistributedType)
        if source.partial is not None:
            raise IRSchemaError(
                "MaterializeLocalShards cannot materialize a partial value."
            )
        owner_count = 1
        for extent in source.placement.hierarchy:
            owner_count *= extent
        return TensorType(
            source.tensor.dtype,
            (owner_count, *local_shape(source)),
            source.tensor.layout,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        source_type = context.types[cls.value.read(node.inputs)]
        if not isinstance(source_type, DistributedType):  # pragma: no cover
            raise IRSchemaError("MaterializeLocalShards input lost its DistributedType.")
        value = cls.value.read(arguments)
        outer_shape = source_type.tensor.shape
        local = local_shape(source_type)
        if any(not dimension.is_fixed for dimension in (*outer_shape, *local)):
            raise IRSchemaError(
                "MaterializeLocalShards reference evaluation requires fixed dimensions."
            )
        target_outer = tuple(dimension.fixed_value for dimension in local)
        owner_coordinates = product(*(
            range(extent) for extent in source_type.placement.hierarchy
        ))
        shards = []
        for coordinates in owner_coordinates:
            descriptor = local_shard_descriptor(source_type, coordinates)
            shard = value
            selected_shape = []
            for tensor_axis, axis in enumerate(descriptor.axes):
                if not axis.active_extent.is_fixed:
                    raise IRSchemaError(
                        "MaterializeLocalShards reference evaluation requires a "
                        "fixed per-owner active extent."
                    )
                active_extent = axis.active_extent.fixed_value
                indices = tuple(
                    axis.map_local_to_global(index).fixed_value
                    for index in range(active_extent)
                )
                selected_shape.append(active_extent)
                index = context.torch.tensor(
                    indices,
                    dtype=context.torch.int64,
                    device=value.device,
                )
                shard = context.torch.index_select(shard, tensor_axis, index)
            if tuple(selected_shape) != target_outer:
                physical_shape = (*target_outer, *tuple(value.shape[len(outer_shape):]))
                padded = context.torch.zeros(
                    physical_shape,
                    dtype=value.dtype,
                    device=value.device,
                )
                padded[tuple(slice(0, extent) for extent in selected_shape)] = shard
                shard = padded
            shards.append(shard)
        return context.torch.stack(tuple(shards), dim=0).contiguous()

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        source_type = context.types[cls.value.read(node.inputs)]
        if not isinstance(source_type, DistributedType):  # pragma: no cover
            raise IRSchemaError("MaterializeLocalShards input lost its DistributedType.")
        value = cls.value.read(arguments)
        outer_shape = source_type.tensor.shape
        local = local_shape(source_type)
        if any(not dimension.is_fixed for dimension in (*outer_shape, *local)):
            raise IRSchemaError(
                "MaterializeLocalShards NumPy materialization requires fixed dimensions."
            )
        target_outer = tuple(dimension.fixed_value for dimension in local)
        owner_coordinates = product(*(
            range(extent) for extent in source_type.placement.hierarchy
        ))
        shards = []
        for coordinates in owner_coordinates:
            descriptor = local_shard_descriptor(source_type, coordinates)
            shard = value
            selected_shape = []
            for tensor_axis, axis in enumerate(descriptor.axes):
                if not axis.active_extent.is_fixed:
                    raise IRSchemaError(
                        "MaterializeLocalShards NumPy materialization requires a "
                        "fixed per-owner active extent."
                    )
                active_extent = axis.active_extent.fixed_value
                indices = tuple(
                    axis.map_local_to_global(index).fixed_value
                    for index in range(active_extent)
                )
                selected_shape.append(active_extent)
                shard = context.numpy.take(shard, indices, axis=tensor_axis)
            if tuple(selected_shape) != target_outer:
                physical_shape = (*target_outer, *tuple(value.shape[len(outer_shape):]))
                padded = context.numpy.zeros(physical_shape, dtype=value.dtype)
                padded[tuple(slice(0, extent) for extent in selected_shape)] = shard
                shard = padded
            shards.append(shard)
        return context.as_contiguous(context.numpy.stack(tuple(shards), axis=0))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(
            bytes_written=tensor_nbytes(node.type),
            notes=("offline-owner-major-sbp-materialization",),
        )

__all__ = ["MaterializeLocalShards"]
