# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Read-only distributed alias over a logical tensor."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_type import sharded_view_error
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "distributed.sharded_view",
    namespace="distributed",
    functional_name="sharded_view",
    display_name="Distributed.ShardedView",
)
class ShardedView(OpDefinition):
    # A view of compile-time storage remains compile-time storage.  Marking
    # this explicitly lets AutoPacking recipes survive AutoDistribution and
    # freeze before TIR lowering instead of becoming runtime reshape kernels.
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    byte_preserving_input_parameters = (value,)
    new_type = attribute_parameter(positional=True)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["new_type"], DistributedType):
            raise IRSchemaError("F.distributed.sharded_view new_type must be DistributedType.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source = cls.value.type_of(inputs)
        target = cls.new_type.read(inputs, attrs)
        reason = sharded_view_error(source, target)
        if reason is not None:
            raise IRSchemaError(reason)
        return target

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.value.read(arguments)

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        del node, context
        return cls.value.read(arguments)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("read-only-sharded-alias",))


__all__ = ["ShardedView"]
