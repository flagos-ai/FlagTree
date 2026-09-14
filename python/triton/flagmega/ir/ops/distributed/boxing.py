# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Materialized distributed reshard operation."""

from typing import Mapping, Sequence
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, TupleType
from triton.flagmega.ir.ops.core import OpCost, OpCostFactors, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_elements, tensor_nbytes
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition(
    "distributed.boxing",
    namespace="distributed",
    functional_name="boxing",
    display_name="Distributed.Boxing",
)
class Boxing(OpDefinition):
    # The destination is explicit, including a plain tensor destination for
    # TensorStore. Broadcast lifting would silently undo that type transition.
    supports_broadcast_lifting = False
    # Reference evaluation represents distributed values by their full logical
    # tensor.  Therefore a boxing edge inside an all-constant island is an
    # identity and is safe to execute while materializing the constant asset.
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_ir_type())
    new_type = attribute_parameter(positional=True)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["new_type"], IRType):
            raise IRSchemaError("F.distributed.boxing new_type must be an IRType.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source = cls.value.type_of(inputs)
        target = cls.new_type.read(inputs, attrs)
        if not _same_logical_type(source, target):
            raise IRSchemaError("Distributed.Boxing cannot change the logical tensor type.")
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
        source = node.metadata.get("source_type")
        return OpCost(communication_bytes=None, notes=("reshard", str(source or "dynamic")))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        source = cls.value.type_of(inputs)
        if not (isinstance(source, DistributedType) and isinstance(return_type, DistributedType)
                and source.partial is not None and return_type.partial is None
                and source.placement == return_type.placement):
            return None
        local = local_tensor_type(return_type)
        size, count = tensor_nbytes(local), tensor_elements(local)
        if size is None or count is None:
            return None
        fan_in = prod(source.placement.hierarchy[axis] for axis in source.partial.axes)
        return OpCostFactors(
            elementwise_operations=count * (fan_in - 1),
            chip_global_memory_load_bytes=size * fan_in,
            chip_global_memory_store_bytes=size,
            grid_synchronizations=1,
        )


def _same_logical_type(lhs: IRType, rhs: IRType) -> bool:
    if isinstance(lhs, TupleType) or isinstance(rhs, TupleType):
        return (
            isinstance(lhs, TupleType)
            and isinstance(rhs, TupleType)
            and len(lhs.fields) == len(rhs.fields)
            and all(
                _same_logical_type(lhs_field, rhs_field)
                for lhs_field, rhs_field in zip(lhs.fields, rhs.fields)
            )
        )
    lhs_tensor = lhs.tensor if isinstance(lhs, DistributedType) else lhs
    rhs_tensor = rhs.tensor if isinstance(rhs, DistributedType) else rhs
    return isinstance(lhs_tensor, TensorType) and lhs_tensor == rhs_tensor


__all__ = ["Boxing"]
