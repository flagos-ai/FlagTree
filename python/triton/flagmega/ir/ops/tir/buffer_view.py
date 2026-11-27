# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed alias of an already planned physical buffer span."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from functools import reduce
from operator import mul

from triton.flagmega.ir.dim_expr import dim
from triton.flagmega.ir.distributed_type import local_shape
from triton.flagmega.ir.model import (
    DistributedType,
    IRModule,
    IRType,
    Node,
    TensorType,
    logical_type,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    NodeRef,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition(
    "tir.buffer_view",
    namespace="tir",
    functional_name="buffer_view",
    display_name="T.BufferView",
)
class BufferView(OpDefinition):
    """A zero-copy logical view backed by the input's MemSpan.

    Bufferization creates this op only after alias analysis has assigned the
    result and its source to the same physical byte range.  Keeping it in TIR
    makes editable Python dumps preserve the logical distributed coordinate
    map without retaining a higher-level distributed dialect operation.
    """

    value = input_parameter(is_ir_type())
    byte_preserving_input_parameters = (value,)
    new_type = attribute_parameter(positional=True)
    alias_kind = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if not isinstance(cls.new_type.read((), attrs), IRType):
            raise IRSchemaError("F.tir.buffer_view new_type must be an IRType.")
        alias_kind = str(cls.alias_kind.read((), attrs))
        if alias_kind not in {"sharded_view", "vector_reinterpret", "reshape"}:
            raise IRSchemaError(
                "F.tir.buffer_view alias_kind must be 'sharded_view' or "
                "a byte-preserving representation view."
            )
        return {"new_type": cls.new_type.read((), attrs), "alias_kind": alias_kind}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source = cls.value.type_of(inputs)
        target = cls.new_type.read(inputs, attrs)
        alias_kind = str(cls.alias_kind.read(inputs, attrs))
        if alias_kind == "sharded_view":
            valid = logical_type(source) == logical_type(target)
        else:
            valid = _same_physical_tensor_extent(source, target)
        if not valid:
            raise IRSchemaError(
                f"T.BufferView {alias_kind!r} cannot alias {source!r} as {target!r}."
            )
        return target

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {"alias_kind": cls.alias_kind.read((), attrs)}

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.value.read(arguments)

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        if set(node.attrs) != {"alias_kind"} or node.attrs["alias_kind"] not in {
            "sharded_view", "vector_reinterpret", "reshape"
        }:
            raise IRVerificationError(
                "tir.buffer_view requires a reviewed alias_kind.", node_id=node.id
            )
        source = module.node_map[node.inputs[0]].type
        if node.attrs["alias_kind"] == "sharded_view":
            valid = logical_type(source) == logical_type(node.type)
        else:
            valid = _same_physical_tensor_extent(source, node.type)
        if not valid:
            raise IRVerificationError(
                "tir.buffer_view source and result have incompatible physical extents.",
                node_id=node.id,
            )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("zero-copy-buffer-alias", str(node.attrs["alias_kind"])))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {
            "alias_kind": node.attrs["alias_kind"],
            "name": node.id,
        }
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall(
            "F.tir.buffer_view",
            (NodeRef(node.inputs[0]), node.type),
            keywords,
        )


def _same_physical_tensor_extent(lhs: IRType, rhs: IRType) -> bool:
    lhs_tensor = lhs.tensor if isinstance(lhs, DistributedType) else lhs
    rhs_tensor = rhs.tensor if isinstance(rhs, DistributedType) else rhs
    if not isinstance(lhs_tensor, TensorType) or not isinstance(rhs_tensor, TensorType):
        return False
    if isinstance(lhs, DistributedType) != isinstance(rhs, DistributedType):
        return False
    if isinstance(lhs, DistributedType) and isinstance(rhs, DistributedType):
        if lhs.placement != rhs.placement:
            return False
        if not _same_extent(
            local_shape(lhs), lhs_tensor.dtype.itemsize,
            local_shape(rhs), rhs_tensor.dtype.itemsize,
        ):
            return False
    return _same_extent(
        lhs_tensor.shape,
        lhs_tensor.dtype.itemsize,
        rhs_tensor.shape,
        rhs_tensor.dtype.itemsize,
    )


def _same_extent(lhs_shape, lhs_itemsize, rhs_shape, rhs_itemsize) -> bool:
    lhs = reduce(mul, lhs_shape, dim(lhs_itemsize))
    rhs = reduce(mul, rhs_shape, dim(rhs_itemsize))
    return lhs.equivalent(rhs)


__all__ = ["BufferView"]
