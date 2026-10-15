# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class TIR markers for readonly parameter-bank storage."""

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import DType, IRModule, IRType, Node, RefType, TensorType, logical_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    PythonCall,
    NodeRef,
)
from triton.flagmega.ir.parameter_bank import ParameterBankSchema
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition(
    "tir.parameter_bank",
    namespace="tir",
    functional_name="parameter_bank",
    display_name="T.ParameterBank",
)
class ParameterBank(OpDefinition):
    constant_source = True
    supports_broadcast_lifting = False
    schema = attribute_parameter()
    source = attribute_parameter()
    rows = attribute_parameter()
    result_type = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        schema = attrs["schema"]
        rows = attrs["rows"]
        if not isinstance(schema, ParameterBankSchema) or not isinstance(attrs["source"], str) \
                or not attrs["source"] or not isinstance(rows, tuple) or not rows:
            raise IRSchemaError("T.ParameterBank requires a schema, source and rows.")
        if any(not isinstance(row, tuple) or len(row) != len(schema.fields) for row in rows):
            raise IRSchemaError("T.ParameterBank rows must match the schema field count.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        declared = attrs.get("result_type")
        return declared if isinstance(declared, IRType) else attrs["schema"].type

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        attrs = cls.normalize_attrs(node.attrs)
        # The schema describes stored layout; AutoDistributed may give the
        # bank value's element fields a distribution, so compare logically
        # exactly as the row access does.
        if logical_type(node.type) != logical_type(attrs["schema"].type):
            raise IRVerificationError("T.ParameterBank type does not match its schema.", node_id=node.id)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {"schema": attrs["schema"], "source": attrs["source"], "rows": attrs["rows"]}

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost.exact_zero(notes=("parameter-bank-storage",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        return PythonCall(
            "F.tir.parameter_bank",
            (),
            {"schema": node.attrs["schema"], "source": node.attrs["source"],
             "rows": node.attrs["rows"],
             **({} if node.type == node.attrs["schema"].type else {"result_type": node.type}),
             "name": node.id, **({"metadata": node.metadata} if node.metadata else {})},
        )


@op_definition("tir.bank_get", namespace="tir", functional_name="bank_get", display_name="T.BankGet")
class BankGet(OpDefinition):
    supports_broadcast_lifting = False
    bank = input_parameter(is_ir_type())
    index = input_parameter(is_ir_type(), parameter_kind="attribute")

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        if len(inputs) != 2 or not isinstance(inputs[0].type, RefType) \
                or inputs[0].type.name != "parameter_bank":
            raise IRSchemaError("T.BankGet requires a ParameterBank value.")
        return dict(inputs[0].type.fields)["element"]

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        inputs = tuple(module.node_map[value] for value in node.inputs)
        if logical_type(inputs[1].type) not in (TensorType(DType.INT32, ()), TensorType(DType.INT64, ())):
            raise IRVerificationError("T.BankGet index must be scalar i32 or i64.", node_id=node.id)
        # AutoDistributed gives the row's fields a distribution while bank
        # storage keeps one readonly schema, so compare the logical element
        # types here exactly as the graph-level BankGet does.
        if logical_type(node.type) != logical_type(cls.infer_type(inputs, node.attrs)):
            raise IRVerificationError("T.BankGet type does not match the bank schema.", node_id=node.id)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost.exact_zero(notes=("parameter-bank-byte-offset",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        return PythonCall(
            "F.tir.bank_get",
            tuple(NodeRef(value) for value in node.inputs),
            {"name": node.id, **({"metadata": node.metadata} if node.metadata else {})},
        )


__all__ = ["BankGet", "ParameterBank"]
