# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Checkpoint-backed parameter bank and dynamic row projection."""

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import DType, IRType, Node, RefType, TensorType, logical_type
from triton.flagmega.ir.ops.core import (
    OpDefinition,
    PythonCall,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.parameter_bank import ParameterBankSchema
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition("builtin.parameter_bank", namespace="builtin", functional_name="parameter_bank",
               display_name="ParameterBank")
class ParameterBank(OpDefinition):
    constant_source = True
    supports_broadcast_lifting = False
    # A bank reference is the one originator that gets destructured downstream:
    # `control.bank_get` reads its element tuple and a `get_item` per field
    # turns each weight into its own call argument, so AutoDistributed gives
    # those leaves -- and therefore this handle -- a distributed contract the
    # logical schema cannot express.  The handle carries it as an optional
    # positional result type, the way `builtin.const_asset` does, so the
    # contract survives a write-out/read-back of the IR instead of collapsing
    # back to the schema's logical type.
    result_type = attribute_parameter(positional=True, default=None)
    schema = attribute_parameter()
    source = attribute_parameter()
    rows = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = super().normalize_attrs(attrs)
        schema = attrs["schema"]
        source = attrs["source"]
        raw_rows = attrs["rows"]
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, (tuple, list)):
            raise IRSchemaError("ParameterBank rows must be a sequence of rows.")
        rows = tuple(tuple(key for key in row) for row in raw_rows if isinstance(row, (tuple, list)))
        if not isinstance(schema, ParameterBankSchema) or not isinstance(source, str) or not source or not rows \
                or len(rows) != len(raw_rows):
            raise IRSchemaError("ParameterBank requires a schema, source and rows.")
        # A field whose program fuses several stored tensors names one key per
        # stored source, so a checkpoint entry is one key or a tuple of keys.
        def valid(field, key):
            keys = key if isinstance(key, tuple) else (key,)
            if field.transform != "checkpoint":
                return isinstance(key, (str, int, float, bool, tuple))
            return len(keys) == len(field.stored_types) and all(
                isinstance(value, str) and value for value in keys)

        if any(
            len(row) != len(schema.fields)
            or any(not isinstance(key, (str, int, float, bool, tuple)) for key in row)
            or any(not valid(field, key) for field, key in zip(schema.fields, row, strict=True))
            for row in rows
        ):
            raise IRSchemaError("ParameterBank rows must match the schema field count.")
        result_type = attrs.get("result_type")
        if result_type is not None:
            if not isinstance(result_type, IRType):
                raise IRSchemaError(
                    "F.builtin.parameter_bank's positional result type must be an IR type."
                )
            if logical_type(result_type) != logical_type(schema.type):
                raise IRSchemaError(
                    "F.builtin.parameter_bank's result type must carry the schema's "
                    "logical shape."
                )
        return {**attrs, "rows": rows, "result_type": result_type}

    @classmethod
    def infer_type(cls, inputs, attrs):
        # A distributed handle records its own contract; the schema stays the
        # logical fallback for a bank AutoDistributed has not reached yet.
        result_type = attrs.get("result_type")
        return attrs["schema"].type if result_type is None else result_type

    @classmethod
    def ir_attrs(cls, attrs):
        return {
            "schema": attrs["schema"],
            "source": attrs["source"],
            "rows": attrs["rows"],
        }

    @classmethod
    def verify(cls, node, module):
        cls.verify_arity(node)
        attrs = cls.normalize_attrs({**node.attrs, "result_type": node.type})
        # The schema describes the stored rows, which are logical: a row is
        # read through `control.bank_get` and destructured into one call
        # argument per field, so AutoDistributed gives those leaves a
        # distributed contract.  The handle then carries that contract while
        # the schema stays logical, and only the logical shape has to agree.
        if logical_type(node.type) != logical_type(attrs["schema"].type):
            raise IRVerificationError("ParameterBank type does not match its schema.", node_id=node.id)

    @classmethod
    def python_call(cls, node):
        # Emit the handle's own type positionally so a distributed contract
        # survives the round trip; `infer_type` would otherwise rebuild the
        # schema's logical type and silently drop it.
        keywords = {
            "schema": node.attrs["schema"],
            "source": node.attrs["source"],
            "rows": node.attrs["rows"],
            "name": node.id,
        }
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall("F.builtin.parameter_bank", (node.type,), keywords)

    @classmethod
    def evaluate(cls, node, arguments, context):
        return ParameterBankValue(node, context)


class ParameterBankValue:
    def __init__(self, node, context):
        self.node = node
        self.context = context

    def row(self, index):
        index = int(index)
        attrs = self.node.attrs
        if not 0 <= index < len(attrs["rows"]):
            raise IndexError("ParameterBank row index is out of range.")
        schema = attrs["schema"]
        values = []
        for field, key in zip(schema.fields, attrs["rows"][index], strict=True):
            if field.transform == "literal":
                from triton.flagmega.ir.ops.builtin.scalar_const import ScalarConst

                values.append(ScalarConst.evaluate(
                    Node(f"{self.node.id}.row{index}.{field.name}", "builtin.scalar_const", (), field.type,
                         attrs={"value": key}), (), self.context))
                continue
            if field.transform not in {"identity", "checkpoint"}:
                raise IRSchemaError(f"Unsupported ParameterBank field transform {field.transform!r}.")
            keys = key if isinstance(key, tuple) else (key,)
            stored = tuple(
                self.context.weight_value(Node(
                    f"{self.node.id}.row{index}.{field.name}.{ordinal}", "builtin.weight", (),
                    stored_type, attrs={"name": name, "source": attrs["source"], "key": name},
                ))
                for ordinal, (name, stored_type) in enumerate(zip(keys, field.stored_types, strict=True))
            )
            values.append(stored[0] if field.program is None else self._apply(field, stored, index))
        return tuple(values)

    def _apply(self, field, stored, index):
        """Replay a field's offline transform for one row.

        The same closure runs for every row, so this evaluates the schema's
        program rather than any per-layer copy of it.

        A field program is its own name scope: its nodes and sources are not
        module nodes, so an op that reads an operand's declared type (a pack
        needs its input rank) cannot find them in the module type map.  Bind
        the program's own types for the replay, the way a constant recipe
        does, or such an op raises KeyError on the source name.
        """
        from triton.flagmega.ir.ops.core import get_definition

        values = dict(zip(field.program.sources, stored, strict=True))
        sources = tuple(
            Node(name, "builtin.weight", (), stored_type, attrs={"name": name})
            for name, stored_type in zip(
                field.program.sources, field.stored_types, strict=True
            )
        )
        with self.context.type_scope((*sources, *field.program.nodes)):
            for node in field.program.nodes:
                arguments = tuple(values[value] for value in node.inputs)
                renamed = Node(f"{self.node.id}.row{index}.{field.name}.{node.id}", node.op, node.inputs,
                               node.type, attrs=dict(node.attrs), effect=node.effect)
                values[node.id] = get_definition(node.op).evaluate(renamed, arguments, self.context)
        return values[field.program.output]


@op_definition("control.bank_get", namespace="control", functional_name="bank_get", display_name="BankGet")
class BankGet(OpDefinition):
    supports_broadcast_lifting = False
    bank = input_parameter(is_ir_type())
    index = input_parameter(is_ir_type(), parameter_kind="attribute")

    @classmethod
    def infer_type(cls, inputs, attrs):
        if len(inputs) != 2 or not isinstance(inputs[0].type, RefType) or inputs[0].type.name != "parameter_bank":
            raise IRSchemaError("BankGet requires a ParameterBank value.")
        return dict(inputs[0].type.fields)["element"]

    @classmethod
    def verify(cls, node, module):
        cls.verify_arity(node)
        inputs = tuple(module.node_map[value] for value in node.inputs)
        if logical_type(inputs[1].type) not in (TensorType(DType.INT32, ()), TensorType(DType.INT64, ())):
            raise IRVerificationError("BankGet index must be scalar i64.", node_id=node.id)
        if logical_type(cls.infer_type(inputs, node.attrs)) != logical_type(node.type):
            raise IRVerificationError("BankGet type does not match the bank schema.", node_id=node.id)

    @classmethod
    def evaluate(cls, node, arguments, context):
        return arguments[0].row(arguments[1])


__all__ = ["BankGet", "ParameterBank", "ParameterBankValue"]
