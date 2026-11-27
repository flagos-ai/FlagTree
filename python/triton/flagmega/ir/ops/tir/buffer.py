# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TIR buffer/weight reference definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import IRModule, IRType, Node
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, PythonCall, attribute_parameter, op_definition, tensor_nbytes


@op_definition("tir.buffer", namespace="tir", functional_name="buffer", display_name="T.Buffer")
class Buffer(OpDefinition):
    result_type = attribute_parameter(positional=True)
    weight_name = attribute_parameter()
    source = attribute_parameter()
    key = attribute_parameter()
    storage = attribute_parameter()
    alignment = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if not isinstance(cls.result_type.read((), attrs), IRType):
            raise IRSchemaError("F.tir.buffer requires one positional IR result type.")
        alignment = int(cls.alignment.read((), attrs))
        if alignment <= 0:
            raise IRSchemaError("F.tir.buffer alignment must be positive.")
        return {
            "result_type": cls.result_type.read((), attrs),
            "weight_name": str(cls.weight_name.read((), attrs)),
            "source": str(cls.source.read((), attrs)),
            "key": str(cls.key.read((), attrs)),
            "storage": str(cls.storage.read((), attrs)),
            "alignment": alignment,
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return cls.result_type.read(inputs, attrs)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "name": cls.weight_name.read((), attrs),
            "source": cls.source.read((), attrs),
            "key": cls.key.read((), attrs),
            "storage": cls.storage.read((), attrs),
            "alignment": cls.alignment.read((), attrs),
        }

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        required = {"name", "source", "key", "storage", "alignment"}
        if set(node.attrs) != required or int(node.attrs["alignment"]) <= 0:
            raise IRVerificationError(f"tir.buffer requires valid attributes {sorted(required)}.", node_id=node.id)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(bytes_read=size, notes=(str(node.attrs["storage"]),))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {
            "name": node.id,
            "weight_name": node.attrs["name"],
            "source": node.attrs["source"],
            "key": node.attrs["key"],
            "storage": node.attrs["storage"],
            "alignment": node.attrs["alignment"],
        }
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall("F.tir.buffer", (node.type,), keywords)
