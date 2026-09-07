# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TIR barrier definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import IRModule, IRType, Node
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, PythonCall, attribute_parameter, op_definition


@op_definition("tir.barrier", namespace="tir", functional_name="barrier", display_name="T.Barrier")
class Barrier(OpDefinition):
    result_type = attribute_parameter(positional=True)
    attrs = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if not isinstance(cls.result_type.read((), attrs), IRType):
            raise IRSchemaError("F.tir.barrier requires one positional IR result type.")
        if not isinstance(cls.attrs.read((), attrs), Mapping):
            raise IRSchemaError("F.tir.barrier requires an attrs mapping.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return cls.result_type.read(inputs, attrs)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return dict(cls.attrs.read((), attrs))

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        if not node.attrs:
            raise IRVerificationError("tir.barrier requires explicit synchronization attributes.", node_id=node.id)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("synchronization",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {"attrs": node.attrs, "name": node.id}
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall("F.tir.barrier", (node.type,), keywords)
