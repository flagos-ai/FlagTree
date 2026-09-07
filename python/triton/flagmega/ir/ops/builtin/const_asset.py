# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opaque reference exported by a frozen constant recipe."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import IRModule, IRType, Node
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    op_definition,
    tensor_nbytes,
)


@op_definition(
    "builtin.const_asset",
    namespace="builtin",
    functional_name="const_asset",
    display_name="ConstAssetRef",
)
class ConstAsset(OpDefinition):
    result_type = attribute_parameter(positional=True)
    recipe = attribute_parameter()
    output = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        result_type = cls.result_type.read((), attrs)
        if not isinstance(result_type, IRType):
            raise IRSchemaError("F.builtin.const_asset requires one positional IR result type.")
        recipe = str(cls.recipe.read((), attrs))
        output = str(cls.output.read((), attrs))
        if not recipe or not output:
            raise IRSchemaError("F.builtin.const_asset requires non-empty recipe and output names.")
        return {"result_type": result_type, "recipe": recipe, "output": output}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return cls.result_type.read(inputs, attrs)

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {"recipe": cls.recipe.read((), attrs), "output": cls.output.read((), attrs)}

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        if set(node.attrs) != {"recipe", "output"}:
            raise IRVerificationError("builtin.const_asset requires recipe/output attributes.", node_id=node.id)

    @classmethod
    def evaluate(cls, node, arguments, context):
        return context.constant_asset_value(node)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(bytes_read=size, notes=("frozen-constant-asset",))

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {
            "recipe": node.attrs["recipe"],
            "output": node.attrs["output"],
            "name": node.id,
        }
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall("F.builtin.const_asset", (node.type,), keywords)


__all__ = ["ConstAsset"]
