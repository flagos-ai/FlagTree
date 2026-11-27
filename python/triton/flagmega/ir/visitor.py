# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Memoized whole-module visitor spanning high-level IR, types and TIR."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from triton.flagmega.ir.dim_expr import DimExpr, Dimension
from triton.flagmega.ir.model import (
    CallableType,
    DistributedType,
    Function,
    IRModule,
    IRType,
    Node,
    RefType,
    TensorType,
    TupleType,
)
from triton.flagmega.ir.ops.core import visit_node as dispatch_op_visit
from triton.flagmega.ir.tir import TIRNode, iter_tir_children


class IRVisitor:
    """Bottom-up visitor with one memo entry per shared IR object.

    High-level op dispatch stays owned by each ``OpDefinition`` and therefore
    supports methods such as ``visit_math_add``.  TIR dispatch uses explicit
    node class names such as ``visit_tir_buffer_load``.  The visitor controls
    traversal and memoization only; it is not a source-generated functor.
    """

    def __init__(
        self,
        *,
        visit_types: bool = True,
        visit_attributes: bool = False,
        visit_constant_recipes: bool = True,
        visit_prim_functions: bool = True,
        visit_dead_nodes: bool = False,
    ) -> None:
        self.visit_types = visit_types
        self.visit_attributes = visit_attributes
        self.visit_constant_recipes = visit_constant_recipes
        self.visit_prim_functions = visit_prim_functions
        self.visit_dead_nodes = visit_dead_nodes
        self.node_memo: dict[int, object] = {}
        self.type_memo: dict[int, object] = {}
        self.tir_memo: dict[int, object] = {}

    def run(self, module: IRModule):
        self.node_memo.clear()
        self.type_memo.clear()
        self.tir_memo.clear()
        node_map = module.node_map
        for function in module.functions:
            for node_id in (*function.parameters, *function.outputs):
                self._visit_value(node_id, module, node_map)
            if self.visit_attributes:
                self._visit_embedded(function.attrs)
            self.visit_function(function, module)
        if self.visit_dead_nodes:
            for node in module.nodes:
                self._visit_value(node.id, module, node_map)
        if self.visit_constant_recipes:
            for recipe in module.constant_recipes:
                recipe_map = recipe.node_map
                for output in recipe.outputs:
                    self._visit_value(output, module, recipe_map)
                self.visit_constant_recipe(recipe, module)
        if self.visit_prim_functions:
            for function in (*module.prim_functions, *module.kernel_definitions, *module.execution_functions):
                self._visit_tir(function)
        if self.visit_attributes:
            self._visit_embedded(module.metadata)
        return self.visit_module(module)

    def _visit_value(self, node_id: str, module: IRModule, node_map: Mapping[str, Node]):
        node = node_map[node_id]
        identity = id(node)
        if identity in self.node_memo:
            return self.node_memo[identity]
        for input_id in node.inputs:
            self._visit_value(input_id, module, node_map)
        if self.visit_types:
            self._visit_type(node.type)
        from triton.flagmega.ir.fusion import iter_fusions
        for body in iter_fusions((node.attrs.get("pre_ops", {}), node.attrs.get("post_ops", ()))):
            self._visit_fusion(body)
        if self.visit_attributes:
            self._visit_embedded(node.attrs)
            self._visit_embedded(node.metadata)
        result = dispatch_op_visit(node, self)
        self.node_memo[identity] = result
        return result

    def _visit_fusion(self, body):
        if id(body) in self.node_memo:
            return self.node_memo[id(body)]
        module = body.as_module()
        self._visit_value(body.output, module, module.node_map)
        result = self.visit_fusion(body)
        self.node_memo[id(body)] = result
        return result

    def _visit_type(self, value: IRType):
        identity = id(value)
        if identity in self.type_memo:
            return self.type_memo[identity]
        if isinstance(value, TensorType):
            for dimension in value.shape:
                self.visit_dimension_tree(dimension)
        elif isinstance(value, TupleType):
            for field in value.fields:
                self._visit_type(field)
        elif isinstance(value, CallableType):
            for parameter in value.parameters:
                self._visit_type(parameter)
            self._visit_type(value.return_type)
        elif isinstance(value, RefType):
            for _, field in value.fields:
                self._visit_type(field)
        elif isinstance(value, DistributedType):
            self._visit_type(value.tensor)
        method = getattr(self, f"visit_{_snake_name(type(value).__name__)}", None)
        result = self.visit_type(value) if method is None else method(value)
        self.type_memo[identity] = result
        return result

    def visit_dimension_tree(self, value: Dimension):
        if isinstance(value, DimExpr):
            for operand in value.operands:
                self.visit_dimension_tree(operand)
        return self.visit_dimension(value)

    def _visit_tir(self, node: TIRNode):
        identity = id(node)
        if identity in self.tir_memo:
            return self.tir_memo[identity]
        for child in iter_tir_children(node):
            self._visit_tir(child)
        from triton.flagmega.ir.fusion import iter_fusions
        for body in iter_fusions(getattr(node, "semantic_attrs", {})):
            self._visit_fusion(body)
        if self.visit_types:
            for field in fields(node):
                self._visit_embedded_types(getattr(node, field.name))
        method = getattr(self, f"visit_tir_{_snake_name(type(node).__name__)}", None)
        result = self.visit_tir_node(node) if method is None else method(node)
        self.tir_memo[identity] = result
        return result

    def _visit_embedded_types(self, value: object) -> None:
        if isinstance(value, IRType):
            self._visit_type(value)
        elif isinstance(value, Dimension):
            self.visit_dimension_tree(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                self._visit_embedded_types(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self._visit_embedded_types(item)
        elif is_dataclass(value) and not isinstance(value, (type, TIRNode)):
            for field in fields(value):
                self._visit_embedded_types(getattr(value, field.name))

    def _visit_embedded(self, value: object) -> None:
        self._visit_embedded_types(value)
        if isinstance(value, TIRNode):
            self._visit_tir(value)

    # Leaf hooks. Op-specific hooks are resolved through OpDefinition.visit.
    def visit_module(self, module: IRModule):
        return None

    def visit_fusion(self, body):
        return None

    def visit_function(self, function: Function, module: IRModule):
        return None

    def visit_constant_recipe(self, recipe, module: IRModule):
        return None

    def visit_node(self, node: Node):
        return None

    def visit_type(self, value: IRType):
        return None

    def visit_dimension(self, value: Dimension):
        return None

    def visit_tir_node(self, node: TIRNode):
        return None


def _snake_name(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


__all__ = ["IRVisitor"]
