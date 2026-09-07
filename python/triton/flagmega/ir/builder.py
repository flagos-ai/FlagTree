# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic builder used by importers and emitted Python checkpoints."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import (
    Effect,
    Function,
    IRModule,
    IRType,
    Node,
    ProvenanceRecord,
    PURE,
    SelectionPoint,
    SelectionRecord,
)
from triton.flagmega.ir.constant_recipe import ConstantRecipe
from triton.flagmega.ir.tir import ExecutionFunction, PrimFunction, KernelDefinition


class IRBuilder:
    def __init__(self, *, dialect: str, stage: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.dialect = dialect
        self.stage = stage
        self.metadata = dict(metadata or {})
        self._nodes: list[Node] = []
        self._node_ids: set[str] = set()
        self._functions: list[Function] = []
        self._prim_functions: list[PrimFunction] = []
        self._execution_functions: list[ExecutionFunction] = []
        self._kernel_definitions: list[KernelDefinition] = []
        self._counter = 0

    def _next_id(self, prefix: str = "n") -> str:
        while f"{prefix}{self._counter}" in self._node_ids:
            self._counter += 1
        result = f"{prefix}{self._counter}"
        self._counter += 1
        return result

    def node(
        self,
        *,
        op: str,
        type: IRType,
        inputs: Sequence[str | Node] = (),
        id: str | None = None,
        effect: Effect = PURE,
        attrs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        node_id = id or self._next_id()
        if node_id in self._node_ids:
            raise IRSchemaError(f"Duplicate node id {node_id!r}.")
        normalized_inputs = tuple(value.id if isinstance(value, Node) else str(value) for value in inputs)
        value = Node(node_id, op, normalized_inputs, type, effect, attrs or {}, metadata or {})
        self._nodes.append(value)
        self._node_ids.add(node_id)
        return value

    def var(self, name: str, type: IRType, *, id: str | None = None, metadata: Mapping[str, Any] | None = None) -> Node:
        return self.node(op="builtin.var", type=type, id=id, attrs={"name": name}, metadata=metadata)

    def weight(
        self,
        name: str,
        type: IRType,
        *,
        source: str,
        key: str,
        id: str | None = None,
        source_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        attrs: dict[str, Any] = {"name": name, "source": source, "key": key}
        if source_hash is not None:
            attrs["source_hash"] = source_hash
        return self.node(op="builtin.weight", type=type, id=id, attrs=attrs, metadata=metadata)

    def call(
        self,
        op: str,
        inputs: Sequence[str | Node],
        type: IRType,
        *,
        id: str | None = None,
        effect: Effect = PURE,
        attrs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.node(
            op=op,
            type=type,
            inputs=inputs,
            id=id,
            effect=effect,
            attrs=attrs,
            metadata=metadata,
        )

    def function(
        self,
        name: str,
        parameters: Sequence[str | Node],
        outputs: Sequence[str | Node],
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> Function:
        if any(function.name == name for function in self._functions):
            raise IRSchemaError(f"Duplicate function name {name!r}.")
        value = Function(
            name,
            tuple(item.id if isinstance(item, Node) else str(item) for item in parameters),
            tuple(item.id if isinstance(item, Node) else str(item) for item in outputs),
            attrs or {},
        )
        self._functions.append(value)
        return value

    def prim_function(self, value: PrimFunction) -> PrimFunction:
        if not isinstance(value, PrimFunction):
            raise TypeError("IRBuilder.prim_function requires a PrimFunction.")
        if any(function.name == value.name for function in self._prim_functions):
            raise IRSchemaError(f"Duplicate PrimFunction name {value.name!r}.")
        self._prim_functions.append(value)
        return value

    def execution_function(self, value: ExecutionFunction) -> ExecutionFunction:
        if not isinstance(value, ExecutionFunction):
            raise TypeError("IRBuilder.execution_function requires an ExecutionFunction.")
        if any(function.name == value.name for function in self._execution_functions):
            raise IRSchemaError(f"Duplicate ExecutionFunction name {value.name!r}.")
        self._execution_functions.append(value)
        return value

    def kernel_definition(self, value: KernelDefinition) -> KernelDefinition:
        if not isinstance(value, KernelDefinition):
            raise TypeError("IRBuilder.kernel_definition requires a KernelDefinition.")
        if any(kernel.name == value.name for kernel in self._kernel_definitions):
            raise IRSchemaError(f"Duplicate KernelDefinition name {value.name!r}.")
        self._kernel_definitions.append(value)
        return value

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Snapshot the nodes built so far for Python constant recipe builders."""

        return tuple(self._nodes)

    def build(
        self,
        *,
        entry: str,
        selection_points: Sequence[SelectionPoint] = (),
        selections: Sequence[SelectionRecord] = (),
        provenance: Sequence[ProvenanceRecord] = (),
        constant_recipes: Sequence[ConstantRecipe] = (),
    ) -> IRModule:
        return IRModule(
            dialect=self.dialect,
            stage=self.stage,
            nodes=tuple(self._nodes),
            functions=tuple(self._functions),
            entry=entry,
            prim_functions=tuple(self._prim_functions),
            execution_functions=tuple(self._execution_functions),
            kernel_definitions=tuple(self._kernel_definitions),
            metadata=self.metadata,
            constant_recipes=tuple(constant_recipes),
            selection_points=tuple(selection_points),
            selections=tuple(selections),
            provenance=tuple(provenance),
        )


class Module:
    """Python-defined IR graph, analogous to a small ``torch.nn.Module``.

    Subclasses implement :meth:`forward` with calls to :meth:`input`,
    :meth:`weight`, :meth:`call`, and :meth:`function`.  Calling
    :meth:`build` executes that Python graph definition and returns the
    immutable compiler IR.  This is the public form used by editable
    checkpoints; it intentionally does not deserialize a data dictionary.
    """

    def __init__(
        self,
        *,
        dialect: str,
        stage: str,
        entry: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.dialect = str(dialect)
        self.stage = str(stage)
        self.entry = str(entry)
        self.metadata = dict(metadata or {})
        self._builder: IRBuilder | None = None

    def forward(self) -> None:
        """Define inputs, weights, operations, and functions in Python."""

        raise NotImplementedError

    @property
    def builder(self) -> IRBuilder:
        if self._builder is None:
            raise IRSchemaError("IR construction helpers can only be used while Module.build() is running.")
        return self._builder

    def input(
        self,
        name: str,
        type: IRType,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.builder.var(name, type, id=id, metadata=metadata)

    def weight(
        self,
        name: str,
        type: IRType,
        *,
        source: str,
        key: str,
        id: str | None = None,
        source_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.builder.weight(
            name,
            type,
            source=source,
            key=key,
            id=id,
            source_hash=source_hash,
            metadata=metadata,
        )

    def node(
        self,
        *,
        op: str,
        type: IRType,
        inputs: Sequence[str | Node] = (),
        id: str | None = None,
        effect: Effect = PURE,
        attrs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.builder.node(
            op=op,
            type=type,
            inputs=inputs,
            id=id,
            effect=effect,
            attrs=attrs,
            metadata=metadata,
        )

    def call(
        self,
        op: str,
        inputs: Sequence[str | Node],
        type: IRType,
        *,
        id: str | None = None,
        effect: Effect = PURE,
        attrs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.builder.call(
            op,
            inputs,
            type,
            id=id,
            effect=effect,
            attrs=attrs,
            metadata=metadata,
        )

    def function(
        self,
        name: str,
        parameters: Sequence[str | Node],
        outputs: Sequence[str | Node],
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> Function:
        return self.builder.function(name, parameters, outputs, attrs=attrs)

    def prim_function(self, value: PrimFunction) -> PrimFunction:
        return self.builder.prim_function(value)

    def execution_function(self, value: ExecutionFunction) -> ExecutionFunction:
        return self.builder.execution_function(value)

    def kernel_definition(self, value: KernelDefinition) -> KernelDefinition:
        return self.builder.kernel_definition(value)

    def build(
        self,
        *,
        selection_points: Sequence[SelectionPoint] = (),
        selections: Sequence[SelectionRecord] = (),
        provenance: Sequence[ProvenanceRecord] = (),
        constant_recipes: Sequence[ConstantRecipe] = (),
    ) -> IRModule:
        if self._builder is not None:
            raise IRSchemaError("A Python IR Module instance cannot be built recursively.")
        self._builder = IRBuilder(dialect=self.dialect, stage=self.stage, metadata=self.metadata)
        try:
            from triton.flagmega.ir.functional import construction_scope

            with construction_scope(self._builder):
                self.forward()
            return self.builder.build(
                entry=self.entry,
                selection_points=selection_points,
                selections=selections,
                provenance=provenance,
                constant_recipes=constant_recipes,
            )
        finally:
            self._builder = None


class ConstantModule:
    """Python builder for frozen constant recipes using the normal ``F`` API.

    Generated checkpoints keep recipe operations readable and editable.  A
    call to :meth:`recipe` closes the nodes emitted since the previous recipe;
    references across recipe boundaries are intentionally forbidden.
    """

    def __init__(self) -> None:
        self._builder: IRBuilder | None = None
        self._recipes: list[ConstantRecipe] = []
        self._recipe_start = 0

    def forward(self) -> None:
        raise NotImplementedError

    @property
    def builder(self) -> IRBuilder:
        if self._builder is None:
            raise IRSchemaError("Constant construction helpers can only be used during ConstantModule.build().")
        return self._builder

    def weight(
        self,
        name: str,
        type: IRType,
        *,
        source: str,
        key: str,
        id: str | None = None,
        source_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.builder.weight(
            name,
            type,
            source=source,
            key=key,
            id=id,
            source_hash=source_hash,
            metadata=metadata,
        )

    def node(
        self,
        *,
        op: str,
        type: IRType,
        inputs: Sequence[str | Node] = (),
        id: str | None = None,
        effect: Effect = PURE,
        attrs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.builder.node(
            op=op,
            type=type,
            inputs=inputs,
            id=id,
            effect=effect,
            attrs=attrs,
            metadata=metadata,
        )

    def recipe(self, name: str, outputs: Sequence[str | Node]) -> ConstantRecipe:
        nodes = self.builder.nodes[self._recipe_start:]
        if not nodes:
            raise IRSchemaError(f"Constant recipe {name!r} has no nodes.")
        node_ids = {node.id for node in nodes}
        output_ids = tuple(value.id if isinstance(value, Node) else str(value) for value in outputs)
        if any(value not in node_ids for value in output_ids):
            raise IRSchemaError(f"Constant recipe {name!r} exports a value outside its node graph.")
        for node in nodes:
            external = set(node.inputs) - node_ids
            if external:
                raise IRSchemaError(
                    f"Constant recipe {name!r} node {node.id!r} has cross-recipe inputs {sorted(external)}.")
        value = ConstantRecipe(name, nodes, output_ids)
        self._recipes.append(value)
        self._recipe_start += len(nodes)
        return value

    def build(self) -> tuple[ConstantRecipe, ...]:
        if self._builder is not None:
            raise IRSchemaError("A Python ConstantModule instance cannot be built recursively.")
        self._builder = IRBuilder(dialect="constant_recipe", stage="frozen_constants")
        self._recipes = []
        self._recipe_start = 0
        try:
            from triton.flagmega.ir.functional import construction_scope

            with construction_scope(self._builder):
                self.forward()
            if self._recipe_start != len(self.builder.nodes):
                raise IRSchemaError("ConstantModule.forward() left nodes outside a recipe.")
            return tuple(self._recipes)
        finally:
            self._builder = None
