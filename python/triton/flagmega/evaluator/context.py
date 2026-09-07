# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed state shared by FlagMega evaluator backends and op definitions.

The context is the Python counterpart of nncase's ``IEvaluateContext``.  It
does not locate handlers (op definitions own their behavior), but it gives a
handler a stable current-call view, named ``ParameterInfo`` access, recursive
evaluation, value memo access and dynamic-dimension bindings.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, TYPE_CHECKING

from triton.flagmega.errors import EvaluationError
from triton.flagmega.ir import IRModule, IRType, Node
from triton.flagmega.ir.ops.core import OpDefinition, ParameterInfo, ParameterKind

if TYPE_CHECKING:
    from triton.flagmega.evaluator.torch_backend import WeightResolver


EvaluateCallback = Callable[[str], Any]


@dataclass(frozen=True)
class EvaluationFrame:
    """One active op evaluation, restored when recursive evaluation returns."""

    node: Node
    arguments: tuple[Any, ...]
    evaluate: EvaluateCallback


class EvaluationContext:
    """Runtime evaluation context with nncase-style named argument access."""

    def __init__(
        self,
        module: IRModule,
        *,
        torch: Any,
        inputs: Mapping[str, Any],
        weights: WeightResolver,
        constant_assets: Mapping[str, Any],
    ) -> None:
        self.module = module
        self.torch = torch
        self.inputs = MappingProxyType(dict(inputs))
        self.weights = weights
        self.constant_assets = constant_assets
        types = {
            node.id: node.type
            for recipe in module.constant_recipes
            for node in recipe.nodes
        }
        types.update((node.id, node.type) for node in module.nodes)
        self.types: Mapping[str, IRType] = MappingProxyType(types)
        self.dimension_bindings: dict[str, int] = {}
        self._frames: list[EvaluationFrame] = []
        self._values: dict[str, Any] = {}

    @property
    def current_node(self) -> Node:
        if not self._frames:
            raise EvaluationError("Current evaluator call is not set.")
        return self._frames[-1].node

    @property
    def current_arguments(self) -> tuple[Any, ...]:
        if not self._frames:
            raise EvaluationError("Current evaluator call is not set.")
        return self._frames[-1].arguments

    @property
    def return_type(self) -> IRType:
        return self.current_node.type

    @property
    def values(self) -> Mapping[str, Any]:
        """Read-only live view of values memoized during this run."""

        return MappingProxyType(self._values)

    @contextmanager
    def call_scope(
        self,
        node: Node,
        arguments: tuple[Any, ...],
        evaluate: EvaluateCallback,
    ) -> Iterator[None]:
        """Make ``node`` the current call and restore the parent on exit."""

        self._frames.append(EvaluationFrame(node, tuple(arguments), evaluate))
        try:
            yield
        finally:
            popped = self._frames.pop()
            if popped.node is not node:
                raise EvaluationError("Evaluator call stack was corrupted.", node_id=node.id)

    def get_argument_value(
        self,
        definition: type[OpDefinition],
        parameter: ParameterInfo,
    ) -> Any:
        """Return a current operand by its declaring ``ParameterInfo``."""

        node = self.current_node
        if parameter.owner is not definition:
            raise EvaluationError(
                f"Parameter {parameter.name!r} does not belong to {definition.op_name!r}.",
                node_id=node.id,
            )
        if parameter.kind != ParameterKind.INPUT:
            raise EvaluationError(
                f"Parameter {definition.op_name}.{parameter.name} is not an input.",
                node_id=node.id,
            )
        if node.op != definition.op_name:
            raise EvaluationError(
                f"Current op is {node.op!r}, not {definition.op_name!r}.",
                node_id=node.id,
            )
        return parameter.read(self.current_arguments)

    def evaluate(self, value: str | Node) -> Any:
        """Evaluate an expression through the active visitor callback."""

        if not self._frames:
            raise EvaluationError("Recursive evaluation requires an active evaluator call.")
        node_id = value.id if isinstance(value, Node) else str(value)
        return self._frames[-1].evaluate(node_id)

    def get_value(self, value: str | Node) -> Any:
        node_id = value.id if isinstance(value, Node) else str(value)
        try:
            return self._values[node_id]
        except KeyError as error:
            raise EvaluationError(f"Value {node_id!r} has not been evaluated.", node_id=node_id) from error

    def record_value(self, node: Node, value: Any) -> None:
        self._values[node.id] = value

    def input_value(self, node: Node) -> Any:
        name = str(node.attrs["name"])
        try:
            return self.inputs[name]
        except KeyError as error:
            raise EvaluationError(f"Missing evaluator input {name!r}.", node_id=node.id) from error

    def weight_value(self, node: Node) -> Any:
        value = self.weights.resolve(node)
        # Import locally to keep the generic context independent of torch at
        # module import time.
        from triton.flagmega.evaluator.torch_backend import physical_shape, shape_product
        from triton.flagmega.ir import DistributedType, TensorType, VectorType

        node_type = node.type.tensor if isinstance(node.type, DistributedType) else node.type
        if isinstance(node_type, TensorType) and isinstance(node_type.dtype, VectorType):
            shape = physical_shape(node_type)
            if shape is not None and tuple(value.shape) != shape and value.numel() == shape_product(shape):
                value = value.reshape(shape)
        return value

    def constant_asset_value(self, node: Node) -> Any:
        output = str(node.attrs["output"])
        try:
            return self.constant_assets[output]
        except KeyError as error:
            raise EvaluationError(
                f"Constant asset {node.id!r} output {output!r} was not materialized.",
                node_id=node.id,
            ) from error

    def validate_value(self, node: Node, value: Any) -> None:
        from triton.flagmega.evaluator.torch_backend import validate_value_type

        validate_value_type(node, value, dimension_bindings=self.dimension_bindings)

    def torch_dtype(self, dtype):
        from triton.flagmega.evaluator.torch_backend import torch_dtype

        return torch_dtype(dtype)


__all__ = ["EvaluationContext", "EvaluationFrame"]
