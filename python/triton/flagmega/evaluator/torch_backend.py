# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PyTorch reference evaluator for Qwen3.8 layer semantics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Mapping, Protocol

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.evaluator.context import EvaluationContext
from triton.flagmega.evaluator.dump import EvaluatorDumpWriter
from triton.flagmega.evaluator.result import EvaluationResult
from triton.flagmega.evaluator.support import inspect_evaluation_support
from triton.flagmega.importer.checkpoint import Checkpoint
from triton.flagmega.ir import (
    AnyType,
    CallableType,
    DType,
    DataType,
    DistributedType,
    IRModule,
    IRType,
    InvalidType,
    Node,
    NoneType,
    RefType,
    TensorType,
    TupleType,
    VectorType,
    verify_module,
)
from triton.flagmega.ir.ops.core import get_definition


class WeightResolver(Protocol):
    def resolve(self, node: Node): ...


class CheckpointWeightResolver:
    def __init__(
        self,
        checkpoint: Checkpoint,
        *,
        device: str = "cpu",
        cache: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.cache = bool(cache)
        self._cache: dict[str, Any] = {}

    def resolve(self, node: Node):
        key = str(node.attrs["key"])
        if not self.cache:
            return self.checkpoint.load_tensor(key, device=self.device)
        if key not in self._cache:
            self._cache[key] = self.checkpoint.load_tensor(key, device=self.device)
        return self._cache[key]


class DictWeightResolver:
    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)

    def resolve(self, node: Node):
        key = str(node.attrs["key"])
        try:
            return self.values[key]
        except KeyError as error:
            raise EvaluationError(f"No reference tensor was provided for weight {key!r}.", node_id=node.id) from error


class TorchEvaluator:
    def __init__(self, weights: WeightResolver) -> None:
        self.weights = weights

    def run(self, module: IRModule, inputs: Mapping[str, Any]) -> tuple[Any, ...]:
        return self.run_result(module, inputs).outputs

    def run_with_trace(self, module: IRModule, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """Evaluate and retain named intermediate values for semantic diagnostics."""
        result = self.run_result(module, inputs)
        return result.outputs, result.trace

    def run_result(self, module: IRModule, inputs: Mapping[str, Any]) -> EvaluationResult:
        """Evaluate a verified module and return outputs with an immutable trace."""

        verify_module(module)
        inspect_evaluation_support(module).require_complete(stage=module.stage)
        constant_assets: dict[str, Any] = {}
        context = EvaluationContext(
            module,
            torch=_torch(),
            inputs=inputs,
            weights=self.weights,
            constant_assets=constant_assets,
        )
        dump_writer = EvaluatorDumpWriter(module)
        _evaluate_constant_recipes(module, context, constant_assets, dump_writer)
        node_map = module.node_map
        active_calls: list[str] = []

        def evaluate_function(function_name: str, arguments: tuple[Any, ...]) -> tuple[Any, ...]:
            if function_name in active_calls:
                cycle = " -> ".join((*active_calls, function_name))
                raise EvaluationError(f"Recursive FlagMega evaluation is unsupported: {cycle}.")
            function = module.function_map[function_name]
            if len(arguments) != len(function.parameters):
                raise EvaluationError(
                    f"Function @{function_name} expects {len(function.parameters)} arguments, "
                    f"got {len(arguments)}."
                )
            local = dict(zip(function.parameters, arguments))
            for parameter_id, value in local.items():
                context.validate_value(node_map[parameter_id], value)
                context.record_value(node_map[parameter_id], value)
            active_calls.append(function_name)
            try:
                def evaluate_node(node_id: str):
                    if node_id in local:
                        return local[node_id]
                    node = node_map[node_id]
                    node_arguments = tuple(evaluate_node(value) for value in node.inputs)
                    dump_token = dump_writer.before(function_name, node, node_arguments)
                    with context.call_scope(node, node_arguments, evaluate_node):
                        if node.op == "builtin.call":
                            callee_outputs = evaluate_function(
                                str(node.attrs["callee"]), node_arguments
                            )
                            value = (
                                callee_outputs[0]
                                if len(callee_outputs) == 1
                                else callee_outputs
                            )
                        else:
                            value = get_definition(node.op).evaluate(
                                node, node_arguments, context
                            )
                    context.validate_value(node, value)
                    dump_writer.after(dump_token, function_name, node, value)
                    local[node_id] = value
                    context.record_value(node, value)
                    return value

                return tuple(evaluate_node(output) for output in function.outputs)
            finally:
                active_calls.pop()

        entry = module.function_map[module.entry]
        entry_arguments = tuple(
            context.input_value(node_map[parameter]) for parameter in entry.parameters
        )
        outputs = evaluate_function(module.entry, entry_arguments)
        trace = {
            node.id: context.values[node.id]
            for node in module.nodes
            if node.id in context.values
        }
        return EvaluationResult(outputs, trace)


def materialize_constant_assets(module: IRModule, weights: WeightResolver) -> Mapping[str, Any]:
    """Evaluate frozen recipes without evaluating the runtime graph."""

    verify_module(module)
    values: dict[str, Any] = {}
    context = EvaluationContext(
        module,
        torch=_torch(),
        inputs={},
        weights=weights,
        constant_assets=values,
    )
    _evaluate_constant_recipes(module, context, values, EvaluatorDumpWriter(module))
    return values


def materialize_constant_recipe(module: IRModule, recipe, weights: WeightResolver) -> tuple[Any, ...]:
    """Evaluate one verified recipe for a bounded backend fallback."""

    verify_module(module)
    if recipe not in module.constant_recipes:
        raise EvaluationError(f"Constant recipe {recipe.id!r} does not belong to the module.")
    context = EvaluationContext(
        module,
        torch=_torch(),
        inputs={},
        weights=weights,
        constant_assets={},
    )
    return _evaluate_constant_recipe(recipe, context, EvaluatorDumpWriter(module))


def iter_materialized_constant_assets(
    module: IRModule,
    weights: WeightResolver,
    *,
    outputs: Iterable[str] | None = None,
) -> Iterator[tuple[str, Any]]:
    """Evaluate frozen assets one recipe at a time with bounded live storage.

    This is the artifact writer's streaming interface.  The ordinary mapping
    API intentionally retains every output for graph evaluation; doing that
    while packing a multi-gigabyte model duplicates the complete readonly
    image in host memory.  A fresh context per recipe releases intermediates
    and resolver values before the next recipe is evaluated.
    """

    verify_module(module)
    requested = None if outputs is None else frozenset(str(value) for value in outputs)
    available = {
        output
        for recipe in module.constant_recipes
        for output in recipe.outputs
    }
    if requested is not None and not requested <= available:
        missing = ", ".join(sorted(requested - available))
        raise EvaluationError(f"Unknown constant recipe outputs: {missing}.")
    dump_writer = EvaluatorDumpWriter(module)
    for recipe in module.constant_recipes:
        selected = (
            recipe.outputs
            if requested is None
            else tuple(output for output in recipe.outputs if output in requested)
        )
        if not selected:
            continue
        context = EvaluationContext(
            module,
            torch=_torch(),
            inputs={},
            weights=weights,
            constant_assets={},
        )
        materialized = _evaluate_constant_recipe(recipe, context, dump_writer)
        values = dict(zip(recipe.outputs, materialized, strict=True))
        for output in selected:
            yield output, values[output]


def _evaluate_constant_recipes(
    module: IRModule,
    context: EvaluationContext,
    outputs: dict[str, Any],
    dump_writer: EvaluatorDumpWriter,
) -> None:
    fingerprint_cache: dict[str, tuple[Any, ...]] = {}
    for recipe in module.constant_recipes:
        cached = fingerprint_cache.get(recipe.fingerprint)
        if cached is not None and len(cached) == len(recipe.outputs):
            outputs.update(zip(recipe.outputs, cached))
            continue
        materialized = _evaluate_constant_recipe(recipe, context, dump_writer)
        fingerprint_cache[recipe.fingerprint] = materialized
        outputs.update(zip(recipe.outputs, materialized))


def _evaluate_constant_recipe(
    recipe,
    context: EvaluationContext,
    dump_writer: EvaluatorDumpWriter,
) -> tuple[Any, ...]:
    """Evaluate one frozen recipe without retaining another recipe's values."""

    local: dict[str, Any] = {}
    for node in recipe.nodes:
        arguments = tuple(local[value] for value in node.inputs)
        dump_token = dump_writer.before(f"@constant:{recipe.id}", node, arguments)
        with context.call_scope(node, arguments, lambda node_id: local[node_id]):
            value = get_definition(node.op).evaluate(node, arguments, context)
        context.validate_value(node, value)
        dump_writer.after(dump_token, f"@constant:{recipe.id}", node, value)
        local[node.id] = value
        context.record_value(node, value)
    return tuple(local[value] for value in recipe.outputs)


def validate_value_type(
    node: Node,
    value: Any,
    *,
    dimension_bindings: dict[str, int] | None = None,
) -> None:
    """Validate one runtime value and bind repeated dynamic dimensions."""

    _validate_ir_value(
        node.type,
        value,
        node_id=node.id,
        path=node.id,
        dimension_bindings={} if dimension_bindings is None else dimension_bindings,
    )


def _validate_ir_value(
    value_type: IRType,
    value: Any,
    *,
    node_id: str,
    path: str,
    dimension_bindings: dict[str, int],
) -> None:
    if isinstance(value_type, DistributedType):
        # Reference evaluation represents a distributed value by its complete
        # logical tensor; local physical shards belong to target simulation.
        value_type = value_type.tensor
    if isinstance(value_type, (AnyType, RefType)):
        return
    if isinstance(value_type, InvalidType):
        raise EvaluationError(
            f"Node {node_id!r} has invalid result type: {value_type.reason}.",
            node_id=node_id,
        )
    if isinstance(value_type, NoneType):
        if value is not None:
            raise EvaluationError(f"Value {path!r} must be None.", node_id=node_id)
        return
    if isinstance(value_type, CallableType):
        if not callable(value):
            raise EvaluationError(f"Value {path!r} must be callable.", node_id=node_id)
        return
    if isinstance(value_type, TupleType):
        if not isinstance(value, (tuple, list)):
            raise EvaluationError(f"Value {path!r} must be a tuple.", node_id=node_id)
        if not value_type.is_variadic and len(value) != len(value_type.fields):
            raise EvaluationError(
                f"Value {path!r} expects {len(value_type.fields)} fields, got {len(value)}.",
                node_id=node_id,
            )
        if value_type.is_variadic:
            if not value_type.fields:
                return
            field_types = (value_type.fields[0],) * len(value)
        else:
            field_types = value_type.fields
        for index, (field_type, field_value) in enumerate(zip(field_types, value)):
            _validate_ir_value(
                field_type,
                field_value,
                node_id=node_id,
                path=f"{path}[{index}]",
                dimension_bindings=dimension_bindings,
            )
        return
    if not isinstance(value_type, TensorType):
        return
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise EvaluationError(f"Value {path!r} expects a tensor.", node_id=node_id)
    actual_shape = tuple(int(dimension) for dimension in value.shape)
    lanes = value_type.dtype.lanes if isinstance(value_type.dtype, VectorType) else ()
    expected_rank = len(value_type.shape) + len(lanes)
    if len(actual_shape) != expected_rank:
        raise EvaluationError(
            f"Value {path!r} expects rank {expected_rank}, got shape {actual_shape}.",
            node_id=node_id,
        )
    for index, (dimension, actual) in enumerate(zip(value_type.shape, actual_shape)):
        if dimension.value is not None:
            expected = dimension.value
        elif dimension.name is not None:
            expected = dimension_bindings.setdefault(dimension.name, actual)
        else:
            try:
                expected = dimension.evaluate(dimension_bindings)
            except IRSchemaError:
                expected = None
        if dimension.minimum is not None and actual < dimension.minimum:
            raise EvaluationError(
                f"Value {path!r} dimension {index} is {actual}, below {dimension.minimum}.",
                node_id=node_id,
            )
        if dimension.maximum is not None and actual > dimension.maximum:
            raise EvaluationError(
                f"Value {path!r} dimension {index} is {actual}, above {dimension.maximum}.",
                node_id=node_id,
            )
        if expected is not None and actual != expected:
            raise EvaluationError(
                f"Value {path!r} dimension {index} expects {expected}, got {actual}.",
                node_id=node_id,
            )
    if tuple(actual_shape[len(value_type.shape):]) != tuple(lanes):
        raise EvaluationError(
            f"Value {path!r} expects vector lanes {tuple(lanes)}, got "
            f"{actual_shape[len(value_type.shape):]}.",
            node_id=node_id,
        )
    expected_dtype = torch_dtype(value_type.dtype)
    if value.dtype != expected_dtype:
        raise EvaluationError(
            f"Value {path!r} expects dtype {expected_dtype}, got {value.dtype}.",
            node_id=node_id,
        )


def torch_dtype(dtype: DataType):
    torch = _torch()
    if isinstance(dtype, VectorType):
        dtype = dtype.elem_type
    mapping = {
        DType.BOOL: torch.bool,
        DType.INT32: torch.int32,
        DType.INT64: torch.int64,
        DType.BFLOAT16: torch.bfloat16,
        DType.FLOAT32: torch.float32,
        DType.FLOAT8_E4M3FN: torch.float8_e4m3fn,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise EvaluationError(f"PyTorch evaluator does not support data type {dtype}.") from error


def physical_shape(value_type: TensorType) -> tuple[int, ...] | None:
    dimensions = tuple(dimension.value for dimension in value_type.shape)
    if any(value is None for value in dimensions):
        return None
    lanes = value_type.dtype.lanes if isinstance(value_type.dtype, VectorType) else ()
    return (*dimensions, *lanes)  # type: ignore[arg-type]


def shape_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("The FlagMega reference evaluator requires PyTorch.") from error
    return torch
