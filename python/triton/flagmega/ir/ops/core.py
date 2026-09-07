# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Descriptor-driven FlagMega op definitions.

This mirrors nncase's useful ``ParameterInfo`` contract in native Python. Each
op declares named input/attribute parameters in its own class body;
``@op_definition`` derives the schema, arity checking, visitor dispatch and
Python emission metadata. Public ``F`` APIs are ordinary handwritten source.
There is deliberately no source generator, IoC container or assembly scan.
"""

from __future__ import annotations

import keyword
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Any, Callable, Iterator, Mapping, Sequence

from triton.flagmega.errors import (
    EvaluationError,
    IRSchemaError,
    IRVerificationError,
    NumpyMaterializationUnsupported,
)
from triton.flagmega.ir.memory_effect import (
    MemoryEffect,
    memory_effect as normalize_memory_effect,
)
from triton.flagmega.ir.model import (
    Effect,
    IRModule,
    IRType,
    Node,
    PURE,
    TensorType,
    canonical_attributes,
    contains_distributed,
    logical_type,
)
from triton.flagmega.ir.type_pattern import TypePattern, is_ir_type


class ParameterKind(str, Enum):
    INPUT = "input"
    ATTRIBUTE = "attribute"


class CostKind(str, Enum):
    """Confidence/source class for an op-local analytic estimate."""

    EXACT = "exact"
    ANALYTIC = "analytic"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


_MISSING = object()


class ParameterInfo:
    """Named op parameter, analogous to nncase ``ParameterInfo``.

    Indices are assigned from class declaration order by ``@op_definition``;
    op authors never maintain integer positions or min/max arity separately.
    """

    def __init__(
        self,
        kind: ParameterKind,
        *,
        name: str | None = None,
        type_pattern: TypePattern | None = None,
        parameter_kind: ParameterKind | str | None = None,
        default: object = _MISSING,
        variadic: bool = False,
        positional: bool = False,
        memory_effect: MemoryEffect | str | object = _MISSING,
    ) -> None:
        self.kind = ParameterKind(kind)
        # Python stores constructor properties in ``Node.attrs`` and call
        # operands in ``Node.inputs``.  nncase's ``ParameterKind`` is an
        # orthogonal semantic distinction: an operand can still be an
        # Attribute (for example a Dimension-valued layer id) and therefore
        # must terminate AutoDistributed instead of acquiring an SBP type.
        # Keep both facts instead of overloading the storage classification.
        self.parameter_kind = ParameterKind(
            self.kind if parameter_kind is None else parameter_kind
        )
        self._name = name
        self.type_pattern = type_pattern or is_ir_type()
        self.default = default
        self.variadic = bool(variadic)
        self.positional = bool(positional)
        self.memory_effect = normalize_memory_effect(
            MemoryEffect.READ
            if (
                memory_effect is _MISSING
                and self.kind is ParameterKind.INPUT
                and self.parameter_kind is ParameterKind.INPUT
            )
            else MemoryEffect.NONE
            if memory_effect is _MISSING
            else memory_effect
        )
        self.owner: type[OpDefinition] | None = None
        self.index: int | None = None
        self.input_index: int | None = None
        self._declared_name: str | None = None
        if self.kind != ParameterKind.INPUT and self.variadic:
            raise ValueError("Only input parameters can be variadic.")
        if (
            self.kind == ParameterKind.ATTRIBUTE
            and self.parameter_kind != ParameterKind.ATTRIBUTE
        ):
            raise ValueError("Node attributes must use ParameterKind.ATTRIBUTE.")
        if self.kind != ParameterKind.INPUT and self.memory_effect != MemoryEffect.NONE:
            raise ValueError("Only input parameters can carry memory effects.")
        if not isinstance(self.type_pattern, TypePattern):
            raise TypeError("ParameterInfo type_pattern must be a TypePattern.")

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._declared_name = name

    @property
    def name(self) -> str:
        name = self._name or self._declared_name
        if name is None:
            raise RuntimeError("ParameterInfo has not been attached to an op class.")
        return name

    @property
    def required(self) -> bool:
        return self.default is _MISSING

    @property
    def pattern(self) -> TypePattern:
        """nncase-compatible alias for the parameter type contract."""

        return self.type_pattern

    def check_type(self, value_type: IRType) -> bool:
        return self.type_pattern.match_leaf(value_type)

    def require_type(self, value_type: IRType) -> IRType:
        owner = "<unbound>" if self.owner is None else self.owner.op_name
        return self.type_pattern.check(value_type, f"{owner}.{self.name}")

    def type_of(self, inputs: Sequence[Node]) -> IRType | tuple[IRType, ...]:
        """Read and validate operand types without restating a type check."""

        value = self.read(inputs)
        if self.variadic:
            return tuple(self.require_type(item.type) for item in value)
        return self.require_type(value.type)

    def bind(self, owner: type[OpDefinition], *, index: int, input_index: int | None) -> None:
        if self.owner is not None and self.owner is not owner:
            raise TypeError(f"ParameterInfo {self.name!r} cannot be shared by multiple op classes.")
        self.owner = owner
        self.index = index
        self.input_index = input_index

    def read(self, inputs: Sequence[Any], attrs: Mapping[str, Any] | None = None) -> Any:
        """Read this parameter without spelling a positional integer."""

        if self.kind == ParameterKind.ATTRIBUTE:
            if attrs is None:
                raise KeyError(f"Attribute values are required to read {self.name!r}.")
            if self.name in attrs:
                return attrs[self.name]
            if not self.required:
                return self.default
            raise KeyError(f"Required attribute {self.name!r} is missing.")
        if self.input_index is None:
            raise RuntimeError(f"Input ParameterInfo {self.name!r} has not been bound.")
        if self.variadic:
            return tuple(inputs[self.input_index:])
        try:
            return inputs[self.input_index]
        except IndexError as error:
            raise IndexError(f"Required input {self.name!r} is missing.") from error

    def __repr__(self) -> str:
        owner = "<unbound>" if self.owner is None else self.owner.op_name
        return f"ParameterInfo({owner}.{self.name}, kind={self.kind.value}, index={self.index})"


def input_parameter(
    type_pattern: TypePattern | None = None,
    *,
    name: str | None = None,
    parameter_kind: ParameterKind | str = ParameterKind.INPUT,
    memory_effect: MemoryEffect | str | object = _MISSING,
) -> ParameterInfo:
    return ParameterInfo(
        ParameterKind.INPUT,
        name=name,
        type_pattern=type_pattern,
        parameter_kind=parameter_kind,
        positional=True,
        memory_effect=memory_effect,
    )


def variadic_input_parameter(
    type_pattern: TypePattern | None = None,
    *,
    name: str | None = None,
    parameter_kind: ParameterKind | str = ParameterKind.INPUT,
    memory_effect: MemoryEffect | str | object = _MISSING,
) -> ParameterInfo:
    return ParameterInfo(
        ParameterKind.INPUT,
        name=name,
        type_pattern=type_pattern,
        parameter_kind=parameter_kind,
        variadic=True,
        positional=True,
        memory_effect=memory_effect,
    )


def attribute_parameter(
    name: str | None = None,
    *,
    default: object = _MISSING,
    positional: bool = False,
) -> ParameterInfo:
    return ParameterInfo(
        ParameterKind.ATTRIBUTE,
        name=name,
        default=default,
        positional=positional,
    )


@dataclass(frozen=True)
class OpCost:
    # ``None`` means unknown, never zero.  This matters because agent-driven
    # selection must not receive fabricated free operations when a model is
    # incomplete.
    flops: int | None = None
    bytes_read: int | None = None
    bytes_written: int | None = None
    communication_bytes: int | None = None
    synchronizations: int | None = None
    kind: CostKind = CostKind.UNKNOWN
    model: str = "op-definition/v1"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.flops,
            self.bytes_read,
            self.bytes_written,
            self.communication_bytes,
            self.synchronizations,
        )
        if any(value is not None and (isinstance(value, bool) or value < 0) for value in values):
            raise ValueError("OpCost factors must be non-negative integers or None.")
        object.__setattr__(self, "kind", CostKind(self.kind))
        object.__setattr__(self, "notes", tuple(str(value) for value in self.notes))
        if not self.model:
            raise ValueError("OpCost requires a non-empty model identifier.")
        if self.kind is CostKind.UNKNOWN and any(value is not None for value in values):
            object.__setattr__(self, "kind", CostKind.ANALYTIC)

    @property
    def unknown_factors(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in (
                ("flops", self.flops),
                ("bytes_read", self.bytes_read),
                ("bytes_written", self.bytes_written),
                ("communication_bytes", self.communication_bytes),
                ("synchronizations", self.synchronizations),
            )
            if value is None
        )

    @property
    def is_complete(self) -> bool:
        return not self.unknown_factors

    @classmethod
    def exact_zero(cls, *, notes: tuple[str, ...] = ()) -> OpCost:
        return cls(0, 0, 0, 0, 0, CostKind.EXACT, "semantic-zero/v1", notes)


@dataclass(frozen=True)
class OpCostFactors:
    """Target-aggregatable execution factors for one candidate call.

    ``OpCost`` remains the architecture-independent metric/coverage record.
    These factors mirror nncase's executable cost-factor vocabulary: an op
    definition computes them from its candidate operand/result types, while a
    target machine decides how bandwidth, latency, parallel blocks and
    synchronization turn them into one search objective.
    """

    cpu_cycles: int = 0
    # Target-scaled arithmetic is kept separate from already-normalized CPU
    # cycles.  This lets an op describe useful work without embedding a
    # machine throughput in its definition; the target folds both paths into
    # the same overlapped compute term.
    elementwise_operations: int = 0
    simt_fma_operations: int = 0
    block_local_memory_load_bytes: int = 0
    block_local_memory_store_bytes: int = 0
    chip_global_memory_load_bytes: int = 0
    chip_global_memory_store_bytes: int = 0
    block_synchronizations: int = 0
    grid_synchronizations: int = 0
    communication_cycles: int = 0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("OpCostFactors values must be non-negative integers.")


@dataclass(frozen=True)
class NodeRef:
    id: str


@dataclass(frozen=True)
class PythonCall:
    function: str
    positional: tuple[object, ...]
    keywords: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedCall:
    inputs: tuple[Node, ...]
    result_type: IRType
    effect: Effect
    attrs: Mapping[str, Any]


_ACTIVE_BUILDER: ContextVar[Any | None] = ContextVar("flagmega_active_ir_builder", default=None)
_DEFINITIONS: dict[str, type[OpDefinition]] = {}
_FUNCTIONALS: dict[tuple[str, str], type[OpDefinition]] = {}


@contextmanager
def construction_scope(builder: Any) -> Iterator[None]:
    token = _ACTIVE_BUILDER.set(builder)
    try:
        yield
    finally:
        _ACTIVE_BUILDER.reset(token)


class OpDefinition:
    op_name = ""
    namespace: str | None = None
    functional_name: str | None = None
    display_name = ""
    parameters: tuple[ParameterInfo, ...] = ()
    input_parameters: tuple[ParameterInfo, ...] = ()
    attribute_parameters: tuple[ParameterInfo, ...] = ()
    # Named operands whose storage may be reused for a tensor result after
    # their last use.  Definitions refer to ParameterInfo objects instead of
    # positional integers, so refactors cannot silently change alias meaning.
    inplace_input_parameters: tuple[ParameterInfo, ...] = ()
    # Tuple-producing ops may independently reuse an operand for each top-level
    # result.  Entries are ParameterInfo objects (or None), aligned with tuple
    # result fields; this keeps alias contracts stable when operand order moves.
    inplace_output_parameters: tuple[ParameterInfo | None, ...] = ()
    # Optional physical effect per top-level tensor/tuple/Ref result. An empty tuple
    # uses the ordinary tensor-write / Ref-identity rule. Collective intrinsics
    # can require chip-visible materialization before memory placement runs.
    result_memory_effects: tuple[MemoryEffect, ...] = ()
    # Compile-time evaluation is an explicit per-op contract.  Purity alone is
    # insufficient (a pure op may still be non-deterministic or target-only).
    constant_source = False
    const_evaluable = False
    numpy_materializable = False
    # A single-result op may declare the named operand whose physical byte
    # sequence is preserved exactly. Constant materialization uses this typed
    # contract to trace a frozen storage view back to checkpoint bytes without
    # restating op names or executing an identity recipe.
    byte_preserving_input_parameters: tuple[ParameterInfo, ...] = ()
    deterministic = True
    # Ordinary tensor operations can be evaluated independently on an
    # all-broadcast placement. Representation-changing distributed operators
    # override this because their result intentionally leaves the distributed
    # type domain.
    supports_broadcast_lifting = True

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        """Evaluate an offline constant recipe using raw NumPy storage.

        Definitions opt in with ``numpy_materializable``. The default remains
        deliberately unavailable so numeric operations never silently acquire
        storage-bit rather than semantic arithmetic.
        """

        raise NumpyMaterializationUnsupported(
            f"{cls.op_name} has no NumPy materializer."
        )

    @classmethod
    def construct(
        cls,
        *arguments: object,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **attributes: object,
    ) -> Node:
        prepared = cls.prepare(arguments, attributes)
        builder = _ACTIVE_BUILDER.get()
        if builder is None:
            raise IRSchemaError("F.* constructors must run inside fm.Module.forward().")
        return builder.call(
            cls.op_name,
            prepared.inputs,
            prepared.result_type,
            id=name,
            effect=prepared.effect,
            attrs=prepared.attrs,
            metadata=metadata,
        )

    @classmethod
    def prepare(cls, arguments: Sequence[object], attributes: Mapping[str, object]) -> PreparedCall:
        inputs, values = cls.split_arguments(arguments, attributes)
        attrs = cls.normalize_attrs(values)
        return PreparedCall(
            inputs,
            cls.infer_call_type(inputs, attrs),
            cls.infer_effect(inputs, attrs),
            cls.ir_attrs(attrs),
        )

    @classmethod
    def infer_call_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        """Infer a call type, including nncase-style broadcast lifting.

        Most op definitions only need to describe their logical tensor
        semantics.  When all distributed input leaves are broadcast on one
        placement, the same operation is valid locally and its tensor result
        leaves are broadcast on that placement as well.  Specialized
        split/partial inference remains owned by the concrete op definition.
        """

        direct_error: Exception | None = None
        try:
            result = cls.infer_type(inputs, attrs)
        except (IRSchemaError, KeyError, TypeError, ValueError, AssertionError, AttributeError) as error:
            direct_error = error
            result = None

        from triton.flagmega.ir.distributed_inference import (
            broadcast_ir_type,
            broadcast_placement_of,
        )

        placement = broadcast_placement_of(*(value.type for value in inputs))
        if (
            cls.supports_broadcast_lifting
            and placement is not None
            and (result is None or not contains_distributed(result))
        ):
            logical_inputs = tuple(
                Node(
                    id=value.id,
                    op=value.op,
                    inputs=value.inputs,
                    type=logical_type(value.type),
                    effect=value.effect,
                    attrs=value.attrs,
                    metadata=value.metadata,
                )
                for value in inputs
            )
            cls.verify_parameter_types(logical_inputs)
            logical_result = cls.infer_type(logical_inputs, attrs)
            return broadcast_ir_type(logical_result, placement)
        if direct_error is not None:
            raise direct_error
        assert result is not None
        return result

    @classmethod
    def split_arguments(
        cls,
        arguments: Sequence[object],
        attributes: Mapping[str, object],
    ) -> tuple[tuple[Node, ...], dict[str, object]]:
        remaining = list(arguments)
        values = dict(attributes)
        inputs: list[Node] = []
        variadic_seen = False
        for parameter in cls.input_parameters:
            if parameter.variadic:
                variadic_seen = True
                reserved = sum(
                    item.positional and item.name not in values
                    for item in cls.attribute_parameters
                )
                take = len(remaining) - reserved
                if take < 0:
                    take = 0
                candidates, remaining = remaining[:take], remaining[take:]
                if any(not isinstance(value, Node) for value in candidates):
                    raise IRSchemaError(f"F.{cls.namespace}.{cls.functional_name} variadic inputs must be IR nodes.")
                for candidate in candidates:
                    parameter.require_type(candidate.type)
                inputs.extend(candidates)  # type: ignore[arg-type]
                continue
            if remaining and parameter.name in values:
                raise IRSchemaError(f"Input {parameter.name!r} was provided both positionally and by name.")
            candidate = remaining.pop(0) if remaining else values.pop(parameter.name, _MISSING)
            if candidate is _MISSING:
                raise IRSchemaError(f"F.{cls.namespace}.{cls.functional_name} is missing input {parameter.name!r}.")
            if not isinstance(candidate, Node):
                raise IRSchemaError(f"Input {parameter.name!r} must be an IR node.")
            parameter.require_type(candidate.type)
            inputs.append(candidate)
        if variadic_seen and any(parameter.variadic for parameter in cls.input_parameters[:-1]):
            raise RuntimeError(f"Variadic input for {cls.op_name!r} must be the last input parameter.")
        for parameter in cls.attribute_parameters:
            if not parameter.positional or not remaining:
                continue
            if parameter.name in values:
                raise IRSchemaError(f"Attribute {parameter.name!r} was provided twice.")
            values[parameter.name] = remaining.pop(0)
        if remaining:
            raise IRSchemaError(
                f"F.{cls.namespace}.{cls.functional_name} received {len(remaining)} extra positional arguments.")
        return tuple(inputs), values

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        remaining = dict(attributes)
        result: dict[str, object] = {}
        for parameter in cls.attribute_parameters:
            if parameter.name in remaining:
                result[parameter.name] = remaining.pop(parameter.name)
            elif parameter.required:
                raise IRSchemaError(f"{cls.op_name} requires attribute {parameter.name!r}.")
            else:
                result[parameter.name] = parameter.default
        if remaining:
            raise IRSchemaError(f"{cls.op_name} does not accept attributes {sorted(remaining)}.")
        return result

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        raise IRSchemaError(f"{cls.op_name} does not implement type inference.")

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return PURE

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return attrs

    @classmethod
    def verify(cls, node: Node, module: IRModule) -> None:
        cls.verify_arity(node)
        inputs = tuple(module.node_map[value] for value in node.inputs)
        try:
            attrs = cls.normalize_attrs(node.attrs)
            distributed_semantic = (
                not node.op.startswith("distributed.")
                and (contains_distributed(node.type) or any(contains_distributed(value.type) for value in inputs))
            )
            distributed_inferred = False
            if distributed_semantic:
                actual_error: Exception | None = None
                try:
                    cls.verify_parameter_types(inputs)
                    distributed_type = cls.infer_call_type(inputs, attrs)
                    distributed_effect = cls.infer_effect(inputs, attrs)
                except (IRSchemaError, KeyError, TypeError, ValueError, AssertionError) as error:
                    actual_error = error
                    distributed_type = None
                    distributed_effect = None
                if distributed_type == node.type:
                    inferred_type = distributed_type
                    inferred_effect = distributed_effect
                    distributed_semantic = False
                    distributed_inferred = True
                elif "distributed_candidate" in node.metadata:
                    detail = "type inference disagreed with the stored candidate"
                    if actual_error is not None:
                        detail = str(actual_error)
                    raise IRSchemaError(
                        f"{node.op} distributed candidate {node.metadata['distributed_candidate']!r} is illegal: {detail}.")
            if distributed_semantic:
                logical_inputs = tuple(
                    Node(
                        id=value.id,
                        op=value.op,
                        inputs=value.inputs,
                        type=logical_type(value.type),
                        effect=value.effect,
                        attrs=value.attrs,
                        metadata=value.metadata,
                    )
                    for value in inputs
                )
                cls.verify_parameter_types(logical_inputs)
                inferred_type = cls.infer_type(logical_inputs, attrs)
                inferred_effect = cls.infer_effect(logical_inputs, attrs)
            elif not distributed_inferred:
                cls.verify_parameter_types(inputs)
                inferred_type = cls.infer_type(inputs, attrs)
                inferred_effect = cls.infer_effect(inputs, attrs)
        except (IRSchemaError, KeyError, TypeError, ValueError) as error:
            raise IRVerificationError(str(error), stage=module.stage, node_id=node.id) from error
        if node.attrs != canonical_attributes(cls.ir_attrs(attrs)):
            raise IRVerificationError(
                f"{node.op} stored attributes are not in canonical op form.",
                stage=module.stage,
                node_id=node.id,
            )
        stored_type = logical_type(node.type) if distributed_semantic else node.type
        if stored_type != inferred_type:
            raise IRVerificationError(
                f"{node.op} stored type does not match its inferred type.",
                stage=module.stage,
                node_id=node.id,
            )
        if node.effect != inferred_effect:
            raise IRVerificationError(
                f"{node.op} stored effect does not match its inferred effect.",
                stage=module.stage,
                node_id=node.id,
            )

    @classmethod
    def verify_arity(cls, node: Node) -> None:
        fixed = sum(not parameter.variadic for parameter in cls.input_parameters)
        variadic = any(parameter.variadic for parameter in cls.input_parameters)
        if len(node.inputs) < fixed or (not variadic and len(node.inputs) != fixed):
            expectation = f"at least {fixed}" if variadic else str(fixed)
            raise IRVerificationError(
                f"Op {node.op!r} expects {expectation} inputs, got {len(node.inputs)}.",
                node_id=node.id,
            )

    @classmethod
    def verify_parameter_types(cls, inputs: Sequence[Node]) -> None:
        """Apply every operand contract declared by ``ParameterInfo``."""

        for parameter in cls.input_parameters:
            parameter.type_of(inputs)

    @classmethod
    def evaluate(cls, node: Node, arguments: Sequence[Any], context: Any) -> Any:
        raise EvaluationError(f"Torch evaluator does not support op {node.op!r}.", node_id=node.id)

    @classmethod
    def visit(cls, node: Node, visitor: object) -> object:
        method = getattr(visitor, f"visit_{cls.op_name.replace('.', '_')}", None)
        if method is None:
            method = getattr(visitor, "visit_node", None)
        if method is None and callable(visitor):
            method = visitor
        if method is None:
            raise TypeError(f"Visitor {type(visitor).__name__} cannot visit {cls.op_name!r}.")
        return method(node)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("not-modeled",))

    @classmethod
    def cost_factors(
        cls,
        inputs: Sequence[Node],
        attrs: Mapping[str, object],
        return_type: IRType,
    ) -> OpCostFactors | None:
        """Return target-aggregatable factors for a typed candidate call.

        Definitions with operand-dependent behavior override this method.
        The default adapts a complete-enough local ``cost`` record so simple
        elementwise, storage-transform and zero-copy ops do not duplicate a
        second evaluator merely to participate in AutoDistribution.
        """

        del inputs
        local_return = _local_cost_type(return_type)
        synthetic = Node(
            "<candidate-cost>",
            cls.op_name,
            (),
            local_return,
            attrs=attrs,
        )
        metric = cls.cost(synthetic)
        if all(
            value is None
            for value in (
                metric.flops,
                metric.bytes_read,
                metric.bytes_written,
                metric.communication_bytes,
                metric.synchronizations,
            )
        ):
            return None
        return OpCostFactors(
            cpu_cycles=metric.flops or 0,
            block_local_memory_load_bytes=metric.bytes_read or 0,
            block_local_memory_store_bytes=metric.bytes_written or 0,
            grid_synchronizations=metric.synchronizations or 0,
            communication_cycles=metric.communication_bytes or 0,
        )

    @classmethod
    def python_call(cls, node: Node) -> PythonCall:
        keywords: dict[str, object] = {}
        for parameter in cls.attribute_parameters:
            if parameter.name in node.attrs:
                keywords[parameter.name] = node.attrs[parameter.name]
        known = set(keywords)
        keywords.update((key, value) for key, value in node.attrs.items() if key not in known)
        keywords["name"] = node.id
        if node.metadata:
            keywords["metadata"] = node.metadata
        return PythonCall(
            f"F.{cls.namespace}.{cls.functional_name}",
            tuple(NodeRef(value) for value in node.inputs),
            keywords,
        )

def op_definition(
    op_name: str,
    *,
    namespace: str | None = None,
    functional_name: str | None = None,
    display_name: str | None = None,
) -> Callable[[type[OpDefinition]], type[OpDefinition]]:
    """Register one explicitly imported definition and bind its parameters."""

    def decorate(definition: type[OpDefinition]) -> type[OpDefinition]:
        if op_name in _DEFINITIONS:
            raise ValueError(f"Op definition {op_name!r} is already registered.")
        if (namespace is None) != (functional_name is None):
            raise ValueError("Functional namespace and name must be provided together.")
        declared_parameters = tuple(
            (name, value)
            for name, value in definition.__dict__.items()
            if isinstance(value, ParameterInfo)
        )
        reserved_names = {
            "parameters",
            "input_parameters",
            "attribute_parameters",
            "inplace_input_parameters",
            "inplace_output_parameters",
            "result_memory_effects",
            "byte_preserving_input_parameters",
            "op_name",
            "namespace",
            "functional_name",
            "display_name",
        }
        conflicting = sorted(name for name, _ in declared_parameters if name in reserved_names)
        if conflicting:
            raise TypeError(
                f"Op {op_name!r} declares reserved ParameterInfo fields {conflicting}; "
                "use a different Python field name and pass the IR name explicitly.")
        parameters = tuple(value for _, value in declared_parameters)
        input_parameters = tuple(value for value in parameters if value.kind == ParameterKind.INPUT)
        attribute_parameters = tuple(value for value in parameters if value.kind == ParameterKind.ATTRIBUTE)
        if sum(parameter.variadic for parameter in input_parameters) > 1:
            raise TypeError(f"Op {op_name!r} can declare at most one variadic input.")
        if any(parameter.variadic for parameter in input_parameters[:-1]):
            raise TypeError(f"Op {op_name!r} variadic input must be the last input parameter.")
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise TypeError(f"Op {op_name!r} parameter names must be unique.")
        invalid_names = sorted(name for name in names if not name.isidentifier() or keyword.iskeyword(name))
        if invalid_names:
            raise TypeError(f"Op {op_name!r} parameter names are not valid Python identifiers: {invalid_names}.")
        if namespace is not None and {"name", "metadata"}.intersection(names):
            raise TypeError(f"Functional op {op_name!r} cannot use reserved name/metadata parameter names.")
        input_index = 0
        for index, parameter in enumerate(parameters):
            parameter.bind(
                definition,
                index=index,
                input_index=input_index if parameter.kind == ParameterKind.INPUT else None,
            )
            if parameter.kind == ParameterKind.INPUT and not parameter.variadic:
                input_index += 1
        inplace_input_parameters = tuple(
            definition.__dict__.get("inplace_input_parameters", ())
        )
        if any(
            parameter not in input_parameters or parameter.variadic
            for parameter in inplace_input_parameters
        ):
            raise TypeError(
                f"Op {op_name!r} inplace_input_parameters must contain its own "
                "non-variadic input ParameterInfo fields."
            )
        inplace_output_parameters = tuple(
            definition.__dict__.get("inplace_output_parameters", ())
        )
        if any(
            parameter is not None
            and (parameter not in input_parameters or parameter.variadic)
            for parameter in inplace_output_parameters
        ):
            raise TypeError(
                f"Op {op_name!r} inplace_output_parameters must contain None or "
                "its own non-variadic input ParameterInfo fields."
            )
        byte_preserving_input_parameters = tuple(
            definition.__dict__.get("byte_preserving_input_parameters", ())
        )
        result_memory_effects = tuple(definition.result_memory_effects)
        if any(not isinstance(effect, MemoryEffect) for effect in result_memory_effects):
            raise TypeError(f"Op {op_name!r} result_memory_effects must be typed MemoryEffects.")
        if len(byte_preserving_input_parameters) > 1 or any(
            parameter not in input_parameters or parameter.variadic
            for parameter in byte_preserving_input_parameters
        ):
            raise TypeError(
                f"Op {op_name!r} byte_preserving_input_parameters must contain "
                "at most one of its own non-variadic input ParameterInfo fields."
            )
        definition.op_name = op_name
        definition.namespace = namespace
        definition.functional_name = functional_name
        definition.display_name = display_name or op_name
        definition.parameters = parameters
        definition.input_parameters = input_parameters
        definition.attribute_parameters = attribute_parameters
        definition.inplace_input_parameters = inplace_input_parameters
        definition.inplace_output_parameters = inplace_output_parameters
        definition.result_memory_effects = result_memory_effects
        definition.byte_preserving_input_parameters = byte_preserving_input_parameters
        _DEFINITIONS[op_name] = definition
        if namespace is not None and functional_name is not None:
            key = (namespace, functional_name)
            if key in _FUNCTIONALS:
                raise ValueError(f"Functional constructor F.{namespace}.{functional_name} is already registered.")
            _FUNCTIONALS[key] = definition
        return definition

    return decorate

def get_definition(op_name: str) -> type[OpDefinition]:
    _ensure_registered()
    try:
        return _DEFINITIONS[op_name]
    except KeyError as error:
        raise KeyError(f"No FlagMega op definition is registered for {op_name!r}.") from error


def definitions() -> tuple[type[OpDefinition], ...]:
    _ensure_registered()
    return tuple(_DEFINITIONS.values())


def functional_definition(namespace: str, name: str) -> type[OpDefinition]:
    _ensure_registered()
    try:
        return _FUNCTIONALS[(namespace, name)]
    except KeyError as error:
        raise AttributeError(f"No FlagMega functional constructor F.{namespace}.{name}.") from error


def visit_node(node: Node, visitor: object) -> object:
    return get_definition(node.op).visit(node, visitor)


def get_cost(node: Node) -> OpCost:
    return get_definition(node.op).cost(node)


def tensor_elements(value: TensorType) -> int | None:
    dimensions = [dimension.value for dimension in value.shape]
    if any(dimension is None for dimension in dimensions):
        return None
    lanes = getattr(value.dtype, "lane_count", 1)
    return prod(dimensions) * lanes  # type: ignore[arg-type]


def tensor_nbytes(value: TensorType) -> int | None:
    dimensions = [dimension.value for dimension in value.shape]
    if any(dimension is None for dimension in dimensions):
        return None
    # A VectorType is one physical tensor element whose ``itemsize`` already
    # includes every lane.  ``tensor_elements`` instead returns the scalar
    # logical element count, so multiplying the two would count lanes twice.
    return prod(dimensions) * value.dtype.itemsize  # type: ignore[arg-type]


def _local_cost_type(value: IRType) -> IRType:
    from triton.flagmega.ir.distributed_type import local_tensor_type
    from triton.flagmega.ir.model import DistributedType, TupleType

    if isinstance(value, DistributedType):
        return local_tensor_type(value)
    if isinstance(value, TupleType):
        return TupleType(tuple(_local_cost_type(field) for field in value.fields))
    return value


def _ensure_registered() -> None:
    from triton.flagmega.ir.ops import ensure_registered

    ensure_registered()


__all__ = [
    "CostKind",
    "MemoryEffect",
    "NodeRef",
    "OpCost",
    "OpCostFactors",
    "OpDefinition",
    "ParameterInfo",
    "ParameterKind",
    "PreparedCall",
    "PythonCall",
    "attribute_parameter",
    "construction_scope",
    "definitions",
    "functional_definition",
    "get_cost",
    "get_definition",
    "input_parameter",
    "op_definition",
    "tensor_elements",
    "tensor_nbytes",
    "variadic_input_parameter",
    "visit_node",
]
