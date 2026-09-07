# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Immutable, serializable FlagMega expression IR.

The in-memory representation deliberately contains only deterministic Python
values.  Canonical ``.py`` checkpoints are emitted by :mod:`.python_ir` and
can reconstruct every object in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.constant_recipe import ConstantRecipe
from triton.flagmega.ir.dim_expr import Dimension, DimConst, DimVar, dim, simplify_dim
from triton.flagmega.ir.distributed_type import (
    Placement,
    SBP,
    SBPPartial,
    is_distributable,
    placement_from_data,
    sbp_from_data,
)
from triton.flagmega.ir.types import (
    DType,
    DataType,
    VectorType,
    data_type,
    data_type_from_data,
    data_type_to_data,
)

if TYPE_CHECKING:
    from triton.flagmega.ir.tir import ExecutionFunction, PrimFunction, KernelDefinition


IR_VERSION = 1


class _FrozenMapping(Mapping[str, Any]):
    """An internally recognizable immutable mapping.

    ``MappingProxyType`` prevents mutation but gives no safe way to distinguish
    a recursively frozen FlagMega value from an arbitrary proxy supplied by a
    caller.  This marker lets dataclass replacement reuse canonical metadata
    in O(1) while ordinary input mappings still receive a deep freeze.
    """

    __slots__ = ("_data", "__weakref__")

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._data = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"_FrozenMapping({self._data!r})"


@dataclass(frozen=True)
class TensorLayout:
    """Logical-to-physical layout facts owned by the compiler."""

    order: tuple[int, ...] = ()
    strides: tuple[int | None, ...] = ()
    vector_lanes: tuple[int, ...] = ()
    tag: str = "dense"

    def to_data(self) -> dict[str, object]:
        return {
            "order": list(self.order),
            "strides": list(self.strides),
            "vector_lanes": list(self.vector_lanes),
            "tag": self.tag,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> TensorLayout:
        return cls(
            order=tuple(int(value) for value in data.get("order", ())),
            strides=tuple(_optional_int(value) for value in data.get("strides", ())),
            vector_lanes=tuple(int(value) for value in data.get("vector_lanes", ())),
            tag=str(data.get("tag", "dense")),
        )


class IRType:
    def to_data(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class AnyType(IRType):
    """Least-specific type used only before inference converges."""

    def to_data(self) -> dict[str, object]:
        return {"kind": "any"}


@dataclass(frozen=True)
class InvalidType(IRType):
    reason: str

    def to_data(self) -> dict[str, object]:
        return {"kind": "invalid", "reason": self.reason}


@dataclass(frozen=True)
class NoneType(IRType):
    def to_data(self) -> dict[str, object]:
        return {"kind": "none"}


@dataclass(frozen=True)
class TensorType(IRType):
    dtype: DataType
    shape: tuple[Dimension, ...]
    layout: TensorLayout = field(default_factory=TensorLayout)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", data_type(self.dtype))
        shape = tuple(simplify_dim(value) for value in self.shape)
        for dimension in shape:
            if dimension.is_fixed and dimension.fixed_value < 0:
                raise IRSchemaError(f"Tensor dimensions cannot be negative, got {dimension.fixed_value}.")
            if dimension.minimum is not None and dimension.minimum < 0:
                raise IRSchemaError(f"Tensor dimension {dimension} has a negative lower bound.")
        object.__setattr__(self, "shape", shape)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "tensor",
            "dtype": data_type_to_data(self.dtype),
            "shape": [value.to_data() for value in self.shape],
            "layout": self.layout.to_data(),
        }


@dataclass(frozen=True)
class TupleType(IRType):
    fields: tuple[IRType, ...]
    is_variadic: bool = False

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "tuple",
            "fields": [field.to_data() for field in self.fields],
            "is_variadic": self.is_variadic,
        }


@dataclass(frozen=True)
class CallableType(IRType):
    return_type: IRType
    parameters: tuple[IRType, ...]

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "callable",
            "return_type": self.return_type.to_data(),
            "parameters": [value.to_data() for value in self.parameters],
        }


@dataclass(frozen=True)
class RefType(IRType):
    name: str
    fields: tuple[tuple[str, IRType], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise IRSchemaError("Reference type requires a name.")
        object.__setattr__(self, "fields", tuple((str(name), value) for name, value in self.fields))

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "ref",
            "name": self.name,
            "fields": [{"name": name, "type": value.to_data()} for name, value in self.fields],
        }


@dataclass(frozen=True)
class DistributedType(IRType):
    tensor: TensorType
    axis_policies: tuple[SBP, ...]
    placement: Placement
    partial: SBPPartial | None = None

    def __post_init__(self) -> None:
        policies = tuple(self.axis_policies)
        if not isinstance(self.tensor, TensorType):
            raise IRSchemaError("DistributedType must wrap a TensorType.")
        if not isinstance(self.placement, Placement):
            raise IRSchemaError("DistributedType requires a Placement.")
        if not all(isinstance(policy, SBP) for policy in policies):
            raise IRSchemaError("DistributedType axis_policies must contain SBP values.")
        if not is_distributable(self.tensor, policies, self.placement):
            raise IRSchemaError("DistributedType policies are not distributable for the tensor and placement.")
        if self.partial is not None:
            if not isinstance(self.partial, SBPPartial):
                raise IRSchemaError("DistributedType partial must be SBPPartial or None.")
            if any(axis >= self.placement.rank for axis in self.partial.axes):
                raise IRSchemaError("DistributedType partial references an out-of-range placement axis.")
        object.__setattr__(self, "axis_policies", policies)

    @property
    def tensor_type(self) -> TensorType:
        return self.tensor

    @property
    def policies(self) -> tuple[SBP, ...]:
        """Compatibility spelling for early FlagMega checkpoints."""

        return self.axis_policies

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "distributed",
            "tensor": self.tensor.to_data(),
            "axis_policies": [policy.to_data() for policy in self.axis_policies],
            "placement": self.placement.to_data(),
            "partial": None if self.partial is None else self.partial.to_data(),
        }


def tensor_type(
    dtype: DataType | str,
    shape: Sequence[int | str | Dimension],
    *,
    layout: TensorLayout | None = None,
) -> TensorType:
    return TensorType(data_type(dtype), tuple(dim(value) for value in shape), layout or TensorLayout())


def logical_type(value: IRType) -> IRType:
    """Erase distribution while preserving tuple/reference structure."""

    if isinstance(value, DistributedType):
        return value.tensor
    if isinstance(value, TupleType):
        return TupleType(
            tuple(logical_type(field) for field in value.fields),
            value.is_variadic,
        )
    if isinstance(value, CallableType):
        return CallableType(
            logical_type(value.return_type),
            tuple(logical_type(parameter) for parameter in value.parameters),
        )
    if isinstance(value, RefType):
        return RefType(value.name, tuple((name, logical_type(field)) for name, field in value.fields))
    return value


def contains_distributed(value: IRType) -> bool:
    if isinstance(value, DistributedType):
        return True
    if isinstance(value, TupleType):
        return any(contains_distributed(field) for field in value.fields)
    if isinstance(value, CallableType):
        return contains_distributed(value.return_type) or any(
            contains_distributed(parameter) for parameter in value.parameters
        )
    if isinstance(value, RefType):
        return any(contains_distributed(field) for _, field in value.fields)
    return False


def type_from_data(data: Mapping[str, Any]) -> IRType:
    kind = data.get("kind")
    if kind == "any":
        return AnyType()
    if kind == "invalid":
        return InvalidType(str(data.get("reason", "")))
    if kind == "none":
        return NoneType()
    if kind == "tensor":
        return TensorType(
            data_type_from_data(data["dtype"]),
            tuple(Dimension.from_data(value) for value in data.get("shape", ())),
            TensorLayout.from_data(data.get("layout", {})),
        )
    if kind == "tuple":
        return TupleType(
            tuple(type_from_data(value) for value in data.get("fields", ())),
            bool(data.get("is_variadic", False)),
        )
    if kind == "callable":
        return CallableType(
            type_from_data(data["return_type"]),
            tuple(type_from_data(value) for value in data.get("parameters", ())),
        )
    if kind == "ref":
        return RefType(
            str(data["name"]),
            tuple((str(value["name"]), type_from_data(value["type"])) for value in data.get("fields", ())),
        )
    if kind == "distributed":
        tensor = type_from_data(data["tensor"])
        if not isinstance(tensor, TensorType):
            raise IRSchemaError("Distributed type must wrap a tensor type.")
        return DistributedType(
            tensor,
            tuple(sbp_from_data(value) for value in data.get("axis_policies", ())),
            placement_from_data(data["placement"]),
            None if data.get("partial") is None else _require_partial(sbp_from_data(data["partial"])),
        )
    raise IRSchemaError(f"Unknown IR type kind {kind!r}.")


class EffectKind(str, Enum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class Effect:
    kind: EffectKind = EffectKind.PURE
    resource: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EffectKind(self.kind))
        if self.kind != EffectKind.PURE and not self.resource:
            raise IRSchemaError(f"Effect {self.kind.value} requires a resource name.")
        if self.kind == EffectKind.PURE and self.resource is not None:
            raise IRSchemaError("A pure effect cannot name a resource.")

    @property
    def is_pure(self) -> bool:
        return self.kind == EffectKind.PURE

    def to_data(self) -> dict[str, object]:
        return {"kind": self.kind.value, "resource": self.resource}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> Effect:
        return cls(EffectKind(str(data.get("kind", "pure"))), data.get("resource"))


PURE = Effect()


def effect(kind: EffectKind | str = EffectKind.PURE, resource: str | None = None) -> Effect:
    return Effect(EffectKind(kind), resource)


@dataclass(frozen=True)
class Candidate:
    id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    facts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))
        object.__setattr__(self, "facts", _freeze_mapping(self.facts))

    def to_data(self) -> dict[str, object]:
        return {"id": self.id, "parameters": _to_data(self.parameters), "facts": _to_data(self.facts)}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> Candidate:
        return cls(str(data["id"]), data.get("parameters", {}), data.get("facts", {}))


@dataclass(frozen=True)
class SelectionPoint:
    id: str
    kind: str
    candidates: tuple[Candidate, ...]
    default_candidate: str
    owner: str | None = None

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "candidates": [value.to_data() for value in self.candidates],
            "default_candidate": self.default_candidate,
            "owner": self.owner,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> SelectionPoint:
        return cls(
            str(data["id"]),
            str(data["kind"]),
            tuple(Candidate.from_data(value) for value in data.get("candidates", ())),
            str(data["default_candidate"]),
            None if data.get("owner") is None else str(data["owner"]),
        )


@dataclass(frozen=True)
class SelectionRecord:
    point_id: str
    candidate_id: str
    origin: str
    policy: str
    rationale: str = ""
    evidence: tuple[str, ...] = ()

    def to_data(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "candidate_id": self.candidate_id,
            "origin": self.origin,
            "policy": self.policy,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> SelectionRecord:
        return cls(
            str(data["point_id"]),
            str(data["candidate_id"]),
            str(data["origin"]),
            str(data["policy"]),
            str(data.get("rationale", "")),
            tuple(str(value) for value in data.get("evidence", ())),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    stage: str
    parent_semantic_hash: str | None
    producer: str
    rationale: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "parent_semantic_hash": self.parent_semantic_hash,
            "producer": self.producer,
            "rationale": self.rationale,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> ProvenanceRecord:
        return cls(
            str(data["stage"]),
            None if data.get("parent_semantic_hash") is None else str(data["parent_semantic_hash"]),
            str(data["producer"]),
            str(data.get("rationale", "")),
        )


@dataclass(frozen=True)
class Node:
    id: str
    op: str
    inputs: tuple[str, ...]
    type: IRType
    effect: Effect = PURE
    attrs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.op:
            raise IRSchemaError("IR nodes require non-empty id and op fields.")
        object.__setattr__(self, "inputs", tuple(str(value) for value in self.inputs))
        object.__setattr__(self, "attrs", canonical_attributes(self.attrs))
        object.__setattr__(self, "metadata", canonical_attributes(self.metadata))

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "type": self.type.to_data(),
            "effect": self.effect.to_data(),
            "attrs": _to_data(self.attrs),
            "metadata": _to_data(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> Node:
        return cls(
            id=str(data["id"]),
            op=str(data["op"]),
            inputs=tuple(str(value) for value in data.get("inputs", ())),
            type=type_from_data(data["type"]),
            effect=Effect.from_data(data.get("effect", {})),
            attrs=data.get("attrs", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Function:
    name: str
    parameters: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(str(value) for value in self.parameters))
        object.__setattr__(self, "outputs", tuple(str(value) for value in self.outputs))
        object.__setattr__(self, "attrs", _freeze_mapping(self.attrs))

    def to_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "outputs": list(self.outputs),
            "attrs": _to_data(self.attrs),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> Function:
        return cls(
            str(data["name"]),
            tuple(str(value) for value in data.get("parameters", ())),
            tuple(str(value) for value in data.get("outputs", ())),
            data.get("attrs", {}),
        )


@dataclass(frozen=True)
class IRModule:
    dialect: str
    stage: str
    nodes: tuple[Node, ...]
    functions: tuple[Function, ...]
    entry: str
    prim_functions: tuple[PrimFunction, ...] = ()
    execution_functions: tuple[ExecutionFunction, ...] = ()
    kernel_definitions: tuple[KernelDefinition, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    constant_recipes: tuple[ConstantRecipe, ...] = ()
    selection_points: tuple[SelectionPoint, ...] = ()
    selections: tuple[SelectionRecord, ...] = ()
    provenance: tuple[ProvenanceRecord, ...] = ()
    ir_version: int = IR_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "functions", tuple(self.functions))
        object.__setattr__(self, "prim_functions", tuple(self.prim_functions))
        object.__setattr__(self, "execution_functions", tuple(self.execution_functions))
        object.__setattr__(self, "kernel_definitions", tuple(self.kernel_definitions))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "constant_recipes", tuple(self.constant_recipes))
        object.__setattr__(self, "selection_points", tuple(self.selection_points))
        object.__setattr__(self, "selections", tuple(self.selections))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if self.ir_version != IR_VERSION:
            raise IRSchemaError(f"Unsupported FlagMega IR version {self.ir_version}; expected {IR_VERSION}.")

    @cached_property
    def node_map(self) -> Mapping[str, Node]:
        # Passes replace the immutable module on edits. Share its index until
        # then, without exposing a mutable backdoor into verification/matching.
        return MappingProxyType({node.id: node for node in self.nodes})

    @property
    def function_map(self) -> dict[str, Function]:
        return {function.name: function for function in self.functions}

    @property
    def prim_function_map(self) -> dict[str, PrimFunction]:
        return {function.name: function for function in self.prim_functions}

    @property
    def kernel_callable_map(self):
        """Kernel contracts, including pre-bufferization legacy PrimFunctions."""
        return {value.name: value for value in (*self.prim_functions, *self.kernel_definitions)}

    @property
    def execution_function_map(self) -> dict[str, ExecutionFunction]:
        return {function.name: function for function in self.execution_functions}

    @property
    def selection_map(self) -> dict[str, SelectionRecord]:
        return {record.point_id: record for record in self.selections}

    def semantic_data(self) -> dict[str, object]:
        data = {
            "ir_version": self.ir_version,
            "dialect": self.dialect,
            "stage": self.stage,
            "nodes": [node.to_data() for node in self.nodes],
            "functions": [function.to_data() for function in self.functions],
            "entry": self.entry,
            "metadata": _to_data(self.metadata),
            "constant_recipes": [value.to_data() for value in self.constant_recipes],
            "selection_points": [value.to_data() for value in self.selection_points],
            "selections": [value.to_data() for value in self.selections],
        }
        # Backward-compatible additive schema: omitting the empty field keeps
        # existing editable v1 checkpoint hashes stable.
        if self.prim_functions:
            data["prim_functions"] = [value.to_data() for value in self.prim_functions]
        if self.kernel_definitions:
            data["kernel_definitions"] = [value.to_data() for value in self.kernel_definitions]
        if self.execution_functions:
            data["execution_functions"] = [
                value.to_data() for value in self.execution_functions
            ]
        return data

    @cached_property
    def semantic_hash(self) -> str:
        payload = _semantic_payload(self)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_data(self) -> dict[str, object]:
        data = self.semantic_data()
        data["provenance"] = [value.to_data() for value in self.provenance]
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> IRModule:
        from triton.flagmega.ir.tir import tir_from_data

        return cls(
            dialect=str(data["dialect"]),
            stage=str(data["stage"]),
            nodes=tuple(Node.from_data(value) for value in data.get("nodes", ())),
            functions=tuple(Function.from_data(value) for value in data.get("functions", ())),
            entry=str(data["entry"]),
            prim_functions=tuple(tir_from_data(value) for value in data.get("prim_functions", ())),
            kernel_definitions=tuple(tir_from_data(value) for value in data.get("kernel_definitions", ())),
            execution_functions=tuple(
                tir_from_data(value) for value in data.get("execution_functions", ())
            ),
            metadata=data.get("metadata", {}),
            constant_recipes=tuple(ConstantRecipe.from_data(value) for value in data.get("constant_recipes", ())),
            selection_points=tuple(SelectionPoint.from_data(value) for value in data.get("selection_points", ())),
            selections=tuple(SelectionRecord.from_data(value) for value in data.get("selections", ())),
            provenance=tuple(ProvenanceRecord.from_data(value) for value in data.get("provenance", ())),
            ir_version=int(data.get("ir_version", IR_VERSION)),
        )


def function_module(
    module: IRModule,
    function_name: str,
    *,
    include_unreferenced: bool = False,
) -> IRModule:
    """Return a self-contained module containing one function's node closure.

    Pass dumps use this view so ``Before``/``After`` directories contain one
    executable Python checkpoint per function, matching nncase's dump model.
    """

    try:
        function = module.function_map[function_name]
    except KeyError as error:
        raise IRSchemaError(f"Unknown function {function_name!r} in module {module.entry!r}.") from error
    node_map = module.node_map

    def closure(value: Function) -> set[str]:
        result = set(value.parameters)
        pending = list(value.outputs)
        while pending:
            node_id = pending.pop()
            if node_id in result:
                continue
            try:
                node = node_map[node_id]
            except KeyError as error:
                raise IRSchemaError(
                    f"Function {value.name!r} references missing node {node_id!r}.") from error
            result.add(node_id)
            pending.extend(node.inputs)
        return result

    reachable = closure(function)
    if module.constant_recipes:
        # Constant recipes are module-level artifact data.  Keep every exported
        # leaf in each per-function dump so the standalone function checkpoint
        # remains globally verifiable even when a recipe is shared by, or only
        # consumed from, another function.
        reachable.update(
            node.id for node in module.nodes
            if node.op == "builtin.const_asset"
            or (node.op == "tir.buffer" and "constant_recipe" in node.metadata)
        )
    if include_unreferenced:
        referenced = set().union(*(closure(value) for value in module.functions))
        reachable.update(node_map.keys() - referenced)
    nodes = tuple(node for node in module.nodes if node.id in reachable)
    points = tuple(
        point for point in module.selection_points
        if point.owner is None or point.owner in reachable
    )
    point_ids = {point.id for point in points}
    selections = tuple(record for record in module.selections if record.point_id in point_ids)
    signatures = {
        value.name: {
            "parameters": tuple(node_map[node_id].type for node_id in value.parameters),
            "outputs": tuple(node_map[node_id].type for node_id in value.outputs),
        }
        for value in module.functions
    }
    metadata = dict(module.metadata)
    if "function_signatures" not in metadata:
        metadata["function_signatures"] = signatures
        metadata["_dump_synthesized_function_signatures"] = True
    # ExecutionFunction schedules form one interprocedural closure and are
    # intentionally duplicated in every per-function checkpoint.  Mark the
    # graph view as a fragment so verification can distinguish this explicit
    # dump contract from an accidentally mismatched ordinary module.
    metadata["_dump_function_fragment"] = True
    return IRModule(
        dialect=module.dialect,
        stage=module.stage,
        nodes=nodes,
        functions=(function,),
        entry=function.name,
        prim_functions=module.prim_functions,
        kernel_definitions=module.kernel_definitions,
        execution_functions=module.execution_functions,
        metadata=metadata,
        constant_recipes=module.constant_recipes,
        selection_points=points,
        selections=selections,
        provenance=module.provenance,
        ir_version=module.ir_version,
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, _FrozenMapping):
        return value
    return _FrozenMapping({
        str(key): _freeze(item)
        for key, item in sorted(value.items())
    })


def canonical_attributes(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the immutable canonical representation used by IR maps."""

    return _freeze_mapping(value)


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IRSchemaError("IR attributes must not contain NaN or infinity.")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, IRType):
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise IRSchemaError(f"Unsupported IR attribute value {type(value).__name__}: {value!r}.")


def _to_data(value: Any) -> Any:
    if isinstance(value, IRType):
        return {"$ir_type": value.to_data()}
    if isinstance(value, Mapping):
        return {str(key): _to_data(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_to_data(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


# Passes construct a new immutable IRModule at every semantic boundary but
# deliberately retain the unchanged Node, SelectionPoint, ConstantRecipe and
# TIR objects.  Building the legacy canonical JSON through ``semantic_data``
# re-expanded those shared trees for every pass.  Cache only their encoded
# fragments by object identity: equal objects reconstructed from an edited
# checkpoint are encoded independently, and weak references prevent this
# compile-time acceleration from retaining obsolete IR.
_SEMANTIC_COMPONENT_JSON: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _semantic_component_json(value: Any) -> str:
    return _semantic_identity_json(value, lambda: _canonical_json(value.to_data()))


def _semantic_mapping_json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, _FrozenMapping):
        return _canonical_json(_to_data(value))
    return _semantic_identity_json(value, lambda: _canonical_json(_to_data(value)))


def _semantic_identity_json(value: Any, encode) -> str:
    identity = id(value)
    cached = _SEMANTIC_COMPONENT_JSON.get(identity)
    if cached is not None and cached[0]() is value:
        return cached[1]
    encoded = encode()

    def discard(reference: weakref.ReferenceType[Any], *, key: int = identity) -> None:
        current = _SEMANTIC_COMPONENT_JSON.get(key)
        if current is not None and current[0] is reference:
            _SEMANTIC_COMPONENT_JSON.pop(key, None)

    try:
        reference = weakref.ref(value, discard)
    except TypeError:
        # All current IR components are weak-referenceable dataclasses.  Keep
        # serialization correct if a future compact component uses slots.
        return encoded
    _SEMANTIC_COMPONENT_JSON[identity] = (reference, encoded)
    return encoded


def _semantic_array(values: Iterable[Any]) -> str:
    return "[" + ",".join(_semantic_component_json(value) for value in values) + "]"


def _semantic_payload(module: IRModule) -> str:
    """Encode exactly the canonical JSON previously produced by semantic_data.

    Keeping this byte-for-byte contract matters because editable checkpoints,
    selection plans, rdata caches and artifacts all use ``semantic_hash`` as a
    durable content identity.
    """

    fields = {
        "constant_recipes": _semantic_array(module.constant_recipes),
        "dialect": _canonical_json(module.dialect),
        "entry": _canonical_json(module.entry),
        "functions": _semantic_array(module.functions),
        "ir_version": _canonical_json(module.ir_version),
        "metadata": _semantic_mapping_json(module.metadata),
        "nodes": _semantic_array(module.nodes),
        "selection_points": _semantic_array(module.selection_points),
        "selections": _semantic_array(module.selections),
        "stage": _canonical_json(module.stage),
    }
    # Match semantic_data's backward-compatible omission of empty additive
    # fields.  json.dumps(sort_keys=True) orders the same top-level keys.
    if module.prim_functions:
        fields["prim_functions"] = _semantic_array(module.prim_functions)
    if module.kernel_definitions:
        fields["kernel_definitions"] = _semantic_array(module.kernel_definitions)
    if module.execution_functions:
        fields["execution_functions"] = _semantic_array(
            module.execution_functions
        )
    return "{" + ",".join(
        _canonical_json(key) + ":" + fields[key]
        for key in sorted(fields)
    ) + "}"


def _require_partial(value: SBP) -> SBPPartial:
    if not isinstance(value, SBPPartial):
        raise IRSchemaError("DistributedType partial must decode to SBPPartial.")
    return value
