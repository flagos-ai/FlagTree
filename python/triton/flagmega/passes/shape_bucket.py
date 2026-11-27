# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bind one declared shape-bucket specialization before target lowering.

FlagMega imports a bucketed input extent (today ``tokens``) as a symbolic
``DimVar`` carrying the declared ``[minimum, maximum]`` range.  nncase's
ShapeBucket compiles one static specialization per segment of such a range and
dispatches at runtime on the actual extent.  This pass is that specialization
step: it substitutes every symbolic dimension with the segment's upper bound,
so every later pass (vectorization, packing, AutoDistributed, Bufferize, TIR
selection and codegen) sees the fixed geometry it already requires.

The declaration lives in the imported module's metadata and mirrors
``nncase.ShapeBucketOptions``.  The selected specialization is recorded in
``shape_bucket_binding`` so a checkpoint or artifact states which bucket it
targets.  A module with no declaration and no symbolic dimension is returned
unchanged (the existing static import path).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import (
    CallableType,
    Dimension,
    DimVar,
    DistributedType,
    IRModule,
    RefType,
    TensorType,
    TupleType,
    substitute_dim,
)

DECLARATION_SCHEMA = "flagmega.shape-bucket/v1"
BINDING_SCHEMA = "flagmega.shape-bucket-binding/v1"
DECLARATION_KEY = "shape_bucket"
BINDING_KEY = "shape_bucket_binding"


@dataclass(frozen=True)
class ShapeBucketSegment:
    """One compiled specialization of every ranged variable."""

    index: int
    lower: int
    upper: int
    bindings: Mapping[str, int]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IRSchemaError(f"Shape-bucket {name} must be a positive integer, got {value!r}.")
    return value


def compute_segment_list(segments_count: int, minimum: int, maximum: int) -> tuple[int, ...]:
    """Port nncase ``ShapeBucketHelper.ComputeSegmentList``.

    ``size = (max - min) // count`` and the first ``count - 1`` inclusive upper
    bounds are ``min + i * size``; the last bound is always ``max``.  Segment
    ``i`` therefore covers ``(bound[i - 1], bound[i]]`` and segment 0 also owns
    the declared minimum, which matches nncase's per-segment compilation.
    """

    count = _positive_int(segments_count, "segments_count")
    low = _positive_int(minimum, "range minimum")
    high = _positive_int(maximum, "range maximum")
    if high < low:
        raise IRSchemaError(
            f"Shape-bucket range minimum {low} exceeds its maximum {high}.")
    size = (high - low) // count
    return tuple([low + index * size for index in range(count - 1)] + [high])


def parse_shape_bucket(data: object) -> dict[str, object]:
    """Validate and canonicalize one ``shape_bucket`` declaration."""

    if not isinstance(data, Mapping):
        raise IRSchemaError("Shape-bucket declaration must be a mapping.")
    schema = data.get("schema", DECLARATION_SCHEMA)
    if schema != DECLARATION_SCHEMA:
        raise IRSchemaError(
            f"Unsupported shape-bucket schema {schema!r}; expected {DECLARATION_SCHEMA!r}.")
    enable = data.get("enable", True)
    if not isinstance(enable, bool):
        raise IRSchemaError("Shape-bucket enable must be boolean.")
    ranges = data.get("range_info", {})
    if not isinstance(ranges, Mapping):
        raise IRSchemaError("Shape-bucket range_info must be a mapping of variable names.")
    canonical_ranges: dict[str, tuple[int, int]] = {}
    for name, bounds in ranges.items():
        if not isinstance(name, str) or not name:
            raise IRSchemaError("Shape-bucket range_info keys must be non-empty variable names.")
        if (not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes))
                or len(bounds) != 2):
            raise IRSchemaError(
                f"Shape-bucket range for {name!r} must contain exactly two bounds.")
        low = _positive_int(bounds[0], "range minimum")
        high = _positive_int(bounds[1], "range maximum")
        if high < low:
            raise IRSchemaError(
                f"Shape-bucket range minimum {low} exceeds its maximum {high} for {name!r}.")
        canonical_ranges[name] = (low, high)
    segments_count = data.get("segments_count", 2)
    if isinstance(segments_count, bool) or not isinstance(segments_count, int) or segments_count < 1:
        raise IRSchemaError("Shape-bucket segments_count must be a positive integer.")
    if enable and segments_count < 2:
        # nncase ShapeBucketRegister.CheckShapeBucketOptions.
        raise IRSchemaError("Shape-bucket segments_count must be at least two when enabled.")
    fixes = data.get("fix_var_map", {})
    if not isinstance(fixes, Mapping):
        raise IRSchemaError("Shape-bucket fix_var_map must be a mapping of variable names.")
    canonical_fixes: dict[str, int] = {}
    for name, value in fixes.items():
        if not isinstance(name, str) or not name:
            raise IRSchemaError("Shape-bucket fix_var_map keys must be non-empty variable names.")
        canonical_fixes[name] = _positive_int(value, f"fix value for {name!r}")
    result: dict[str, object] = {
        "schema": DECLARATION_SCHEMA,
        "enable": enable,
        "range_info": canonical_ranges,
        "segments_count": segments_count,
        "fix_var_map": canonical_fixes,
    }
    segment = data.get("segment")
    if segment is not None:
        if isinstance(segment, bool) or not isinstance(segment, int) or segment < 0:
            raise IRSchemaError("Shape-bucket segment must be a non-negative integer.")
        result["segment"] = segment
    return result


def shape_bucket_declaration(module: IRModule) -> dict[str, object] | None:
    """Return the validated declaration of ``module``, if it has one."""

    data = module.metadata.get(DECLARATION_KEY)
    if data is None:
        return None
    return parse_shape_bucket(data)


def shape_bucket_segments(declaration: Mapping[str, object]) -> tuple[ShapeBucketSegment, ...]:
    """Return every specialization declared by one parsed declaration.

    Every ranged variable is compiled at the same segment index, as nncase
    does.  ``lower``/``upper`` report the envelope of that segment for dispatch
    purposes; the authoritative per-variable values are ``bindings``.
    """

    parsed = parse_shape_bucket(declaration)
    count = int(parsed["segments_count"])
    ranges = parsed["range_info"]
    fixes = parsed["fix_var_map"]
    # All ranged variables are segmented over the same number of segments but
    # may have different bounds; segment i is the i-th upper bound of each.
    bounds = {name: compute_segment_list(count, low, high) for name, (low, high) in ranges.items()}
    segments = []
    for index in range(count):
        bindings = {name: values[index] for name, values in bounds.items()}
        # A variable named in both maps is fixed for every specialization;
        # nncase treats fix_var_map as the practical value of a dynamic dim.
        bindings.update(fixes)
        if ranges:
            lower = min(low for low, _ in ranges.values()) if index == 0 else min(
                # Segment i starts just after the previous segment's bound.
                values[index - 1] + 1 for values in bounds.values())
            upper = max(values[index] for values in bounds.values())
        else:
            lower = upper = 1
        segments.append(ShapeBucketSegment(index, lower, upper, bindings))
    return tuple(segments)


def shape_bucket_bindings(
    module: IRModule,
    *,
    segment: int | None = None,
) -> dict[str, int] | None:
    """Return the bindings the selected specialization applies, or ``None``."""

    declaration = shape_bucket_declaration(module)
    if declaration is None or not declaration["enable"]:
        return None
    segments = shape_bucket_segments(declaration)
    index = _segment_index(declaration, segments, segment)
    return dict(segments[index].bindings)


def bucket_index_for(
    segments: Sequence[ShapeBucketSegment],
    *,
    value: int,
) -> int:
    """Return the bucket a runtime extent dispatches to.

    The dispatch rule is the nncase one: choose the first segment whose
    inclusive upper bound covers ``value``; segment 0 owns the declared
    minimum and values outside every declared range are runtime errors.
    """

    if not segments:
        raise IRSchemaError("Shape-bucket dispatch requires at least one segment.")
    low = min(segment.lower for segment in segments)
    high = max(segment.upper for segment in segments)
    if value < low:
        raise IRSchemaError(f"Extent {value} is below the declared minimum {low}.")
    if value > high:
        raise IRSchemaError(f"Extent {value} is above the declared maximum {high}.")
    for segment in segments:
        if value <= segment.upper:
            return segment.index
    raise IRSchemaError(f"Extent {value} does not fit any declared shape-bucket segment.")


def _segment_index(
    declaration: Mapping[str, object],
    segments: Sequence[ShapeBucketSegment],
    segment: int | None,
) -> int:
    if segment is None:
        segment = declaration.get("segment")
    if segment is None:
        # The largest bucket is the default specialization: it covers the
        # complete declared range's upper bound and is the geometry a single
        # static artifact most often targets.
        segment = len(segments) - 1
    if isinstance(segment, bool) or not isinstance(segment, int) or not 0 <= segment < len(segments):
        raise IRSchemaError(
            f"Shape-bucket segment {segment!r} is outside the declared range "
            f"0..{len(segments) - 1}.")
    return int(segment)


def _bind_value_type(value_type, bindings: Mapping[str, int]):
    if isinstance(value_type, TensorType):
        shape = tuple(substitute_dim(dimension, bindings) for dimension in value_type.shape)
        return value_type if shape == value_type.shape else replace(value_type, shape=shape)
    if isinstance(value_type, DistributedType):
        tensor = _bind_value_type(value_type.tensor, bindings)
        return value_type if tensor is value_type.tensor else replace(value_type, tensor=tensor)
    if isinstance(value_type, TupleType):
        fields = tuple(_bind_value_type(value, bindings)
                       for value in value_type.fields)
        return value_type if fields == value_type.fields else replace(value_type, fields=fields)
    if isinstance(value_type, CallableType):
        return_type = _bind_value_type(value_type.return_type, bindings)
        parameters = tuple(_bind_value_type(value, bindings)
                           for value in value_type.parameters)
        if return_type == value_type.return_type and parameters == value_type.parameters:
            return value_type
        return replace(value_type, return_type=return_type, parameters=parameters)
    if isinstance(value_type, RefType):
        fields = tuple((name, _bind_value_type(value, bindings))
                       for name, value in value_type.fields)
        return value_type if fields == value_type.fields else replace(value_type, fields=fields)
    return value_type


def _dimension_symbols(value_type, result: set[str]) -> None:
    if isinstance(value_type, TensorType):
        for dimension in value_type.shape:
            _dimension_symbols(dimension, result)
        return
    if isinstance(value_type, DistributedType):
        _dimension_symbols(value_type.tensor, result)
        return
    if isinstance(value_type, TupleType):
        for field in value_type.fields:
            _dimension_symbols(field, result)
        return
    if isinstance(value_type, CallableType):
        _dimension_symbols(value_type.return_type, result)
        for parameter in value_type.parameters:
            _dimension_symbols(parameter, result)
        return
    if isinstance(value_type, RefType):
        for _, field in value_type.fields:
            _dimension_symbols(field, result)
        return
    if isinstance(value_type, DimVar):
        result.add(value_type.symbol)
        return
    if isinstance(value_type, Dimension) and not value_type.is_fixed:
        for operand in getattr(value_type, "operands", ()):
            _dimension_symbols(operand, result)


def module_dimension_symbols(module: IRModule) -> tuple[str, ...]:
    """Every symbolic dimension symbol carried by a graph node type."""

    symbols: set[str] = set()
    for node in module.nodes:
        _dimension_symbols(node.type, symbols)
    return tuple(sorted(symbols))


def bind_shape_buckets(
    module: IRModule,
    *,
    segment: int | None = None,
) -> IRModule:
    """Substitute the selected bucket's values into every symbolic dimension.

    A module without a declaration and without symbolic dimensions is returned
    unchanged, so the static ``seq_len=1`` import path is byte-identical.  A
    module with symbolic dimensions but no declaration is an error: nothing
    declares which specialization to compile.
    """

    declaration = shape_bucket_declaration(module)
    if declaration is None:
        missing = module_dimension_symbols(module)
        if missing:
            raise IRSchemaError(
                "Symbolic dimensions "
                f"{list(missing)} have no shape-bucket declaration; add a "
                f"{DECLARATION_KEY!r} metadata entry before compilation.")
        return module
    if not declaration["enable"]:
        return module
    segments = shape_bucket_segments(declaration)
    index = _segment_index(declaration, segments, segment)
    selected = segments[index]
    bindings = dict(selected.bindings)

    missing = tuple(symbol for symbol in module_dimension_symbols(module) if symbol not in bindings)
    if missing:
        raise IRSchemaError(
            f"Shape-bucket declaration does not bind symbolic dimensions {list(missing)}; "
            f"add them to range_info or fix_var_map.")

    previous_binding = module.metadata.get(BINDING_KEY)
    if previous_binding is not None:
        previous_values = dict(previous_binding.get("bindings", {}))
        if not module_dimension_symbols(module):
            if previous_values == bindings:
                # Idempotent re-entry of the same specialization.
                return module
            raise IRSchemaError(
                "Module is already specialized to shape bucket "
                f"{previous_values} and its dimensions are fixed; it cannot be "
                f"re-specialized to {bindings}.")

    nodes = []
    changed = False
    for node in module.nodes:
        value_type = _bind_value_type(node.type, bindings)
        if value_type is node.type:
            nodes.append(node)
        else:
            nodes.append(replace(node, type=value_type))
            changed = True
    if changed:
        module = replace(module, nodes=tuple(nodes))

    previous = module.metadata.get(BINDING_KEY)
    binding = {
        "schema": BINDING_SCHEMA,
        "segment": index,
        "segments": [value.upper for value in segments],
        "range_info": {name: list(bounds) for name, bounds in declaration["range_info"].items()},
        "fix_var_map": dict(declaration["fix_var_map"]),
        "bindings": bindings,
        "tokens_per_call": bindings.get("tokens"),
    }
    metadata = {**dict(module.metadata), BINDING_KEY: binding}
    if bindings.get("tokens") is not None and "tokens_per_call" in metadata:
        # The imported model uses the conventional token-count key; an
        # artifact must never claim the unbound range after specialization.
        metadata["tokens_per_call"] = bindings["tokens"]
    if previous == binding and metadata.get("tokens_per_call") == module.metadata.get("tokens_per_call"):
        return module
    return replace(module, metadata=metadata)


__all__ = [
    "BINDING_KEY",
    "BINDING_SCHEMA",
    "DECLARATION_KEY",
    "DECLARATION_SCHEMA",
    "ShapeBucketSegment",
    "bind_shape_buckets",
    "bucket_index_for",
    "compute_segment_list",
    "module_dimension_symbols",
    "parse_shape_bucket",
    "shape_bucket_bindings",
    "shape_bucket_declaration",
    "shape_bucket_segments",
]
