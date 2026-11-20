# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

PROGRAM_GRID_TRANSFORMS_ATTR = "hacc.program_grid_transforms"
PROGRAM_GRID_TRANSFORMS_VERSION = 2

GRAPH_OPTIMIZATION_KNOWN_RULE_MASK = 0xFBFF
INDEPENDENT_AXIS_TENSORIZE_RULE_BIT = 1 << 9
PERSISTENT_TASK_STRIP_MINING_RULE_BIT = 1 << 11
DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK = (511 | INDEPENDENT_AXIS_TENSORIZE_RULE_BIT
                                        | PERSISTENT_TASK_STRIP_MINING_RULE_BIT)

RUNTIME_ORIGINAL_GRID = "runtime_original_grid"
HIDDEN_EXTENT_AXES = (0, 1)
HIDDEN_ARGUMENT_ORDER = ("originalGridX", "originalGridY")
HIDDEN_ARGUMENT_TYPES = ("i32", "i32")


class ProgramGridContractError(ValueError):
    pass


_TOP_LEVEL_KEYS = frozenset((
    "version",
    "extent_source",
    "hidden_extent_axes",
    "hidden_argument_order",
    "hidden_argument_types",
    "transforms",
))
_TRANSFORM_KEYS = frozenset((
    "order",
    "kind",
    "axis",
    "factor",
    "persistent_coverage",
    "grid_stride_abi_verified",
))
_SUPPORTED_SEQUENCES = frozenset((
    ((1, 16, False, False), ),
    ((1, 16, False, False), (0, 4, True, True)),
    ((0, 64, True, True), ),
    ((0, 2, False, False), ),
    ((0, 4, False, False), ),
    ((0, 8, False, False), ),
    ((0, 16, False, False), ),
    ((0, 32, False, False), ),
))


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramGridContractError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ProgramGridContractError(f"{name} must be >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramGridContractError(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramGridContractError(f"{name} must be a mapping")
    return value


def _exact_sequence(value: Any, expected: tuple[Any, ...], name: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProgramGridContractError(f"{name} must be a sequence")
    if tuple(value) != expected:
        raise ProgramGridContractError(f"{name} must equal {list(expected)!r}")


def normalize_graph_optimization_rule_mask(raw: Any) -> int:
    mask = _integer(raw, "rule_mask", minimum=0)
    if mask & ~GRAPH_OPTIMIZATION_KNOWN_RULE_MASK:
        raise ProgramGridContractError("rule_mask contains unsupported graph-optimization bits")
    return mask


def normalize_program_grid_transforms(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProgramGridContractError("program_grid_transforms must be valid JSON when encoded as text") from error

    contract = _mapping(raw, "program_grid_transforms")
    if set(contract) != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - set(contract))
        unknown = sorted(set(contract) - _TOP_LEVEL_KEYS)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ProgramGridContractError("program_grid_transforms has an invalid top-level schema" +
                                       (": " + "; ".join(detail) if detail else ""))

    version = _integer(contract["version"], "version", minimum=1)
    if version != PROGRAM_GRID_TRANSFORMS_VERSION:
        raise ProgramGridContractError(f"unsupported program_grid_transforms version {version}; "
                                       f"expected {PROGRAM_GRID_TRANSFORMS_VERSION}")
    if contract["extent_source"] != RUNTIME_ORIGINAL_GRID:
        raise ProgramGridContractError("extent_source must be runtime_original_grid")
    _exact_sequence(contract["hidden_extent_axes"], HIDDEN_EXTENT_AXES, "hidden_extent_axes")
    _exact_sequence(contract["hidden_argument_order"], HIDDEN_ARGUMENT_ORDER, "hidden_argument_order")
    _exact_sequence(contract["hidden_argument_types"], HIDDEN_ARGUMENT_TYPES, "hidden_argument_types")

    transforms_raw = contract["transforms"]
    if isinstance(transforms_raw, (str, bytes)) or not isinstance(transforms_raw, Sequence):
        raise ProgramGridContractError("transforms must be a sequence")
    if len(transforms_raw) not in (1, 2):
        raise ProgramGridContractError("transforms must contain one or two entries")

    transforms: list[dict[str, Any]] = []
    sequence: list[tuple[int, int, bool, bool]] = []
    seen_axes: set[int] = set()
    persistent_count = 0
    for expected_order, raw_transform in enumerate(transforms_raw):
        transform = _mapping(raw_transform, f"transforms[{expected_order}]")
        if set(transform) != _TRANSFORM_KEYS:
            missing = sorted(_TRANSFORM_KEYS - set(transform))
            unknown = sorted(set(transform) - _TRANSFORM_KEYS)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unknown:
                detail.append("unknown " + ", ".join(unknown))
            raise ProgramGridContractError(f"transforms[{expected_order}] has an invalid schema" +
                                           (": " + "; ".join(detail) if detail else ""))
        order = _integer(transform["order"], f"transforms[{expected_order}].order", minimum=0)
        if order != expected_order:
            raise ProgramGridContractError("transform order must be contiguous from zero")
        if transform["kind"] != "ceil_div":
            raise ProgramGridContractError(f"transforms[{expected_order}].kind must be 'ceil_div'")
        axis = _integer(transform["axis"], f"transforms[{expected_order}].axis", minimum=0)
        if axis not in HIDDEN_EXTENT_AXES:
            raise ProgramGridContractError("dynamic program-grid transforms support only axes 0 and 1")
        if axis in seen_axes:
            raise ProgramGridContractError("dynamic program-grid transforms may not repeat an axis")
        seen_axes.add(axis)
        factor = _integer(transform["factor"], f"transforms[{expected_order}].factor", minimum=2)
        persistent = _boolean(transform["persistent_coverage"], f"transforms[{expected_order}].persistent_coverage")
        grid_stride = _boolean(transform["grid_stride_abi_verified"],
                               f"transforms[{expected_order}].grid_stride_abi_verified")
        if persistent != grid_stride:
            raise ProgramGridContractError("persistent coverage and grid-stride ABI verification must agree")
        persistent_count += int(persistent)
        if persistent_count > 1:
            raise ProgramGridContractError("at most one transform may be persistent")
        sequence.append((axis, factor, persistent, grid_stride))
        transforms.append({
            "order": order,
            "kind": "ceil_div",
            "axis": axis,
            "factor": factor,
            "persistent_coverage": persistent,
            "grid_stride_abi_verified": grid_stride,
        })

    if tuple(sequence) not in _SUPPORTED_SEQUENCES:
        raise ProgramGridContractError("unsupported dynamic IAT/PTSM transform sequence")
    return {
        "version": PROGRAM_GRID_TRANSFORMS_VERSION,
        "extent_source": RUNTIME_ORIGINAL_GRID,
        "hidden_extent_axes": list(HIDDEN_EXTENT_AXES),
        "hidden_argument_order": list(HIDDEN_ARGUMENT_ORDER),
        "hidden_argument_types": list(HIDDEN_ARGUMENT_TYPES),
        "transforms": transforms,
    }


def canonical_program_grid_transforms_json(raw: Any) -> str:
    return json.dumps(normalize_program_grid_transforms(raw), sort_keys=True, separators=(",", ":"))


def get_persistent_transform(contract: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for transform in normalize_program_grid_transforms(contract)["transforms"]:
        if transform["persistent_coverage"]:
            return transform
    return None


def apply_program_grid_transforms(
    grid: Sequence[int],
    contract: Mapping[str, Any],
    *,
    physical_core_count: int | None = None,
) -> tuple[int, int, int]:
    if len(grid) != 3:
        raise ProgramGridContractError("grid must contain exactly three dimensions")
    launch_grid = [_integer(value, f"grid[{axis}]", minimum=1) for axis, value in enumerate(grid)]
    normalized = normalize_program_grid_transforms(contract)
    for transform in normalized["transforms"]:
        axis = transform["axis"]
        launch_grid[axis] = (launch_grid[axis] + transform["factor"] - 1) // transform["factor"]
    persistent = get_persistent_transform(normalized)
    if persistent is not None:
        cores = _integer(physical_core_count, "physical_core_count", minimum=1)
        axis = persistent["axis"]
        other_programs = 1
        for other_axis, extent in enumerate(launch_grid):
            if other_axis != axis:
                other_programs *= extent
        launch_grid[axis] = min(launch_grid[axis], max(1, cores // max(1, other_programs)))
    return tuple(launch_grid)
