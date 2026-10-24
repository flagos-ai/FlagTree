# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1201 `s_prefetch_data` weight prefetch.

Two cases:
- Next invocation: a reused decode-layer function reads its weights through a
  per-call offset record. After its last GEMV only latency-bound phases remain,
  so HBM is idle; prefetching the next invocation's QKV slab there lets that
  GEMV start from cache.
- Split-K GEMV tiles: the owners of one mesh row stream the same weight region
  tile by tile in lockstep; each prefetches its share of a tile a few K steps
  ahead.

A prefetch has no architectural side effect, so generated values are unchanged.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, MutableMapping, Sequence
from math import prod

import numpy as np

from triton.flagmega.codegen.triton.function_pool_abi import (
    CALL_RECORD,
    OFFSET_TABLE,
    pool_parameter,
)

PREFETCH_CHUNK_BYTES = 4096  # one `s_prefetch_data ..., 31` request (32 x 128 B)
MAX_PREFETCH_BYTES = 128 * 1024
# R9700: the MLP down projection (12 K tiles) is fastest 4 tiles ahead (1/2/6/8
# measured slower); o-proj (4 K tiles) uses 2. Half the K tiles, capped at 4.
MAX_TILE_PREFETCH_DISTANCE = 4
_N8_K16_ATOM = 8 * 16


def add_next_invocation_qkv_prefetch(
    device_functions: Sequence[MutableMapping],
    encoded_calls: Mapping[str, Sequence[Mapping]],
    *,
    platform: str,
    architecture: str,
) -> bool:
    """Insert prefetch events into eligible functions; return whether any was added."""

    if platform != "amd" or architecture != "gfx1201":
        return False
    added = False
    for definition in device_functions:
        event = _prefetch_event(definition, encoded_calls.get(str(definition.get("function")), ()))
        if event is None:
            continue
        events = definition["schedule"]["consumer_events"]
        events.insert(_last_collective_index(events), event)
        added = True
    return added


def _prefetch_event(definition: Mapping, calls: Sequence[Mapping]) -> dict | None:
    plan = definition.get("function_pool_abi")
    if not plan or not int(plan.get("record_width", 0)):
        return None
    events = definition["schedule"]["consumer_events"]
    qkv_events = [event for event in events if event.get("family") == "qkv_parallel_linear"]
    if len(qkv_events) != 1:
        return None
    call = next((c for c in calls if str(c.get("call")) == str(qkv_events[0].get("call"))), None)
    if call is None:
        return None
    weight = _input_buffer(call, "q_weight_qkv_fused")
    if weight is None:
        return None
    field = next((f for f in plan["bindings"]
                  if str(f["formal"]) == str(weight["runtime_argument"]) and "record_column" in f), None)
    if field is None:
        return None
    width = int(plan["record_width"])
    rows = plan["calls"]
    offsets = [int(row["record_offset"]) for row in rows]
    if (len(rows) < 2 or len({row["caller"] for row in rows}) != 1
            or offsets != list(range(offsets[0], offsets[0] + width * len(rows), width))):
        return None
    index = _last_collective_index(events)
    if index is None or _previous_kernel_family(events, index) != "dense_matmul":
        return None
    abi = weight["abi"]
    try:
        slab_bytes = int(abi["scalar_storage_strides"][0]) * int(abi["scalar_itemsize"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if slab_bytes <= 0:
        return None
    displacement = f"tl.load({CALL_RECORD} + {width + int(field['record_column'])}, volatile=True)"
    if int(field.get("byte_offset", 0)):
        displacement = f"({displacement} + {int(field['byte_offset'])})"
    nbytes = min(slab_bytes, MAX_PREFETCH_BYTES)
    return {
        "kind": "prefetch",
        "call": str(qkv_events[0]["call"]),
        # Triton rejects pointer comparisons, so compare the record addresses.
        "guard": (f"{CALL_RECORD}.to(tl.int64, bitcast=True) != "
                  f"({OFFSET_TABLE}.to(tl.pointer_type(tl.int64)) + {offsets[-1]}).to(tl.int64, bitcast=True)"),
        "base": f"{pool_parameter(str(field['storage']))}.to(tl.pointer_type(tl.uint8)) + {displacement}",
        "slab_bytes": slab_bytes,
        "chunks": -(-nbytes // PREFETCH_CHUNK_BYTES),
    }


def add_split_k_weight_tile_prefetch(
    calls: Sequence[MutableMapping],
    mesh: Mapping,
    *,
    platform: str,
    architecture: str,
) -> bool:
    """Attach ``weight_tile_prefetch`` to eligible split-K GEMV calls; return whether any was."""

    if platform != "amd" or architecture != "gfx1201" or mesh.get("mesh_requires_device_dptr"):
        return False
    added = False
    for call in calls:
        if call.get("family") != "dense_matmul" or call.get("variant") != "split_k_n_packed_k_major_gemv":
            continue
        try:
            plan = _split_k_tile_prefetch(call, mesh)
        except (ArithmeticError, KeyError, NameError, SyntaxError, TypeError, ValueError):
            plan = None
        if plan is not None:
            call["weight_tile_prefetch"] = plan
            added = True
    return added


def _split_k_tile_prefetch(call: Mapping, mesh: Mapping) -> dict | None:
    """Derive the cooperative prefetch addressing by evaluating the call's own
    K/N mappings and weight offsets for every owner, or return None when the
    row's tile regions do not split into equal shares with affine addresses."""

    if (int(call.get("local_m_capacity", 0)) != 1 or call.get("packed_layout") != "k_major_n8_k16"
            or int(call.get("weight_k_contiguity", 1)) != 16 or not call.get("weight_itemsize")):
        return None
    itemsize = int(call["weight_itemsize"])
    if PREFETCH_CHUNK_BYTES % itemsize:
        return None
    chunk = PREFETCH_CHUNK_BYTES // itemsize
    n_capacity, k_capacity = int(call["local_n_capacity"]), int(call["local_k_capacity"])
    tile_n, block_k = int(call["tile_n"]), int(call["block_k"])
    if n_capacity % tile_n or k_capacity % block_k or tile_n % 8 or block_k % 16:
        return None
    n_tiles, k_tiles = n_capacity // tile_n, k_capacity // block_k
    distance = min(MAX_TILE_PREFETCH_DISTANCE, k_tiles // 2)
    if distance < 1:
        return None
    hierarchy = tuple(int(value) for value in mesh.get("mesh_hierarchy", ()))
    levels = tuple(mesh.get("mesh_levels", ()))
    if not hierarchy or len(levels) != len(hierarchy):
        return None
    extents = tuple(extent if level == "b" else 1 for extent, level in zip(hierarchy, levels))
    weight_offset = str(call["weight_offset"])
    for array, scalar in (("dense_global_k[None, :]", "__k"), ("dense_global_n[:, None]", "__n")):
        weight_offset = weight_offset.replace(array, scalar)
    if "[" in weight_offset:
        return None
    k_code = compile(str(call["global_k"]), "<split-K global K>", "eval")
    n_code = compile(str(call["global_n"]), "<split-K global N>", "eval")
    w_code = compile(weight_offset, "<split-K weight offset>", "eval")

    def shard_names(owner):
        names = {"__builtins__": {}}
        linear = 0
        for axis, value in enumerate(owner):
            names[f"shard_coord{axis}"] = value
            linear = linear * hierarchy[axis] + value
        if len(owner) == 2:
            names["shard_y"], names["shard_x"] = owner
        names["shard_index"] = linear
        return names

    def mapping(code, owner, local_name, capacity):
        value = eval(code, shard_names(owner), {local_name: np.arange(capacity, dtype=np.int64)})
        return np.broadcast_to(np.asarray(value, dtype=np.int64), (capacity,))

    def weight_offsets(k, n):
        return np.asarray(eval(w_code, {"__builtins__": {}}, {"__k": k, "__n": n}), dtype=np.int64)

    def with_axis(owner, axis, value):
        return owner[:axis] + (value,) + owner[axis + 1:]

    owners = list(itertools.product(*(range(extent) for extent in extents)))
    k_of = {owner: mapping(k_code, owner, "dense_local_k_offsets", k_capacity) for owner in owners}
    n_of = {owner: mapping(n_code, owner, "dense_local_n_offsets", n_capacity) for owner in owners}

    # Owners reading the same K slice share every weight tile: they form a row,
    # indexed by the row axes; the remaining (rank) axes order its members.
    origin = owners[0]
    row_axes = tuple(
        axis for axis, extent in enumerate(extents)
        if any(not np.array_equal(k_of[origin], k_of[with_axis(origin, axis, value)])
               for value in range(1, extent))
    )
    rank_axes = tuple(axis for axis, extent in enumerate(extents) if extent > 1 and axis not in row_axes)
    if not rank_axes:
        return None

    def row_of(owner):
        for axis in rank_axes:
            owner = with_axis(owner, axis, 0)
        return owner

    if any(not np.array_equal(k_of[owner], k_of[row_of(owner)]) for owner in owners):
        return None
    group = prod(extents[axis] for axis in rank_axes)

    starts = {}
    run_shape = None
    for row in sorted({row_of(owner) for owner in owners}):
        members = [owner for owner in owners if row_of(owner) == row]
        k_groups = [np.unique(tile // 16) * 16 for tile in k_of[row].reshape(k_tiles, block_k)]
        for n_tile in range(n_tiles):
            n_groups = np.unique(np.concatenate([
                n_of[owner][n_tile * tile_n:(n_tile + 1) * tile_n] // 8 for owner in members
            ])) * 8
            for k_tile in range(k_tiles):
                first = weight_offsets(k_groups[k_tile][None, :], n_groups[:, None])
                last = weight_offsets(k_groups[k_tile][None, :] + 15, n_groups[:, None] + 7)
                if not np.all(last - first == _N8_K16_ATOM - 1):
                    return None
                atoms = np.unique(first)
                pieces = np.split(atoms, np.flatnonzero(np.diff(atoms) != _N8_K16_ATOM) + 1)
                runs = [(int(piece[0]), len(piece) * _N8_K16_ATOM) for piece in pieces]
                lengths = {length for _, length in runs}
                strides = {b[0] - a[0] for a, b in zip(runs, runs[1:])}
                if len(lengths) != 1 or len(strides) > 1:
                    return None
                shape = (len(runs), lengths.pop(), strides.pop() if strides else 0)
                if run_shape not in (None, shape):
                    return None
                run_shape = shape
                starts[row, n_tile, k_tile] = runs[0][0]
    run_count, run_length, run_stride = run_shape
    share, remainder = divmod(run_count * run_length, group * chunk)
    if remainder or share < 1 or run_length % (share * chunk):
        return None
    shares_per_run = run_length // (share * chunk)

    base_row = row_of(origin)
    base = starts[base_row, 0, 0]
    k_stride = starts[base_row, 0, 1] - base if k_tiles > 1 else 0
    n_stride = starts[base_row, 1, 0] - base if n_tiles > 1 else 0
    row_strides = {axis: starts[with_axis(base_row, axis, 1), 0, 0] - base for axis in row_axes}
    for (row, n_tile, k_tile), start in starts.items():
        expected = (base + n_tile * n_stride + k_tile * k_stride
                    + sum(row_strides[axis] * row[axis] for axis in row_axes))
        if start != expected:
            return None

    # The template forms the tile term from an int32 loop counter.
    if (n_tiles - 1) * abs(n_stride) + (k_tiles - 1) * abs(k_stride) >= 2**31:
        return None
    rank = ""
    for axis in rank_axes:
        rank = f"shard_coord{axis}" if not rank else f"({rank} * {extents[axis]} + shard_coord{axis})"
    terms = [str(base)] if base else []
    terms += [f"{row_strides[axis]} * shard_coord{axis}" for axis in row_axes]
    if shares_per_run < group:
        terms.append(f"({rank} // {shares_per_run}) * {run_stride}")
    terms.append(f"({rank} % {shares_per_run}) * {share * chunk}")
    return {
        "distance": distance,
        "tiles": n_tiles * k_tiles,
        "k_tiles": k_tiles,
        "base": " + ".join(terms),
        "k_tile_stride": k_stride,
        "n_tile_stride": n_stride,
        "chunks": share,
    }


def _input_buffer(call: Mapping, formal: str) -> Mapping | None:
    for parameter in call.get("inputs", ()):
        if isinstance(parameter, Mapping) and parameter.get("formal") == formal:
            buffers = parameter.get("buffers")
            if isinstance(buffers, (tuple, list)) and buffers and isinstance(buffers[0], Mapping):
                return buffers[0]
    return None


def _last_collective_index(events: Sequence[Mapping]) -> int | None:
    indices = [index for index, event in enumerate(events) if event.get("execution_kind") == "collective"]
    return indices[-1] if indices else None


def _previous_kernel_family(events: Sequence[Mapping], index: int) -> str | None:
    for event in reversed(events[:index]):
        if event.get("kind") == "barrier":
            continue
        return event.get("family")
    return None


__all__ = ["add_next_invocation_qkv_prefetch", "add_split_k_weight_tile_prefetch"]
