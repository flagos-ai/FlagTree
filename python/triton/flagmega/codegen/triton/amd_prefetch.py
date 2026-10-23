# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1201: warm the next layer's QKV weights while the current one finishes.

A reused decode-layer function reads its weights through a per-call offset
record. After its last GEMV only latency-bound phases remain, so HBM is idle;
issuing `s_prefetch_data` there for the next invocation's QKV slab lets that
GEMV start from cache. The prefetch has no architectural side effect, so the
generated values are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

from triton.flagmega.codegen.triton.function_pool_abi import (
    CALL_RECORD,
    OFFSET_TABLE,
    pool_parameter,
)

PREFETCH_CHUNK_BYTES = 4096  # one `s_prefetch_data ..., 31` request (32 x 128 B)
MAX_PREFETCH_BYTES = 128 * 1024


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


__all__ = ["add_next_invocation_qkv_prefetch"]
