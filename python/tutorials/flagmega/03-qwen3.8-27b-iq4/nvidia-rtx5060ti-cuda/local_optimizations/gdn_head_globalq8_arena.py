# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Final arena reconciliation for the explicit head dual-Q8/r8 experiment.

Call this after both r8 configurators. This does not move planner spans or
replace Vector's ABI validation. Frames reuse the arena only across complete
synchronous calls; the selected templates drain async copies and publish/reuse
at all-CTA barriers before returning.
"""
from triton.flagmega.errors import CodegenError
from .gdn_head_globalq8_96_abi import IMPLEMENTATION, program_contexts, validate_selection
from .gdn_head_globalq8_qkregs_96_abi import IMPLEMENTATION as QKREGS_IMPLEMENTATION, validate_selection as validate_qkregs_selection
from .gdn_head_globalq8_qkregs_fullhead48_abi import (IMPLEMENTATION as FULLHEAD48_IMPLEMENTATION,
    validate_selection as validate_fullhead48_selection, program_contexts as fullhead48_program_contexts)


def _require(value, reason):
    if not value:
        raise CodegenError("Head/r8 arena: " + reason)


def configure_descriptor(descriptor):
    selected_fullhead48 = descriptor.get("local_iq4_recipe", {}).get("gdn_head_projection") == "gdn_head_globalq8_qkregs_fullhead48"
    selected_qkregs = descriptor.get("local_iq4_recipe", {}).get("gdn_head_projection") == "dual_q8_global_qkregs"
    selected_implementation = QKREGS_IMPLEMENTATION if selected_qkregs else IMPLEMENTATION
    selected_validation = validate_qkregs_selection if selected_qkregs else validate_selection
    if selected_fullhead48:
        selected_implementation = FULLHEAD48_IMPLEMENTATION
        selected_validation = validate_fullhead48_selection
    contexts = (fullhead48_program_contexts if selected_fullhead48 else program_contexts)(descriptor)
    if contexts is not None:
        # Shared is allocated/reused within each launched entry only.  U4 Q8
        # and matrix partials in other entries never join the GDN head arena.
        arenas = []
        for context in contexts:
            if any(c.get("family") == "gdn_recurrent" for c in context.get("render_calls", ())):
                configure_descriptor(context)
                arenas.append({"entry_id": context["entry_id"],
                               **context["local_iq4_recipe"]["combined_arena"]})
        _require(bool(arenas), "separate programs lost all GDN entries")
        descriptor["local_iq4_recipe"]["program_arenas"] = arenas
        descriptor["local_iq4_recipe"]["gdn_head_projection_sites"] = selected_validation(descriptor)
        return
    recipe = descriptor["local_iq4_recipe"]
    _require(recipe.get("async_weights") == "uniform128_r8"
             and recipe.get("async_interleaved") is True,
             "requires both r8 configurators")
    root = descriptor["function_schedule"]["function"]
    schedules = {root: descriptor["entry_events"]}
    for function in descriptor["device_functions"]:
        name, schedule = function["function"], function["schedule"]
        _require(name not in schedules, "duplicate function identity")
        _require(schedule.get("schema") == "flagmega.triton-pipeline-source-schedule/v3"
                 and not any(schedule.get(k) for k in (
                     "stages", "handoffs", "producer_events", "auxiliary_events",
                     "trip_scoped_drains", "trip_scoped_closes")),
                 "overlapping role schedules cannot share this arena")
        schedules[name] = schedule["consumer_events"]
    reached, events, invocation_counts = set(), {}, {}

    def walk(name, stack):
        _require(name in schedules and name not in stack, "unknown or recursive function")
        reached.add(name)
        for ordinal, event in enumerate(schedules[name]):
            _require(event.get("kind") in ("barrier", "function_call", "tir.kernel_call")
                     and not event.get("pipeline_channels")
                     and event.get("participation_active", "True") == "True",
                     "requires sequential, fully participating events")
            if event["kind"] == "function_call":
                walk(event["callee"], (*stack, name))
            elif event["kind"] == "tir.kernel_call":
                symbol = event["symbol"]
                owner = (name, ordinal)
                _require(symbol not in events or events[symbol] == owner,
                         "one leaf symbol has multiple scheduled owners")
                events[symbol] = owner
                invocation_counts[symbol] = invocation_counts.get(symbol, 0) + 1

    walk(root, ())
    _require(reached == set(schedules), "unreachable device function")
    calls = descriptor["render_calls"]
    _require(len({c["symbol"] for c in calls}) == len(calls)
             and {c["symbol"] for c in calls} == set(events),
             "render calls must exactly cover reachable scheduled leaves")
    frames, seen_ids = [], set()
    arena = descriptor["local_iq4_shared_bytes"]
    for call in calls:
        head = call.get("implementation") == selected_implementation
        if not head and not call.get("local_iq4_async"):
            _require(not call.get("shared_workspaces"),
                     "unrecognized Shared workspace user")
            continue
        if not head:
            _require(call.get("local_iq4_async_r8") is True,
                     "matrix call was not configured by r8")
        spans = call.get("shared_workspaces", ())
        _require(len(spans) == (2 if selected_fullhead48 or not head else 1), "unexpected planned Shared spans")
        caller, ordinal = events[call["symbol"]]
        intervals = []
        for index, span in enumerate(spans):
            start, size = span.get("offset_bytes"), span.get("nbytes")
            expected = f"execution:{caller}:{call['call']}:shared:{call['callee']}:{index}:{span['name']}"
            _require(type(start) is int and type(size) is int and start >= 0
                     and size > 0 and start % 16 == 0
                     and span.get("alignment_bytes") == 16,
                     "invalid planned span")
            _require(span.get("physical_buffer") == expected and expected not in seen_ids,
                     "stale, duplicated or foreign Shared identity")
            seen_ids.add(expected)
            intervals.append((start, start + size))
            arena = max(arena, start + size)
        if len(intervals) == 2:
            a, b = intervals
            _require(a[1] <= b[0] or b[1] <= a[0], "live spans overlap within a call")
        frames.append({"symbol": call["symbol"], "call": call["call"],
                       "caller": caller, "callee": call["callee"], "event_ordinal": ordinal,
                       "invocations_per_entry": invocation_counts[call["symbol"]],
                       "implementation": call["implementation"], "shared_workspaces": spans,
                       "lifetime": "within this synchronous leaf invocation",
                       "release": ("all-CTA statistics/core/RMS publication; final output reads before caller publication/reuse"
                                   if head else "async waits and all-CTA reuse barrier before return")})
    _require(arena <= min(49152, descriptor["launch_resource_limits"]["max_shared_memory_bytes"]),
             "combined arena exceeds static/target limit")
    descriptor["local_iq4_shared_bytes"] = arena
    recipe["scratch"].update(nbytes=arena,
        aliases="one arena reused by sequential r8 matrix and dual-Q8 head call frames",
        planned_spans="combined_arena.frames[].shared_workspaces; original physical IDs and offsets retained")
    recipe["combined_arena"] = {"name": "_local_iq4_q8", "nbytes": arena,
        "alignment_bytes": 16, "frames": frames, "reachable_functions": sorted(reached),
        "proof": "static acyclic synchronous call graph; nonoverlapping live spans within each frame",
        "native_resource_admission": "pending full static Shared and cooperative occupancy verification"}
    recipe["gdn_head_projection_sites"] = selected_validation(descriptor)
