# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical Triton schedule derived from bufferized function-call ABIs.

Logical functions remain the unit of IR and device-code reuse. Synchronization
belongs to the function schedule or the kernel body that owns it. The TLE
device-call ABI passes scratch frame bases (not CTA-relative pointers), so a
grid barrier no longer requires expanding the callee into the entry.
"""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.codegen.triton.call_abi import describe_function_call_abi
from triton.flagmega.codegen.triton.distributed_abi import KernelExecutionKind
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Barrier, IRModule, execution_calls_of, execution_items_of


FUNCTION_SCHEDULE_SCHEMA = "flagmega.triton-function-schedule/v2"


def describe_function_schedule(
    module: IRModule,
    *,
    function_name: str | None = None,
    call_abi_cache: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Plan one logical function without model- or distribution-specific roles.

    Every ordinary kernel is scheduled over the dense local buffer ABI on all
    placement owners.  Only an explicit execution kind introduces a physical
    synchronization boundary. Regions describe ordering within each function,
    not synthetic outlined segments or permission to expand the call graph.
    """

    root = module.entry if function_name is None else str(function_name)
    cache: dict[str, dict[str, object]] = {}
    abis = {} if call_abi_cache is None else call_abi_cache

    def call_abi(name: str) -> dict[str, object]:
        result = abis.get(name)
        if result is None:
            result = describe_function_call_abi(module, function_name=name)
            abis[name] = result
        return result

    def plan(name: str, active: tuple[str, ...]) -> dict[str, object]:
        if name in cache:
            return cache[name]
        if name in active:
            cycle = " -> ".join((*active, name))
            raise CodegenError(f"Recursive Triton function schedule: {cycle}.")

        abi = call_abi(name)
        kernel_by_id = {
            str(call["call"]): call for call in abi["kernel_calls"]
        }
        memory_barriers = memory_barriers_for_function(module, name)
        regions: list[dict[str, object]] = []
        pending_local: list[str] = []
        pending_local_barrier = False

        def flush_local() -> None:
            nonlocal pending_local_barrier
            if not pending_local:
                return
            ordinal = len(regions)
            regions.append({
                "kind": "local_segment",
                "name": f"{name}_local_{ordinal}",
                "calls": tuple(pending_local),
                "execution_domain": "dense_local_shard",
                "owner_participation": "all",
                "lowering": "function_body",
                "barrier_before": pending_local_barrier,
                "memory_dependencies": tuple(
                    memory_barriers.get(pending_local[0], ())
                ) if pending_local_barrier else (),
            })
            pending_local.clear()
            pending_local_barrier = False

        requires_synchronization = False
        for event in _execution_events(module, name, abi):
            if event["kind"] == "kernel_call":
                call_id = str(event["call"])
                call = kernel_by_id[call_id]
                kind = KernelExecutionKind(str(event["execution_kind"]))
                needs_memory_barrier = call_id in memory_barriers
                if kind is KernelExecutionKind.LOCAL_SHARD:
                    if needs_memory_barrier:
                        flush_local()
                        pending_local_barrier = True
                        requires_synchronization = True
                    pending_local.append(call_id)
                    continue

                flush_local()
                requires_synchronization = True
                regions.append({
                    "kind": (
                        "collective_kernel"
                        if kind is KernelExecutionKind.COLLECTIVE
                        else "synchronized_local_kernel"
                    ),
                    "call": call_id,
                    "semantic_op": str(call["semantic_op"]),
                    "execution_domain": "dense_local_shard",
                    "owner_participation": "all",
                    "lowering": "function_body",
                    "barrier_before": (
                        kind is KernelExecutionKind.COLLECTIVE
                        or needs_memory_barrier
                    ),
                    "barrier_owner": (
                        "function_schedule"
                        if (
                            kind is KernelExecutionKind.COLLECTIVE
                            or needs_memory_barrier
                        )
                        else "inline_kernel_body"
                    ),
                    "memory_dependencies": tuple(
                        memory_barriers.get(call_id, ())
                    ),
                })
                continue

            if event["kind"] != "function_call":
                raise CodegenError(
                    f"Function @{name} has unknown executable event "
                    f"{event['kind']!r}."
                )
            flush_local()
            call_id = str(event["call"])
            needs_memory_barrier = call_id in memory_barriers
            callee = str(event["callee"])
            callee_plan = plan(callee, (*active, name))
            callee_requires_sync = bool(
                callee_plan["requires_synchronization"]
            )
            requires_synchronization |= callee_requires_sync
            requires_synchronization |= needs_memory_barrier
            regions.append({
                "kind": "function_schedule_call",
                "call": call_id,
                "callee": callee,
                "lowering": "direct_device_call",
                "arguments": event["arguments"],
                "results": event["results"],
                "memory_pools": event["memory_pools"],
                "barrier_before": needs_memory_barrier,
                "memory_dependencies": tuple(
                    memory_barriers.get(call_id, ())
                ),
            })
        flush_local()

        requested_reuse = bool(abi["reusable"] or abi["noinline"])
        result = {
            "schema": FUNCTION_SCHEDULE_SCHEMA,
            "function": name,
            "calling_convention": abi["calling_convention"],
            "requested_noinline": abi["noinline"],
            "requested_reuse": requested_reuse,
            "requires_synchronization": requires_synchronization,
            "direct_noinline_safe": True,
            "physical_strategy": (
                "direct_noinline"
                if requested_reuse
                else "entry_schedule"
            ),
            "regions": regions,
        }
        cache[name] = result
        return result

    schedule = plan(root, ())
    reachable = {
        name: cache[name]
        for name in cache
        if name != root
    }
    return {
        **schedule,
        "reachable_functions": reachable,
    }


def _execution_events(module, function_name, abi):
    """Join typed execution order with the detailed physical call ABI."""

    function = module.execution_function_map.get(function_name)
    if function is None:
        # Compatibility for standalone pre-schedule fixtures and v1 dumps.
        return tuple(abi["events"])
    by_id = {str(value["call"]): value for value in abi["events"]}
    calls = execution_calls_of(function)
    if {call.call_id for call in calls} != set(by_id):
        raise CodegenError(
            f"ExecutionFunction @{function_name} call closure differs from its "
            "bufferized ABI."
        )
    return tuple(by_id[call.call_id] for call in calls)


def memory_barriers_for_function(
    module: IRModule,
    function_name: str,
) -> dict[str, tuple[dict[str, object], ...]]:
    execution = module.execution_function_map.get(function_name)
    if execution is not None:
        result: dict[str, list[dict[str, object]]] = {}
        for item in execution_items_of(execution):
            if not isinstance(item, Barrier):
                continue
            result.setdefault(item.before, []).append({
                "function": function_name,
                "after": item.after[-1],
                "before": item.before,
                "scope": (
                    "grid" if item.scope.value == "chip" else "block"
                ),
                "hazards": tuple(item.hazards),
                "axis_group_axes": tuple(item.axis_group_axes),
                "ranges": tuple({
                    "storage": value.storage,
                    "physical_id": value.physical_id,
                    "offset": value.offset,
                    "nbytes": value.nbytes,
                    "access": value.mode,
                } for value in item.ranges),
            })
        return {name: tuple(values) for name, values in result.items()}
    synchronization = module.metadata.get("memory_synchronization")
    if not isinstance(synchronization, Mapping):
        return {}
    raw_events = synchronization.get("events", ())
    if not isinstance(raw_events, (tuple, list)):
        raise CodegenError(
            "memory_synchronization.events must be a sequence."
        )
    grouped: dict[str, list[dict[str, object]]] = {}
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise CodegenError(
                "memory_synchronization event must be a mapping."
            )
        if str(raw.get("function")) != function_name:
            continue
        before = str(raw.get("before", ""))
        if not before:
            raise CodegenError(
                f"Memory synchronization in @{function_name} has no before call."
            )
        grouped.setdefault(before, []).append(dict(raw))
    return {name: tuple(events) for name, events in grouped.items()}


__all__ = [
    "FUNCTION_SCHEDULE_SCHEMA",
    "describe_function_schedule",
    "memory_barriers_for_function",
]
