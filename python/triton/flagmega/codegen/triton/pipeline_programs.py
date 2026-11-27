# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Give complete transfer stages their program's shared lifetime."""

from triton.flagmega.errors import CodegenError


def _project_function_handoffs(schedule, calls):
    """Keep a one-shot callee handoff with its complete two-role invocation."""
    result = []
    for handoff in schedule["handoffs"]:
        # Loop-carried and root-level edges need explicit program-boundary
        # semantics. This projection handles a complete reusable call only.
        if (not handoff["semantic"] or handoff.get("repeated")
                or handoff.get("wrap_around")):
            raise CodegenError("Pipeline program cuts cannot cross a reusable handoff.")

        def owners(role, endpoint):
            return {e["call"] for e in schedule[role + "_events"]
                    if e["kind"] == "function_call"
                    and endpoint in e.get("endpoint_bindings", {}).values()}

        consumer = owners("consumer", handoff["writer_name"])
        producer = owners("producer", handoff["reader_name"])
        if len(consumer) != 1 or consumer != producer:
            raise CodegenError("Pipeline program cuts cannot split a handoff across invocations.")
        if consumer <= calls:
            result.append(handoff)
    return result


def project_pipeline_schedule(module, schedule, segment, *, complete_entry=False):
    """Preserve complete lifetimes or project isolated transfer stages.

    Stage slots and endpoints remain exactly those of the physical schedule.
    Kernel retirement replaces only the incoming CTA/grid fence. A handoff or
    reusable stream spanning programs requires a separate inter-program ABI.
    """
    if schedule is None:
        return None
    if complete_entry:
        # Exact call coverage was checked by describe_kernel_programs. There
        # is no cut through this entry, so preserve its complete physical
        # lifetime, including root/loop handoffs, drains and auxiliary roles.
        # Only the incoming local fence may be replaced by kernel entry.
        from triton.flagmega.codegen.triton.kernel_programs import _incoming_completion
        if segment != _incoming_completion(schedule["consumer_events"]):
            raise CodegenError("Pipeline program must preserve its complete entry synchronization lifetime.")
        return {**schedule, "consumer_events": segment}
    if schedule["trip_scoped_closes"] or schedule["trip_scoped_drains"]:
        raise CodegenError("Pipeline program cuts cannot cross a handoff or trip-scoped stream.")
    if any(e["kind"] not in {"barrier", "drain", "tir.kernel_call", "pipeline_kernel_call", "function_call"}
           for e in segment):
        raise CodegenError("Pipeline programs require complete straight-line stages.")
    calls = {e["call"] for e in segment if e["kind"] not in {"barrier", "drain"}}
    handoffs = _project_function_handoffs(schedule, calls)
    # A reusable function keeps its leaf call IDs. Its instantiated endpoint
    # bindings identify the caller-owned stages without cloning its body.
    bound = {name for e in segment for name in e.get("endpoint_bindings", {}).values()}
    stages = [s for s in schedule["stages"] if s["call"] in calls or any(
        c["reader_name"] in bound for c in s["channels"])]
    if not stages:
        if any(e["kind"] == "drain" for e in segment):
            raise CodegenError("A pipeline drain must retire its owning program's resources.")
        return None
    channels = [c for s in stages for c in s["channels"]]
    consumers = [c["reader_name"] for c in channels]
    consumers.extend(w["variable"] for s in stages for w in s["consumer_workspaces"])
    consumers.extend(h["writer_name"] for h in handoffs)
    producers = [c["writer_name"] for c in channels]
    producers.extend(h["reader_name"] for h in handoffs)
    barriers = [e for e in segment if e["kind"] == "barrier"]

    def role_events(role):
        result = []
        for event in schedule[role + "_events"]:
            if event["kind"] == "barrier":
                if event in barriers:
                    result.append(event)
            elif event.get("call") in calls:
                result.append(event)
            elif event["kind"] == "drain":
                owned = set(producers if role == "producer" else consumers)
                endpoints = set(event["endpoints"])
                if endpoints & owned:
                    if not endpoints <= owned:
                        raise CodegenError("Pipeline program cuts cannot split a shared arena drain.")
                    result.append(event)
        return result

    if any(s["auxiliary_consumer"] is not None for s in stages):
        raise CodegenError("Pipeline program cuts require an explicit auxiliary-role projection.")
    required = max(w["offset_bytes"] + w["nbytes"] for s in stages for w in s["workspaces"])
    alignment = max(max(w["alignment_bytes"], w["allocation_alignment_bytes"])
                    for s in stages for w in s["workspaces"])
    from triton.flagmega.ir import verify_buffer_plan
    shared = next(s for s in verify_buffer_plan(module).memory_spaces if s.name == "shared")
    allocated = shared.allocation_bytes(required)
    return {**schedule, "stages": stages, "handoffs": handoffs,
            "consumer_events": segment, "producer_events": role_events("producer"),
            "auxiliary_events": [], "consumer_endpoints": consumers,
            "producer_endpoints": producers, "auxiliary_endpoints": [],
            "shared_arena_required_nbytes": required, "shared_arena_nbytes": allocated,
            "shared_arena_alignment_bytes": alignment}
