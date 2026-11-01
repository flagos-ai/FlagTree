"""Elect disjoint CTA replicas for independent pure projection calls."""
from contextlib import contextmanager
from copy import deepcopy
from . import activate_base as activate_tutorial
activate_tutorial()
from .replica_work import replica_group
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError


@contextmanager
def activate(roles):
    original = TritonTemplateRegistry.render
    receipts = []

    def rendered(self, name, context):
        if name != "module.py.jinja":
            return original(self, name, context)
        # Outlined programs reuse function schedules. Elect roles in a local
        # copy so rendering one program cannot change another program's input.
        context = {**context, "entry_events": deepcopy(context.get("entry_events", ())),
                   "device_functions": deepcopy(context.get("device_functions", ()))}
        if context.get("pipeline_schedule"):
            context["pipeline_schedule"] = deepcopy(context["pipeline_schedule"])
        guards = {}
        selected = []
        calls = {c["call"]: c for c in context.get("render_calls", ())}
        def physical(call, role):
            return {b["abi"]["physical_buffer"] for a in call[role] for b in a["buffers"]}
        for call in context.get("render_calls", ()):
            if call["call"] not in roles:
                continue
            if (call["family"] != "dense_matmul" or call["variant"] != "packed_k_major_mma_prefetch"
                    or call.get("pipeline_channels") or call.get("shared_workspaces")):
                raise CodegenError("Replica projection roles require pure CTA-local MMA prefetch calls.")
            group = replica_group(call, context)
            role = tuple(roles[call["call"]])
            if group is None or set(group["axes"]) != {axis for axis, _ in role}:
                raise CodegenError(f"No complete physical replica proof for {call['call']}: {group}")
            if any(not 0 <= coordinate < context["mesh_hierarchy"][axis] for axis, coordinate in role):
                raise CodegenError("Replica role coordinate exceeds the physical mesh.")
            guards[call["call"]] = " & ".join(f"(shard_coord{axis} == {coordinate})" for axis, coordinate in role)
            receipts.append(dict(call=call["call"], group=group, role=role))
            selected.append(call)
        if guards:
            if len({c["source"] for c in selected}) != 1:
                raise CodegenError("Replica projection group requires one shared, already published activation.")
            reads = set.union(*(physical(c, "inputs") for c in selected))
            writes = [physical(c, "outputs") for c in selected]
            if any(w & reads for w in writes) or any(a & b for i,a in enumerate(writes) for b in writes[i+1:]):
                raise CodegenError("Replica projection results must be disjoint and independent of every group input.")
        schedules = [context.get("entry_events", ())]
        if context.get("pipeline_schedule"):
            schedules.append(context["pipeline_schedule"]["consumer_events"])
        schedules.extend(f["schedule"]["consumer_events"] for f in context.get("device_functions", ()))
        changed = 0
        for events in schedules:
            positions = [i for i,e in enumerate(events) if e.get("kind")=="tir.kernel_call" and e.get("call") in guards]
            if positions:
                if len(positions)!=len(guards) or positions != list(range(positions[0],positions[-1]+1)):
                    raise CodegenError("Replica projection group must be consecutive independent call events.")
                # Changing which replica computes a value must not create a
                # new cross-CTA dependency before or after the group.
                for direction, segment in (("before", reversed(events[:positions[0]])),
                                           ("after", events[positions[-1]+1:])):
                    for event in segment:
                        if (event.get("kind")=="barrier" and event.get("scope") in {"grid","device"}
                                and not event.get("axis_group_axes") and not event.get("barrier_axis_group_key")):
                            break
                        if event.get("kind")=="function_call":
                            raise CodegenError("Replica group cannot cross an unproved function effect.")
                        if event.get("kind")!="tir.kernel_call":
                            continue
                        other = calls[event["call"]]
                        conflict = (physical(other,"outputs") & reads if direction=="before" else
                                    physical(other,"inputs") & set.union(*writes))
                        if conflict:
                            raise CodegenError("Replica group needs an existing publication barrier for its producer/consumer.")
            for event in events:
                if event.get("kind") == "tir.kernel_call" and event.get("call") in guards:
                    assert event.get("participation_active", "True") == "True"
                    event["participation_active"] = guards[event["call"]]
                    changed += 1
        if guards and changed != len(guards):
            raise CodegenError("Requested replica calls must each have exactly one ordinary call event.")
        return original(self, name, context)

    TritonTemplateRegistry.render = rendered
    try:
        yield receipts
    finally:
        TritonTemplateRegistry.render = original
