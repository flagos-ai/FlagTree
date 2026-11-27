# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Project root call bindings independently of reusable GPU entry definitions."""

from copy import deepcopy

from triton.flagmega.codegen.triton.function_pool_abi import CALL_RECORD, OFFSET_TABLE, ROW_INDEX, pool_parameter
from triton.flagmega.codegen.triton.physical_access import emit_triton_scalar_type
from triton.flagmega.codegen.triton.runtime_binding import (
    DEVICE_CONTEXT_ARGUMENTS, DEVICE_CONTEXT_HANDLES, DEVICE_CONTEXT_MODE, describe_function_runtime_binding,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import verify_buffer_plan


KERNEL_PROGRAM_FUNCTION_SCHEMA = "flagmega.kernel-programs/v4-functions"


def pipeline_entry_identity(pipeline):
    """Include every emitted resource/control property in entry reuse."""
    if pipeline is None:
        return None

    def freeze(value):
        if isinstance(value, dict):
            # These describe the invocation's physical binding or origin;
            # code uses the projected arguments, spans and canonical names.
            labels = {"call", "buffer_name", "physical_buffer", "descriptor_argument_bindings"}
            return tuple((key, freeze(item)) for key, item in sorted(value.items()) if key not in labels)
        if isinstance(value, (tuple, list)):
            return tuple(freeze(item) for item in value)
        return value

    keys = ("stages", "handoffs", "producer_events", "consumer_endpoints", "producer_endpoints",
            "auxiliary_endpoints", "auxiliary_events", "shared_arena_required_nbytes",
            "shared_arena_nbytes", "shared_arena_alignment_bytes", "consumer_warps",
            "producer_warps", "producer_registers")
    return freeze({key: pipeline.get(key) for key in keys})


def _canonical_pipeline(program, definition, call, names):
    """Rename a complete invocation's resources without changing its spans."""
    actual = program["pipeline_schedule"]
    formal = definition["schedule"]
    producer_calls = [e for e in actual["producer_events"] if e["kind"] == "function_call"]
    if len(producer_calls) != 1 or producer_calls[0]["call"] != call["call"]:
        raise CodegenError("Reusable pipeline entries require one complete producer/consumer invocation.")
    producer = producer_calls[0]
    bindings = {}
    for role, event in (("consumer", call), ("producer", producer)):
        bound = event.get("endpoint_bindings", {})
        if set(bound) != set(formal[role + "_endpoints"]) or set(bound.values()) != set(actual[role + "_endpoints"]):
            raise CodegenError("Reusable pipeline entry endpoints differ from the callee's complete ABI.")
        bindings.update(bound)
    if len(set(bindings.values())) != len(bindings):
        raise CodegenError("Reusable pipeline entry endpoint bindings must be one-to-one.")
    rename = {value: key for key, value in bindings.items()}
    if len(actual["stages"]) != len(formal["stages"]) or len(actual["handoffs"]) != len(formal["handoffs"]):
        raise CodegenError("Reusable pipeline entries require all callee resources.")
    matched = set()
    for stage in formal["stages"]:
        first = bindings[stage["channels"][0]["reader_name"]]
        candidates = [s for s in actual["stages"] if any(c["reader_name"] == first for c in s["channels"])]
        if len(candidates) != 1:
            raise CodegenError("Reusable pipeline entry has no unique bound stage.")
        instance = candidates[0]
        matched.add(instance["id"])
        rename[instance["id"]] = stage["id"]
        for key in ("producer_warps", "producer_registers", "auxiliary_consumer"):
            if instance.get(key) != stage.get(key):
                raise CodegenError("Reusable pipeline entry changes its selected role contract.")
        if len(instance["workspaces"]) != len(stage["workspaces"]) or len(instance["channels"]) != len(stage["channels"]):
            raise CodegenError("Reusable pipeline entry changes its Shared workspace ABI.")
        for workspace, bound in zip(stage["workspaces"], instance["workspaces"], strict=True):
            for key in ("shape", "dtype", "element_type", "strides", "nbytes", "matrix_compatible", "pipe_owned", "consumer_owned"):
                if bound.get(key) != workspace.get(key):
                    raise CodegenError("Reusable pipeline entry changes its Shared payload ABI.")
            rename[bound["variable"]] = workspace["variable"]
        for channel, bound in zip(stage["channels"], instance["channels"], strict=True):
            if (channel["name"] != bound["name"] or channel["capacity"] != bound["capacity"]
                    or [f["name"] for f in channel["fields"]] != [f["name"] for f in bound["fields"]]
                    or any(bindings[channel[key]] != bound[key] for key in ("reader_name", "writer_name"))):
                raise CodegenError("Reusable pipeline entry changes its channel ABI.")
            rename[bound["pipe_name"]] = channel["pipe_name"]
    if len(matched) != len(actual["stages"]):
        raise CodegenError("Reusable pipeline entry aliases distinct callee stages.")
    for handoff in formal["handoffs"]:
        candidates = [h for h in actual["handoffs"] if h["reader_name"] == bindings[handoff["reader_name"]]]
        if len(candidates) != 1 or candidates[0]["writer_name"] != bindings[handoff["writer_name"]]:
            raise CodegenError("Reusable pipeline entry changes its handoff ownership.")
        bound = candidates[0]
        if any(bound.get(key) != handoff.get(key) for key in ("semantic", "repeated", "wrap_around")):
            raise CodegenError("Reusable pipeline entry changes its handoff lifetime.")
        rename[bound["id"]] = handoff["id"]
        rename[bound["pipe_name"]] = handoff["pipe_name"]

    def replace_names(value):
        if isinstance(value, str):
            return rename.get(value, value)
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = replace_names(item)
        elif isinstance(value, list):
            value[:] = [replace_names(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(replace_names(item) for item in value)
        return value

    result = replace_names(deepcopy(actual))
    for role in ("consumer", "producer"):
        rewritten = []
        for event in result[role + "_events"]:
            if event["kind"] == "function_call":
                arguments = (*formal[role + "_endpoints"], *names)
                event = {**event, "arguments": ", ".join(arguments), "argument_expressions": arguments,
                         "endpoint_bindings": {name: name for name in formal[role + "_endpoints"]}}
            rewritten.append(event)
        result[role + "_events"] = rewritten
    return result


def _constant(module, buffer_id):
    plan = verify_buffer_plan(module)
    seen = set()
    while buffer_id not in seen:
        seen.add(buffer_id)
        buffer = plan.buffer_map[buffer_id]
        node = module.node_map.get(str(buffer.source_node or buffer_id))
        if node is not None and node.op in {"builtin.scalar_const", "tir.scalar_const"}:
            return node.attrs["value"]
        if buffer.alias_of is None:
            break
        buffer_id = buffer.alias_of
    raise CodegenError("A shared GPU entry requires a typed constant or root scalar argument.")


def _pointer(source, dtype, offset=0, stride=0):
    return dict(kind="pointer", source=source, dtype=dtype, byte_offset=offset, scope_stride_bytes=stride)


def _actual_bindings(module, root, definition, call, signature, callee, physical):
    """Use typed edges, pool frames and resource bindings, never emitted text."""
    edges = {e["formal"]: e for e in (*physical["arguments"], *physical["results"])}
    actuals = {}
    for argument in callee["arguments"]:
        if argument["name"] not in signature:
            continue
        edge = edges[argument["buffer"]]
        kind, abi = edge["actual_runtime_value_kind"], edge["actual_abi"]
        if kind == "immediate":
            value = dict(kind="immediate", dtype=edge["formal_abi"]["scalar_dtype"],
                         value=_constant(module, str(edge["actual"])))
        elif kind == "scalar":
            value = dict(kind="scalar_argument", source=edge["actual_runtime_argument"],
                         dtype=edge["formal_abi"]["scalar_dtype"], add=0)
        else:
            if "view_byte_offset" in abi:
                raise CodegenError("Shared GPU entries require a root-addressable view offset.")
            value = _pointer(edge["actual_runtime_argument"], abi["scalar_dtype"],
                             int(abi.get("pool_byte_offset", 0)), int(abi.get("pool_scope_stride_bytes", 0)))
        actuals[argument["name"]] = value
    pools = {p["storage"]: p for p in root["pools"]}
    frames = {f["memory_space"]: f for f in physical["memory_pools"]}
    for pool in callee["pools"]:
        if pool["name"] not in signature:
            continue
        storage = pool["storage"]
        parent = pools[storage]
        frame = frames.get(storage)
        actuals[pool["name"]] = _pointer(
            parent["name"], "uint8", int(frame["offset"]) if frame else 0,
            int(parent.get("scope_nbytes", 0)) if frame and int(parent.get("scope_count", 1)) > 1 else 0)
    actuals.update(call["descriptor_argument_bindings"])
    plan = definition["function_pool_abi"]
    if OFFSET_TABLE in signature:
        actuals[OFFSET_TABLE] = dict(kind="argument", source=OFFSET_TABLE)
    if plan is not None:
        for pool in plan["pools"]:
            actuals[pool_parameter(pool["storage"])] = dict(kind="argument", source=pools[pool["storage"]]["name"])
        if plan["record_width"]:
            row = next(r for r in plan["calls"] if r["caller"] == module.entry and r["call"] == call["call"])
            actuals[CALL_RECORD] = dict(kind="immediate", dtype="int64", value=row["record_offset"])
    if ROW_INDEX in signature:
        raise CodegenError("A shared root GPU entry cannot invent a parameter-bank loop induction value.")
    return actuals


def project_function_programs(module, descriptor, programs, launches):
    """Keep invocation IDs/DAGs; canonicalize code only by its complete ABI."""
    if not any(sum(e["kind"] not in {"barrier", "drain"} for e in p["entry_events"]) == 1
               and any(e["kind"] == "function_call" and (not e.get("endpoint_bindings") or p.get("pipeline_schedule"))
                       for e in p["entry_events"]) for p in programs):
        return False
    root = descriptor["runtime_binding"]
    signature = descriptor["signature_arguments"]
    indices = {name: i for i, name in enumerate(signature)}
    dynamic = set(descriptor["dynamic_argument_indices"])
    entries = {}
    callees = {}
    physical_calls = {e["call"]: e for e in root["call_abi"]["events"] if e["kind"] == "function_call"}
    for program, launch in zip(programs, launches, strict=True):
        events = program["entry_events"]
        body = [e for e in events if e["kind"] not in {"barrier", "drain"}]
        function = body[0]["callee"] if (len(body) == 1 and body[0]["kind"] == "function_call"
                                         and (not body[0].get("endpoint_bindings") or program.get("pipeline_schedule"))) else None
        launch["entry_function"] = function
        if function is None:
            launch["entry_symbol"] = program["symbol"]
            launch["argument_bindings"] = [dict(kind="argument", source=i) for i in range(len(signature))]
            continue
        definition = next(d for d in program["device_functions"] if d["function"] == function)
        names = definition["signature_arguments"]
        pipeline = _canonical_pipeline(program, definition, body[0], names) if program.get("pipeline_schedule") else None
        if pipeline is not None:
            events = pipeline["consumer_events"]
            body = [e for e in events if e["kind"] not in {"barrier", "drain"}]
        if function not in callees:
            callees[function] = describe_function_runtime_binding(module, function_name=function)
        actuals = _actual_bindings(module, root, definition, body[0], names,
                                   callees[function], physical_calls[body[0]["call"]])
        bindings, parameters, typed, scalars, scoped = [], [], [], [], []
        for name in names:
            if name == "device_dptr":
                context = DEVICE_CONTEXT_ARGUMENTS if descriptor["device_context_abi"] == "runtime-handles/v2" else DEVICE_CONTEXT_HANDLES
                for field in context:
                    bindings.append(dict(kind="argument", source=indices[field]))
                    parameters.append(field)
                    typed.append(f"{field}: tl.constexpr" if field == DEVICE_CONTEXT_MODE else f"{field}: tl.int64")
                    if field != DEVICE_CONTEXT_MODE:
                        scalars.append(field)
                continue
            value = dict(actuals[name])
            if value["kind"] != "immediate":
                value["source"] = indices[value["source"]]
            bindings.append(value)
            parameters.append(name)
            scalar = value["kind"] in {"immediate", "scalar_argument"}
            typed.append(f"{name}: {emit_triton_scalar_type(value['dtype'])}" if scalar else name)
            if scalar:
                scalars.append(name)
            elif value["kind"] == "pointer":
                stride = value["scope_stride_bytes"]
                if stride:
                    scoped.append(dict(name=name, stride=stride, dtype=emit_triton_scalar_type(value["dtype"])))
        # Fences can differ at the first use or after a peer completion.
        # Their executable scope/axis is part of entry identity, not call IDs.
        fences = tuple(("call",) if e is body[0] else ("drain", tuple(e["endpoints"])) if e["kind"] == "drain" else
                       ("barrier", e["scope"], e.get("barrier_axis_group_key")) for e in events)
        key = (function, launch["num_warps"], tuple(launch["grid"]), fences,
               tuple((v["name"], v["stride"], v["dtype"]) for v in scoped),
               tuple(typed), body[0].get("participation_active", "True"), pipeline_entry_identity(pipeline))
        canonical = entries.setdefault(key, program["symbol"])
        launch["entry_symbol"] = canonical
        launch["argument_bindings"] = bindings
        launch["dynamic_argument_indices"] = [i for i, v in enumerate(bindings)
                                               if v["kind"] != "immediate" and v["source"] in dynamic]
        if canonical != program["symbol"]:
            program["entry_alias"] = canonical
            continue
        program.update(signature=",\n    ".join(typed), signature_arguments=parameters,
                       typed_signature_arguments=typed, scalar_arguments=scalars,
                       scoped_entry_arguments=scoped, entry_inline_function=function)
        # This entry invokes the complete root once. Keeping another device
        # call here forces all invocation parameters into a fixed call ABI
        # before their first use. Inline this wrapper so constants can be
        # materialized where used; nested device-call boundaries stay intact.
        if pipeline is not None:
            program["pipeline_schedule"] = pipeline
        # Runtime preparation retains and validates the compiled pointer
        # alignment on every dynamic rebinding. Keep those facts at reusable
        # entries as well: losing them forces scalar state loads in a callee
        # that otherwise satisfies its register residency contract.
        program.pop("unspecialized_pointer_arguments", None)
        # Pass the existing function signature through unchanged. Each call
        # supplies its own values; no per-layer address becomes constexpr.
        arguments = (*pipeline["consumer_endpoints"], *names) if pipeline is not None else tuple(names)
        replacement = {**body[0], "arguments": ", ".join(arguments), "argument_expressions": tuple(arguments)}
        program["entry_events"] = [replacement if e is body[0] else e for e in events]
        if pipeline is not None:
            pipeline["consumer_events"] = program["entry_events"]
    return True
