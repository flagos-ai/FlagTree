# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit execution programs over a verified physical TIR schedule.

The target owns the requested cut and warp geometry. This projection preserves
all calls and their order, retains device rendezvous, and replaces only an
incoming chip/CTA boundary with the preceding kernel's stream completion.
Explicit concurrency requires the corresponding lifetime/dependency proof.
"""

import keyword
from collections.abc import Mapping

from triton.flagmega.codegen.triton.package_plan import plain_package_value
from triton.flagmega.errors import CodegenError, IRVerificationError

KERNEL_PROGRAM_SCHEMA = "flagmega.kernel-programs/v1"
KERNEL_PROGRAM_DAG_SCHEMA = "flagmega.kernel-programs/v2-dag"
KERNEL_PROGRAM_DAG_COMPLETION_SCHEMA = "flagmega.kernel-programs/v3-dag-completion"
KERNEL_PROGRAM_MESH_SCHEMA = "flagmega.kernel-programs/v5-mesh"
KERNEL_PROGRAM_MESH_DAG_SCHEMA = "flagmega.kernel-programs/v7-mesh-dag"


def _valid_symbol(symbol):
    return (
        isinstance(symbol, str)
        and symbol.isidentifier()
        and not keyword.iskeyword(symbol)
        and not symbol.startswith("_flagmega_build_")
        and symbol not in {"FLAGMEGA_KERNEL_PROGRAMS", "FLAGMEGA_FUNCTION_OFFSET_TABLES", "FLAGMEGA_PROGRAM_EXECUTION"}
    )


def target_kernel_programs(module, capability):
    """Normalize explicit target requests while enforcing machine limits."""
    raw = module.metadata.get("kernel_program_partition")
    if raw is None:
        return None
    if not isinstance(raw, (tuple, list)) or not raw:
        raise IRVerificationError("Kernel program partition must be a non-empty sequence.", stage=module.stage)
    from triton.flagmega.ir.program_execution import verify_program_execution
    execution = verify_program_execution(module)
    programs = []
    symbols = set()
    for request in raw:
        if not isinstance(request, Mapping) or set(request) - {"symbol", "calls", "num_warps", "grid", "depends_on", "tasks_per_owner", "execution_mesh", "program_kind", "data_placement", "launch_recipe", "maxnreg", "function"}:
            raise IRVerificationError("Kernel program request has unknown fields.", stage=module.stage)
        symbol = request.get("symbol")
        calls = request.get("calls")
        warps = request.get("num_warps")
        if not _valid_symbol(symbol) or symbol in symbols:
            raise IRVerificationError("Kernel program symbols must be unique Python identifiers.", stage=module.stage)
        if not isinstance(calls, (tuple, list)) or not calls or any(not isinstance(c, str) or not c for c in calls):
            raise IRVerificationError(
                "Kernel program calls must be a non-empty sequence of call ids.", stage=module.stage
            )
        if (
            type(warps) is not int
            or warps not in (1, 2, 4, 8, 16, 32)
            or warps * capability.warp_size > capability.max_threads_per_block
        ):
            raise IRVerificationError(
                "Kernel program num_warps exceeds the target resource contract.", stage=module.stage
            )
        grid = request.get("grid")
        tasks = request.get("tasks_per_owner", 1)
        if type(tasks) is not int or tasks < 1:
            raise IRVerificationError("Kernel program tasks_per_owner must be a positive integer.", stage=module.stage)
        if grid is not None and (
            not isinstance(grid, (tuple, list))
            or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)
        ):
            raise IRVerificationError("Kernel program grid must contain three positive integers.", stage=module.stage)
        symbols.add(symbol)
        maxnreg = request.get('maxnreg')
        if maxnreg is not None:
            if (type(maxnreg) is not int or maxnreg < 8 or maxnreg > 256 or maxnreg % 8
                    or 'maxnreg' not in getattr(capability, 'supported_program_compile_options', ())):
                raise IRVerificationError('Program register cap is unsupported by this target/resource contract.', stage=module.stage)
        mesh_fields = {"execution_mesh", "program_kind", "data_placement"}
        present = mesh_fields.intersection(request)
        if present and present != mesh_fields:
            raise IRVerificationError('Execution program mesh/kind/data mapping must be supplied together.', stage=module.stage)
        if present:
            from triton.flagmega.codegen.triton.execution_programs import verify_program_mesh
            from triton.flagmega.ir.execution_mesh import KernelProgramKind
            verify_program_mesh(module, request)
            try:
                KernelProgramKind(request['program_kind'])
            except (ValueError, TypeError) as exc:
                raise IRVerificationError('Unknown kernel program kind.', stage=module.stage) from exc
        programs.append(
            {
                "symbol": symbol,
                "calls": tuple(calls),
                "num_warps": warps,
                **({'function': request['function']} if execution is not None else {}),
                **({key: request[key] for key in mesh_fields} if present else {}),
                **({'launch_recipe': request['launch_recipe']} if 'launch_recipe' in request else {}),
                **({'maxnreg': maxnreg} if maxnreg is not None else {}),
                **({"grid": tuple(request["grid"])} if "grid" in request else {}),
                **({"tasks_per_owner": tasks} if "tasks_per_owner" in request else {}),
                **({"depends_on": tuple(request["depends_on"])} if "depends_on" in request
                   and isinstance(request["depends_on"], (tuple, list)) else {}),
            }
        )
    from triton.flagmega.ir.program_order import program_ancestors
    program_ancestors(raw)
    return tuple(programs)


def _events(events):
    for event in events:
        yield event
        if event["kind"] == "loop":
            yield from _events(event["events"])


def _incoming_completion(events):
    events = [dict(v) for v in events]
    while events and events[0]["kind"] == "barrier" and events[0]["scope"] in {"grid", "block"}:
        events.pop(0)
    if events and events[0]["kind"] != "barrier" and events[0].get("barrier_scope") in {"grid", "block"}:
        events[0]["barrier_before"] = False
    return events


def _entry_item_id(event, function):
    # Match the structured identity used by describe_function_schedule.
    # A loop is one indivisible region; its body is never unrolled for cuts.
    return f"{function}.{event['loop_var']}" if event["kind"] == "loop" else str(event["call"])


def _task_owner_partition(request, descriptor, selected):
    """Validate interleaved CTA tasks against each callee's write partition."""
    tasks = request.get("tasks_per_owner", 1)
    if type(tasks) is not int or tasks < 1:
        raise CodegenError("Kernel program tasks_per_owner must be a positive integer.")
    grid = list(request.get("grid", descriptor["grid"]))
    logical = descriptor["grid"]
    if 'launch_recipe' in request:
        from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
        recipe = TaskLaunchRecipe.from_data(request['launch_recipe'])
        if (tasks != 1 or tuple(grid) != recipe.grid_upper_bound or len(selected) != 1
                or plain_package_value(selected[0].get('task_launch_recipe')) != recipe.to_data()):
            raise CodegenError('Dynamic program task/tail coverage differs from its selected callee ABI.')
        return None
    if 'execution_mesh' in request:
        from triton.flagmega.ir.distributed_type import placement_from_data
        domain = placement_from_data(request['data_placement'])
        logical = [domain.physical_level_size('b'), 1, 1]
    if (len(grid) != 3 or any(type(v) is not int or v <= 0 for v in grid)
            or tasks == 1 and grid != logical
            or tasks > 1 and (logical[1:] != [1, 1] or grid != [logical[0] * tasks, 1, 1])):
        raise CodegenError("Kernel program physical grid requires an explicit task-to-owner ABI.")
    if tasks == 1:
        if any(c.get("task_partition", {}).get("tasks_per_owner", 1) != 1 for c in selected):
            raise CodegenError("Kernel program task count differs from its callee write partition.")
        return None
    if not descriptor["distributed_entry"] or descriptor["grid_barrier_axis_groups"]:
        raise CodegenError("Task-to-owner programs require a flat full grid and owner-addressable pools.")
    if not selected or any(
        c.get("task_partition", {}).get("tasks_per_owner") != tasks
        or c["task_partition"].get("disjoint_writes") is not True
        or c["task_partition"].get("owner_extent") != tasks * c["task_partition"].get("task_extent", 0)
        or c.get("internal_grid_barriers", 0)
        for c in selected
    ):
        raise CodegenError("Every task-to-owner callee must prove complete disjoint output writes.")
    if any(
        int(buffer.get("abi", {}).get("pool_scope_stride_bytes", 0))
        for call in selected for side in ("inputs", "outputs")
        for binding in call[side] for buffer in binding.get("buffers", ())
    ):
        raise CodegenError("Task-to-owner callees cannot address writable CTA-private pool slices.")
    return {"tasks_per_owner": tasks, "owner_count": logical[0], "order": "owner-then-task"}


def describe_kernel_programs(module, plan, descriptor):
    """Bind complete entries or project cuts without changing buffer ownership."""
    raw = plan.get("programs")
    if module.metadata.get('kernel_program_execution') is not None:
        from .structured_programs import describe_structured_programs
        return describe_structured_programs(module, plan, descriptor)
    requested = module.metadata.get("kernel_program_partition")
    if raw is None:
        if requested is not None:
            raise CodegenError("Kernel program request has no matching target package plan.")
        if any(c.get("task_partition", {}).get("tasks_per_owner", 1) > 1 for c in descriptor["render_calls"]):
            raise CodegenError("Task-partitioned callees require an explicit execution program.")
        return {}
    if plain_package_value(raw) != plain_package_value(requested):
        raise CodegenError("Kernel program plan differs from its target-owned partition request.")
    return _describe_program_partition(module, raw, descriptor)


def _describe_program_partition(module, raw, descriptor, *, function_name=None, complete_scope=True):
    """Project an already verified partition in its actual function scope."""
    function_name = module.entry if function_name is None else function_name
    from triton.flagmega.codegen.triton.program_dependencies import verify_program_dependencies, elide_completed_device_barriers
    from triton.flagmega.ir.program_order import program_stream_indices, program_ancestors
    streams = program_stream_indices(raw)
    if streams is not None:
        external_writes, device_completion, accesses = verify_program_dependencies(
            module, raw, descriptor["runtime_binding"], with_accesses=True)
    if (
        not isinstance(raw, (tuple, list))
        or not raw
        or any(
            not isinstance(p, Mapping) or not isinstance(p.get("calls"), (tuple, list)) or not p["calls"] for p in raw
        )
    ):
        raise CodegenError("Kernel program cuts must be non-empty mappings with calls.")
    complete_entry = len(raw) == 1 and complete_scope
    events = descriptor["entry_events"]
    control_events = {"barrier", "drain", "handoff"}
    if any(e["kind"] not in control_events | {"tir.kernel_call", "pipeline_kernel_call", "function_call", "loop"}
           for e in events):
        raise CodegenError("Kernel program cuts require complete executable entry regions.")
    if not complete_entry and any(e["kind"] == "handoff" for e in events):
        raise CodegenError("Kernel program cuts cannot cross a root handoff without an inter-program lifetime ABI.")
    root_calls = [_entry_item_id(e, function_name) for e in events if e["kind"] not in control_events]
    flat = [str(c) for request in raw for c in request["calls"]]
    if flat != root_calls or len(flat) != len(set(flat)):
        raise CodegenError(
            "Kernel program calls must exactly partition executable entry calls in their original order."
        )
    calls = {str(c["call"]): c for c in descriptor["render_calls"]}
    definitions = {str(d["function"]): d for d in descriptor["device_functions"]}
    programs = []
    launches = []
    cursor = 0
    symbols = set()
    device_channels = {}
    for request in raw:
        symbol = request["symbol"]
        warps = request["num_warps"]
        if (
            not _valid_symbol(symbol)
            or symbol in symbols
            or type(warps) is not int
            or warps not in (1, 2, 4, 8, 16, 32)
        ):
            raise CodegenError("Kernel program symbols/num_warps are invalid.")
        symbols.add(symbol)
        grid = list(request.get("grid", descriptor["grid"]))
        segment = []
        remaining = len(request["calls"])
        while remaining:
            event = events[cursor]
            cursor += 1
            segment.append(event)
            remaining -= event["kind"] not in control_events
        # Drains use the preceding call's pipe instances. Keep them with
        # that call so their endpoint pointers never escape its kernel.
        while cursor < len(events) and events[cursor]["kind"] == "drain":
            segment.append(events[cursor])
            cursor += 1
        if len(programs) == len(raw) - 1:
            segment.extend(events[cursor:])
            cursor = len(events)
        segment = _incoming_completion(segment)
        completion_elisions = []
        if streams is not None:
            segment, completion_elisions = elide_completed_device_barriers(
                segment, symbol, raw, accesses, device_completion)
            segment = _incoming_completion(segment)
        selected = {}
        used_functions = {}
        visited_events = []

        def visit(local_events, *, selected=selected, used_functions=used_functions, visited_events=visited_events):
            for event in _events(local_events):
                visited_events.append(event)
                if event["kind"] in {"tir.kernel_call", "pipeline_kernel_call"}:
                    selected[event["call"]] = calls[event["call"]]
                elif event["kind"] == "function_call":
                    name = str(event["callee"])
                    if name in used_functions:
                        continue
                    definition = definitions[name]
                    used_functions[name] = definition
                    schedule = definition["schedule"]
                    visit(schedule["consumer_events"])

        visit(segment)
        from triton.flagmega.codegen.triton.pipeline_programs import project_pipeline_schedule
        pipeline = project_pipeline_schedule(module, descriptor["pipeline_schedule"], segment,
                                             complete_entry=complete_entry)
        task_owner = _task_owner_partition(request, descriptor, list(selected.values()))
        completes_device = streams is not None and device_completion[symbol]
        if completes_device:
            segment.append({"kind": "barrier", "scope": "device", "reason": "program_device_completion"})
        device_channels[symbol] = completes_device or any(
            e["kind"] == "barrier" and e.get("scope") == "device"
            or e.get("barrier_before") and e.get("barrier_scope") == "device" for e in visited_events)
        if any(warps < int(c.get("parameters", {}).get("compute_num_warps", 1)) for c in selected.values()):
            raise CodegenError("Kernel program warp count is below a selected callee requirement.")
        synchronized = any(
            (e["kind"] == "barrier" and e.get("scope") in {"grid", "device"})
            or (e.get("barrier_before") and e.get("barrier_scope") in {"grid", "device"})
            for e in visited_events
        )
        synchronized |= any(
            int(c.get("facts", {}).get("internal_grid_barriers", 0)) > 0
            or "grid_sync" in c.get("facts", {}).get("requires", ())
            for c in selected.values()
        )
        synchronized |= completes_device
        if 'launch_recipe' in request and synchronized:
            raise CodegenError('Independent dynamic tiles cannot contain grid/device waits.')
        launch = {
            "symbol": symbol,
            "grid": grid,
            "num_warps": warps,
            "dynamic_argument_indices": list(descriptor["dynamic_argument_indices"]),
            "requires_grid_wide_sync": bool(synchronized),
            "calls": list(request["calls"]),
            **({key: plain_package_value(request[key]) for key in ('execution_mesh', 'program_kind', 'data_placement')}
               if 'execution_mesh' in request else {}),
            **({'launch_recipe': plain_package_value(request['launch_recipe'])} if 'launch_recipe' in request else {}),
            **({'maxnreg': request['maxnreg']} if 'maxnreg' in request else {}),
            **({"task_owner_partition": task_owner} if task_owner else {}),
            **({"depends_on": list(request["depends_on"]), "stream_index": streams[symbol],
                "external_writes": list(external_writes[symbol]),
                "device_completion": bool(completes_device)} if streams is not None else {}),
        }
        launches.append(launch)
        mesh_context = {}
        if 'execution_mesh' in request:
            from triton.flagmega.codegen.triton.tir_package import _mesh_context
            mesh_context = _mesh_context(request['data_placement'])
            mesh_context.update(distributed_entry=True, use_tle=True,
                                dynamic_execution=request['execution_mesh']['kind'] == 'dynamic')
        programs.append(
            {
                "symbol": symbol,
                "num_warps": warps,
                "entry_events": segment,
                "render_calls": list(selected.values()),
                "device_functions": list(used_functions.values()),
                "pipeline_schedule": pipeline,
                **({"tasks_per_owner": task_owner["tasks_per_owner"]} if task_owner else {}),
                "completed_device_barriers": completion_elisions,
                **({'mesh_context': mesh_context} if mesh_context else {}),
            }
        )
    if streams is not None:
        ancestors = program_ancestors(raw)
        for index, request in enumerate(raw):
            if device_channels[request["symbol"]] and any(
                device_channels[previous["symbol"]] and previous["symbol"] not in ancestors[request["symbol"]]
                for previous in raw[:index]
            ):
                raise CodegenError("Unordered device rendezvous require distinct distributed synchronization channels.")
    from triton.flagmega.codegen.triton.function_programs import project_function_programs, KERNEL_PROGRAM_FUNCTION_SCHEMA
    has_mesh = any('execution_mesh' in p for p in raw)
    if has_mesh and not all('execution_mesh' in p for p in raw):
        raise CodegenError('Mesh program plans require complete mesh contracts.')
    shared = streams is not None and not has_mesh and project_function_programs(module, descriptor, programs, launches)
    if any(p.get("tasks_per_owner", 1) > 1 and p.get("scoped_entry_arguments") for p in programs):
        raise CodegenError("Task-to-owner programs cannot share writable CTA-private entry frames.")
    return {"kernel_program_schema": KERNEL_PROGRAM_MESH_DAG_SCHEMA if has_mesh and streams is not None else
            KERNEL_PROGRAM_MESH_SCHEMA if has_mesh else KERNEL_PROGRAM_FUNCTION_SCHEMA if shared else
            KERNEL_PROGRAM_DAG_COMPLETION_SCHEMA if streams is not None else KERNEL_PROGRAM_SCHEMA,
            "kernels": launches, "programs": programs}
