# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Render a bufferized TIR function graph without model reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
import re
from .descriptor_resources import DescriptorResources

from triton.flagmega.codegen.triton.function_call import (
    emit_function_call_arguments,
)
from triton.flagmega.codegen.triton.function_schedule import (
    describe_function_schedule,
    memory_barriers_for_function,
)
from triton.flagmega.codegen.triton.kernel_call_renderers import (
    prepare_kernel_calls,
)
from triton.flagmega.codegen.triton.package_plan import (
    plain_package_value,
    require_package_plan,
)
from triton.flagmega.codegen.triton.pipeline_source import (
    build_pipeline_source_schedule,
    function_call_storage_roots,
)
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.codegen.triton.physical_access import emit_triton_scalar_type
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
    module_template_target,
)
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.ir import IRModule, verify_buffer_plan, verify_module
from triton.flagmega.ir import execution_calls_of


TIR_PACKAGE_DESCRIPTOR_SCHEMA = "flagmega.triton-package-descriptor/v1"


def describe_tir_package(module: IRModule) -> dict[str, object]:
    """Derive source-generation facts solely from executable TIR contracts."""

    try:
        verify_module(module)
    except IRVerificationError as error:
        raise CodegenError(
            str(error), stage=error.stage, node_id=error.node_id
        ) from error
    verify_buffer_plan(module)
    package_plan = require_package_plan(module, "tir_call_graph")
    launch = module.metadata.get("launch_contract")
    if not isinstance(launch, Mapping):
        raise CodegenError(
            "Bufferized TIR has no target launch contract.", stage=module.stage
        )
    plan_launch = package_plan.get("launch")
    if not isinstance(plan_launch, Mapping):
        raise CodegenError("TIR package plan has no launch resource contract.")

    call_abi_cache: dict[str, dict[str, object]] = {}
    root_schedule = describe_function_schedule(
        module,
        call_abi_cache=call_abi_cache,
    )
    function_names = [module.entry, *root_schedule["reachable_functions"]]
    bindings = {
        name: describe_function_runtime_binding(
            module,
            function_name=name,
            call_abi=call_abi_cache.get(name),
        )
        for name in function_names
    }
    encoded_calls = {
        name: prepare_kernel_calls(
            bindings[name]["call_abi"]["kernel_calls"],
            function_name=name,
        )
        for name in function_names
    }
    _verify_package_call_closure(package_plan, bindings)

    root_binding = bindings[module.entry]
    replicated_block_runtime_pool = any(
        int(pool.get("scope_count", 1)) > 1
        and pool.get("scope") == "block"
        and pool.get("scope_index") == "program_id_x"
        for pool in root_binding["pools"]
    )
    root_signature = [str(value) for value in root_binding["signature"]]
    generated_descriptor_specs = DescriptorResources()
    root_actuals = {str(name): str(name) for name in root_signature}
    root_storage = {
        str(name): (str(name), 0) for name in root_signature
    }
    source_schedule = build_pipeline_source_schedule(
        module,
        bindings,
        encoded_calls,
        lambda function_name, call, actuals, storage, path, descriptor_specs: (
            _kernel_source_event(
                call,
                actuals,
                storage,
                descriptor_specs,
                path,
                memory_dependencies=(),
            )
        ),
        _instantiate_descriptor_requests,
    )
    pipeline_schedule = source_schedule if source_schedule and source_schedule["stages"] else None
    if source_schedule is not None:
        generated_descriptor_specs.extend(source_schedule["host_tensor_descriptor_specs"])
    entry_events = (
        _expand_entry_events(
            module,
            module.entry,
            bindings,
            encoded_calls,
            root_actuals,
            (),
            root_storage,
            generated_descriptor_specs,
            (),
        )
        if source_schedule is None
        else source_schedule["consumer_events"]
    )
    wrappers = [
        call
        for name in function_names
        for call in encoded_calls[name]
    ]
    cooperative = bool(plan_launch.get("cooperative_grid", False))
    mesh = (
        _mesh_context(launch.get("grid_mesh"))
        if cooperative
        else _local_mesh(launch.get("grid_mesh"))
    )
    grid_barrier_axis_groups = _attach_grid_barrier_axis_groups(
        entry_events,
        source_schedule,
        mesh,
    )
    owner_count = int(mesh["mesh_size"])
    for call in wrappers:
        if call["family"] == "add_norm_stats":
            call["owner_count"] = owner_count

    platform, architecture = module_template_target(module)
    raw_templates = package_plan.get("kernel_templates")
    if not isinstance(raw_templates, (tuple, list)):
        raise CodegenError("TIR package plan has no kernel-template closure.")
    specs = tuple(
        KernelTemplateSpec(
            str(value["kernel"]),
            str(value["variant"]),
            platform,
            architecture,
        )
        for value in raw_templates
    )
    registry = TritonTemplateRegistry()
    kernel_templates = tuple(registry.resolve(spec) for spec in specs)
    descriptor_specs = _merge_host_tensor_descriptor_specs(
        _host_tensor_descriptor_specs(package_plan, root_signature),
        tuple(generated_descriptor_specs),
        root_signature,
    )
    descriptor_names = [str(value["name"]) for value in descriptor_specs]
    signature = [*root_signature, *descriptor_names]
    scalar_arguments = {
        str(value["name"]): str(value["scalar_dtype"])
        for value in root_binding["arguments"]
        if value.get("runtime_value_kind") == "scalar"
    }
    typed_signature = ",\n    ".join(
        f"{name}: {emit_triton_scalar_type(scalar_arguments[name])}" if name in scalar_arguments else name
        for name in signature
    )
    external_names = {
        str(value["name"]) for value in root_binding["arguments"]
    }
    descriptor_base = len(root_signature)
    dynamic_descriptors = [
        descriptor_base + index
        for index, spec in enumerate(descriptor_specs)
        if str(spec["source"]) in external_names
    ]
    return {
        "schema": TIR_PACKAGE_DESCRIPTOR_SCHEMA,
        "symbol": str(launch["entry"]),
        "grid": [int(value) for value in plan_launch["grid"]],
        "num_warps": int(plan_launch["num_warps"]),
        "dynamic_argument_indices": list(
            range(len(root_binding["arguments"]))
        ) + dynamic_descriptors,
        "signature_arguments": signature,
        "signature": typed_signature,
        "scalar_arguments": list(scalar_arguments),
        "host_tensor_descriptor_specs": plain_package_value(
            descriptor_specs
        ),
        "function_schedule": plain_package_value(root_schedule),
        "runtime_binding": plain_package_value(root_binding),
        "replicated_block_runtime_pool": replicated_block_runtime_pool,
        "render_calls": wrappers,
        "entry_events": entry_events,
        "pipeline_schedule": pipeline_schedule,
        "device_functions": source_schedule["device_functions"] if source_schedule else [],
        "use_tle": pipeline_schedule is not None or cooperative,
        "kernel_templates": kernel_templates,
        "kernel_template_specs": plain_package_value(raw_templates),
        "platform": platform,
        "target_architecture": architecture,
        "distributed_entry": cooperative,
        "grid_barrier_axis_groups": grid_barrier_axis_groups,
        **mesh,
        **_shared_template_context(wrappers),
    }


def _host_tensor_descriptor_specs(
    package_plan: Mapping[str, object],
    root_signature: list[str],
) -> tuple[Mapping[str, object], ...]:
    raw = package_plan.get("host_tensor_descriptor_specs", ())
    if not isinstance(raw, (tuple, list)):
        raise CodegenError(
            "TIR package host_tensor_descriptor_specs must be a sequence."
        )
    specs = tuple(raw)
    if any(not isinstance(value, Mapping) for value in specs):
        raise CodegenError(
            "TIR package host tensor descriptor specs must be mappings."
        )
    names = tuple(str(value.get("name", "")) for value in specs)
    sources = tuple(str(value.get("source", "")) for value in specs)
    if (
        any(not value for value in (*names, *sources))
        or len(set(names)) != len(names)
        or set(names).intersection(root_signature)
    ):
        raise CodegenError(
            "TIR package host tensor descriptors require unique non-empty ABI "
            "names disjoint from pointer arguments."
        )
    unknown_sources = sorted(set(sources) - set(root_signature))
    if unknown_sources:
        raise CodegenError(
            "TIR package host tensor descriptors reference unknown roots: "
            f"{unknown_sources}."
        )
    return specs


def _merge_host_tensor_descriptor_specs(
    explicit: tuple[Mapping[str, object], ...],
    generated: tuple[Mapping[str, object], ...],
    root_signature: list[str],
) -> tuple[Mapping[str, object], ...]:
    specs = (*explicit, *generated)
    names = tuple(str(value.get("name", "")) for value in specs)
    sources = tuple(str(value.get("source", "")) for value in specs)
    if any(not value for value in (*names, *sources)):
        raise CodegenError(
            "TIR package host tensor descriptors require non-empty names and sources."
        )
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise CodegenError(
            f"TIR package host tensor descriptor names are duplicated: {duplicates}."
        )
    conflicts = sorted(set(names).intersection(root_signature))
    if conflicts:
        raise CodegenError(
            "TIR package host tensor descriptor names conflict with pointer "
            f"arguments: {conflicts}."
        )
    unknown_sources = sorted(set(sources) - set(root_signature))
    if unknown_sources:
        raise CodegenError(
            "TIR package host tensor descriptors reference unknown roots: "
            f"{unknown_sources}."
        )
    return tuple(specs)


def render_tir_package(
    descriptor: dict[str, object], renderer_version: str
) -> str:
    context = {
        **descriptor,
        "renderer_version": renderer_version,
        "entry_template": "entrypoints/call_graph.py.jinja",
    }
    return TritonTemplateRegistry().render("module.py.jinja", context)


def _expand_entry_events(
    module: IRModule,
    function_name: str,
    bindings: Mapping[str, Mapping[str, object]],
    encoded_calls: Mapping[str, list[dict[str, object]]],
    actual_by_formal: Mapping[str, str],
    active: tuple[str, ...],
    storage_roots: Mapping[str, tuple[str, int]],
    descriptor_specs: list[Mapping[str, object]],
    call_path: tuple[str, ...],
) -> list[dict[str, object]]:
    if function_name in active:
        raise CodegenError(
            "Recursive TIR source schedule: "
            + " -> ".join((*active, function_name))
        )
    binding = bindings[function_name]
    call_by_id = {
        str(value["call"]): value for value in encoded_calls[function_name]
    }
    synchronization_before = memory_barriers_for_function(
        module, function_name
    )
    result: list[dict[str, object]] = []
    for event in _execution_events(module, function_name, binding):
        if event["kind"] == "kernel_call":
            call = call_by_id[str(event["call"])]
            result.append(_kernel_source_event(
                call,
                actual_by_formal,
                storage_roots,
                descriptor_specs,
                (*call_path, str(event["call"])),
                memory_dependencies=synchronization_before.get(
                    str(event["call"]), ()
                ),
            ))
            continue
        if event["kind"] != "function_call":
            raise CodegenError(
                f"TIR function @{function_name} has unknown event "
                f"{event['kind']!r}."
            )
        callee = str(event["callee"])
        if callee not in bindings:
            raise CodegenError(
                f"TIR function @{function_name} calls unplanned @{callee}."
            )
        local_actuals = emit_function_call_arguments(module, binding, event)
        callee_signature = bindings[callee]["signature"]
        if len(local_actuals) != len(callee_signature):
            raise CodegenError(
                f"TIR call {event['call']!r} does not match @{callee} signature."
            )
        callee_actuals = {
            str(formal): _substitute(actual, actual_by_formal)
            for formal, actual in zip(
                callee_signature, local_actuals, strict=True
            )
        }
        callee_storage_roots = function_call_storage_roots(
            binding,
            bindings[callee],
            event,
            storage_roots,
        )
        nested = _expand_entry_events(
            module,
            callee,
            bindings,
            encoded_calls,
            callee_actuals,
            (*active, function_name),
            callee_storage_roots,
            descriptor_specs,
            (*call_path, str(event["call"])),
        )
        dependencies = synchronization_before.get(str(event["call"]), ())
        if nested and dependencies:
            nested[0]["memory_dependencies"] = plain_package_value((
                *tuple(nested[0].get("memory_dependencies", ())),
                *dependencies,
            ))
            scope, axes = _merge_barrier_requirements(
                (
                    str(nested[0].get("barrier_scope", "block")),
                    tuple(nested[0].get("barrier_axis_group_axes", ())),
                ) if nested[0].get("barrier_before") else None,
                _dependency_barrier_requirement(dependencies),
            )
            nested[0]["barrier_before"] = True
            nested[0]["barrier_scope"] = scope
            nested[0]["barrier_axis_group_axes"] = axes
        for value in nested:
            value["call"] = f"{event['call']}::{value['call']}"
        result.extend(nested)
    return result


def _kernel_source_event(
    call: Mapping[str, object],
    actual_by_formal: Mapping[str, str],
    storage_roots: Mapping[str, tuple[str, int]],
    descriptor_specs: list[Mapping[str, object]],
    call_path: tuple[str, ...],
    *,
    memory_dependencies,
) -> dict[str, object]:
    descriptor_arguments = _instantiate_descriptor_requests(
        call, storage_roots, descriptor_specs, call_path
    )
    descriptor_parameters = tuple(call.get("descriptor_parameters", ()))
    pointer_argument_count = len(call["arguments"]) - len(
        descriptor_parameters
    )
    pointer_arguments = call["arguments"][:pointer_argument_count]
    dependencies = tuple(memory_dependencies)
    requirement = _dependency_barrier_requirement(
        dependencies,
        force_full_grid=call["execution_kind"] == "collective",
    )
    barrier_scope, barrier_axis_group_axes = (
        (None, ()) if requirement is None else requirement
    )
    return {
        "kind": "tir.kernel_call",
        "call": str(call["call"]),
        "family": str(call["family"]),
        "variant": str(call["variant"]),
        "execution_kind": str(call["execution_kind"]),
        "barrier_before": (
            call["execution_kind"] == "collective" or bool(dependencies)
        ),
        "barrier_scope": barrier_scope,
        "barrier_axis_group_axes": barrier_axis_group_axes,
        "memory_dependencies": plain_package_value(dependencies),
        "symbol": str(call["symbol"]),
        "pipeline_channels": call.get("pipeline_channels", ()),
        "pipeline_consumer_workspaces": call.get(
            "pipeline_consumer_workspaces", ()
        ),
        "pipeline_consumer_parameters": call.get(
            "pipeline_consumer_parameters", ()
        ),
        "pipeline_producer_parameters": call.get(
            "pipeline_producer_parameters", ()
        ),
        "pipeline_auxiliary_consumer": call.get(
            "pipeline_auxiliary_consumer"
        ),
        "arguments": ", ".join((
            *(
                _substitute(value, actual_by_formal)
                for value in pointer_arguments
            ),
            *descriptor_arguments,
        )),
    }


def _dependency_barrier_requirement(
    dependencies,
    *,
    force_full_grid: bool = False,
) -> tuple[str, tuple[int, ...]] | None:
    if force_full_grid:
        return "grid", ()
    dependencies = tuple(dependencies)
    if not dependencies:
        return None
    grid_axes = []
    for dependency in dependencies:
        if str(dependency.get("scope", "grid")) != "grid":
            continue
        axes = tuple(int(value) for value in dependency.get("axis_group_axes", ()))
        if not axes:
            return "grid", ()
        grid_axes.extend(axes)
    if grid_axes:
        return "grid", tuple(sorted(set(grid_axes)))
    return "block", ()


def _merge_barrier_requirements(
    lhs: tuple[str, tuple[int, ...]] | None,
    rhs: tuple[str, tuple[int, ...]] | None,
) -> tuple[str, tuple[int, ...]]:
    if lhs is None:
        assert rhs is not None
        return rhs
    if rhs is None:
        return lhs
    if lhs[0] == "block":
        return rhs
    if rhs[0] == "block":
        return lhs
    if not lhs[1] or not rhs[1]:
        return "grid", ()
    return "grid", tuple(sorted(set((*lhs[1], *rhs[1]))))


def _attach_grid_barrier_axis_groups(
    entry_events: list[dict[str, object]],
    pipeline_schedule: Mapping[str, object] | None,
    mesh: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Validate axis-group barriers, attach symbols, and deduplicate globals."""

    event_lists = [entry_events]
    if pipeline_schedule is not None:
        event_lists = [
            pipeline_schedule["consumer_events"],
            pipeline_schedule["producer_events"],
            pipeline_schedule["auxiliary_events"],
        ]
        event_lists.extend(
            definition["schedule"][role]
            for definition in pipeline_schedule["device_functions"]
            for role in ("consumer_events", "producer_events", "auxiliary_events")
        )
    physical_axes = tuple(
        int(value["placement_axis"])
        for value in mesh.get("mesh_axes", ())
        if str(value["level"]) == "b"
    )
    hierarchy = tuple(int(value) for value in mesh.get("mesh_hierarchy", ()))
    axis_names = tuple(str(value) for value in mesh.get("mesh_axis_names", ()))
    groups: dict[tuple[int, ...], dict[str, object]] = {}
    seen_events: set[int] = set()
    for events in event_lists:
        for event in events:
            # consumer_events is also entry_events for a pipeline package.
            if id(event) in seen_events:
                continue
            seen_events.add(id(event))
            axes = tuple(int(value) for value in (
                event.get("axis_group_axes", ())
                if event.get("kind") == "barrier"
                else event.get("barrier_axis_group_axes", ())
            ))
            if not axes:
                continue
            if tuple(sorted(set(axes))) != axes:
                raise CodegenError(
                    f"Grid barrier axis-group axes must be sorted and unique: {axes}."
                )
            unknown = tuple(axis for axis in axes if axis not in physical_axes)
            if unknown:
                raise CodegenError(
                    f"Grid barrier axes {unknown} are not physical block axes "
                    f"of the launch mesh {physical_axes}."
                )
            if axes == physical_axes:
                axes = ()
                if event.get("kind") == "barrier":
                    event["axis_group_axes"] = ()
                else:
                    event["barrier_axis_group_axes"] = ()
                continue
            shape = tuple(hierarchy[axis] for axis in axes)
            key = "_".join(
                f"{axis}x{extent}"
                for axis, extent in zip(axes, shape, strict=True)
            )
            event["barrier_axis_group_key"] = key
            groups[axes] = {
                "key": key,
                "axis_names_repr": repr(tuple(axis_names[axis] for axis in axes)),
                "shape_repr": repr(shape),
                "axes": axes,
                "shape": shape,
            }
    return tuple(groups[axes] for axes in sorted(groups))


def _instantiate_descriptor_requests(
    call: Mapping[str, object],
    storage_roots: Mapping[str, tuple[str, int]],
    descriptor_specs: list[Mapping[str, object]],
    call_path: tuple[str, ...],
) -> tuple[str, ...]:
    requests = call.get("host_tensor_descriptor_requests", ())
    if not isinstance(requests, (tuple, list)):
        raise CodegenError(
            "TIR call host tensor descriptor requests must be a sequence."
        )
    result: list[str] = []
    for request in requests:
        if not isinstance(request, Mapping):
            raise CodegenError(
                "TIR call host tensor descriptor request must be a mapping."
            )
        parameter = str(request.get("parameter", ""))
        local_source = str(request.get("source", ""))
        try:
            root_source, root_offset = storage_roots[local_source]
        except KeyError as error:
            raise CodegenError(
                f"TIR descriptor parameter {parameter!r} cannot resolve local "
                f"storage source {local_source!r} to an entry ABI root."
            ) from error
        kind = str(request.get("kind", ""))
        name = _descriptor_instance_name(call_path, parameter)
        expected = (
            {
                "parameter", "source", "kind", "offset_bytes", "dtype",
                "shape", "strides", "block_shape", "source_shape_axes", "padding",
            }
            if kind == "single"
            else {
                "parameter", "source", "kind", "dtype", "block_shape",
                "padding", "swizzle_mode", "entry_size_bytes", "entries",
            }
            if kind == "table"
            else None
        )
        if expected is None:
            raise CodegenError(
                f"Automatically generated descriptor {parameter!r} has "
                f"unsupported kind {kind!r}."
            )
        fields = set(request)
        if "rebase_axis" in fields:
            expected = expected | {"rebase_axis"}
        if kind == "single" and request.get("storage") == "device":
            expected = expected | {"storage", "box_shape", "swizzle_mode"}
        if fields != expected:
            raise CodegenError(
                f"TIR descriptor request {parameter!r} fields differ: "
                f"missing={sorted(expected - fields)}, "
                f"unexpected={sorted(fields - expected)}."
            )
        if kind == "single":
            spec = {
                key: value
                for key, value in request.items()
                if key not in {"parameter", "source", "offset_bytes", "rebase_axis"}
            }
            spec.update({
                "name": name,
                "source": root_source,
                "offset_bytes": root_offset + int(request["offset_bytes"]),
            })
        else:
            entries = request["entries"]
            if not isinstance(entries, (tuple, list)) or not entries:
                raise CodegenError(
                    f"TIR descriptor table {parameter!r} requires entries."
                )
            rebased_entries = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise CodegenError(
                        f"TIR descriptor table {parameter!r} entry {index} "
                        "must be a mapping."
                    )
                entry_expected = {
                    "offset_bytes", "shape", "strides", "source_shape_axes",
                }
                if set(entry) != entry_expected:
                    raise CodegenError(
                        f"TIR descriptor table {parameter!r} entry {index} "
                        "fields differ."
                    )
                rebased_entries.append({
                    **dict(entry),
                    "offset_bytes": root_offset + int(entry["offset_bytes"]),
                })
            spec = {
                key: value
                for key, value in request.items()
                if key not in {"parameter", "source", "entries", "rebase_axis"}
            }
            spec.update({
                "name": name,
                "source": root_source,
                "entries": tuple(rebased_entries),
            })
        if isinstance(descriptor_specs, DescriptorResources):
            result.extend(descriptor_specs.bind(spec, request.get("rebase_axis")))
            continue
        if "rebase_axis" in request:
            raise CodegenError("Rebased descriptor calls require a DescriptorResources ABI.")
        # Legacy callers without an origin ABI can still intern exact views.
        identical = next((value for value in descriptor_specs if {
            key: item for key, item in value.items() if key != "name"
        } == {key: item for key, item in spec.items() if key != "name"}), None)
        if identical is not None:
            result.append(str(identical["name"]))
            continue
        if any(str(value.get("name", "")) == name for value in descriptor_specs):
            raise CodegenError(
                f"Duplicate instantiated host tensor descriptor {name!r}."
            )
        descriptor_specs.append(spec)
        result.append(name)
    return tuple(result)


def _descriptor_instance_name(
    call_path: tuple[str, ...], parameter: str
) -> str:
    raw = "__".join((*call_path, parameter))
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    if not stem:
        raise CodegenError("A host tensor descriptor requires a named call path.")
    if stem[0].isdigit():
        stem = f"call_{stem}"
    return f"_flagmega_{stem}_descriptor"


def _execution_events(module, function_name, binding):
    abi_events = binding["call_abi"]["events"]
    function = module.execution_function_map.get(function_name)
    if function is None:
        return tuple(abi_events)
    by_id = {str(value["call"]): value for value in abi_events}
    calls = execution_calls_of(function)
    if {call.call_id for call in calls} != set(by_id):
        raise CodegenError(
            f"ExecutionFunction @{function_name} call closure differs from its "
            "runtime binding ABI."
        )
    return tuple(by_id[call.call_id] for call in calls)


def _substitute(expression: str, values: Mapping[str, str]) -> str:
    if expression in values:
        return values[expression]
    names = tuple(sorted(values, key=len, reverse=True))
    if not names:
        return expression
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, names))
        + r")(?![A-Za-z0-9_])"
    )
    return pattern.sub(lambda match: f"({values[match.group(1)]})", expression)


def _verify_package_call_closure(
    package_plan: Mapping[str, object],
    bindings: Mapping[str, Mapping[str, object]],
) -> None:
    raw_calls = package_plan.get("calls")
    if not isinstance(raw_calls, (tuple, list)):
        raise CodegenError("TIR package plan has no selected call closure.")
    planned = _call_closure(raw_calls, source="target package plan")
    actual = _call_closure(
        (
            call
            for binding in bindings.values()
            for call in binding["call_abi"]["kernel_calls"]
        ),
        source="bufferized function graph",
    )
    if planned != actual:
        raise CodegenError(
            "Target TIR package call closure differs from the bufferized "
            f"function graph: planned={planned}, actual={actual}."
        )


def _call_closure(values, *, source: str) -> dict[str, tuple[str, str]]:
    """Return an order-independent implementation closure keyed by call id.

    A module stores functions in structural order while the runtime binding
    walks them in call-graph order.  Neither ordering is an execution schedule;
    execution order lives in each function ABI.  Requiring those two incidental
    orders to agree made otherwise identical multi-function packages fail.
    """

    result: dict[str, tuple[str, str]] = {}
    for value in values:
        call = str(value["call"])
        implementation = (str(value["family"]), str(value["variant"]))
        if call in result:
            raise CodegenError(
                f"Duplicate TIR call id {call!r} in {source}."
            )
        result[call] = implementation
    return result


def _local_mesh(value: object = None) -> dict[str, object]:
    if value is None:
        return {
            "grid_mesh": None,
            "mesh_hierarchy": (),
            "mesh_axis_names": (),
            "mesh_levels": (),
            "mesh_axes": (),
            "mesh_rank": 0,
            "mesh_size": 1,
            "mesh_axes_repr": "[]",
        }
    topology = _mesh_context(value)
    return {
        **topology,
        "grid_mesh": None,
        # An ordinary launch executes one logical owner.  Retain placement
        # coordinates for generated address/writer expressions but do not turn
        # their logical topology into a multi-CTA launch.
        "mesh_size": 1,
    }


def _mesh_context(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CodegenError(
            "Cooperative TIR codegen requires grid_mesh metadata."
        )
    hierarchy = tuple(int(item) for item in value.get("hierarchy", ()))
    names = str(value.get("name", ""))
    levels = str(value.get("hierarchy_levels", ""))
    if not hierarchy or len(names) != len(hierarchy) or len(levels) != len(hierarchy):
        raise CodegenError(
            "Triton distributed codegen requires a non-empty placement whose "
            "axis names and hierarchy levels match its rank."
        )
    if any(item <= 0 for item in hierarchy):
        raise CodegenError("Triton grid-mesh dimensions must be positive.")
    if any(level not in {"c", "d", "b"} for level in levels):
        raise CodegenError("Triton grid-mesh levels must be drawn from c/d/b.")
    if any(
        level != "b" and hierarchy[index] != 1
        for index, level in enumerate(levels)
    ):
        raise CodegenError(
            "Triton launches can materialize only non-trivial physical block axes."
        )
    if "b" not in levels:
        raise CodegenError("Triton distributed launch has no physical block axis.")
    axis_names = tuple(f"block_{name}" for name in names)
    physical_names = tuple(
        axis_names[index]
        for index, level in enumerate(levels)
        if level == "b"
    )
    if len(set(physical_names)) != len(physical_names):
        raise CodegenError("Triton physical grid-mesh axis names must be unique.")
    axes = tuple(
        {
            "placement_axis": index,
            "name": name,
            "size": hierarchy[index],
            "level": levels[index],
        }
        for index, name in enumerate(axis_names)
    )
    return {
        "grid_mesh": {
            "hierarchy": list(hierarchy),
            "name": names,
            "hierarchy_levels": levels,
        },
        "mesh_hierarchy": hierarchy,
        "mesh_axis_names": axis_names,
        "mesh_levels": tuple(levels),
        "mesh_axes": axes,
        "mesh_rank": len(hierarchy),
        "mesh_size": prod(hierarchy),
        "mesh_axes_repr": repr([
            (axis_names[index], hierarchy[index])
            for index, level in enumerate(levels)
            if level == "b"
        ]),
    }


def _shared_template_context(
    calls: list[dict[str, object]],
) -> dict[str, object]:
    matrix_calls = [
        call for call in calls
        if call["family"] in {"block_fp8", "matmul_glu"}
    ]
    recurrent = next(
        (call for call in calls if call["family"] == "gdn_recurrent"),
        {},
    )
    elementwise = next(
        (call for call in calls if call["family"] == "elementwise"),
        {},
    )
    return {
        "shared_silu": any(
            call["family"] in {"gdn_convolution", "gdn_recurrent", "matmul_glu"}
            for call in calls
        ),
        "tn": max(
            (int(call.get("tile_n", 128)) for call in matrix_calls),
            default=128,
        ),
        "recurrent_value_tile": int(recurrent.get("value_tile", 4)),
        "head_block": int(recurrent.get("head_block", 128)),
        "projection_tile": int(recurrent.get("projection_tile", 128)),
        "query_scale_repr": str(recurrent.get("query_scale", "1.0")),
        "elementwise_block": int(elementwise.get("tile", 256)),
        "elementwise_packed_axes": False,
        "fastest_axis_order": (),
        "padded_shape": (),
        "logical_strides": (),
        "logical_shape": (),
    }


__all__ = [
    "TIR_PACKAGE_DESCRIPTOR_SCHEMA",
    "describe_tir_package",
    "render_tir_package",
]
