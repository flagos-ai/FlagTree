# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Render a bufferized TIR function graph without model reconstruction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from math import prod

from triton.flagmega.codegen.triton.amd_prefetch import (
    add_next_invocation_qkv_prefetch,
    add_split_k_weight_tile_prefetch,
)
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
from triton.flagmega.codegen.triton.physical_access import emit_triton_scalar_type
from triton.flagmega.codegen.triton.pipeline_source import (
    build_pipeline_source_schedule,
    function_call_storage_roots,
)
from triton.flagmega.codegen.triton.runtime_binding import (
    DEVICE_CONTEXT_ABI,
    DEVICE_CONTEXT_ARGUMENTS,
    DEVICE_CONTEXT_HANDLES,
    DEVICE_CONTEXT_MODE,
    describe_function_runtime_binding,
)
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
    module_template_target,
)
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.ir import IRModule, execution_calls_of, verify_buffer_plan, verify_module
from triton.flagmega.ir.ops.math._block_scaled import block_scaled_precision_supports_target

from .descriptor_resources import DescriptorResources
from .kernel_programs import describe_kernel_programs

TIR_PACKAGE_DESCRIPTOR_SCHEMA = "flagmega.triton-package-descriptor/v1"


def describe_tir_package(
    module: IRModule, *, pool_function_arguments: bool = True,
    specialize_readonly_layout: bool = False,
    triton_trace=None,
    readonly_layout_policy=None,
) -> dict[str, object]:
    """Derive source-generation facts solely from executable TIR contracts.

    ``pool_function_arguments`` forwards the reusable-function calling
    convention to the source schedule; see
    ``build_pipeline_source_schedule``. A backend whose device functions take
    one pointer per buffer clears it.

    ``specialize_readonly_layout`` enables Triton's proved device weight
    addresses. C++ describes canonical calls here and selects its own layout
    later, so sharing this descriptor builder does not repack its pool twice.
    """

    if module.heterogeneous_plans or module.heterogeneous_programs:
        raise CodegenError("This backend cannot lower a heterogeneous execution program.")
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
    if module.metadata.get('r004_standalone_request') is not None:
        return _describe_r004_phase_package(module,package_plan,bindings)
    from .function_pool_abi import bank_row_strides, function_pool_origins
    owned_origins = (function_pool_origins(
        bindings, bank_row_strides(module, verify_buffer_plan(module)), entry=module.entry)
        if pool_function_arguments else {})
    encoded_calls = {
        name: prepare_kernel_calls(
            bindings[name]["call_abi"]["kernel_calls"],
            function_name=name,
            owned_pool_origins=owned_origins.get(name),
        )
        for name in function_names
    }
    nvidia_readonly_plan = None
    if readonly_layout_policy is not None:
        from .nvidia_readonly_layout import plan_readonly_layout, finish_layout_data
        nvidia_readonly_plan = plan_readonly_layout(
            module, (call for calls in encoded_calls.values() for call in calls), readonly_layout_policy)
        encoded_calls = {
            name: prepare_kernel_calls(bindings[name]["call_abi"]["kernel_calls"], function_name=name,
                                       readonly_device_views=nvidia_readonly_plan.views,
                                       owned_pool_origins=owned_origins.get(name))
            for name in function_names
        }
    trace_plan = None
    if triton_trace is not None:
        from .trace_config import TracePlan
        trace_plan = TracePlan(module, bindings, encoded_calls, triton_trace)
    _verify_package_call_closure(package_plan, bindings)

    cooperative = bool(plan_launch.get("cooperative_grid", False))
    mesh = (
        _mesh_context(launch.get("grid_mesh"))
        if cooperative
        else _local_mesh(launch.get("grid_mesh"))
    )
    # A chip/node coordinate comes from the FlagCX runtime context rather
    # than the per-rank Triton launch grid.  Thread that context through every
    # generated wrapper as one tuple of runtime handles so nested calls and
    # kernel helpers observe the same topology without address specialization.
    needs_device_dptr = bool(mesh.get("mesh_requires_device_dptr", False))
    if needs_device_dptr:
        for binding in bindings.values():
            binding["signature"] = [*binding["signature"], "device_dptr"]
        for calls in encoded_calls.values():
            for call in calls:
                descriptor_parameters = tuple(call.get("descriptor_parameters", ()))
                pointer_count = len(call["arguments"]) - len(descriptor_parameters)
                pointer_arguments = list(call["arguments"][:pointer_count])
                descriptors = list(call["arguments"][pointer_count:])
                if "device_dptr" not in pointer_arguments:
                    pointer_arguments.append("device_dptr")
                call["arguments"] = [*pointer_arguments, *descriptors]
                call["signature"] = ", ".join(call["arguments"])

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
        pool_function_arguments=pool_function_arguments,
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
    device_function_names = {
        definition["function"]
        for definition in source_schedule.get("device_functions", ())
    } if source_schedule is not None else set()
    for name, calls in encoded_calls.items():
        for call in calls:
            call["device_function_parent"] = name in device_function_names
    # A stage the schedule invokes once per loop trip shares one pipe instance
    # across trips, so its template renders a trip-base offset and leaves the
    # stream open; mark those wrappers before any template sees them.
    if source_schedule is not None:
        trip_scoped, trip_dynamic = _trip_scoped_stage_calls(source_schedule)
        for call in wrappers:
            call_id = str(call.get("call"))
            if call_id in trip_scoped:
                call["trip_scoped_pipe"] = True
            if call_id in trip_dynamic:
                call["trip_scoped_dynamic"] = True
    grid_barrier_axis_groups = _attach_grid_barrier_axis_groups(
        entry_events,
        source_schedule,
        mesh,
    )
    device_window_storage = _device_window_storage(wrappers, root_binding)
    # Ordinary entries execute one program. Cooperative wrappers retain the
    # chip-local reduction groups derived from their physical buffer ABI;
    # logical mesh_size also counts device/node replicas of the allocation.
    if not cooperative:
        for call in wrappers:
            if call["family"] == "add_norm_stats":
                call["owner_count"] = 1
                call["owner_index"] = "0"
            elif call["family"] == "greedy_sample":
                call["partial_owner_count"] = 1
                call["partial_owner_index"] = "0"

    platform, architecture = module_template_target(module)
    readonly_layout = (None if nvidia_readonly_plan is None else
                       dict(nvidia_readonly_plan.data["device_layout"]))
    if platform == "corex" and architecture == "bi_v150" and int(plan_launch["num_warps"]) == 16:
        from triton.flagmega.codegen.triton.corex_sme_abi import attach_flat_sme_calls
        attach_flat_sme_calls(encoded_calls, source_schedule)
        if specialize_readonly_layout:
            from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
            readonly_layout = select_readonly_layouts(module, wrappers)
    amd_prefetch = add_next_invocation_qkv_prefetch(
        source_schedule["device_functions"] if source_schedule else (),
        encoded_calls,
        platform=platform,
        architecture=architecture,
    )
    if cooperative:
        amd_prefetch = add_split_k_weight_tile_prefetch(
            wrappers, mesh, platform=platform, architecture=architecture,
        ) or amd_prefetch
    if (platform != "nvidia" or architecture != "sm90") and any(
        call.get("decode_schedule") == "bf16_mma" for calls in encoded_calls.values() for call in calls
    ):
        raise CodegenError("DSA bf16_mma requires the NVIDIA SM90 target.")
    if (platform != "nvidia" or architecture != "sm90") and any(
        call.get("kda_reduction_layout") == "warp_k" for calls in encoded_calls.values() for call in calls
    ):
        raise CodegenError("KDA warp_k requires the NVIDIA SM90 target.")
    if (platform != "nvidia" or architecture != "sm90") and any(
        call.get("norm_stats_reduction_algorithm") == "fma_1024"
        for calls in encoded_calls.values() for call in calls
    ):
        raise CodegenError("NormStats fma_1024 requires the NVIDIA SM90 target.")
    if (platform != "nvidia" or architecture != "sm90") and any(
        call.get("norm_stats_reduction_layout") == "contiguous_8"
        or call.get("norm_partial_reduction") == "global_tree"
        for calls in encoded_calls.values() for call in calls
    ):
        raise CodegenError("Contiguous NormStats/global-tree finalization requires the NVIDIA SM90 target.")
    if int(plan_launch["num_warps"]) < 2 and any(
        call.get("norm_stats_reduction_algorithm") == "fma_1024"
        for calls in encoded_calls.values() for call in calls
    ):
        raise CodegenError("NormStats fma_1024 requires at least two compute warps.")
    # A resumed raw TIR call also has to satisfy the approximation's target
    # contract; candidate filtering alone cannot protect this entry point.
    if any(not block_scaled_precision_supports_target(call.get("semantic_attrs", {}), platform, architecture)
           for calls in encoded_calls.values() for call in calls):
        raise CodegenError("SM90 reciprocal activation quantization requires the NVIDIA SM90 target.")
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
    offset_tables = source_schedule["function_offset_tables"] if source_schedule else []
    entry_signature = [field for name in root_signature
                       for field in (DEVICE_CONTEXT_ARGUMENTS if name == "device_dptr" else (name,))]
    signature = [*entry_signature, *descriptor_names, *(table["name"] for table in offset_tables)]
    if len(signature) != len(set(signature)):
        raise CodegenError(
            "TIR entry ABI requires unique argument names across buffers, "
            "device context, descriptors and function offset tables."
        )
    scalar_arguments = {
        str(value["name"]): str(value["scalar_dtype"])
        for value in root_binding["arguments"]
        if value.get("runtime_value_kind") == "scalar"
    }
    if needs_device_dptr:
        scalar_arguments.update({name: "int64" for name in DEVICE_CONTEXT_HANDLES})
    typed_signature_arguments = [
        f"{name}: tl.constexpr" if needs_device_dptr and name == DEVICE_CONTEXT_MODE else name
        for name in signature
    ]
    typed_signature = ",\n    ".join(
        f"{name}: {emit_triton_scalar_type(scalar_arguments[name])}" if name in scalar_arguments else typed_name
        for name, typed_name in zip(signature, typed_signature_arguments, strict=True)
    )
    external_names = {
        str(value["name"]) for value in root_binding["arguments"]
    }
    descriptor_base = len(entry_signature)
    dynamic_descriptors = [
        descriptor_base + index
        for index, spec in enumerate(descriptor_specs)
        if str(spec["source"]) in external_names
    ]
    descriptor = {
        "schema": TIR_PACKAGE_DESCRIPTOR_SCHEMA,
        "symbol": str(launch["entry"]),
        "grid": [int(value) for value in plan_launch["grid"]],
        "num_warps": int(plan_launch["num_warps"]),
        "dynamic_argument_indices": list(
            range(len(root_binding["arguments"]))
        ) + dynamic_descriptors,
        "signature_arguments": [
            name for name in signature
        ],
        "typed_signature_arguments": typed_signature_arguments,
        "signature": typed_signature,
        "scalar_arguments": list(scalar_arguments),
        "host_tensor_descriptor_specs": plain_package_value(
            descriptor_specs
        ),
        "function_offset_tables": plain_package_value(offset_tables),
        "function_schedule": plain_package_value(root_schedule),
        # The logical context is not an external buffer or memory-pool root.
        # Only the generated entry expands it into two runtime scalar handles.
        "runtime_binding": plain_package_value({
            **root_binding,
            "signature": [
                name for name in root_binding["signature"]
                if name != "device_dptr"
            ],
        }),
        "replicated_block_runtime_pool": replicated_block_runtime_pool,
        "render_calls": wrappers,
        **({"triton_readonly_device_layout": readonly_layout} if readonly_layout is not None else {}),
        **({"triton_readonly_layout": finish_layout_data(nvidia_readonly_plan, plain_package_value(descriptor_specs))}
           if nvidia_readonly_plan is not None else {}),
        "entry_events": entry_events,
        "pipeline_schedule": pipeline_schedule,
        "device_functions": source_schedule["device_functions"] if source_schedule else [],
        "amd_prefetch": amd_prefetch,
        "use_tle": pipeline_schedule is not None or cooperative,
        "kernel_templates": kernel_templates,
        "kernel_template_specs": plain_package_value(raw_templates),
        "platform": platform,
        "target_architecture": architecture,
        "distributed_entry": cooperative,
        "distributed_device_dptr": needs_device_dptr,
        "device_context_abi": DEVICE_CONTEXT_ABI if needs_device_dptr else None,
        # A kernel that reads a peer chip's instance addresses it inside the
        # FlagCX symmetric window, so the runtime has to register this pool
        # itself rather than a placeholder allocation.
        "device_window_storage": device_window_storage,
        "grid_barrier_axis_groups": grid_barrier_axis_groups,
        **mesh,
        **_shared_template_context(wrappers),
    }
    if trace_plan is not None:
        trace_plan.finish(descriptor)
    descriptor.update(describe_kernel_programs(module, package_plan, descriptor))
    from .r004_native_model import project_model_programs
    descriptor = project_model_programs(module, descriptor)
    if "persistent_window_resources" in module.metadata:
        from triton.flagmega.ir.persistent_window import plan_window_resources

        descriptor["persistent_window_resources"] = plan_window_resources(
            module.metadata["persistent_window_resources"], root_binding["pools"],
            descriptor.get("kernels"), device_window_storage,
        )
    return descriptor


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


def _trip_scoped_stage_calls(source_schedule):
    """Return call ids of stages invoked once per counted-loop trip.

    The first set scales its pipe iteration by a statically known commit
    count; the second has to compute its own base at run time because the
    stream length is only known on the device.
    """

    found: set[str] = set()
    dynamic: set[str] = set()

    def walk(events):
        for event in events or ():
            if event.get("trip_scoped_pipe"):
                found.add(str(event.get("call")))
            if event.get("trip_scoped_dynamic"):
                dynamic.add(str(event.get("call")))
            if event.get("kind") == "loop":
                walk(event.get("events"))
    for role in ("consumer_events", "producer_events", "auxiliary_events"):
        walk(source_schedule.get(role))
    for definition in source_schedule.get("device_functions", ()) or ():
        schedule = definition.get("schedule") or {}
        for role in ("consumer_events", "producer_events", "auxiliary_events"):
            walk(schedule.get(role))
    return found, dynamic


def prepare_tir_package(
    descriptor: dict[str, object], renderer_version: str
) -> dict[str, object]:
    """Render each namespace once for both the complete and split sources."""
    if not descriptor.get("programs"):
        return descriptor
    context = {
        **descriptor,
        "renderer_version": renderer_version,
        "entry_template": "entrypoints/call_graph.py.jinja",
    }
    registry = TritonTemplateRegistry()
    mesh_contexts = any("mesh_context" in program for program in descriptor["programs"])
    programs = []
    for program in descriptor["programs"]:
        if program.get("entry_alias") or program.get("rendered_source"):
            programs.append(program)
            continue
        if mesh_contexts:
            local = {**context, **program, **program.get("mesh_context", {}), "programs": ()}
        else:
            # Match the explicit namespace bindings in programs.py.jinja.
            inherited = ("symbol", "num_warps", "entry_events", "render_calls", "device_functions",
                         "signature", "signature_arguments", "typed_signature_arguments",
                         "scalar_arguments", "device_context_abi")
            local = {**context, "programs": (),
                     **{name: program[name] if name in program else context[name] for name in inherited},
                     "pipeline_schedule": program.get("pipeline_schedule"),
                     "unspecialized_pointer_arguments": program.get("unspecialized_pointer_arguments", ()),
                     "scoped_entry_arguments": program.get("scoped_entry_arguments", ()),
                     "entry_inline_function": program.get("entry_inline_function"),
                     "tasks_per_owner": program.get("tasks_per_owner", 1)}
        programs.append({**program, "rendered_source": registry.render("module.py.jinja", local)})
    return {**descriptor, "programs": programs}


def render_tir_package(
    descriptor: dict[str, object], renderer_version: str
) -> str:
    context = {
        **descriptor,
        "renderer_version": renderer_version,
        "entry_template": "entrypoints/call_graph.py.jinja",
    }
    programs = descriptor.get('programs', ())
    if programs:
        # Each outlined entry has its own complete template namespace and
        # schedule transformations. Assembling already rendered namespaces
        # must not run those transformations again on the package descriptor.
        if any(not p.get('rendered_source') and not p.get('entry_alias') for p in programs):
            context['programs'] = prepare_tir_package(descriptor, renderer_version)['programs']
        return TritonTemplateRegistry().render("programs.py.jinja", context)
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
        local_actuals = emit_function_call_arguments(module, binding, event, callee_binding=bindings[callee])
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
    if "trace_layer_actual" in call:
        from .trace_config import LAYER_ARGUMENT
        actual_by_formal = {
            **actual_by_formal,
            LAYER_ARGUMENT: _substitute(str(call["trace_layer_actual"]), actual_by_formal),
        }
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
        "participation_active": str(call.get("participation_active", "True")),
        "barrier_before": (
            call["execution_kind"] == "collective" or bool(dependencies)
        ),
        "barrier_scope": barrier_scope,
        "barrier_axis_group_axes": barrier_axis_group_axes,
        "memory_dependencies": plain_package_value(dependencies),
        "symbol": str(call["symbol"]),
        "pipeline_channels": call.get("pipeline_channels", ()),
        # How many pipe iterations one invocation consumes; a counted
        # loop needs it to advance the stage's iteration base per trip.
        "pipeline_commits_per_invocation": call.get(
            "pipeline_commits_per_invocation", 0
        ),
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
    dependencies = tuple(dependencies)
    # A device dependency outranks the collective promotion below: a chip-local
    # grid barrier orders nothing between chips.
    if any(str(value.get("scope", "grid")) == "device" for value in dependencies):
        return "device", ()
    if force_full_grid:
        return "grid", ()
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
    if "device" in {lhs[0], rhs[0]}:
        return "device", ()
    if lhs[0] == "block":
        return rhs
    if rhs[0] == "block":
        return lhs
    if not lhs[1] or not rhs[1]:
        return "grid", ()
    return "grid", tuple(sorted(set((*lhs[1], *rhs[1]))))


def _device_window_storage(
    wrappers: list[dict[str, object]],
    root_binding: Mapping[str, object],
) -> str | None:
    """Return the pool a kernel reads through peer chips, if any.

    Remote reads address the peer's instance at this pool's own offsets, so the
    registered FlagCX symmetric window has to be the pool allocation itself.
    One window exists per process, so two pools read remotely cannot both be
    registered.
    """

    storages = {
        str(leaf["remote_window_pool"])
        for call in wrappers
        for leaf in (call, *call.get("leaves", ()))
        if "remote_window_pool" in leaf
    }
    if not storages:
        return None
    if len(storages) > 1:
        raise CodegenError(
            "One FlagCX symmetric window exists per process, but cross-chip "
            f"reads span memory spaces {sorted(storages)}."
        )
    storage = storages.pop()
    pools = {str(pool["storage"]): pool for pool in root_binding["pools"]}
    pool = pools.get(storage)
    if pool is None:
        raise CodegenError(
            f"Cross-chip reads target memory space {storage!r}, which the entry "
            "runtime binding does not allocate as a pool."
        )
    if int(pool.get("scope_count", 1)) != 1:
        raise CodegenError(
            f"Cross-chip reads require a chip-scoped pool, but {storage!r} is "
            f"replicated per {pool.get('scope')}."
        )
    return storage


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
            "mesh_axis_sources": (),
            "mesh_coordinate_strides": (),
            "mesh_device_extent": 1,
            "mesh_node_extent": 1,
            "mesh_requires_device_dptr": False,
            "mesh_topology_repr": "{}",
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
    if "b" not in levels:
        raise CodegenError("Triton distributed launch has no physical block axis.")
    # ``c``/``d`` coordinates identify a device/node rank through FlagCX;
    # only ``b`` coordinates are materialized by the per-rank Triton grid.
    # Keep placement rank/order for address arithmetic while exposing the
    # canonical TLE names for runtime shard queries.
    axis_names = tuple(
        "device" if level == "c" else
        "node" if level == "d" else
        f"block_{name}"
        for name, level in zip(names, levels, strict=True)
    )
    if len({name for name, level in zip(axis_names, levels, strict=True) if level == "b"}) != levels.count("b"):
        raise CodegenError("Triton physical grid-mesh axis names must be unique.")
    physical_names = tuple(
        axis_names[index]
        for index, level in enumerate(levels)
        if level == "b"
    )
    device_extent = prod(
        hierarchy[index] for index, level in enumerate(levels) if level == "c"
    )
    node_extent = prod(
        hierarchy[index] for index, level in enumerate(levels) if level == "d"
    )
    topology: dict[str, object] = {}
    if node_extent > 1:
        topology["node"] = node_extent
    if device_extent > 1:
        topology["device"] = device_extent
    topology["block"] = [
        (axis_names[index], hierarchy[index])
        for index, level in enumerate(levels)
        if level == "b"
    ]
    mesh_axis_sources = tuple(
        "device" if level == "c" and hierarchy[index] > 1 else
        "node" if level == "d" and hierarchy[index] > 1 else
        None
        for index, level in enumerate(levels)
    )
    mesh_coordinate_strides = tuple(
        prod(
            hierarchy[next_index]
            for next_index in range(index + 1, len(hierarchy))
            if mesh_axis_sources[next_index] == mesh_axis_sources[index]
        )
        if mesh_axis_sources[index] is not None else 1
        for index in range(len(hierarchy))
    )
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
        "mesh_axis_sources": mesh_axis_sources,
        "mesh_coordinate_strides": mesh_coordinate_strides,
        "mesh_device_extent": device_extent,
        "mesh_node_extent": node_extent,
        "mesh_requires_device_dptr": device_extent > 1 or node_extent > 1,
        "mesh_topology_repr": repr(topology),
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
        "gdn_helper_variant": recurrent.get("variant", "persistent"),
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


def _describe_r004_phase_package(module, package_plan, bindings):
    """Lower a selected coarse ABI into normal native entry template namespaces."""
    import hashlib
    import json
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import (
        standalone_request, validate_phase_plan, phase_runtime_binding, verify_phase_runtime_binding,
        phase_writer_proof, phase_plan_for, PROGRAM_SCHEMA, IMPLEMENTATION, DYNAMIC_ROOTS, PLAN_BODY_SHA256, DELTA_SHA256,
    )
    from triton.flagmega.ir.distributed_type import Placement
    from triton.flagmega.codegen.triton.kernel_programs import _task_owner_partition
    request=standalone_request(module)
    if request is None:
        raise CodegenError('Phase implementation requires an explicit standalone request.')
    raw_calls=[call for binding in bindings.values() for call in binding['call_abi']['kernel_calls']]
    if len(raw_calls)!=1 or raw_calls[0].get('implementation')!=phase_plan_for(request['module'])['implementation']:
        raise CodegenError('Standalone C must contain exactly one selected coarse GDN call.')
    raw=raw_calls[0]
    parameters=raw.get('parameters',{})
    if raw.get('facts',{}).get('internal_grid_barriers')!=0 or 'grid_sync' in raw.get('facts',{}).get('requires',()):
        raise CodegenError('Selected coarse phase contract contains an internal grid barrier.')
    plan=validate_phase_plan(parameters.get('phase_plan',raw.get('semantic_attrs',{}).get('phase_plan')))
    DYNAMIC_ROOTS = plan['root_dynamic_indices']
    binding=module.metadata.get('r004_runtime_binding')
    if binding is None:
        binding=phase_runtime_binding(request['readonly_layer'],module.metadata.get('r004_readonly_provenance',{}),plan)
    verify_phase_runtime_binding(binding)
    if binding['readonly_layer']!=request['readonly_layer']:
        raise CodegenError('Phase runtime readonly row differs from owning policy.')
    _verify_r004_bufferized_abi(module,raw,plan)
    _verify_r004_native_definitions(module,plan)
    from triton.flagmega.codegen.triton.kernels.gdn_recurrent.r004_contract_v4 import native_arguments
    formals=native_arguments(plan)
    programs=[]; launches=[]; render_calls=[]
    for entry in plan['entries']:
        phase=entry['phase']; names=formals[phase]
        shape=entry['execution_mesh']['shape']
        domain=Placement(tuple(shape),'xyz'[:len(shape)],'b'*len(shape)).to_data()
        selected={**raw,'call':raw['call']+'.'+phase,'arguments':names,
                  'parameters':{**parameters,'phase_plan':plan,'r004_phase':phase,
                                'program_grid':entry['grid'],'program_owner_mapping':'x-fast'},
                  'facts':{**raw.get('facts',{}),'internal_grid_barriers':0,
                           'scalar_writer_proof':phase_writer_proof(phase,plan)}}
        call=prepare_kernel_calls([selected],function_name=module.entry)[0]
        _task_owner_partition({**entry,'data_placement':domain},{'grid':entry['grid']},[call])
        if call['internal_grid_barriers'] or 'grid_sync' in call.get('facts',{}).get('requires',()):
            raise CodegenError('Phase ordinary entry cannot contain an internal barrier.')
        render_calls.append(call)
        typed=[name+': tl.int32' if name=='Layer' else (name+': tl.float32' if name in ('Scale','Bias') else name) for name in names]
        launch={**{key:plain_package_value(entry[key]) for key in (
            'symbol','grid','num_warps','maxnreg','enable_static_shared','requires_grid_wide_sync',
            'program_kind','depends_on','condition','dynamic_argument_indices','argument_bindings','execution_mesh')},
                'calls':[selected['call']],'data_placement':domain,'stream_index':0,
                'runtime_binding':{'signature':names,'root_dynamic_indices':DYNAMIC_ROOTS.copy(),
                                   'argument_bindings':entry['argument_bindings']},
                'scalar_writer_proof':phase_writer_proof(phase,plan),'internal_grid_barriers':0}
        launches.append(launch)
        programs.append({**launch,'signature_arguments':names,'typed_signature_arguments':typed,
                         'signature':',\n    '.join(typed),'scalar_arguments':[n for n in names if n in ('Layer','Scale','Bias')],
                         'render_calls':[call],'device_functions':[],'pipeline_schedule':None,
                         'entry_events':[{'kind':'tir.kernel_call','call':selected['call'],'family':call['family'],
                             'variant':call['variant'],'symbol':call['symbol'],'arguments':', '.join(names),
                             'barrier_before':False,'execution_kind':'ordinary'}],
                         'mesh_context':{**_local_mesh(), 'use_tle':True,'distributed_entry':False,
                                         'grid_barrier_axis_groups':[]}})
    raw_templates=package_plan['kernel_templates']
    platform,architecture=module_template_target(module)
    registry=TritonTemplateRegistry()
    templates=[registry.resolve(KernelTemplateSpec(str(v['kernel']),str(v['variant']),platform,architecture)) for v in raw_templates]
    # Normal source/toolchain artifact hashes are checked by loader. This key
    # adds the interface, selected source identity and ALL entry compile options.
    key_material={'interface':plan['schema'],'module_semantic_hash':module.semantic_hash,
                  'plan_body_sha256':plan['plan_body_sha256'],'delta_sha256':DELTA_SHA256,
                  'variants':plan['variants'],'runtime_binding':binding,
                  'options':launches,'templates':raw_templates,'toolchain':module.metadata.get('toolchain',{}),
                  'renderer_sources':_r004_source_hashes(plan),'runtime_toolchain':_r004_toolchain_identity()}
    key=hashlib.sha256(json.dumps(plain_package_value(key_material),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    signature=[r['name'] for r in plan['roots']]
    return {'schema':TIR_PACKAGE_DESCRIPTOR_SCHEMA,'symbol':plan['entries'][0]['symbol'],'grid':[48,1,1],
            'num_warps':4,'dynamic_argument_indices':DYNAMIC_ROOTS.copy(),'signature_arguments':signature,
            'typed_signature_arguments':[n+': tl.int32' if n in ('gdn_layer','clear_required') else
                    (n+': tl.float32' if n in ('clear_scale','clear_bias') else n) for n in signature],
            'signature':', '.join(signature),'scalar_arguments':['gdn_layer'],'runtime_binding':plain_package_value(binding),
            'host_tensor_descriptor_specs':[],'function_offset_tables':[], 'function_schedule':{},
            'render_calls':render_calls,'entry_events':[],'device_functions':[],'pipeline_schedule':None,
            'kernel_templates':templates,'kernel_template_specs':plain_package_value(raw_templates),
            'platform':platform,'target_architecture':architecture,'use_tle':True,
            'distributed_entry':False,'distributed_device_dptr':False,'device_context_abi':None,
            'device_window_storage':None,'grid_barrier_axis_groups':[], 'replicated_block_runtime_pool':False,
            'kernel_program_schema':PROGRAM_SCHEMA,'kernels':launches,'programs':programs,
            'phase_interface':plan['schema'],'phase_cache_key':key,'phase_variants':plan['variants'],
            'plan_body_sha256':plan['plan_body_sha256'],'delta_sha256':DELTA_SHA256, **_local_mesh()}


def _r004_source_hashes(plan=None):
    import hashlib
    from pathlib import Path
    root=Path(__file__).parent
    paths=[root/'tir_package.py', root/'kernel_programs.py', root/'kernel_call_renderers.py',
           root/'kernels/gdn_recurrent/r004_reference_v4.py.jinja',
           root/'kernels/gdn_recurrent/r004_helpers_v4.py.jinja',
           root/'kernels/gdn_recurrent/r004_contract_v4.py',
           root.parent.parent/'targets/nvidia/gdn.py',
           root.parent.parent/'passes/tir/materialize_kernel_definitions.py',
           root.parent.parent/'rules/ntt/r004_gdn_phase_bindings.py',
           root/'execution_programs.py',root.parent.parent/'runtime/module.py',
           root.parent.parent/'targets/ntt.py',root.parent.parent/'targets/portable_triton_implementations.py',
           root.parent.parent/'ir/execution_mesh.py',root.parent.parent/'ir/tir/serialization.py',
           root.parent.parent/'ir/ops/nn/gdn_recurrent_core.py']
    if plan is not None and plan['module'] == 'r004_T_full_p5_v4':
        paths += [root/'kernels/gdn_recurrent/r004_column_v4.py.jinja', root.parent.parent/'rules/ntt/r004_column_phase_bindings.py', root.parent.parent/'runtime/r004_phase_pool.py']
    if plan is not None and plan['module'] == 'r004_T_full_p5_v6':
        paths += [root/'kernels/gdn_recurrent/r004_column_v6.py.jinja',
                  root/'kernels/gdn_recurrent/r004_norm_v6.py.jinja',
                  root.parent.parent/'rules/ntt/r004_column_phase_bindings_v6.py',
                  root.parent.parent/'runtime/r004_phase_pool.py']
    return {str(p.relative_to(root.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _verify_r004_bufferized_abi(module,raw,plan):
    """Verify actual coarse buffers and typed phase arena before projecting views."""
    from math import prod
    buffer_plan=verify_buffer_plan(module)
    for side in ('inputs','outputs','workspaces'):
        for parameter in raw.get(side,()):
            for value in parameter.get('buffers',()):
                abi=value.get('abi',{})
                if int(abi.get('pool_scope_stride_bytes',0)):
                    raise CodegenError('Standalone phase buffers must use CHIP storage without scope stride.')
    # Arena may be one aggregate U8 requirement with typed phase views, or nine
    # explicitly resolved F32 allocations. Both must resolve signed offsets.
    workspaces={w['formal']:w for w in raw.get('workspaces',())}
    if 'phase_pool' in workspaces:
        values=workspaces['phase_pool'].get('buffers',())
        if len(values)!=1:
            raise CodegenError('Phase pool must have one typed backing buffer.')
        buffer=buffer_plan.buffer_map[values[0]['actual']]
        if buffer.dtype.value!='uint8' or int(buffer.nbytes)!=plan['pool']['nbytes']:
            raise CodegenError('Phase pool backing span/dtype differs from frozen ABI.')
    else:
        raise CodegenError('Selected coarse GDN is missing its typed phase_pool workspace.')
    state=next((p for p in raw['inputs'] if p['formal']=='state'),None)
    if state is None or len(state.get('buffers',()))!=1:
        raise CodegenError('Standalone C state must expose exactly its recurrent layer view.')
    state_buffer=buffer_plan.buffer_map[state['buffers'][0]['actual']]
    bindings=state_buffer.offset_bindings
    from triton.flagmega.ir.dim_expr import dim
    from triton.flagmega.ir.types import VectorType
    if (not isinstance(state_buffer.dtype,VectorType) or state_buffer.dtype.elem_type.value!='float32'
            or tuple(state_buffer.dtype.lanes)!=(4,) or state_buffer.shape!=(1,48,128,32)
            or state_buffer.strides!=(196608,4096,32,1)
            or state_buffer.nbytes!=3145728 or state_buffer.mem_span.buffer.nbytes!=150994944
            or len(bindings)!=1 or bindings[0][1]!='gdn_layer'
            or not state_buffer.mem_span.start.equivalent(dim(bindings[0][0])*3145728)):
        raise CodegenError('Phase state layer view does not bind the exact full parent/layer stride.')
    for layer in range(48):
        begin=state_buffer.mem_span.start.evaluate({bindings[0][0]:layer})
        if begin!=layer*3145728 or begin+state_buffer.nbytes>state_buffer.mem_span.buffer.nbytes:
            raise CodegenError('Phase state View escapes its selected layer/parent.')
    outputs={p['formal']:p for p in raw['outputs']}
    y=outputs.get('result_0',{})
    if len(y.get('buffers',()))!=1:
        raise CodegenError('Phase final output root is missing.')
    y_buffer=buffer_plan.buffer_map[y['buffers'][0]['actual']]
    if y_buffer.dtype.value!='bfloat16' or y_buffer.shape!=(1,6144) or y_buffer.nbytes!=12288:
        raise CodegenError('Phase final output root shape/span differs from v4.')
    from triton.flagmega.ir.distributed_inference import tensor_of
    expected_formals={'qkv':0,'z':1,'projection_input':2,'b_weight':6,'a_weight':7,'a_log':8,'dt_bias':9,'norm_weight':10}
    for formal,index in expected_formals.items():
        parameter=next((p for p in raw['inputs'] if p['formal']==formal),None)
        if parameter is None or len(parameter.get('buffers',()))!=1:
            raise CodegenError('Phase input root missing from typed bufferized ABI.')
        buffer=buffer_plan.buffer_map[parameter['buffers'][0]['actual']]
        root=plan['roots'][index]
        from triton.flagmega.ir.types import VectorType
        dtype=buffer.dtype.elem_type if isinstance(buffer.dtype,VectorType) else buffer.dtype
        if dtype.value!=root['dtype'] or int(buffer.nbytes)!=root['nbytes']:
            raise CodegenError('Phase input typed span differs from signed scalar root inventory.')


def _r004_toolchain_identity():
    from pathlib import Path
    import triton
    from triton._C import libtriton
    from triton.backends.nvidia import compiler
    paths=[Path(libtriton.__file__),Path(compiler.__file__)]
    return {'triton_version':triton.__version__,
            'files':{p.name:_r004_cached_file_digest(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths}}


from functools import lru_cache as _r004_lru_cache


@_r004_lru_cache(maxsize=8)
def _r004_cached_file_digest(path, size, mtime):
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_r004_native_definitions(module, plan):
    from triton.flagmega.ir.memory_effect import MemoryAccessMode
    from triton.flagmega.ir import TensorType
    from triton.flagmega.ir.tir import PrimParameterRole
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import phase_writer_proof
    views={v['byte_offset']:v for v in plan['pool']['views']}
    from triton.flagmega.codegen.triton.kernels.gdn_recurrent.r004_contract_v4 import native_arguments
    NATIVE_ARGUMENTS = native_arguments(plan)
    for entry in plan['entries']:
        definition=module.kernel_callable_map.get(entry['symbol'])
        if definition is None or len(definition.parameters)!=len(entry['argument_bindings']):
            raise CodegenError('Missing typed native phase definition/signature.')
        dispatch=definition.dispatch
        if (dispatch.parameters.get('r004_phase')!=entry['phase']
                or list(dispatch.parameters.get('program_grid',()))!=entry['grid']
                or dispatch.parameters.get('program_owner_mapping')!='x-fast'
                or dispatch.facts.get('internal_grid_barriers')!=0
                or plain_package_value(dispatch.facts.get('scalar_writer_proof'))!=phase_writer_proof(entry['phase'],plan)):
            raise CodegenError('Typed native phase definition has stale owner/barrier/writer facts.')
        effects=dict(dispatch.memory_effects)
        native_order=NATIVE_ARGUMENTS[entry['phase']]
        by_name={p.name:p for p in definition.parameters}
        if (set(by_name)!=set(native_order) or
                plain_package_value(definition.attrs.get('native_parameter_order'))!=list(native_order)):
            raise CodegenError('Typed native parameter projection differs from the exact named native ABI.')
        for i,(name,binding) in enumerate(zip(native_order,entry['argument_bindings'],strict=True)):
            parameter=by_name[name]
            value=views[binding['byte_offset']] if binding['kind']=='pointer' else plan['roots'][binding['source']]
            tensor=parameter.type
            write_positions={'clear':{0},'gather':{3},'core':{6,8},'pb':{2},'pa':{2},
                             'beta':{1},'bias':{2},'gate':{2},'q':{1},'k':{1},'rms':{2},'y':{2}}
            if plan['module'] in ('r004_T_full_p5_v4', 'r004_T_full_p5_v6'):
                write_positions.update(prep={6,7,8,9}, core={5,7}, y={3})
            role=PrimParameterRole.OUTPUT if i in write_positions[entry['phase']] else PrimParameterRole.INPUT
            mode=(MemoryAccessMode.READ_WRITE if (entry['phase']=='clear' and i==0 or plan['module'] in ('r004_T_full_p5_v4', 'r004_T_full_p5_v6') and entry['phase']=='core' and i==5) else
                  (MemoryAccessMode.WRITE if role is PrimParameterRole.OUTPUT else MemoryAccessMode.READ))
            if (not isinstance(tensor,TensorType) or tensor.dtype.value!=value['dtype']
                    or tuple(d.fixed_value for d in tensor.shape)!=tuple(value['shape'])
                    or parameter.role is not role or parameter.name not in effects
                    or effects[parameter.name].physical_mode!=mode):
                raise CodegenError('Typed phase View/root shape, lifetime or state effect differs from v4.')
