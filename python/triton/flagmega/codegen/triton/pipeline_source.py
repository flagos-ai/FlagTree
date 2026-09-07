# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Role-preserving source schedule for first-class producer/consumer TIR."""

from __future__ import annotations

from dataclasses import dataclass
import keyword
import re
from typing import Callable, Mapping

from triton.flagmega.codegen.triton.function_call import (
    emit_function_call_arguments,
)
from triton.flagmega.codegen.triton.pool_abi import emit_pool_scope_base
from triton.flagmega.codegen.triton.descriptor_resources import DescriptorResources
from triton.flagmega.codegen.triton.pipeline_function_abi import (
    descriptor_requests, instantiate_resources,
)
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir import (
    Barrier,
    Buffer,
    IRModule,
    PipelineDrain,
    PipelineHandoff,
    PipelineStage,
    PrimFunctionCall,
    KernelInvoke,
    ProducerConsumerRegion,
    Sequential,
    execution_calls_of,
    kernel_dispatch_of,
    verify_buffer_plan,
)


KernelEventFactory = Callable[
    [
        str,
        Mapping[str, object],
        Mapping[str, str],
        Mapping[str, tuple[str, int]],
        tuple[str, ...],
        list[Mapping[str, object]],
    ],
    dict[str, object],
]


@dataclass(frozen=True)
class _Context:
    function_name: str
    actual_by_formal: Mapping[str, str]
    storage_roots: Mapping[str, tuple[str, int]]
    shared_by_name: Mapping[str, Buffer]
    call_path: tuple[str, ...]


def build_pipeline_source_schedule(
    module: IRModule,
    bindings: Mapping[str, Mapping[str, object]],
    encoded_calls: Mapping[str, list[dict[str, object]]],
    kernel_event_factory: KernelEventFactory,
    descriptor_instantiator: Callable,
) -> dict[str, object] | None:
    """Project the entry's explicit TIR region into producer/consumer source.

    This is intentionally a visitor over ``ProducerConsumerRegion`` rather
    than a scan for implementation names. Each ExecutionFunction has one
    formal role body; calls instantiate only its physical resource interface.
    """

    execution = module.execution_function_map.get(module.entry)
    if execution is None:
        return None
    definitions: dict[str, dict[str, object]] = {}
    active: list[str] = []

    def build_function(name):
        if name in definitions:
            return definitions[name]
        if name in active:
            raise CodegenError("Recursive pipeline function: " + " -> ".join((*active, name)))
        active.append(name)
        signature = tuple(str(value) for value in bindings[name]["signature"])
        context = _Context(name, {value: value for value in signature},
                           {value: (value, 0) for value in signature}, {}, ())
        function = module.execution_function_map[name]
        body = _top_level_region(function.body, name) or function.body
        builder = _PipelineSourceBuilder(
            module, bindings, encoded_calls, kernel_event_factory,
            build_function, descriptor_instantiator,
        )
        builder.descriptor_specs = DescriptorResources(formal_origins=name != module.entry)
        schedule = builder.build(body, context)
        result = {
            "function": name,
            "symbol": f"_flagmega_function_{_identifier(name)}",
            "signature_arguments": [*signature, *builder.descriptor_specs.parameters()],
            "host_tensor_descriptor_specs": builder.descriptor_specs,
            "schedule": schedule,
        }
        definitions[name] = result
        active.pop()
        return result

    root = build_function(module.entry)
    return {
        **root["schedule"],
        "host_tensor_descriptor_specs": root["host_tensor_descriptor_specs"],
        "device_functions": [definition for name, definition in definitions.items() if name != module.entry],
    }


class _PipelineSourceBuilder:
    def __init__(
        self,
        module: IRModule,
        bindings: Mapping[str, Mapping[str, object]],
        encoded_calls: Mapping[str, list[dict[str, object]]],
        kernel_event_factory: KernelEventFactory,
        function_builder: Callable,
        descriptor_instantiator: Callable,
    ) -> None:
        self.module = module
        self.bindings = bindings
        self.encoded_calls = encoded_calls
        self.kernel_event_factory = kernel_event_factory
        self.function_builder = function_builder
        self.descriptor_instantiator = descriptor_instantiator
        self.descriptor_specs = DescriptorResources()
        self.function_instances: dict[tuple[str, ...], dict[str, object]] = {}
        self.drained_stage_ids: set[str] = set()
        self.stages: list[dict[str, object]] = []
        self.stage_by_key: dict[tuple[str, ...], dict[str, object]] = {}
        self.logical_stage_leaves: dict[tuple[str, ...], tuple[str, ...]] = {}
        self.handoffs: list[dict[str, object]] = []
        self.handoff_by_key: dict[tuple[str, ...], dict[str, object]] = {}
        self.event_cache: dict[tuple[str, ...], dict[str, object]] = {}
        self.consumer_events: list[dict[str, object]] = []
        self.producer_events: list[dict[str, object]] = []
        self.auxiliary_events: list[dict[str, object]] = []
        self.auxiliary_event_keys: set[tuple[str, ...]] = set()

    def build(
        self, region: ProducerConsumerRegion | Sequential, context: _Context
    ) -> dict[str, object]:
        self._visit_body(
            region.consume_body if isinstance(region, ProducerConsumerRegion) else region, "consumer", context,
            (context.function_name,), self.consumer_events,
        )
        if isinstance(region, ProducerConsumerRegion):
            self._visit_body(
                region.produce_body, "producer", context,
                (context.function_name,), self.producer_events,
            )
        if isinstance(region, ProducerConsumerRegion) and not self.stages:
            raise CodegenError(
                f"Pipeline entry @{context.function_name} has no leaf transfer stage."
            )
        self._resolve_drains(self.consumer_events)
        self._resolve_drains(self.producer_events)
        self._resolve_drains(self.auxiliary_events)
        producer_warps = {int(value["producer_warps"]) for value in self.stages}
        producer_registers = {
            int(value["producer_registers"]) for value in self.stages
        }
        if len(producer_warps) > 1 or len(producer_registers) > 1:
            raise CodegenError(
                "One producer task cannot combine stages with different worker "
                "warp/register contracts."
            )
        auxiliary_stages = [
            value for value in self.stages
            if value["auxiliary_consumer"] is not None
        ]
        auxiliary_warps = {
            int(value["auxiliary_consumer"]["warps"])
            for value in auxiliary_stages
        }
        auxiliary_registers = {
            int(value["auxiliary_consumer"]["registers"])
            for value in auxiliary_stages
        }
        if (
            len(auxiliary_warps) > 1
            or len(auxiliary_registers) > 1
        ):
            raise CodegenError(
                "One auxiliary consumer task cannot combine stages with "
                "different warp/register contracts."
            )
        shared_arena_required_nbytes = max((
            int(workspace["offset_bytes"]) + int(workspace["nbytes"])
            for stage in self.stages
            for workspace in stage["workspaces"]
        ), default=0)
        shared_arena_alignment_bytes = max((
            max(int(workspace["alignment_bytes"]),
                int(workspace["allocation_alignment_bytes"]))
            for stage in self.stages
            for workspace in stage["workspaces"]
        ), default=16)
        shared_space = next(
            value for value in verify_buffer_plan(self.module).memory_spaces
            if value.name == "shared"
        )
        try:
            shared_arena_nbytes = shared_space.allocation_bytes(
                shared_arena_required_nbytes
            )
        except IRSchemaError as error:
            raise CodegenError(
                "Triton Shared arena allocation violates the target "
                f"memory-space contract: {error}"
            ) from error
        return {
            "schema": "flagmega.triton-pipeline-source-schedule/v3",
            "drained_stage_ids": tuple(sorted(self.drained_stage_ids)),
            "stages": self.stages,
            "handoffs": self.handoffs,
            "consumer_events": self.consumer_events,
            "producer_events": self.producer_events,
            "auxiliary_events": self.auxiliary_events,
            "consumer_endpoints": [
                *(
                    str(channel["reader_name"])
                    for stage in self.stages
                    for channel in stage["channels"]
                ),
                *(
                    str(workspace["variable"])
                    for stage in self.stages
                    for workspace in stage["consumer_workspaces"]
                ),
                *(
                    str(endpoint)
                    for stage in self.stages
                    if stage["auxiliary_consumer"] is not None
                    for endpoint in stage["auxiliary_consumer"][
                        "primary_endpoints"
                    ]
                ),
                *(
                    str(value["writer_name"])
                    for value in self.handoffs
                    if value["semantic"]
                ),
            ],
            "producer_endpoints": [
                *(
                    str(channel["writer_name"])
                    for stage in self.stages
                    for channel in stage["channels"]
                ),
                *(
                    str(value["reader_name"])
                    for value in self.handoffs
                    if value["semantic"]
                ),
            ],
            "auxiliary_endpoints": [
                str(endpoint)
                for stage in self.stages
                if stage["auxiliary_consumer"] is not None
                for endpoint in stage["auxiliary_consumer"]["endpoints"]
            ],
            "shared_arena_nbytes": shared_arena_nbytes,
            "shared_arena_alignment_bytes": shared_arena_alignment_bytes,
            "shared_arena_required_nbytes": shared_arena_required_nbytes,
            "producer_warps": producer_warps.pop() if producer_warps else None,
            "producer_registers": producer_registers.pop() if producer_registers else None,
            "auxiliary_consumer_warps": (
                None if not auxiliary_warps else auxiliary_warps.pop()
            ),
            "auxiliary_consumer_registers": (
                None if not auxiliary_registers else auxiliary_registers.pop()
            ),
        }

    def _visit_body(
        self,
        body: Sequential,
        role: str,
        context: _Context,
        region_path: tuple[str, ...],
        output: list[dict[str, object]],
    ) -> None:
        for statement in body.fields:
            if isinstance(statement, Sequential):
                self._visit_body(statement, role, context, region_path, output)
                continue
            if isinstance(statement, Barrier):
                if role != "consumer":
                    raise CodegenError(
                        "A pipeline producer body cannot contain an ordinary barrier."
                    )
                output.append({
                    "kind": "barrier",
                    "scope": "grid" if statement.scope.value == "chip" else "block",
                    "axis_group_axes": tuple(statement.axis_group_axes),
                    "after": tuple(statement.after),
                    "before": statement.before,
                    "hazards": tuple(statement.hazards),
                    "ranges": tuple(value.to_data() for value in statement.ranges),
                })
                continue
            if isinstance(statement, PipelineHandoff):
                handoff = self._handoff((*region_path, statement.handoff_id))
                output.append({
                    "kind": "handoff",
                    "endpoint": handoff[
                        "writer_name" if role == "consumer" else "reader_name"
                    ],
                    "action": "commit" if role == "consumer" else "wait",
                })
                continue
            if isinstance(statement, PipelineDrain):
                output.append({
                    "kind": "drain",
                    "logical_stage": (*region_path, statement.stage_id),
                    "role": role,
                })
                if role == "consumer":
                    self.auxiliary_events.append({
                        "kind": "drain",
                        "logical_stage": (*region_path, statement.stage_id),
                        "role": "auxiliary",
                    })
                continue
            if isinstance(statement, PipelineStage):
                before = len(self.stages)
                before_events = len(output)
                self._visit_stage(
                    statement, role, context, region_path, output
                )
                key = (*region_path, statement.stage_id)
                if role == "consumer":
                    leaves = tuple(
                        str(value["id"]) for value in self.stages[before:]
                    )
                    if not leaves:
                        # A nested stage may reference leaf stages already
                        # registered by an equivalent path only in malformed IR.
                        raise CodegenError(
                            f"Pipeline stage {statement.stage_id!r} produced no "
                            "leaf transfer stages."
                        )
                    # A nested pipeline function exports only pipe endpoints
                    # whose lifetimes remain open at its boundary. Drains in
                    # the callee have already consumed their endpoints and
                    # must not be repeated when the caller drains the
                    # enclosing PipelineStage. This matches nncase's
                    # PipelineDeviceFunctionInterface residual drain ABI.
                    internally_drained: set[str] = set()
                    for event in output[before_events:]:
                        internally_drained.update(event.get("drained_stages", ()))
                        if event["kind"] != "drain":
                            continue
                        inner_key = tuple(
                            str(value) for value in event["logical_stage"]
                        )
                        try:
                            internally_drained.update(
                                self.logical_stage_leaves[inner_key]
                            )
                        except KeyError as error:
                            raise CodegenError(
                                "Nested pipeline drain references an unknown "
                                f"logical stage {inner_key}."
                            ) from error
                    self.logical_stage_leaves[key] = tuple(
                        leaf for leaf in leaves
                        if leaf not in internally_drained
                    )
                continue
            if isinstance(statement, (PrimFunctionCall, KernelInvoke)):
                if role != "consumer":
                    raise CodegenError(
                        "A pipeline producer body may contain calls only through "
                        "PipelineStage."
                    )
                self._visit_call(statement, role, context, output, staged=False)
                continue
            raise CodegenError(
                f"Pipeline source generation does not support "
                f"{type(statement).__name__} in @{context.function_name}; lower "
                "structured task control flow to explicit source-schedule nodes."
            )

    def _visit_stage(
        self,
        stage: PipelineStage,
        role: str,
        context: _Context,
        region_path: tuple[str, ...],
        output: list[dict[str, object]],
    ) -> None:
        operation = stage.operation
        if not isinstance(operation, (PrimFunctionCall, KernelInvoke)):
            raise CodegenError(
                "Entry pipeline source generation expects materialized "
                "PrimFunctionCall stages."
            )
        self._visit_call(operation, role, context, output, staged=True)

    def _visit_call(
        self,
        call: PrimFunctionCall,
        role: str,
        context: _Context,
        output: list[dict[str, object]],
        *,
        staged: bool,
    ) -> None:
        primitive = self.module.kernel_callable_map.get(call.callee)
        if primitive is not None:
            dispatch = kernel_dispatch_of(primitive)
            if dispatch is None:
                raise CodegenError(
                    f"Pipeline call {call.call_id!r} references general TIR "
                    f"PrimFunction @{call.callee}; only a selected kernel ABI is supported."
                )
            is_pipeline = (
                dispatch.microkernel is not None
                and dispatch.microkernel.transfer_pipeline is not None
            )
            if staged != is_pipeline:
                raise CodegenError(
                    f"Pipeline role structure for call {call.call_id!r} differs "
                    "from its selected transfer contract."
                )
            event = self._kernel_event(context, call)
            if not is_pipeline:
                output.append(event)
                return
            stage_plan = self._leaf_stage(context, call, event)
            endpoints = [
                str(channel[
                    "reader_name" if role == "consumer" else "writer_name"
                ])
                for channel in stage_plan["channels"]
            ]
            if role == "consumer":
                endpoints.extend(
                    str(workspace["variable"])
                    for workspace in stage_plan["consumer_workspaces"]
                )
                auxiliary = stage_plan["auxiliary_consumer"]
                if auxiliary is not None:
                    endpoints.extend(
                        str(value) for value in auxiliary["primary_endpoints"]
                    )
            expected = tuple(event[f"pipeline_{role}_parameters"])
            if len(endpoints) != len(expected):
                raise CodegenError(
                    f"Pipeline stage {call.call_id!r} {role} endpoint arity "
                    "differs from its prepared helper ABI."
                )
            output.append({
                **event,
                "kind": "pipeline_kernel_call",
                "role": role,
                "symbol": f"{event['symbol']}__{role}",
                "arguments": _join_arguments(endpoints, str(event["arguments"])),
            })
            if role == "consumer":
                self._append_auxiliary_event(
                    (*context.call_path, call.call_id), event, stage_plan
                )
            return

        execution = self.module.execution_function_map.get(call.callee)
        if execution is None:
            raise CodegenError(
                f"Pipeline call {call.call_id!r} references unknown @{call.callee}."
            )
        nested = self._nested_context(context, call, execution.name)
        region = _top_level_region(execution.body, execution.name)
        if staged != (region is not None):
            raise CodegenError(
                f"Call {call.call_id!r} PipelineStage placement differs from "
                f"ExecutionFunction @{execution.name}."
            )
        key = (*context.call_path, call.call_id)
        instance = self.function_instances.get(key)
        definition = self.function_builder(execution.name)
        if instance is None:
            resources, names = instantiate_resources(
                definition["schedule"], nested.shared_by_name,
                f"_flagmega_{_identifier('__'.join(key))}",
            )
            self.stages.extend(resources["stages"])
            self.handoffs.extend(resources["handoffs"])
            descriptors = self.descriptor_instantiator(
                {"host_tensor_descriptor_requests": descriptor_requests(
                    definition["host_tensor_descriptor_specs"])},
                nested.storage_roots, self.descriptor_specs, key,
            )
            actuals = [
                nested.actual_by_formal[str(formal)]
                for formal in self.bindings[execution.name]["signature"]
            ]
            instance = {
                "names": names,
                "arguments": [*actuals, *descriptors],
                "drained_stages": tuple(names[value] for value in definition["schedule"]["drained_stage_ids"]),
            }
            self.function_instances[key] = instance
            self.drained_stage_ids.update(instance["drained_stages"])
            if definition["schedule"]["auxiliary_events"]:
                self.auxiliary_events.append(self._function_event(call, definition, instance, "auxiliary"))
        output.append(self._function_event(call, definition, instance, role))

    @staticmethod
    def _function_event(call, definition, instance, role):
        names = instance["names"]
        endpoints = [names[value] for value in definition["schedule"][f"{role}_endpoints"]]
        suffix = "auxiliary_consumer" if role == "auxiliary" else role
        return {
            "kind": "function_call",
            "call": call.call_id,
            "callee": call.callee,
            "symbol": f"{definition['symbol']}__{suffix}",
            "arguments": ", ".join((*endpoints, *instance["arguments"])),
            "endpoint_bindings": dict(zip(
                definition["schedule"][f"{role}_endpoints"], endpoints, strict=True,
            )),
            "barrier_before": False,
            "drained_stages": instance["drained_stages"],
        }

    def _kernel_event(
        self, context: _Context, call: PrimFunctionCall
    ) -> dict[str, object]:
        key = (*context.call_path, call.call_id)
        cached = self.event_cache.get(key)
        if cached is not None:
            return dict(cached)
        try:
            encoded = next(
                value for value in self.encoded_calls[context.function_name]
                if str(value["call"]) == call.call_id
            )
        except (KeyError, StopIteration) as error:
            raise CodegenError(
                f"Pipeline call {call.call_id!r} has no encoded kernel ABI in "
                f"@{context.function_name}."
            ) from error
        event = self.kernel_event_factory(
            context.function_name,
            encoded,
            context.actual_by_formal,
            context.storage_roots,
            key,
            self.descriptor_specs,
        )
        # Pipeline source is built only from a materialized ExecutionFunction;
        # all required synchronization is already present as Barrier nodes in
        # that body.  ``barrier_before`` is retained by the flat package path
        # for legacy/pre-schedule modules, but must not be emitted a second
        # time here.
        event["barrier_before"] = False
        event["barrier_scope"] = None
        event["barrier_axis_group_axes"] = ()
        self.event_cache[key] = dict(event)
        return event

    def _leaf_stage(
        self,
        context: _Context,
        call: PrimFunctionCall,
        event: Mapping[str, object],
    ) -> dict[str, object]:
        key = (*context.call_path, call.call_id)
        existing = self.stage_by_key.get(key)
        if existing is not None:
            return existing
        encoded = next(
            value for value in self.encoded_calls[context.function_name]
            if str(value["call"]) == call.call_id
        )
        contract = encoded.get("pipeline_contract")
        local_workspaces = encoded.get("shared_workspaces")
        if not isinstance(contract, Mapping) or not isinstance(
            local_workspaces, (tuple, list)
        ):
            raise CodegenError(
                f"Pipeline stage {call.call_id!r} has no encoded resource contract."
            )
        channels = contract.get("channels")
        prepared_channels = encoded.get("pipeline_channels")
        consumer_workspace_indices = contract.get(
            "consumer_shared_workspace_indices", ()
        )
        if (
            not isinstance(channels, (tuple, list))
            or not channels
            or not isinstance(prepared_channels, (tuple, list))
            or len(prepared_channels) != len(channels)
        ):
            raise CodegenError(
                "A transfer-pipelined source stage must expose matching typed "
                "and prepared channels."
            )
        if not isinstance(consumer_workspace_indices, (tuple, list)):
            raise CodegenError(
                "Transfer-pipeline consumer Shared workspace indices must be a sequence."
            )
        if len(local_workspaces) != len(call.shared_workspace_buffers):
            raise CodegenError(
                f"Pipeline stage {call.call_id!r} Shared ABI arity differs from "
                "its caller-owned buffers."
            )
        actual_buffers = tuple(
            context.shared_by_name.get(value.name, value)
            for value in call.shared_workspace_buffers
        )
        workspaces = []
        stem = _identifier("__".join(key))
        for index, (local, actual) in enumerate(zip(
            local_workspaces, actual_buffers, strict=True
        )):
            shape = tuple(int(value) for value in local["shape"])
            actual_shape = tuple(value.fixed_value for value in actual.dimensions)
            if shape != actual_shape or str(local["dtype"]) != actual.elem_type.value:
                raise CodegenError(
                    f"Pipeline stage {call.call_id!r} Shared workspace {index} "
                    "differs from its selected typed descriptor."
                )
            workspaces.append({
                **dict(local),
                "name": str(local["name"]),
                "buffer_name": actual.name,
                "variable": f"_flagmega_{stem}_shared_{index}",
                "dtype": _triton_dtype(actual.elem_type.value),
                "element_type": actual.elem_type.value,
                "shape": actual_shape,
                "strides": tuple(value.fixed_value for value in actual.strides),
                "offset_bytes": actual.mem_span.absolute_start.fixed_value,
                "nbytes": actual.mem_span.size.fixed_value,
                "physical_buffer": actual.mem_span.buffer.id,
                # The view's layout requirement and its allocation's base
                # alignment are distinct, especially after edit/resume.
                "allocation_alignment_bytes": actual.mem_span.buffer.alignment,
                "matrix_compatible": bool(
                    local.get("matrix_compatible", False)
                ),
            })
        raw_capacity = contract.get("capacity")
        capacity = int(
            encoded["num_stages"]
            if raw_capacity is None
            else raw_capacity
        )
        ordinal = len(self.stages)
        channel_plans = []
        owned_indices: set[int] = set()
        for channel_index, channel in enumerate(channels):
            if not isinstance(channel, Mapping):
                raise CodegenError("Transfer-pipeline channel must be a mapping.")
            channel_name = str(channel.get("name", ""))
            if not channel_name.isidentifier() or keyword.iskeyword(channel_name):
                raise CodegenError(
                    f"Pipeline channel {channel_name!r} is not a Python identifier."
                )
            try:
                indices = tuple(
                    int(value) for value in channel["shared_workspace_indices"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise CodegenError(
                    f"Pipeline channel {channel_name!r} has invalid Shared indices."
                ) from error
            if not indices or any(
                index < 0 or index >= len(workspaces) or index in owned_indices
                for index in indices
            ):
                raise CodegenError(
                    f"Pipeline channel {channel_name!r} has invalid or multiply "
                    "owned Shared workspaces."
                )
            owned_indices.update(indices)
            for index in indices:
                workspace = workspaces[index]
                if (
                    not workspace["shape"]
                    or int(workspace["shape"][0]) != capacity
                ):
                    raise CodegenError(
                        f"Pipeline stage {call.call_id!r} capacity {capacity} "
                        f"differs from Shared workspace {workspace['name']!r} "
                        "leading dimension."
                    )
            channel_stem = (
                stem if len(channels) == 1
                else f"{stem}_{_identifier(channel_name)}"
            )
            prepared = prepared_channels[channel_index]
            if (
                not isinstance(prepared, Mapping)
                or str(prepared.get("name", "")) != channel_name
                or tuple(prepared.get("workspace_names", ()))
                != tuple(workspaces[index]["name"] for index in indices)
            ):
                raise CodegenError(
                    f"Pipeline channel {channel_name!r} prepared helper ABI "
                    "differs from its typed workspace contract."
                )
            raw_fields = prepared.get("fields")
            if not isinstance(raw_fields, (tuple, list)) or len(raw_fields) != len(indices):
                raise CodegenError(
                    f"Pipeline channel {channel_name!r} has no one-to-one pipe fields."
                )
            fields = []
            for field, index in zip(raw_fields, indices, strict=True):
                if (
                    not isinstance(field, Mapping)
                    or str(field.get("workspace_name", ""))
                    != str(workspaces[index]["name"])
                ):
                    raise CodegenError(
                        f"Pipeline channel {channel_name!r} pipe field differs "
                        "from its Shared workspace binding."
                    )
                field_name = str(field.get("name", ""))
                if not field_name.isidentifier() or keyword.iskeyword(field_name):
                    raise CodegenError(
                        f"Pipeline channel {channel_name!r} field {field_name!r} "
                        "is not a Python identifier."
                    )
                fields.append({
                    "name": field_name,
                    "workspace": workspaces[index],
                })
            channel_plans.append({
                "name": channel_name,
                "pipe_name": f"_flagmega_{channel_stem}_pipe",
                "reader_name": f"_flagmega_{channel_stem}_reader",
                "writer_name": f"_flagmega_{channel_stem}_writer",
                "capacity": capacity,
                "fields": fields,
            })
        try:
            consumer_workspaces = [
                workspaces[int(index)] for index in consumer_workspace_indices
            ]
        except (IndexError, TypeError, ValueError) as error:
            raise CodegenError(
                "Transfer pipeline references an invalid consumer Shared workspace."
            ) from error
        consumer_index_set = {int(value) for value in consumer_workspace_indices}
        if (
            len(consumer_index_set) != len(consumer_workspaces)
            or owned_indices & consumer_index_set
            or owned_indices | consumer_index_set != set(range(len(workspaces)))
        ):
            raise CodegenError(
                "Every Shared workspace must have exactly one transfer-channel "
                "or consumer owner."
            )
        for index, workspace in enumerate(workspaces):
            workspace["pipe_owned"] = index in owned_indices
            workspace["consumer_owned"] = index in consumer_index_set
        auxiliary_plan = self._auxiliary_consumer_plan(
            key,
            encoded,
            channel_plans,
            workspaces,
            consumer_index_set,
        )
        plan = {
            "id": f"stage_{ordinal}_{stem}",
            "call": str(event["call"]),
            "workspaces": workspaces,
            "channels": channel_plans,
            "consumer_workspaces": consumer_workspaces,
            "auxiliary_consumer": auxiliary_plan,
            "producer_warps": int(encoded["producer_warps"]),
            "producer_registers": int(encoded["producer_registers"]),
        }
        self.stages.append(plan)
        self.stage_by_key[key] = plan
        return plan

    def _auxiliary_consumer_plan(
        self,
        key: tuple[str, ...],
        encoded: Mapping[str, object],
        channels: list[dict[str, object]],
        workspaces: list[dict[str, object]],
        consumer_indices: set[int],
    ) -> dict[str, object] | None:
        contract = encoded["pipeline_contract"].get("auxiliary_consumer")
        prepared = encoded.get("pipeline_auxiliary_consumer")
        if contract is None:
            if prepared is not None:
                raise CodegenError(
                    "Prepared auxiliary consumer has no typed contract."
                )
            return None
        if not isinstance(contract, Mapping) or not isinstance(prepared, Mapping):
            raise CodegenError(
                "Auxiliary consumer requires matching typed and prepared ABI."
            )
        try:
            channel_indices = tuple(
                int(value) for value in contract["channel_indices"]
            )
            workspace_indices = tuple(
                int(value)
                for value in contract.get(
                    "consumer_shared_workspace_indices", ()
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CodegenError(
                "Auxiliary consumer has invalid channel/workspace indices."
            ) from error
        if (
            not channel_indices
            or len(set(channel_indices)) != len(channel_indices)
            or any(index < 0 or index >= len(channels) for index in channel_indices)
            or len(set(workspace_indices)) != len(workspace_indices)
            or any(index not in consumer_indices for index in workspace_indices)
            or tuple(prepared.get("channel_indices", ())) != channel_indices
            or tuple(prepared.get("consumer_shared_workspace_indices", ()))
            != workspace_indices
        ):
            raise CodegenError(
                "Auxiliary consumer ownership differs from its prepared helper ABI."
            )
        parameters = encoded.get("parameters")
        if not isinstance(parameters, Mapping):
            raise CodegenError(
                "Auxiliary consumer requires implementation worker parameters."
            )
        try:
            warps = int(parameters["auxiliary_consumer_warps"])
            registers = int(parameters["auxiliary_consumer_registers"])
        except (KeyError, TypeError, ValueError) as error:
            raise CodegenError(
                "Auxiliary consumer requires integer warp/register resources."
            ) from error
        if warps <= 0 or registers <= 0:
            raise CodegenError(
                "Auxiliary consumer warp/register resources must be positive."
            )
        start = self._handoff((*key, "auxiliary", "start"))
        done = self._handoff((*key, "auxiliary", "done"))
        start["semantic"] = False
        done["semantic"] = False
        selected_channels = [channels[index] for index in channel_indices]
        selected_workspaces = [workspaces[index] for index in workspace_indices]
        return {
            "channel_indices": channel_indices,
            "consumer_shared_workspace_indices": workspace_indices,
            "primary_endpoints": (
                start["writer_name"], done["reader_name"],
            ),
            "endpoints": (
                *(value["reader_name"] for value in selected_channels),
                start["reader_name"],
                done["writer_name"],
                *(value["variable"] for value in selected_workspaces),
            ),
            "drain_endpoints": tuple(
                value["reader_name"] for value in selected_channels
            ),
            "warps": warps,
            "registers": registers,
        }

    def _append_auxiliary_event(
        self,
        key: tuple[str, ...],
        event: Mapping[str, object],
        stage: Mapping[str, object],
    ) -> None:
        auxiliary = stage["auxiliary_consumer"]
        if auxiliary is None or key in self.auxiliary_event_keys:
            return
        prepared = event.get("pipeline_auxiliary_consumer")
        if not isinstance(prepared, Mapping):
            raise CodegenError(
                "Auxiliary consumer stage has no prepared helper ABI."
            )
        endpoints = tuple(str(value) for value in auxiliary["endpoints"])
        if len(endpoints) != len(tuple(prepared.get("parameters", ()))):
            raise CodegenError(
                "Auxiliary consumer endpoint arity differs from its helper ABI."
            )
        self.auxiliary_events.append({
            **event,
            "kind": "pipeline_kernel_call",
            "role": "auxiliary_consumer",
            "symbol": f"{event['symbol']}__auxiliary_consumer",
            "arguments": _join_arguments(endpoints, str(event["arguments"])),
        })
        self.auxiliary_event_keys.add(key)

    def _nested_context(
        self,
        context: _Context,
        call: PrimFunctionCall,
        callee: str,
    ) -> _Context:
        caller_binding = self.bindings[context.function_name]
        callee_binding = self.bindings[callee]
        event = next(
            value for value in caller_binding["call_abi"]["events"]
            if value["kind"] == "function_call"
            and str(value["call"]) == call.call_id
        )
        local_actuals = emit_function_call_arguments(
            self.module, caller_binding, event
        )
        signature = callee_binding["signature"]
        if len(local_actuals) != len(signature):
            raise CodegenError(
                f"Pipeline call {call.call_id!r} does not match @{callee} ABI."
            )
        actual_by_formal = {
            str(formal): _substitute(str(actual), context.actual_by_formal)
            for formal, actual in zip(signature, local_actuals, strict=True)
        }
        storage_roots = function_call_storage_roots(
            caller_binding, callee_binding, event, context.storage_roots
        )
        local_shared = _execution_shared_buffers(
            self.module.execution_function_map[callee]
        )
        actual_shared = tuple(
            context.shared_by_name.get(value.name, value)
            for value in call.shared_workspace_buffers
        )
        if len(local_shared) != len(actual_shared):
            raise CodegenError(
                f"Pipeline call {call.call_id!r} exposes {len(actual_shared)} "
                f"Shared buffers, but @{callee} owns {len(local_shared)}."
            )
        shared_by_name = {
            local.name: actual
            for local, actual in zip(local_shared, actual_shared, strict=True)
        }
        return _Context(
            callee,
            actual_by_formal,
            storage_roots,
            shared_by_name,
            (*context.call_path, call.call_id),
        )

    def _handoff(self, key: tuple[str, ...]) -> dict[str, object]:
        existing = self.handoff_by_key.get(key)
        if existing is not None:
            return existing
        stem = _identifier("__".join(key))
        handoff = {
            "id": f"handoff_{len(self.handoffs)}_{stem}",
            "pipe_name": f"_flagmega_{stem}_handoff",
            "reader_name": f"_flagmega_{stem}_handoff_reader",
            "writer_name": f"_flagmega_{stem}_handoff_writer",
            "semantic": True,
        }
        self.handoffs.append(handoff)
        self.handoff_by_key[key] = handoff
        return handoff

    def _resolve_drains(self, events: list[dict[str, object]]) -> None:
        resolved = []
        for event in events:
            if event["kind"] != "drain":
                resolved.append(event)
                continue
            key = tuple(str(value) for value in event.pop("logical_stage"))
            try:
                leaves = self.logical_stage_leaves[key]
            except KeyError as error:
                raise CodegenError(
                    f"Pipeline drain references unknown logical stage {key}."
                ) from error
            by_id = {str(value["id"]): value for value in self.stages}
            role = event.pop("role")
            self.drained_stage_ids.update(leaves)
            endpoints = []
            for leaf in leaves:
                stage = by_id[leaf]
                auxiliary = stage["auxiliary_consumer"]
                if role == "producer":
                    endpoints.extend(
                        channel["writer_name"] for channel in stage["channels"]
                    )
                elif role == "auxiliary":
                    if auxiliary is not None:
                        endpoints.extend(auxiliary["drain_endpoints"])
                else:
                    excluded = (
                        set(auxiliary["channel_indices"])
                        if auxiliary is not None else set()
                    )
                    endpoints.extend(
                        channel["reader_name"]
                        for index, channel in enumerate(stage["channels"])
                        if index not in excluded
                    )
            if endpoints:
                event["endpoints"] = [str(value) for value in endpoints]
                resolved.append(event)
        events[:] = resolved


def function_call_storage_roots(
    caller_binding: Mapping[str, object],
    callee_binding: Mapping[str, object],
    event: Mapping[str, object],
    caller_roots: Mapping[str, tuple[str, int]],
) -> dict[str, tuple[str, int]]:
    """Propagate verified physical storage roots through one function call."""

    edges = {
        str(edge["formal"]): edge
        for edge in (*event["arguments"], *event["results"])
    }
    result: dict[str, tuple[str, int]] = {}
    for argument in callee_binding["arguments"]:
        formal_buffer = str(argument["buffer"])
        try:
            edge = edges[formal_buffer]
        except KeyError as error:
            raise CodegenError(
                f"TIR nested call {event['call']!r} has no storage edge for "
                f"callee buffer {formal_buffer!r}."
            ) from error
        if edge.get("actual_runtime_value_kind") != "pointer":
            continue
        local_root = str(edge["actual_runtime_argument"])
        try:
            root_name, root_offset = caller_roots[local_root]
        except KeyError as error:
            raise CodegenError(
                f"TIR nested call {event['call']!r} cannot resolve caller "
                f"storage root {local_root!r}."
            ) from error
        actual_abi = edge["actual_abi"]
        storage = str(actual_abi.get("storage"))
        byte_offset = (
            int(actual_abi.get("pool_byte_offset", 0))
            if bool(actual_abi.get("pooled", False))
            else 0
        )
        result[str(argument["name"])] = (
            root_name, root_offset + byte_offset
        )

    caller_pools = {
        str(pool["storage"]): pool
        for pool in caller_binding["pools"]
    }
    frames = {
        str(frame["memory_space"]): frame
        for frame in event["memory_pools"]
    }
    for pool in callee_binding["pools"]:
        storage = str(pool["storage"])
        try:
            caller_pool = caller_pools[storage]
            caller_pool_name = str(caller_pool["name"])
            root_name, root_offset = caller_roots[caller_pool_name]
        except KeyError as error:
            raise CodegenError(
                f"TIR nested call {event['call']!r} cannot bind callee "
                f"{storage!r} storage."
            ) from error
        frame = frames.get(storage)
        if frame is not None:
            root_name = emit_pool_scope_base(caller_pool, root_name)
            root_offset += int(frame["offset"])
        result[str(pool["name"])] = (root_name, root_offset)
    return result


def _top_level_region(
    body: Sequential, function_name: str
) -> ProducerConsumerRegion | None:
    regions = tuple(
        value for value in body.fields if isinstance(value, ProducerConsumerRegion)
    )
    if not regions:
        return None
    if len(body.fields) != 1 or len(regions) != 1:
        raise CodegenError(
            f"Pipeline function @{function_name} must contain exactly one "
            "top-level ProducerConsumerRegion."
        )
    return regions[0]


def _execution_shared_buffers(function) -> tuple[Buffer, ...]:
    result: list[Buffer] = []
    seen: set[str] = set()
    for call in execution_calls_of(function):
        for buffer in call.shared_workspace_buffers:
            if buffer.name not in seen:
                seen.add(buffer.name)
                result.append(buffer)
    return tuple(result)


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


def _identifier(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not stem:
        raise CodegenError("Pipeline source identity cannot be empty.")
    if stem[0].isdigit() or keyword.iskeyword(stem):
        stem = f"pipeline_{stem}"
    return stem


def _triton_dtype(value: str) -> str:
    mapping = {
        "bfloat16": "tl.bfloat16",
        "float16": "tl.float16",
        "float32": "tl.float32",
        "int8": "tl.int8",
        "uint8": "tl.uint8",
        "int16": "tl.int16",
        "int32": "tl.int32",
        "int64": "tl.int64",
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise CodegenError(
            f"Pipeline Shared workspace has unsupported Triton dtype {value!r}."
        ) from error


def _join_arguments(first: list[str], rest: str) -> str:
    prefix = ", ".join(first)
    if not prefix:
        return rest
    return prefix if not rest else f"{prefix}, {rest}"


__all__ = [
    "build_pipeline_source_schedule",
    "function_call_storage_roots",
]
