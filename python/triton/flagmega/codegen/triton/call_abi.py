# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Concrete call-site ABI for bufferized Triton execution functions.

ExecutionFunction owns op/call order and formal/actual edges, KernelDefinition
owns kernel semantics, and BufferPlan owns physical storage. The high-level graph
is retained for edit/resume but is not the executable schedule.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from math import gcd, prod

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import (
    DistributedBufferStorageKind,
    IRModule,
    execution_calls_of,
    kernel_dispatch_of,
    kernel_dispatch_for_call,
    local_shape as distributed_local_shape,
    placement_owner_count,
    verify_buffer_plan,
)
from triton.flagmega.ir import VectorType
from triton.flagmega.ir import local_shard_descriptor
from triton.flagmega.codegen.triton.dimension_expression import emit_dimension
from triton.flagmega.codegen.triton.distributed_abi import kernel_execution_kind
from triton.flagmega.codegen.triton.kernel_dispatch import selected_kernel_node
from triton.flagmega.codegen.triton.package_plan import plain_package_value
from triton.flagmega.codegen.triton.pool_abi import memory_scope_counts
from triton.flagmega.codegen.triton.readonly_scalars import annotate_readonly_scalars
from triton.flagmega.passes.functions.graph import function_nodes


CALL_ABI_SCHEMA = "flagmega.triton-call-abi/v1"
LOCAL_BUFFER_ABI_SCHEMA = "flagmega.local-buffer-abi/v2"
FUNCTION_CALL_ABI_SCHEMA = "flagmega.function-call-abi/v2"


def describe_function_call_abi(
    module: IRModule,
    *,
    function_name: str | None = None,
) -> dict[str, object]:
    """Describe one bufferized function without reconstructing model roles.

    The ordered event stream contains only executable kernel calls and nested
    function calls. Ordinary graph/view nodes remain represented by the
    value-to-buffer bindings in ``FunctionBufferPlan``.
    """

    plan = verify_buffer_plan(module)
    function_name = module.entry if function_name is None else str(function_name)
    try:
        function = module.function_map[function_name]
        function_plan = plan.function_map[function_name]
    except KeyError as error:
        raise CodegenError(
            f"Cannot describe call ABI for unknown function {function_name!r}."
        ) from error

    kernel_calls = describe_kernel_call_abis(
        module, function_name=function_name
    )
    pool_scope_counts = memory_scope_counts(module, plan)
    kernel_by_id = {str(value["call"]): value for value in kernel_calls}
    nested_by_id = {value.call: value for value in function_plan.calls}
    events: list[dict[str, object]] = []
    execution = module.execution_function_map.get(function_name)
    if execution is None:
        for node in function_nodes(module, function):
            kernel = kernel_by_id.get(node.id)
            if kernel is not None:
                events.append({
                    "kind": "kernel_call",
                    "call": node.id,
                    "execution_kind": kernel["execution_kind"],
                })
                continue
            nested = nested_by_id.get(node.id)
            if nested is not None:
                events.append(_nested_event(
                    plan,
                    nested.call,
                    nested.callee,
                    nested.arguments,
                    nested.results,
                    nested.memory_pools,
                    pool_scope_counts,
                ))
    else:
        for call in execution_calls_of(execution):
            kernel = kernel_by_id.get(call.call_id)
            if kernel is not None:
                events.append({
                    "kind": "kernel_call",
                    "call": call.call_id,
                    "execution_kind": kernel["execution_kind"],
                })
                continue
            if call.callee not in module.execution_function_map:
                raise CodegenError(
                    f"Execution call {call.call_id!r} references non-executable "
                    f"callee @{call.callee}."
                )
            events.append(_nested_event(
                plan,
                call.call_id,
                call.callee,
                tuple((value.formal, value.actual) for value in call.arguments),
                tuple((value.formal, value.actual) for value in call.results),
                call.memory_pools,
                pool_scope_counts,
            ))

    described_kernel_ids = set(kernel_by_id)
    event_kernel_ids = {
        str(value["call"])
        for value in events
        if value["kind"] == "kernel_call"
    }
    described_nested_ids = (
        set(nested_by_id)
        if execution is None
        else {
            call.call_id
            for call in execution_calls_of(execution)
            if call.callee in module.execution_function_map
        }
    )
    event_nested_ids = {
        str(value["call"])
        for value in events
        if value["kind"] == "function_call"
    }
    if (
        event_kernel_ids != described_kernel_ids
        or event_nested_ids != described_nested_ids
    ):
        raise CodegenError(
            f"Function @{function_name} executable event order is incomplete: "
            f"kernels={sorted(described_kernel_ids - event_kernel_ids)}, "
            f"calls={sorted(described_nested_ids - event_nested_ids)}."
        )

    return {
        "schema": FUNCTION_CALL_ABI_SCHEMA,
        "function": function_name,
        "calling_convention": str(
            (execution.attrs if execution is not None else function.attrs).get(
                "calling_convention",
                "entry" if function_name == module.entry else "device",
            )
        ),
        "noinline": bool(
            (execution.attrs if execution is not None else function.attrs).get(
                "noinline", False
            )
        ),
        "reusable": bool(
            (execution.attrs if execution is not None else function.attrs).get(
                "reusable", False
            )
        ),
        "parameters": [
            _function_value_binding(plan, name, buffers, pool_scope_counts)
            for name, buffers in function_plan.parameters
        ],
        "outputs": [
            _function_value_binding(plan, name, buffers, pool_scope_counts)
            for name, buffers in function_plan.outputs
        ],
        "result_aliases": [
            {"result": result, "parameter": parameter}
            for result, parameter in function_plan.result_aliases
        ],
        "memory_pools": [
            {
                **pool.to_data(),
                "scope_count": pool_scope_counts.get(pool.memory_space, 1),
            }
            for pool in function_plan.memory_pools
        ],
        "kernel_calls": list(kernel_calls),
        "events": events,
    }


def describe_kernel_call_abis(
    module: IRModule,
    *,
    function_name: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Bind every selected kernel call to its actual physical buffers.

    The result follows execution order and deliberately includes collectives. An
    ordinary call receives local buffer domains regardless of its
    ``DistributedType``; only ``execution_kind`` determines whether owners
    must communicate.
    """

    plan = verify_buffer_plan(module)
    pool_scope_counts = memory_scope_counts(module, plan)
    function_name = module.entry if function_name is None else str(function_name)
    execution = module.execution_function_map.get(function_name)
    if execution is not None:
        return annotate_readonly_scalars(tuple(
            _execution_kernel_call_abi(
                module, plan, call, pool_scope_counts
            )
            for call in execution_calls_of(execution)
            if call.callee in module.kernel_callable_map
        ), module, plan)
    try:
        graph_function = module.function_map[function_name]
        function_plan = plan.function_map[function_name]
    except KeyError as error:
        raise CodegenError(
            f"Cannot describe kernel calls for unknown function {function_name!r}."
        ) from error
    values = dict(function_plan.values)
    workspace_calls = plan.kernel_call_map
    calls: list[dict[str, object]] = []
    graph_nodes = {node.id for node in function_nodes(module, graph_function)}
    for node in module.nodes:
        if node.id not in graph_nodes:
            continue
        dispatch = kernel_dispatch_for_call(module, node)
        if dispatch is None:
            continue
        selected = selected_kernel_node(module, node)
        try:
            primitive = module.kernel_callable_map[str(node.attrs["callee"])]
        except KeyError as error:
            raise CodegenError(
                f"Kernel call {node.id!r} references a missing PrimFunction."
            ) from error

        inputs: list[dict[str, object]] = []
        for formal, actual_node_id in zip(
            primitive.runtime_parameters, node.inputs, strict=True
        ):
            actual_buffers = (
                ()
                if formal.role.value == "metadata"
                else values.get(actual_node_id, ())
            )
            inputs.append(
                _bind_parameter(
                    plan,
                    node.id,
                    formal,
                    actual_node_id,
                    actual_buffers,
                    pool_scope_counts,
                )
            )

        output_buffer_ids = values.get(node.id, ())
        outputs: list[dict[str, object]] = []
        cursor = 0
        for formal in primitive.output_parameters:
            count = len(formal.buffers)
            actual_buffers = output_buffer_ids[cursor : cursor + count]
            cursor += count
            outputs.append(
                _bind_parameter(
                    plan,
                    node.id,
                    formal,
                    node.id,
                    actual_buffers,
                    pool_scope_counts,
                )
            )
        if cursor != len(output_buffer_ids):
            raise CodegenError(
                f"Kernel call {node.id!r} binds {len(output_buffer_ids)} result "
                f"buffers, but @{primitive.name} consumes {cursor}."
            )

        workspaces: list[dict[str, object]] = []
        workspace_binding = workspace_calls.get(node.id)
        workspace_map = (
            {} if workspace_binding is None else dict(workspace_binding.workspaces)
        )
        for formal in primitive.workspaces:
            try:
                actual_id = workspace_map[formal.name]
            except KeyError as error:
                raise CodegenError(
                    f"Kernel call {node.id!r} has no actual workspace for "
                    f"formal {formal.name!r}."
                ) from error
            workspaces.append(
                _bind_parameter(
                    plan,
                    node.id,
                    formal,
                    actual_id,
                    (actual_id,),
                    pool_scope_counts,
                )
            )

        parameters = selected.attrs["parameters"]
        facts = selected.attrs["facts"]
        calls.append(
            {
                "schema": CALL_ABI_SCHEMA,
                "call": node.id,
                "callee": primitive.name,
                "semantic_op": selected.attrs["semantic_op"],
                "implementation": selected.attrs["candidate"],
                "family": str(parameters["family"]),
                "variant": str(parameters["variant"]),
                "execution_kind": kernel_execution_kind(
                    str(selected.attrs["semantic_op"]), facts
                ).value,
                "inputs": inputs,
                "outputs": outputs,
                "workspaces": workspaces,
                "semantic_attrs": plain_package_value(
                    selected.attrs.get("semantic_attrs", {})
                ),
                "parameters": plain_package_value(parameters),
                "facts": plain_package_value(facts),
                **_transfer_pipeline_abi(
                    dispatch, dispatch.shared_workspace_buffers
                ),
            }
        )
    return annotate_readonly_scalars(tuple(calls), module, plan)


def _execution_kernel_call_abi(
    module, plan, call, pool_scope_counts
) -> dict[str, object]:
    try:
        primitive = module.kernel_callable_map[call.callee]
    except KeyError as error:
        raise CodegenError(
            f"Kernel execution call {call.call_id!r} references missing "
            f"PrimFunction @{call.callee}."
        ) from error
    dispatch = kernel_dispatch_of(primitive)
    if dispatch is None or dispatch.microkernel is None:
        raise CodegenError(
            f"Kernel execution call {call.call_id!r} has no single selected "
            "KernelDispatch."
        )

    def bind_group(parameters, bindings, group):
        by_formal = {value.formal: value.actual for value in bindings}
        result = []
        for parameter in parameters:
            try:
                actuals = tuple(
                    by_formal[buffer.name] for buffer in parameter.buffers
                )
            except KeyError as error:
                raise CodegenError(
                    f"Kernel execution call {call.call_id!r} has no {group} "
                    f"binding for {error.args[0]!r}."
                ) from error
            result.append(_bind_parameter(
                plan,
                call.call_id,
                parameter,
                call.call_id,
                actuals,
                pool_scope_counts,
            ))
        expected = {
            buffer.name for parameter in parameters for buffer in parameter.buffers
        }
        if set(by_formal) != expected:
            raise CodegenError(
                f"Kernel execution call {call.call_id!r} {group} ABI differs "
                f"from @{primitive.name}."
            )
        return result

    selection = dispatch.microkernel
    parameters = dispatch.resolved_parameters
    facts = dispatch.resolved_facts
    return {
        "schema": CALL_ABI_SCHEMA,
        "call": call.call_id,
        "callee": primitive.name,
        "semantic_op": dispatch.semantic_op,
        "implementation": selection.implementation,
        "family": selection.family,
        "variant": selection.variant,
        "execution_kind": kernel_execution_kind(
            dispatch.semantic_op, facts
        ).value,
        "inputs": bind_group(
            primitive.runtime_parameters, call.arguments, "input"
        ),
        "outputs": bind_group(
            primitive.output_parameters, call.results, "output"
        ),
        "workspaces": bind_group(
            primitive.workspaces, call.workspaces, "workspace"
        ),
        "semantic_attrs": plain_package_value(dispatch.semantic_attrs),
        "parameters": plain_package_value(parameters),
        "facts": plain_package_value(facts),
        **_transfer_pipeline_abi(dispatch, call.shared_workspace_buffers),
    }


def _transfer_pipeline_abi(dispatch, shared_buffers) -> dict[str, object]:
    """Serialize selected transfer resources without a target/model lookup."""

    selection = dispatch.microkernel
    pipeline = None if selection is None else selection.transfer_pipeline
    if selection is None:
        return {}
    descriptors = selection.shared_workspaces
    buffers = tuple(shared_buffers)
    if len(descriptors) != len(buffers):
        raise CodegenError(
            f"Kernel {dispatch.candidate!r} transfer-pipeline Shared ABI has "
            f"{len(buffers)} buffers, expected {len(descriptors)}."
        )
    encoded_workspaces = []
    for descriptor, buffer in zip(descriptors, buffers, strict=True):
        try:
            shape = tuple(value.fixed_value for value in buffer.dimensions)
            strides = tuple(value.fixed_value for value in buffer.strides)
            offset = buffer.mem_span.absolute_start.fixed_value
            nbytes = buffer.mem_span.size.fixed_value
        except ValueError as error:
            raise CodegenError(
                f"Kernel {dispatch.candidate!r} Shared workspace "
                f"{descriptor.name!r} must have a fixed post-Bufferize ABI."
            ) from error
        encoded_workspaces.append({
            "name": descriptor.name,
            "dtype": buffer.elem_type.value,
            "shape": shape,
            "strides": strides,
            "offset_bytes": offset,
            "nbytes": nbytes,
            "alignment_bytes": descriptor.alignment_bytes,
            "matrix_compatible": descriptor.matrix_compatible,
            "physical_buffer": buffer.mem_span.buffer.id,
        })
    result = {"shared_workspaces": encoded_workspaces}
    if pipeline is not None:
        result["transfer_pipeline"] = {
            "capacity": pipeline.capacity,
            "channels": [
                {
                    "name": channel.name,
                    "source_argument_indices": list(
                        channel.source_argument_indices
                    ),
                    "shared_workspace_indices": list(
                        channel.shared_workspace_indices
                    ),
                    "source_alignment_bytes": channel.source_alignment_bytes,
                }
                for channel in pipeline.channels
            ],
            "consumer_shared_workspace_indices": list(
                pipeline.consumer_shared_workspace_indices
            ),
            "auxiliary_consumer": (
                None
                if pipeline.auxiliary_consumer is None
                else {
                    "channel_indices": list(
                        pipeline.auxiliary_consumer.channel_indices
                    ),
                    "consumer_shared_workspace_indices": list(
                        pipeline.auxiliary_consumer
                        .consumer_shared_workspace_indices
                    ),
                }
            ),
        }
    return result


def _nested_event(
    plan,
    call_id,
    callee,
    arguments,
    results,
    memory_pools,
    pool_scope_counts,
) -> dict[str, object]:
    encoded_pools = [
        {
            "memory_space": pool.memory_space,
            "allocation": pool.allocation,
            "offset": pool.offset,
            "scope_bytes": (
                pool.scope_bytes
                if hasattr(pool, "scope_bytes")
                else pool.nbytes
            ),
            "scope_count": pool_scope_counts.get(pool.memory_space, 1),
        }
        for pool in memory_pools
    ]
    return {
        "kind": "function_call",
        "call": call_id,
        "callee": callee,
        "arguments": [
            _buffer_edge(plan, formal, actual, pool_scope_counts)
            for formal, actual in arguments
        ],
        "results": [
            _buffer_edge(plan, formal, actual, pool_scope_counts)
            for formal, actual in results
        ],
        "memory_pools": encoded_pools,
    }


def describe_local_buffer_abi(
    buffer, plan=None, *, pool_scope_count: int = 1
) -> dict[str, object]:
    """Serialize the per-owner view which an ordinary kernel consumes."""

    # Buffer descriptors are immutable. The same descriptor appears in many
    # formal/actual edges and is independently visited by schedule and runtime
    # binding analyses. Cache the target-independent local-shard derivation,
    # then copy before attaching plan-specific memory-space fields.
    abi = deepcopy(_describe_local_buffer_abi_base(buffer))
    # Effective alignment belongs to the view, not merely its pool. In
    # particular an aligned allocation may have a byte-offset alias.
    start = buffer.mem_span.start
    if not start.is_fixed:
        abi["view_byte_offset"] = start.to_data()
        abi["offset_bindings"] = dict(buffer.offset_bindings)
    abi["alignment_bytes"] = (
        gcd(buffer.mem_span.buffer.alignment, start.fixed_value) if start.is_fixed else 1
    )
    return _with_memory_space_abi(abi, buffer, plan, pool_scope_count)


@lru_cache(maxsize=2048)
def _describe_local_buffer_abi_base(buffer) -> dict[str, object]:
    """Derive immutable-buffer local ABI fields without retaining BufferPlan."""

    lane_shape = (
        tuple(buffer.dtype.lanes)
        if isinstance(buffer.dtype, VectorType)
        else ()
    )
    lane_count = prod(lane_shape, start=1)
    scalar_dtype = (
        buffer.dtype.elem_type
        if isinstance(buffer.dtype, VectorType)
        else buffer.dtype
    )
    scalar_strides = tuple(value * lane_count for value in buffer.strides)
    distributed = buffer.distributed_type
    if distributed is None:
        return {
            "schema": LOCAL_BUFFER_ABI_SCHEMA,
            "buffer": buffer.id,
            "physical_buffer": buffer.physical_id,
            "storage": buffer.storage,
            "pool_byte_offset": buffer.offset if buffer.mem_span.start.is_fixed else buffer.mem_span.buffer.offset,
            "storage_kind": DistributedBufferStorageKind.COMPACT_LOCAL.value,
            "dtype": buffer.dtype.value,
            "scalar_dtype": scalar_dtype.value,
            "scalar_itemsize": scalar_dtype.itemsize,
            "logical_shape": tuple(buffer.shape),
            "local_capacity_shape": tuple(buffer.shape),
            "active_shape_expressions": tuple(str(value) for value in buffer.shape),
            "logical_coordinate_expressions": tuple(f"local_coord_{axis}" for axis in range(len(buffer.shape))),
            "storage_coordinate_expressions": tuple(f"local_coord_{axis}" for axis in range(len(buffer.shape))),
            "storage_strides": tuple(buffer.strides),
            "scalar_storage_strides": scalar_strides,
            "scalar_lane_shape": lane_shape,
            "scalar_lane_count": lane_count,
            "component_stride_elements": 0,
            "component_stride_scalar_elements": 0,
            "owner_count": 1,
            # A plain tensor has no owner-dependent projection: its physical
            # shape is the complete logical tensor.  Calling this coordinate
            # space ``local`` makes a tensor_load into a sharded destination
            # look like a compact-owner remap even though the source pointer
            # is canonical global storage.
            "coordinate_space": "canonical_global",
            "distributed_type": None,
            "distributed_backing_type": None,
        }
    local_shape = tuple(buffer.local_shape)
    backing_type = buffer.distributed_backing_type
    storage_type = backing_type or distributed
    storage_local_shape = tuple(
        value.fixed_value if value.is_fixed else value.maximum
        for value in distributed_local_shape(storage_type)
    )
    if any(value is None for value in storage_local_shape):
        raise CodegenError(
            f"Buffer {buffer.id!r} has an unbounded storage component shape."
        )
    owner_count = placement_owner_count(distributed)
    coordinate_names = tuple(
        f"shard_coord_{axis}" for axis in range(distributed.placement.rank)
    )
    shard = local_shard_descriptor(distributed, coordinate_names)
    storage_shard = local_shard_descriptor(storage_type, coordinate_names)
    component_stride = (
        prod(storage_local_shape, start=1)
        if buffer.distributed_storage_kind
        is DistributedBufferStorageKind.COMPACT_PER_OWNER
        else 0
    )
    return {
        "schema":
        LOCAL_BUFFER_ABI_SCHEMA,
        "buffer":
        buffer.id,
        "physical_buffer":
        buffer.physical_id,
        "storage":
        buffer.storage,
        "pool_byte_offset":
        buffer.offset if buffer.mem_span.start.is_fixed else buffer.mem_span.buffer.offset,
        "storage_kind":
        buffer.distributed_storage_kind.value,
        "dtype":
        buffer.dtype.value,
        "scalar_dtype":
        scalar_dtype.value,
        "scalar_itemsize":
        scalar_dtype.itemsize,
        "logical_shape":
        tuple(buffer.shape),
        "local_capacity_shape":
        local_shape,
        "active_shape_expressions":
        tuple(emit_dimension(value) for value in shard.active_shape),
        "logical_coordinate_expressions":
        tuple(
            emit_dimension(axis.map_local_to_global(f"local_coord_{index}")) for index, axis in enumerate(shard.axes)),
        "storage_coordinate_expressions":
        tuple(
            emit_dimension(
                semantic_axis.map_local_to_global(f"local_coord_{index}") - storage_axis.map_local_to_global(0))
            for index, (semantic_axis, storage_axis) in enumerate(zip(shard.axes, storage_shard.axes, strict=True))),
        "storage_strides":
        tuple(buffer.strides),
        "scalar_storage_strides":
        scalar_strides,
        "scalar_lane_shape":
        lane_shape,
        "scalar_lane_count":
        lane_count,
        "component_stride_elements":
        component_stride,
        "component_stride_scalar_elements":
        component_stride * lane_count,
        "owner_count":
        owner_count,
        "coordinate_space": ("canonical_global" if buffer.distributed_storage_kind.exposes_logical_coordinates else
                             "parent_shard_local" if backing_type is not None else "local"),
        # Complete staged SBP mapping. Renderers derive active extents and
        # local-to-global coordinates from this instead of a layout whitelist.
        "distributed_type":
        plain_package_value(distributed.to_data()),
        "distributed_backing_type": (None if backing_type is None else plain_package_value(backing_type.to_data())),
    }


def _with_memory_space_abi(abi, buffer, plan, pool_scope_count):
    if plan is None:
        return abi
    try:
        space = plan.memory_space_map[buffer.mem_span.buffer.memory_space]
    except KeyError as error:
        raise CodegenError(
            f"Buffer {buffer.id!r} uses an unknown memory space."
        ) from error
    abi["memory_space"] = space.name
    abi["memory_sharing_scope"] = space.sharing_scope.value
    if (
        space.allocation_scope.value == "function"
        and space.kind != "shared"
    ):
        if buffer.function is None:
            raise CodegenError(
                f"Pooled buffer {buffer.id!r} has no owning function."
            )
        abi["pooled"] = True
        if space.sharing_scope.value == "block" and pool_scope_count > 1:
            if (
                buffer.distributed_storage_kind
                is DistributedBufferStorageKind.COMPACT_PER_OWNER
            ):
                raise CodegenError(
                    f"Block-scoped buffer {buffer.id!r} cannot also contain "
                    "all owner components."
                )
            abi["pool_scope"] = "block"
            abi["pool_scope_count"] = pool_scope_count
            abi["pool_scope_stride_bytes"] = plan.function_memory_space_bytes(
                buffer.function, space.name
            )
            abi["pool_scope_index"] = "shard_index"
    elif space.allocation_scope.value == "module" and space.kind != "shared":
        abi["pooled"] = True
    return abi


def _bind_parameter(
    plan,
    call_id,
    formal,
    actual_node_id,
    actual_buffer_ids,
    pool_scope_counts,
):
    if len(actual_buffer_ids) != len(formal.buffers):
        raise CodegenError(
            f"Kernel call {call_id!r} formal {formal.name!r} binds "
            f"{len(actual_buffer_ids)} buffers, expected {len(formal.buffers)}."
        )
    buffers = []
    for formal_buffer, actual_id in zip(
        formal.buffers, actual_buffer_ids, strict=True
    ):
        try:
            actual = plan.buffer_map[actual_id]
        except KeyError as error:
            raise CodegenError(
                f"Kernel call {call_id!r} references missing buffer {actual_id!r}."
            ) from error
        buffers.append(
            {
                "formal": formal_buffer.name,
                "actual": actual_id,
                "abi": describe_local_buffer_abi(
                    actual,
                    plan,
                    pool_scope_count=pool_scope_counts.get(
                        actual.mem_span.buffer.memory_space, 1
                    ),
                ),
            }
        )
    return {
        "formal": formal.name,
        "role": formal.role.value,
        "actual_value": actual_node_id,
        "buffers": buffers,
    }


def _function_value_binding(
    plan, node_id, buffer_ids, pool_scope_counts
) -> dict[str, object]:
    return {
        "value": str(node_id),
        "buffers": [
            {
                "buffer": str(buffer_id),
                "abi": describe_local_buffer_abi(
                    plan.buffer_map[str(buffer_id)],
                    plan,
                    pool_scope_count=pool_scope_counts.get(
                        plan.buffer_map[str(buffer_id)].mem_span.buffer.memory_space,
                        1,
                    ),
                ),
            }
            for buffer_id in buffer_ids
        ],
    }


def _buffer_edge(
    plan, formal: str, actual: str, pool_scope_counts
) -> dict[str, object]:
    try:
        formal_buffer = plan.buffer_map[formal]
        actual_buffer = plan.buffer_map[actual]
    except KeyError as error:
        raise CodegenError(
            f"Nested function call references unknown buffer {error.args[0]!r}."
        ) from error
    return {
        "formal": formal,
        "actual": actual,
        "formal_abi": describe_local_buffer_abi(
            formal_buffer,
            plan,
            pool_scope_count=pool_scope_counts.get(
                formal_buffer.mem_span.buffer.memory_space, 1
            ),
        ),
        "actual_abi": describe_local_buffer_abi(
            actual_buffer,
            plan,
            pool_scope_count=pool_scope_counts.get(
                actual_buffer.mem_span.buffer.memory_space, 1
            ),
        ),
    }


__all__ = [
    "CALL_ABI_SCHEMA",
    "FUNCTION_CALL_ABI_SCHEMA",
    "LOCAL_BUFFER_ABI_SCHEMA",
    "describe_function_call_abi",
    "describe_kernel_call_abis",
    "describe_local_buffer_abi",
]
