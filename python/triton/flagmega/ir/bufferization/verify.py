# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structural and MemSpan verification for buffer-plan v6."""

from __future__ import annotations

from typing import Mapping
from weakref import ReferenceType, ref

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization.mem_span import MemSpan
from triton.flagmega.ir.bufferization.memory import MemorySharingScope
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.bufferization.plan import (
    BUFFER_PLAN_SCHEMA,
    LEGACY_BUFFER_PLAN_SCHEMA,
    BufferPlan,
)
from triton.flagmega.ir.model import (
    DistributedType,
    IRModule,
    NoneType,
    RefType,
    TensorType,
    TupleType,
)


_MEMORY_SPACE_METADATA = "bufferization.memory_space"
_VERIFIED_BUFFER_PLANS: dict[int, tuple[ReferenceType[IRModule], BufferPlan]] = {}
_PARSED_BUFFER_PLANS: dict[int, tuple[ReferenceType[Mapping], BufferPlan]] = {}


def _parse_buffer_plan(data: Mapping) -> BufferPlan:
    """Parse one canonical immutable metadata object once.

    Validation below intentionally remains module-specific: execution
    functions and graph ABIs can change while retaining the same serialized
    allocation plan during late TIR passes.
    """

    identity = id(data)
    cached = _PARSED_BUFFER_PLANS.get(identity)
    if cached is not None and cached[0]() is data:
        return cached[1]
    plan = BufferPlan.from_data(data)

    def discard(reference: ReferenceType[Mapping], *, key: int = identity) -> None:
        current = _PARSED_BUFFER_PLANS.get(key)
        if current is not None and current[0] is reference:
            _PARSED_BUFFER_PLANS.pop(key, None)

    try:
        reference = ref(data, discard)
    except TypeError:
        return plan
    _PARSED_BUFFER_PLANS[identity] = (reference, plan)
    return plan


def verify_buffer_plan(module: IRModule) -> BufferPlan:
    identity = id(module)
    cached = _VERIFIED_BUFFER_PLANS.get(identity)
    if cached is not None and cached[0]() is module:
        return cached[1]
    data = module.metadata.get("buffer_plan")
    if not isinstance(data, Mapping):
        raise IRVerificationError(
            f"bufferized TIR requires {BUFFER_PLAN_SCHEMA} metadata.",
            stage=module.stage,
        )
    if data.get("schema") not in {BUFFER_PLAN_SCHEMA, LEGACY_BUFFER_PLAN_SCHEMA}:
        raise IRVerificationError(
            f"bufferized TIR requires {BUFFER_PLAN_SCHEMA} metadata; "
            f"{LEGACY_BUFFER_PLAN_SCHEMA} is accepted for resume.",
            stage=module.stage,
        )
    plan = _parse_buffer_plan(data)
    buffers = plan.buffer_map
    physical_buffers = plan.physical_buffer_map
    functions = plan.function_map
    spaces = {value.name: value for value in plan.memory_spaces}
    if len(spaces) != len(plan.memory_spaces):
        raise IRVerificationError(
            "Buffer plan memory-space identities must be unique.",
            stage=module.stage,
        )
    workspace_space = plan.workspace_memory_space
    if (
        workspace_space.strategy.value != "sat"
        or workspace_space.allocation_scope.value != "function"
        or workspace_space.kind == "shared"
    ):
        raise IRVerificationError(
            f"Default workspace {workspace_space.name!r} must be a non-shared "
            "function-scoped SAT memory space."
        )
    if len(buffers) != len(plan.buffers):
        raise IRVerificationError("Buffer plan logical ids must be unique.", stage=module.stage)
    if len(physical_buffers) != len(plan.physical_buffers):
        raise IRVerificationError("Buffer plan physical ids must be unique.", stage=module.stage)
    if len(functions) != len(plan.functions) or set(functions) != set(module.function_map):
        raise IRVerificationError(
            "Buffer plan must contain exactly one ABI for every IR function.", stage=module.stage
        )
    call_ids = [call.call for function in plan.functions for call in function.calls]
    if len(call_ids) != len(set(call_ids)):
        raise IRVerificationError("Call buffer ABI ids must be unique.", stage=module.stage)
    kernel_call_ids = [
        call.call for function in plan.functions for call in function.kernel_calls
    ]
    if len(kernel_call_ids) != len(set(kernel_call_ids)):
        raise IRVerificationError("Kernel-call workspace ABI ids must be unique.", stage=module.stage)
    if plan.allocator != "ortools-cp-sat/no-overlap-2d":
        raise IRVerificationError(f"Unsupported buffer allocator {plan.allocator!r}.")

    for allocation in plan.physical_buffers:
        try:
            space = spaces[allocation.memory_space]
        except KeyError as error:
            raise IRVerificationError(
                f"Allocation {allocation.id!r} uses missing memory space {allocation.memory_space!r}."
            ) from error
        if allocation.size.minimum is None or allocation.size.minimum < 0:
            raise IRVerificationError(f"Allocation {allocation.id!r} has invalid bounds.")
        if allocation.start.minimum is None or allocation.start.minimum < 0:
            raise IRVerificationError(f"Allocation {allocation.id!r} has invalid start.")
        if allocation.alignment <= 0:
            raise IRVerificationError(f"Allocation {allocation.id!r} is not aligned.")
        if allocation.start.is_fixed and allocation.start.fixed_value % allocation.alignment:
            raise IRVerificationError(f"Allocation {allocation.id!r} is not aligned.")
        if allocation.start.maximum is None or allocation.size.maximum is None:
            raise IRVerificationError(f"Allocation {allocation.id!r} requires finite byte bounds.")
        if allocation.start.maximum + allocation.size.maximum > space.maximum_bytes:
            raise IRVerificationError(f"Allocation {allocation.id!r} exceeds {space.name!r} capacity.")
        if (
            space.strategy.value == "sat"
            and space.allocation_scope.value == "function"
            and space.kind != "shared"
        ) and (
            allocation.function not in functions
            or allocation.live_start is None
            or allocation.live_end is None
            or allocation.live_start < 0
            or allocation.live_end < allocation.live_start
        ):
            raise IRVerificationError(
                f"Workspace allocation {allocation.id!r} has no valid function lifetime."
            )
        if allocation.memory_space == "shared" and (
            allocation.function not in module.kernel_callable_map
            or allocation.live_start is None
            or allocation.live_end is None
            or allocation.live_start < 0
            or allocation.live_end < allocation.live_start
            or allocation.role != "microkernel_shared_workspace"
        ):
            raise IRVerificationError(
                f"Shared allocation {allocation.id!r} has no valid PrimFunction lifetime."
            )

    for descriptor in plan.buffers:
        span = descriptor.mem_span
        if span.start.minimum is None or span.start.minimum < 0:
            raise IRVerificationError(f"Buffer {descriptor.id!r} has invalid physical bounds.")
        if span.size.minimum is None or span.size.minimum < 0:
            raise IRVerificationError(f"Buffer {descriptor.id!r} has invalid physical size.")
        if descriptor.alignment <= 0:
            raise IRVerificationError(f"Buffer {descriptor.id!r} is not aligned.")
        allocation = physical_buffers.get(span.buffer.id)
        if allocation is None:
            raise IRVerificationError(
                f"Buffer {descriptor.id!r} references missing PhysicalBuffer {span.buffer.id!r}."
            )
        if allocation != span.buffer:
            raise IRVerificationError(
                f"Buffer {descriptor.id!r} does not reference the canonical PhysicalBuffer object."
            )
        if span.absolute_start.is_fixed and span.absolute_start.fixed_value % descriptor.alignment:
            raise IRVerificationError(f"Buffer {descriptor.id!r} is not aligned.")
        if not span.is_within(MemSpan(allocation)):
            raise IRVerificationError(
                f"Buffer {descriptor.id!r} MemSpan exceeds PhysicalBuffer {allocation.id!r}."
            )
        if (
            descriptor.distributed_storage_kind
            is DistributedBufferStorageKind.REPLICATED_LOCAL
        ):
            space = spaces.get(allocation.memory_space)
            if (
                space is None
                or space.sharing_scope is not MemorySharingScope.BLOCK
                or space.allocation_scope.value != "function"
                or space.kind == "shared"
            ):
                raise IRVerificationError(
                    f"Replicated-local buffer {descriptor.id!r} requires a "
                    "non-shared function-scoped block pool."
                )
        if (
            descriptor.storage in spaces
            and spaces[descriptor.storage].allocation_scope.value != "external"
        ):
            expected = descriptor.storage
            if allocation.memory_space != expected:
                raise IRVerificationError(
                    f"Buffer {descriptor.id!r} uses {allocation.memory_space!r}, expected {expected!r}."
                )
        elif allocation.memory_space != "external":
            raise IRVerificationError(
                f"External buffer {descriptor.id!r} uses non-external PhysicalBuffer {allocation.id!r}."
            )
        if descriptor.alias is not None:
            try:
                source = buffers[descriptor.alias.source]
            except KeyError as error:
                raise IRVerificationError(
                    f"Buffer {descriptor.id!r} aliases missing {descriptor.alias.source!r}."
                ) from error
            if descriptor.alias.kind.value in {"identity", "inplace", "parameter", "result"}:
                valid_alias = span.must_alias(source.mem_span)
            else:
                valid_alias = span.is_within(source.mem_span)
            if not valid_alias:
                raise IRVerificationError(
                    f"Alias {descriptor.id!r} has an invalid MemSpan relative to {source.id!r}."
                )
        group = (descriptor.rdata_group, descriptor.group_index, descriptor.group_count)
        if any(value is not None for value in group) and not (
            descriptor.storage == "rdata"
            and descriptor.rdata_group is not None
            and descriptor.group_index is not None
            and descriptor.group_count is not None
            and 0 <= descriptor.group_index < descriptor.group_count
        ):
            raise IRVerificationError(f"Buffer {descriptor.id!r} has an invalid rdata group.")

    _verify_nonoverlap(plan)
    _verify_function_abis(module, plan)
    _verify_prim_function_abis(module, plan)
    _verify_prim_function_shared_workspaces(module, plan)
    _verify_bindings(plan.entry_inputs, buffers, "entry input")
    _verify_bindings(plan.entry_outputs, buffers, "entry output")
    entry_pool = functions[module.entry].memory_pool_map.get(plan.default_workspace)
    if entry_pool is None:
        raise IRVerificationError(
            f"Entry function has no default {plan.default_workspace!r} memory pool."
        )
    if plan.workspace_bytes != entry_pool.scope_bytes:
        raise IRVerificationError("Module workspace size must equal the entry function workspace.")
    if plan.alignment != entry_pool.alignment:
        raise IRVerificationError("Module workspace alignment must equal the entry function ABI.")
    rdata_end = max((
        value.offset + value.nbytes for value in plan.physical_buffers
        if value.memory_space == "rdata"
    ), default=0)
    if rdata_end > plan.rdata_bytes:
        raise IRVerificationError("Readonly allocations exceed the declared rdata image.")
    def discard(reference: ReferenceType[IRModule], *, key: int = identity) -> None:
        current = _VERIFIED_BUFFER_PLANS.get(key)
        if current is not None and current[0] is reference:
            _VERIFIED_BUFFER_PLANS.pop(key, None)

    _VERIFIED_BUFFER_PLANS[identity] = (ref(module, discard), plan)
    return plan


def _verify_nonoverlap(plan: BufferPlan) -> None:
    values = [
        value for value in plan.physical_buffers
        if plan.memory_space_map[value.memory_space].strategy.value == "sat"
    ]
    for index, lhs in enumerate(values):
        for rhs in values[index + 1:]:
            if lhs.function != rhs.function:
                continue
            if lhs.memory_space != rhs.memory_space:
                continue
            time_overlap = (
                int(lhs.live_start) <= int(rhs.live_end)
                and int(rhs.live_start) <= int(lhs.live_end)
            )
            byte_overlap = (
                lhs.offset < rhs.offset + rhs.nbytes
                and rhs.offset < lhs.offset + lhs.nbytes
            )
            if time_overlap and byte_overlap and lhs.nbytes and rhs.nbytes:
                raise IRVerificationError(
                    f"Physical allocations {lhs.id!r} and {rhs.id!r} overlap while live in "
                    f"@{lhs.function}."
                )


def _verify_function_abis(module: IRModule, plan: BufferPlan) -> None:
    from triton.flagmega.ir.tir import kernel_dispatch_for_call

    buffers = plan.buffer_map
    allocations = plan.physical_buffer_map
    for function in module.functions:
        abi = plan.function_map[function.name]
        default_pool = abi.memory_pool_map.get(plan.default_workspace)
        if default_pool is None:
            raise IRVerificationError(
                f"@{function.name} has no default {plan.default_workspace!r} memory pool."
            )
        if tuple(value for value, _ in abi.parameters) != function.parameters:
            raise IRVerificationError(f"@{function.name} parameter ABI is out of order.")
        if tuple(value for value, _ in abi.outputs) != function.outputs:
            raise IRVerificationError(f"@{function.name} result ABI is out of order.")
        _verify_bindings(abi.parameters, buffers, f"@{function.name} parameter")
        _verify_bindings(abi.outputs, buffers, f"@{function.name} result")
        _verify_bindings(abi.values, buffers, f"@{function.name} value")
        _verify_typed_bindings(module, abi.parameters, buffers, f"@{function.name} parameter")
        _verify_typed_bindings(module, abi.outputs, buffers, f"@{function.name} result")
        _verify_typed_bindings(module, abi.values, buffers, f"@{function.name} value")
        _verify_explicit_memory_placements(module, plan, abi.values)
        _verify_inplace_memory_domains(module, plan, abi.values)
        if len(dict(abi.values)) != len(abi.values):
            raise IRVerificationError(f"@{function.name} value bindings contain duplicate nodes.")
        expected_kernel_calls = tuple(
            node_id
            for node_id, _ in abi.values
            if kernel_dispatch_for_call(module, module.node_map[node_id]) is not None
        )
        if tuple(value.call for value in abi.kernel_calls) != expected_kernel_calls:
            raise IRVerificationError(
                f"@{function.name} kernel-call workspace ABIs are missing or out of order."
            )
        listed_allocations = tuple(
            allocation
            for pool in abi.memory_pools
            for allocation in pool.allocations
        )
        if len(set(listed_allocations)) != len(listed_allocations):
            raise IRVerificationError(
                f"@{function.name} lists one PhysicalBuffer in multiple memory pools."
            )
        expected_allocations = {
            allocation.id
            for allocation in plan.physical_buffers
            if allocation.function == function.name
            and plan.memory_space_map[allocation.memory_space].allocation_scope.value
            == "function"
            and plan.memory_space_map[allocation.memory_space].kind != "shared"
        }
        if set(listed_allocations) != expected_allocations:
            raise IRVerificationError(
                f"@{function.name} memory-pool allocation closure differs from its "
                "PhysicalBuffers."
            )
        for pool in abi.memory_pools:
            space = plan.memory_space_map.get(pool.memory_space)
            if (
                space is None
                or space.allocation_scope.value != "function"
                or space.kind == "shared"
            ):
                raise IRVerificationError(
                    f"@{function.name} pool {pool.memory_space!r} is not a "
                    "function-scoped runtime memory space."
                )
            for allocation_id in pool.allocations:
                allocation = allocations.get(allocation_id)
                if (
                    allocation is None
                    or allocation.function != function.name
                    or allocation.memory_space != pool.memory_space
                ):
                    raise IRVerificationError(
                        f"@{function.name} pool {pool.memory_space!r} references "
                        f"invalid allocation {allocation_id!r}."
                    )
                if allocation.offset + allocation.nbytes > pool.scope_bytes:
                    raise IRVerificationError(
                        f"@{function.name} allocation {allocation_id!r} exceeds "
                        f"its {pool.memory_space!r} pool."
                    )
                if allocation.alignment > pool.alignment:
                    raise IRVerificationError(
                        f"@{function.name} allocation {allocation_id!r} exceeds "
                        f"its {pool.memory_space!r} base alignment."
                    )
        for allocation_id in listed_allocations:
            allocation = allocations.get(allocation_id)
            if allocation is None or allocation.function != function.name:
                raise IRVerificationError(
                    f"@{function.name} references invalid allocation {allocation_id!r}."
                )
        for result, parameter in abi.result_aliases:
            parameter_ids = {value for _, values in abi.parameters for value in values}
            result_ids = {value for _, values in abi.outputs for value in values}
            if result not in result_ids or parameter not in parameter_ids:
                raise IRVerificationError(f"@{function.name} result alias references a missing buffer.")
            if not buffers[result].mem_span.must_alias(buffers[parameter].mem_span):
                raise IRVerificationError(f"@{function.name} result alias has a different MemSpan.")
        for call in abi.calls:
            if call.caller != function.name or call.callee not in plan.function_map:
                raise IRVerificationError(f"Call ABI {call.call!r} has invalid ownership.")
            callee = plan.function_map[call.callee]
            expected_arguments = tuple(
                value for _, values in callee.parameters for value in values
            )
            expected_results = tuple(
                value for _, values in callee.outputs for value in values
            )
            if tuple(formal for formal, _ in call.arguments) != expected_arguments:
                raise IRVerificationError(f"Call {call.call!r} formal argument ABI is out of order.")
            if tuple(formal for formal, _ in call.results) != expected_results:
                raise IRVerificationError(f"Call {call.call!r} formal result ABI is out of order.")
            expected_pools = {
                pool.memory_space: pool
                for pool in callee.memory_pools
                if pool.scope_bytes
            }
            if set(call.memory_pool_map) != set(expected_pools):
                raise IRVerificationError(
                    f"Call {call.call!r} memory-pool frame set differs from @{call.callee}."
                )
            for memory_space, pool in call.memory_pool_map.items():
                callee_pool = expected_pools[memory_space]
                if pool.scope_bytes != callee_pool.scope_bytes:
                    raise IRVerificationError(
                        f"Call {call.call!r} has the wrong {memory_space!r} frame size."
                    )
                if pool.allocation is None:
                    raise IRVerificationError(
                        f"Call {call.call!r} is missing its {memory_space!r} frame."
                    )
                allocation = allocations.get(pool.allocation)
                if (
                    allocation is None
                    or allocation.role not in {"call_workspace", "call_memory_pool"}
                    or allocation.function != function.name
                    or allocation.memory_space != memory_space
                    or allocation.nbytes != pool.scope_bytes
                ):
                    raise IRVerificationError(
                        f"Call {call.call!r} has an invalid {memory_space!r} frame."
                    )
                if pool.offset != allocation.offset:
                    raise IRVerificationError(
                        f"Call {call.call!r} {memory_space!r} frame offset is inconsistent."
                    )
            for formal, actual in (*call.arguments, *call.results):
                if formal not in buffers or actual not in buffers:
                    raise IRVerificationError(f"Call {call.call!r} binding references a missing buffer.")
        for call in abi.kernel_calls:
            node = module.node_map[call.call]
            dispatch = kernel_dispatch_for_call(module, node)
            if (
                call.caller != function.name
                or dispatch is None
                or call.callee != str(node.attrs.get("callee", ""))
            ):
                raise IRVerificationError(
                    f"Kernel call workspace ABI {call.call!r} has invalid ownership."
                )
            primitive = module.kernel_callable_map[call.callee]
            expected = tuple(value.name for value in primitive.workspaces)
            if tuple(formal for formal, _ in call.workspaces) != expected:
                raise IRVerificationError(
                    f"Kernel call {call.call!r} workspace ABI is out of order."
                )
            if len(dict(call.workspaces)) != len(call.workspaces):
                raise IRVerificationError(
                    f"Kernel call {call.call!r} has duplicate workspace bindings."
                )
            for parameter, (formal, actual_id) in zip(primitive.workspaces, call.workspaces):
                actual = buffers.get(actual_id)
                if (
                    formal != parameter.name
                    or actual is None
                    or actual.function != function.name
                    or actual.source_node != call.call
                    or actual.storage != parameter.memory_space
                    or actual.role != "kernel_workspace"
                    or actual.mem_span.buffer.memory_space != parameter.memory_space
                ):
                    raise IRVerificationError(
                        f"Kernel call {call.call!r} workspace {formal!r} has an invalid allocation."
                    )


def _verify_explicit_memory_placements(module, plan, values) -> None:
    """Keep editable placement annotations authoritative after resume.

    ``bufferization.memory_space`` is part of the editable Python IR contract,
    rather than a transient allocator hint.  A resumed/corrupted checkpoint
    must therefore fail when its serialized BufferPlan no longer implements
    the annotated placement.
    """

    buffers = plan.buffer_map
    spaces = plan.memory_space_map
    for node_id, binding in values:
        requested = module.node_map[node_id].metadata.get(
            _MEMORY_SPACE_METADATA
        )
        if requested is None:
            continue
        name = str(requested)
        space = spaces.get(name)
        if (
            space is None
            or space.strategy.value != "sat"
            or space.allocation_scope.value != "function"
            or space.kind == "shared"
        ):
            raise IRVerificationError(
                f"Node {node_id!r} requests invalid function memory space "
                f"{name!r}.",
                stage=module.stage,
                node_id=node_id,
            )
        for buffer_id in binding:
            descriptor = buffers[buffer_id]
            if (
                descriptor.storage != name
                or descriptor.mem_span.buffer.memory_space != name
            ):
                raise IRVerificationError(
                    f"Node {node_id!r} requests memory space {name!r}, but "
                    f"buffer {buffer_id!r} is placed in "
                    f"{descriptor.mem_span.buffer.memory_space!r}.",
                    stage=module.stage,
                    node_id=node_id,
                )


def _verify_inplace_memory_domains(module, plan, values) -> None:
    """A resumed optional alias must honor the implicit default domain too."""

    buffers = plan.buffer_map
    spaces = plan.memory_space_map
    default = spaces[plan.default_workspace]
    for node_id, binding in values:
        if _MEMORY_SPACE_METADATA in module.node_map[node_id].metadata:
            continue  # Explicit placement is checked separately.
        for buffer_id in binding:
            descriptor = buffers[buffer_id]
            if (
                descriptor.source_node != node_id
                or descriptor.alias is None
                or descriptor.alias.kind.value != "inplace"
            ):
                continue
            space = spaces.get(descriptor.storage)
            if (
                space is not None
                and space.strategy.value == "sat"
                and space.allocation_scope.value == "function"
                and space.kind != "shared"
                and (space.kind != default.kind or space.sharing_scope is not default.sharing_scope)
            ):
                raise IRVerificationError(
                    f"In-place result {node_id!r} in {space.name!r} does not "
                    f"preserve the default memory domain {default.name!r}.",
                    stage=module.stage, node_id=node_id,
                )


def _verify_bindings(values, buffers, label):
    for node_id, binding in values:
        if not binding or any(value not in buffers for value in binding):
            raise IRVerificationError(f"{label} {node_id!r} has an invalid buffer binding.")


def _verify_typed_bindings(module, values, buffers, label) -> None:
    """Check graph-value ABI types, independently of selected PrimFunction use.

    A graph parameter may feed a zero-copy distribution view before reaching a
    selected kernel, so PrimFunction formal checks alone cannot validate its
    descriptor.  Every serialized logical binding remains authoritative and
    must agree with the corresponding Python IR type.
    """

    for node_id, binding in values:
        leaves = _buffer_leaf_types(module.node_map[node_id].type)
        if len(leaves) != len(binding):
            raise IRVerificationError(
                f"{label} {node_id!r} type/buffer arity differs."
            )
        for leaf, buffer_id in zip(leaves, binding):
            descriptor = buffers[buffer_id]
            tensor = leaf.tensor if isinstance(leaf, DistributedType) else leaf
            shape = tuple(
                dimension.fixed_value if dimension.is_fixed else dimension.maximum
                for dimension in tensor.shape
            )
            expected_distributed = leaf if isinstance(leaf, DistributedType) else None
            if (
                None in shape
                or descriptor.dtype != tensor.dtype
                or descriptor.shape != shape
                or descriptor.distributed_type != expected_distributed
            ):
                raise IRVerificationError(
                    f"{label} {node_id!r} buffer {buffer_id!r} does not match "
                    "its logical IR type."
                )


def _buffer_leaf_types(value_type):
    if isinstance(value_type, NoneType):
        return ()
    if isinstance(value_type, (TensorType, DistributedType)):
        return (value_type,)
    if isinstance(value_type, RefType):
        return tuple(
            leaf
            for _, field_type in value_type.fields
            for leaf in _buffer_leaf_types(field_type)
        )
    if isinstance(value_type, TupleType):
        return tuple(
            leaf
            for field_type in value_type.fields
            for leaf in _buffer_leaf_types(field_type)
        )
    raise IRVerificationError(
        f"Buffer plan cannot bind logical type {type(value_type).__name__}."
    )


def _verify_prim_function_abis(module: IRModule, plan: BufferPlan) -> None:
    from triton.flagmega.ir.tir.kernel_dispatch import kernel_dispatch_for_call
    from triton.flagmega.ir.tir.prim_function import PrimParameterRole

    descriptors = plan.buffer_map
    for graph_function in module.functions:
        function_plan = plan.function_map[graph_function.name]
        values = dict(function_plan.values)
        kernel_calls = {value.call: value for value in function_plan.kernel_calls}
        for node_id, actual_results in values.items():
            node = module.node_map[node_id]
            if kernel_dispatch_for_call(module, node) is None:
                continue
            primitive = module.kernel_callable_map[str(node.attrs["callee"])]
            for formal, actual_node in zip(primitive.runtime_parameters, node.inputs):
                if formal.role is PrimParameterRole.METADATA:
                    if formal.buffers:
                        raise IRVerificationError(
                            f"Kernel call {node.id!r} metadata input "
                            f"{formal.name!r} unexpectedly owns buffers."
                        )
                    continue
                if not _buffer_leaf_types(formal.type):
                    if formal.buffers:
                        raise IRVerificationError(
                            f"Kernel call {node.id!r} non-physical input "
                            f"{formal.name!r} unexpectedly owns buffers."
                        )
                    continue
                try:
                    actual = values[actual_node]
                except KeyError as error:
                    raise IRVerificationError(
                        f"Kernel call {node.id!r} input {actual_node!r} has no physical binding."
                    ) from error
                _verify_formal_buffers(formal.buffers, actual, descriptors, node.id, formal.name)
            formal_outputs = tuple(
                buffer
                for parameter in primitive.output_parameters
                for buffer in parameter.buffers
            )
            _verify_formal_buffers(
                formal_outputs,
                actual_results,
                descriptors,
                node.id,
                "results",
            )
            workspace_record = kernel_calls[node.id]
            for parameter, (formal_name, actual_id) in zip(
                primitive.workspaces,
                workspace_record.workspaces,
            ):
                if parameter.name != formal_name:
                    raise IRVerificationError(
                        f"Kernel call {node.id!r} workspace formal order differs."
                    )
                _verify_formal_buffers(
                    parameter.buffers,
                    (actual_id,),
                    descriptors,
                    node.id,
                    parameter.name,
                )


def _verify_prim_function_shared_workspaces(module: IRModule, plan: BufferPlan) -> None:
    from triton.flagmega.ir.tir import kernel_dispatch_of
    from triton.flagmega.passes.tir.shared_workspace import (
        shared_workspace_buffer_id,
    )

    descriptors = plan.buffer_map
    physical_buffers = plan.physical_buffer_map
    expected_ids = set()
    for function in (*module.prim_functions, *module.kernel_definitions):
        dispatch = kernel_dispatch_of(function)
        if dispatch is None:
            continue
        for buffer in dispatch.shared_workspace_buffers:
            descriptor_id = shared_workspace_buffer_id(function.name, buffer.name)
            expected_ids.add(descriptor_id)
            descriptor = descriptors.get(descriptor_id)
            physical = physical_buffers.get(buffer.mem_span.buffer.id)
            if (
                descriptor is None
                or descriptor.storage != "shared"
                or descriptor.function != function.name
                or descriptor.role != "microkernel_shared_workspace"
                or descriptor.mem_span.buffer != physical
                or buffer.mem_span.buffer != physical
                or buffer.mem_span.start != descriptor.mem_span.start
                or buffer.mem_span.size != descriptor.mem_span.size
            ):
                raise IRVerificationError(
                    f"PrimFunction @{function.name} shared workspace {buffer.name!r} "
                    "does not match its buffer-plan allocation."
                )
    actual_ids = {
        value.id for value in plan.buffers
        if value.role == "microkernel_shared_workspace"
    }
    if actual_ids != expected_ids:
        raise IRVerificationError(
            "Buffer plan shared workspace set differs from selected PrimFunctions."
        )


def _verify_formal_buffers(formals, actual_ids, descriptors, call_id, label) -> None:
    if not formals or len(formals) != len(actual_ids):
        raise IRVerificationError(
            f"Kernel call {call_id!r} {label} formal/actual buffer arity differs."
        )
    for formal, actual_id in zip(formals, actual_ids):
        actual = descriptors[actual_id]
        shape = tuple(
            value.fixed_value if value.is_fixed else value.maximum
            for value in formal.dimensions
        )
        if (
            actual.dtype != formal.elem_type
            or actual.shape != shape
            or actual.alignment < formal.mem_span.buffer.alignment
            or actual.distributed_type != formal.distributed_type
            or actual.distributed_storage_kind != formal.distributed_storage_kind
            or actual.distributed_backing_type
            != formal.distributed_backing_type
            or actual.strides != tuple(value.fixed_value for value in formal.strides)
        ):
            raise IRVerificationError(
                f"Kernel call {call_id!r} {label} buffer {actual_id!r} does not match "
                f"formal {formal.name!r}."
            )


__all__ = ["verify_buffer_plan"]
