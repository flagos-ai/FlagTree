# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Alias-aware, per-function physical bufferization planner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import (
    AllocationPolicy,
    AllocationStrategy,
    AliasInfo,
    AliasKind,
    BufferDescriptor,
    BufferPlan,
    CallBufferBinding,
    CallMemoryPoolBinding,
    FunctionBufferPlan,
    FunctionMemoryPool,
    KernelCallBufferBinding,
    MemorySpace,
    MemoryAllocationScope,
    MemorySharingScope,
    MemSpan,
    PhysicalAllocation,
    PhysicalBuffer,
)
from triton.flagmega.ir.model import (
    DistributedType,
    Function,
    IRModule,
    IRType,
    NoneType,
    RefType,
    TensorType,
    TupleType,
    logical_type,
)
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import (
    is_fully_replicated,
    is_fully_sharded_across_placement,
    is_local_shard_subview,
    local_shape,
    placement_owner_count,
)
from triton.flagmega.passes.tir.bufferize.graph import callee_first_functions, function_nodes
from triton.flagmega.passes.tir.bufferize.alias_analysis import AliasAnalysis
from triton.flagmega.passes.tir.bufferize.call_input_ownership import non_consumable_parameters
from triton.flagmega.passes.tir.bufferize.alignment import (
    transfer_source_alignment_requirements,
)
from triton.flagmega.passes.tir.bufferize.sat_allocator import BufferLifetime, SATBufferAllocator
from triton.flagmega.passes.tir.shared_workspace import (
    shared_workspace_buffer_id,
)


@dataclass(frozen=True)
class BufferizationOptions:
    memory_spaces: tuple[MemorySpace, ...]
    workspace: str = "workspace"
    readonly_data: str = "rdata"
    solver_time_seconds: float = 30.0
    block_local: str | None = None

    @classmethod
    def generic(cls, *, alignment: int = 256) -> BufferizationOptions:
        if alignment <= 0 or alignment & (alignment - 1):
            raise IRVerificationError(
                f"Buffer alignment must be a positive power of two, got {alignment}."
            )
        maximum = (1 << 62) - 1
        return cls((
            MemorySpace(
                "workspace", "device", alignment, maximum,
                AllocationStrategy.SAT, AllocationPolicy.GRANULARITY_ALIGNED,
                MemoryAllocationScope.FUNCTION, MemorySharingScope.BLOCK,
            ),
            MemorySpace(
                "rdata", "readonly_device", alignment, maximum,
                AllocationStrategy.LINEAR, AllocationPolicy.GRANULARITY_ALIGNED,
                MemoryAllocationScope.MODULE, MemorySharingScope.CHIP,
            ),
            MemorySpace(
                "external", "external", 1, maximum,
                AllocationStrategy.EXTERNAL, AllocationPolicy.GRANULARITY_ALIGNED,
                MemoryAllocationScope.EXTERNAL, MemorySharingScope.CHIP,
            ),
        ), block_local="workspace")

    @property
    def memory_space_map(self) -> Mapping[str, MemorySpace]:
        return {value.name: value for value in self.memory_spaces}


class BufferPlanner:
    """Build a MemSpan-based buffer plan without mutating semantic TIR nodes."""

    def __init__(self, module: IRModule, options: BufferizationOptions, *, reuse_preferences=None) -> None:
        self.module = module
        self.options = options
        self.reuse_preferences = reuse_preferences or {}
        spaces = options.memory_space_map
        try:
            self.workspace_space = spaces[options.workspace]
            self.rdata_space = spaces[options.readonly_data]
        except KeyError as error:
            raise IRVerificationError(f"Missing required buffer memory space {error.args[0]!r}.") from error
        if self.workspace_space.strategy is not AllocationStrategy.SAT:
            raise IRVerificationError("Workspace memory must use SAT allocation.")
        if self.rdata_space.strategy is not AllocationStrategy.LINEAR:
            raise IRVerificationError("Readonly data memory must use linear allocation.")
        self.function_pool_spaces = {
            value.name: value
            for value in options.memory_spaces
            if value.strategy is AllocationStrategy.SAT
            and value.allocation_scope is MemoryAllocationScope.FUNCTION
            and value.kind != "shared"
        }
        if self.workspace_space.name not in self.function_pool_spaces:
            raise IRVerificationError(
                "Default workspace must be a non-shared function-scoped SAT pool."
            )
        self.allocator = SATBufferAllocator(maximum_time_seconds=options.solver_time_seconds)
        self.shared_space = spaces.get("shared")
        self.alias_analysis = AliasAnalysis()
        self.non_consumable_parameters = non_consumable_parameters(module)
        self.non_consumable_physical_buffers: set[str] = set()
        self.required_alignments = transfer_source_alignment_requirements(module)
        self.descriptors: dict[str, BufferDescriptor] = {}
        self.bindings: dict[str, tuple[str, ...]] = {}
        self.physical_buffers: dict[str, PhysicalBuffer] = {}
        self.function_plans: dict[str, FunctionBufferPlan] = {}
        self.rdata_offset = 0
        self._physical_serial = 0

    def _is_function_pool_storage(self, storage: str) -> bool:
        return storage in self.function_pool_spaces

    def _temporary_memory_space(self, node) -> MemorySpace | None:
        raw = node.metadata.get("bufferization.memory_space")
        if raw is None:
            return None
        name = str(raw)
        try:
            return self.function_pool_spaces[name]
        except KeyError as error:
            raise IRVerificationError(
                f"Node {node.id!r} requests invalid function memory space "
                f"{name!r}.",
                stage=self.module.stage,
                node_id=node.id,
            ) from error

    def run(self) -> BufferPlan:
        self._plan_prim_function_shared_workspaces()
        self._plan_rdata()
        for function in callee_first_functions(self.module):
            self._plan_function(function)
        entry_plan = self.function_plans[self.module.entry]
        entry_workspace = entry_plan.memory_pool_map[self.workspace_space.name]
        entry = self.module.function_map[self.module.entry]
        return BufferPlan(
            buffers=tuple(self.descriptors.values()),
            physical_buffers=tuple(self.physical_buffers.values()),
            memory_spaces=self.options.memory_spaces,
            functions=tuple(self.function_plans[value.name] for value in self.module.functions),
            workspace_bytes=entry_workspace.scope_bytes,
            rdata_bytes=self.rdata_space.allocation_bytes(self.rdata_offset),
            alignment=entry_workspace.alignment,
            entry_inputs=tuple((node_id, self.bindings[node_id]) for node_id in entry.parameters),
            entry_outputs=tuple((node_id, self.bindings[node_id]) for node_id in entry.outputs),
            allocator=self.allocator.name,
            default_workspace=self.workspace_space.name,
        )

    def _plan_prim_function_shared_workspaces(self) -> None:
        """Allocate selected-kernel shared buffers per PrimFunction."""

        from triton.flagmega.ir.tir import kernel_dispatch_of

        functions = []
        for function in self.module.kernel_callable_map.values():
            dispatch = kernel_dispatch_of(function)
            if dispatch is not None and dispatch.shared_workspace_buffers:
                functions.append((function, dispatch.shared_workspace_buffers))
        if not functions:
            return
        if self.shared_space is None:
            raise IRVerificationError(
                "Selected microkernel shared workspaces have no 'shared' memory space."
            )
        if self.shared_space.strategy is not AllocationStrategy.SAT:
            raise IRVerificationError("Shared memory must use SAT allocation.")

        for function, buffers in functions:
            lifetimes = tuple(
                BufferLifetime(
                    value.mem_span.buffer.id,
                    _finite_dimension(
                        value.mem_span.size,
                        f"@{function.name} shared workspace {value.name!r} size",
                    ),
                    value.mem_span.buffer.alignment,
                    0,
                    0,
                    "microkernel_shared_workspace",
                )
                for value in buffers
            )
            allocated = self.allocator.allocate(lifetimes, self.shared_space)
            offsets = allocated.offset_map
            for value, lifetime in zip(buffers, lifetimes):
                physical = PhysicalBuffer(
                    lifetime.id,
                    self.shared_space.name,
                    lifetime.nbytes,
                    max(self.shared_space.granularity, lifetime.alignment),
                    offsets[lifetime.id],
                    function.name,
                    lifetime.live_start,
                    lifetime.live_end,
                    lifetime.role,
                )
                if physical.id in self.physical_buffers:
                    raise IRVerificationError(
                        f"Duplicate shared PhysicalBuffer {physical.id!r}."
                    )
                self.physical_buffers[physical.id] = physical
                shape = tuple(
                    _finite_dimension(
                        dimension,
                        f"@{function.name} shared workspace {value.name!r} dimension",
                    )
                    for dimension in value.dimensions
                )
                descriptor_id = shared_workspace_buffer_id(
                    function.name, value.name
                )
                descriptor = BufferDescriptor(
                    id=descriptor_id,
                    dtype=value.elem_type,
                    shape=shape,
                    strides=_dense_strides(shape),
                    storage="shared",
                    alignment=physical.alignment,
                    mem_span=MemSpan(physical, 0, lifetime.nbytes),
                    source_node=function.name,
                    function=function.name,
                    live_start=0,
                    live_end=0,
                    role="microkernel_shared_workspace",
                )
                self.descriptors[descriptor_id] = descriptor
                self.alias_analysis.define_span(descriptor_id, descriptor.mem_span)

    def _plan_rdata(self) -> None:
        nodes = tuple(
            node for node in self.module.nodes
            if node.op in {"builtin.weight", "builtin.const_asset", "tir.buffer"}
        )
        grouped: dict[str, list[tuple[object, Mapping[str, object]]]] = {}
        group_order: list[str] = []
        ungrouped = []
        for node in nodes:
            hint = node.metadata.get("rdata_group")
            if hint is None:
                ungrouped.append(node)
                continue
            if not isinstance(hint, Mapping) or not str(hint.get("name", "")):
                raise IRVerificationError(
                    "rdata_group metadata requires a non-empty name.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            name = str(hint["name"])
            if name not in grouped:
                grouped[name] = []
                group_order.append(name)
            grouped[name].append((node, hint))

        # Identical ungrouped WeightRefs share one immutable physical range.
        deduplicated: dict[tuple[object, ...], str] = {}
        for node in ungrouped:
            key = self._rdata_key(node)
            leaves = _leaf_types(node.id, node.type)
            ids = []
            for buffer_id, value_type, field in leaves:
                shape = _maximum_shape(value_type, self.module.stage, node.id)
                required_alignment = self._required_alignment(node.id)
                identity = (
                    str(node.attrs.get("source", "")),
                    str(node.attrs.get("source_hash", "")),
                    key,
                    field,
                    _tensor_type(value_type).dtype,
                    shape,
                    required_alignment,
                )
                physical_id = deduplicated.get(identity)
                if physical_id is None:
                    physical_id = self._new_physical("rdata")
                    deduplicated[identity] = physical_id
                    self._allocate_rdata_physical(
                        physical_id,
                        _physical_nbytes(value_type, shape, "rdata", "readonly_data"),
                        alignment=required_alignment,
                    )
                ids.append(self._add_descriptor(
                    buffer_id, value_type, shape, "rdata", node.id,
                    field=field, physical_id=physical_id, weight_key=key,
                    offset=self._allocation(physical_id).offset,
                    role="readonly_data",
                    alignment=required_alignment,
                ))
            self.bindings[node.id] = tuple(ids)

        for group_name in group_order:
            members = grouped[group_name]
            counts = {int(hint.get("count", -1)) for _, hint in members}
            if len(counts) != 1 or next(iter(counts)) <= 0:
                raise IRVerificationError(
                    f"rdata group {group_name!r} must declare one positive member count.",
                    stage=self.module.stage,
                )
            count = next(iter(counts))
            by_index = {int(hint.get("index", -1)): node for node, hint in members}
            if set(by_index) != set(range(count)) or len(by_index) != len(members):
                raise IRVerificationError(
                    f"rdata group {group_name!r} requires indices 0..{count - 1} "
                    "with no duplicates.",
                    stage=self.module.stage,
                )
            reference = by_index[0].type
            for index in range(count):
                node = by_index[index]
                if node.type != reference:
                    raise IRVerificationError(
                        f"rdata group {group_name!r} members must have one physical type.",
                        stage=self.module.stage,
                        node_id=node.id,
                    )
                ids = []
                for buffer_id, value_type, field in _leaf_types(node.id, node.type):
                    shape = _maximum_shape(value_type, self.module.stage, node.id)
                    required_alignment = self._required_alignment(node.id)
                    physical_id = self._new_physical("rdata")
                    self._allocate_rdata_physical(
                        physical_id,
                        _physical_nbytes(
                            value_type, shape, "rdata", "readonly_data_group_member"
                        ),
                        alignment=required_alignment,
                    )
                    ids.append(self._add_descriptor(
                        buffer_id, value_type, shape, "rdata", node.id,
                        field=field, physical_id=physical_id, weight_key=self._rdata_key(node),
                        offset=self._allocation(physical_id).offset,
                        rdata_group=group_name, group_index=index, group_count=count,
                        role="readonly_data_group_member",
                        alignment=required_alignment,
                    ))
                self.bindings[node.id] = tuple(ids)

    def _plan_function(self, function: Function) -> None:
        nodes = function_nodes(self.module, function)
        metadata_only_structural_values = _metadata_only_structural_values(
            self.module, nodes, function
        )
        local_index = {node.id: index for index, node in enumerate(nodes)}
        last_use = dict(local_index)
        for index, node in enumerate(nodes):
            for input_id in node.inputs:
                if input_id in local_index:
                    last_use[input_id] = max(last_use[input_id], index)
        function_end = len(nodes)
        for output_id in function.outputs:
            last_use[output_id] = function_end
        get_items = {
            node.id: (node.inputs[0], int(node.attrs["index"]))
            for node in nodes if node.op == "builtin.get_item"
        }
        get_item_users: dict[tuple[str, int], list[str]] = {}
        for get_item_id, source in get_items.items():
            get_item_users.setdefault(source, []).append(get_item_id)
        output_leaf_ids = self._output_leaf_ids(function, get_items)
        function_argument_leaves = _function_argument_leaf_requirements(
            nodes,
            self.module.node_map,
            frozenset(self.module.function_map),
        )

        for parameter_id in function.parameters:
            if parameter_id in self.bindings and self.module.node_map[parameter_id].op != "builtin.var":
                continue
            node = self.module.node_map[parameter_id]
            ids = []
            for buffer_id, value_type, field in _leaf_types(node.id, node.type):
                shape = _maximum_shape(value_type, self.module.stage, node.id)
                scalar = not shape
                storage = "scalar" if scalar else (
                    "state" if isinstance(logical_type(node.type), RefType) and function.name == self.module.entry
                    else "input" if function.name == self.module.entry
                    else "parameter"
                )
                ids.append(self._add_descriptor(
                    buffer_id, value_type, shape, storage, node.id,
                    field=field, physical_id=f"external:{function.name}:{buffer_id}",
                    function=function.name, role="scalar_parameter" if scalar else "parameter",
                    alignment=self._required_alignment(node.id),
                ))
            self.bindings[node.id] = tuple(ids)

            if node.id in self.non_consumable_parameters[function.name]:
                self.non_consumable_physical_buffers.update(
                    self.descriptors[value].mem_span.buffer.id for value in ids
                )

        lifetimes_by_space: dict[str, list[BufferLifetime]] = {
            name: [] for name in self.function_pool_spaces
        }
        call_records: list[CallBufferBinding] = []
        kernel_records: list[KernelCallBufferBinding] = []
        for index, node in enumerate(nodes):
            if node.id in function.parameters or node.op in {
                "builtin.weight", "builtin.const_asset", "tir.buffer"
            }:
                continue
            if node.id in self.bindings:
                # Constants and their zero-copy views may be reachable from a
                # reusable callee and its caller after function-boundary
                # propagation.  Reuse only storage that is physically global;
                # overlapping ownership of function-local workspace remains a
                # verifier error instead of silently sharing an allocation.
                descriptors = tuple(
                    self.descriptors[value] for value in self.bindings[node.id]
                )
                if all(
                    value.mem_span.buffer.memory_space
                    not in self.function_pool_spaces
                    for value in descriptors
                ):
                    continue
                raise IRVerificationError(
                    f"Value {node.id!r} is owned by multiple functions but uses "
                    "function-local workspace.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            if (
                node.id in metadata_only_structural_values
            ):
                # Structural values used exclusively by metadata-only kernel
                # operands carry type/shape information but have no physical
                # executable ABI.  Recording an empty binding avoids forcing
                # their source alias group into canonical-global storage.
                self.bindings[node.id] = ()
                continue
            if node.op == "builtin.get_item":
                source, field_index = get_items[node.id]
                self.bindings[node.id] = _field_binding(
                    self.bindings[source], self.module.node_map[source].type, field_index
                )
                continue
            if node.op == "builtin.tuple":
                # A tuple is an SSA aggregate, not a tensor-producing
                # operation. Its leaf ABI is exactly the concatenation of its
                # operands' existing buffer identities. Allocating fresh
                # leaves here disconnects a later get_item from the producer
                # that populated the operand (for example packed Q/K/V views).
                ids = tuple(
                    buffer_id
                    for input_id in node.inputs
                    for buffer_id in self.bindings[input_id]
                )
                expected_arity = len(_leaf_types(node.id, node.type))
                if len(ids) != expected_arity:
                    raise IRVerificationError(
                        f"Tuple {node.id!r} has {len(ids)} physical leaves; "
                        f"its type requires {expected_arity}.",
                        stage=self.module.stage,
                        node_id=node.id,
                    )
                self.bindings[node.id] = ids
                continue
            if node.op in {"distributed.sharded_view", "tir.buffer_view"}:
                if len(node.inputs) != 1 or len(self.bindings[node.inputs[0]]) != 1:
                    raise IRVerificationError(
                        "A buffer view requires one tensor buffer binding.",
                        stage=self.module.stage,
                        node_id=node.id,
                    )
                alias_kind = (
                    "sharded_view"
                    if node.op == "distributed.sharded_view"
                    else str(node.attrs.get("alias_kind", ""))
                )
                source_id = self.bindings[node.inputs[0]][0]
                sharded_storage_kind = None
                sharded_backing_type = None
                if alias_kind == "sharded_view":
                    (
                        sharded_storage_kind,
                        sharded_backing_type,
                    ) = self._ensure_sharded_view_backing(source_id, node.id)
                self.bindings[node.id] = self._plan_value(
                    function,
                    node.id,
                    node.type,
                    index,
                    last_use[node.id],
                    output_leaf_ids,
                    prefix=node.id,
                    alias=source_id,
                    role=alias_kind,
                    reinterpret=alias_kind in {"vector_reinterpret", "reshape"},
                    alias_kind=AliasKind.VIEW,
                    distributed_storage_kind=(
                        sharded_storage_kind
                        if alias_kind == "sharded_view"
                        else None
                    ),
                    distributed_backing_type=(
                        sharded_backing_type
                        if alias_kind == "sharded_view"
                        else None
                    ),
                    memory_space=self._temporary_memory_space(node),
                )
                continue
            if (
                node.op == "builtin.call"
                or (
                    node.op == "tir.call"
                    and str(node.attrs.get("callee", "")) in self.module.function_map
                )
            ):
                call, call_lifetimes = self._plan_call(
                    function,
                    node,
                    index,
                    last_use,
                    get_item_users,
                    output_leaf_ids,
                    function_argument_leaves,
                )
                call_records.append(call)
                for memory_space, values in call_lifetimes.items():
                    lifetimes_by_space[memory_space].extend(values)
                continue
            if _is_scalar_constant(node.op, node.type):
                ids = []
                for buffer_id, value_type, field in _leaf_types(node.id, node.type):
                    ids.append(self._add_descriptor(
                        buffer_id, value_type, (), "scalar", node.id,
                        field=field, physical_id=f"scalar:{buffer_id}",
                        function=None, role="immediate",
                    ))
                self.bindings[node.id] = tuple(ids)
                continue
            ids = []
            if isinstance(node.type, TupleType):
                leaf_offset = 0
                for field_index, field_type in enumerate(node.type.fields):
                    field_leaf_count = len(_leaf_types("field", field_type))
                    next_leaf_offset = leaf_offset + field_leaf_count
                    is_function_argument = any(
                        ordinal in function_argument_leaves.get(node.id, ())
                        for ordinal in range(leaf_offset, next_leaf_offset)
                    )
                    leaf_offset = next_leaf_offset
                    ref_alias = self._ref_alias(node.inputs, field_type)
                    if ref_alias is not None:
                        # Ref values are physical identities, not newly
                        # allocated logical tensors.  Forwarding the binding
                        # also makes entry state outputs identical to inputs.
                        ids.extend(ref_alias)
                        continue
                    alias = self._named_inplace_alias(
                        function,
                        node,
                        index,
                        last_use,
                        output_leaf_ids,
                        result_index=field_index,
                    )
                    field_ids = self._plan_value(
                        function, node.id, field_type, index,
                        self._field_live_end(node.id, field_index, index, last_use, get_item_users),
                        output_leaf_ids, prefix=f"{node.id}.{field_index}",
                        field=str(field_index), alias=alias,
                        role=(
                            "function_argument"
                            if is_function_argument else None
                        ),
                        memory_space=self._temporary_memory_space(node),
                    )
                    ids.extend(field_ids)
            elif isinstance(logical_type(node.type), RefType):
                ref_alias = self._ref_alias(node.inputs, node.type)
                if ref_alias is None:
                    raise IRVerificationError(
                        f"Ref value {node.id!r} has no reference identity source.",
                        stage=self.module.stage,
                        node_id=node.id,
                    )
                # A state update returns the same physical reference.  Forward
                # its leaf bindings verbatim so function result_aliases and
                # nested CallBufferBinding edges preserve the MemSpan identity.
                ids.extend(ref_alias)
            else:
                alias = self._named_inplace_alias(
                    function, node, index, last_use, output_leaf_ids
                )
                ids.extend(self._plan_value(
                    function, node.id, node.type, index, last_use[node.id], output_leaf_ids,
                    prefix=node.id, alias=alias,
                    role=(
                        "function_argument"
                        if function_argument_leaves.get(node.id) else None
                    ),
                    memory_space=self._temporary_memory_space(node),
                ))
            self.bindings[node.id] = tuple(ids)
            kernel_record = self._plan_kernel_workspaces(
                function, node, index, function_end
            )
            if kernel_record is not None:
                kernel_records.append(kernel_record)

        # Build one lifetime per physical allocation.  Alias chains extend the
        # root interval instead of creating overlapping rectangles. Identity-
        # forwarded Ref/call results do not create descriptors, so collect
        # physical uses from every resolved value binding as well as SSA ids.
        physical_use_end: dict[str, int] = {}
        for use_index, node in enumerate(nodes):
            if node.op == "builtin.get_item":
                # Projection is an SSA binding operation, not a read of every
                # field in the tuple. Consumers of the projected binding carry
                # the selected field's real physical use.
                continue
            for input_id in node.inputs:
                for buffer_id in self.bindings.get(input_id, ()):
                    descriptor = self.descriptors[buffer_id]
                    if (
                        descriptor.function == function.name
                        and self._is_function_pool_storage(descriptor.storage)
                    ):
                        physical_use_end[descriptor.mem_span.buffer.id] = max(
                            physical_use_end.get(descriptor.mem_span.buffer.id, 0), use_index
                        )
        for output_id in function.outputs:
            for buffer_id in self.bindings[output_id]:
                descriptor = self.descriptors[buffer_id]
                if (
                    descriptor.function == function.name
                    and self._is_function_pool_storage(descriptor.storage)
                ):
                    physical_use_end[descriptor.mem_span.buffer.id] = function_end
        physical_descriptors: dict[str, list[BufferDescriptor]] = {}
        for descriptor in self.descriptors.values():
            if (
                descriptor.function == function.name
                and self._is_function_pool_storage(descriptor.storage)
            ):
                physical_descriptors.setdefault(descriptor.mem_span.buffer.id, []).append(descriptor)
        for physical_id, values in physical_descriptors.items():
            memory_space = values[0].mem_span.buffer.memory_space
            lifetimes_by_space[memory_space].append(BufferLifetime(
                physical_id,
                max(value.mem_span.buffer.nbytes for value in values),
                max(value.alignment for value in values),
                min(int(value.live_start) for value in values),
                max(
                    max(int(value.live_end) for value in values),
                    physical_use_end.get(physical_id, 0),
                ),
            ))

        allocation_results = {
            name: self.allocator.allocate(
                tuple(lifetimes_by_space[name]), space,
                avoid_reuse=self.reuse_preferences.get((function.name, name), ()),
            )
            for name, space in self.function_pool_spaces.items()
        }
        offsets = {
            allocation_id: offset
            for result in allocation_results.values()
            for allocation_id, offset in result.offsets
        }
        function_allocations: dict[str, list[str]] = {
            name: [] for name in self.function_pool_spaces
        }
        for memory_space, lifetimes in lifetimes_by_space.items():
            for lifetime in lifetimes:
                allocation = PhysicalBuffer(
                    lifetime.id,
                    memory_space,
                    lifetime.nbytes,
                    lifetime.alignment,
                    offsets[lifetime.id],
                    function.name,
                    lifetime.live_start,
                    lifetime.live_end,
                    lifetime.role,
                )
                self.physical_buffers[allocation.id] = allocation
                function_allocations[memory_space].append(allocation.id)
        for buffer_id, descriptor in tuple(self.descriptors.items()):
            if (
                descriptor.function == function.name
                and self._is_function_pool_storage(descriptor.storage)
            ):
                physical = self.physical_buffers[descriptor.mem_span.buffer.id]
                self.descriptors[buffer_id] = replace(
                    descriptor,
                    mem_span=MemSpan(
                        physical,
                        descriptor.mem_span.start,
                        descriptor.mem_span.size,
                    ),
                )
        resolved_calls = tuple(
            replace(
                call,
                memory_pools=tuple(
                    replace(
                        pool,
                        offset=(
                            0
                            if pool.allocation is None
                            else offsets[pool.allocation]
                        ),
                    )
                    for pool in call.memory_pools
                ),
            )
            for call in call_records
        )
        parameters = tuple((node_id, self.bindings[node_id]) for node_id in function.parameters)
        outputs = tuple((node_id, self.bindings[node_id]) for node_id in function.outputs)
        parameter_ids = {value for _, values in parameters for value in values}
        result_ids = {value for _, values in outputs for value in values}
        result_aliases = tuple(
            (result, parameter)
            for result in sorted(result_ids)
            for parameter in sorted(parameter_ids)
            if self.descriptors[result].mem_span.must_alias(
                self.descriptors[parameter].mem_span
            )
        )
        self.function_plans[function.name] = FunctionBufferPlan(
            name=function.name,
            parameters=parameters,
            outputs=outputs,
            memory_pools=(
                FunctionMemoryPool(
                    name,
                    allocation_results[name].pool_bytes,
                    max(
                        (value.alignment for value in lifetimes_by_space[name]),
                        default=space.granularity,
                    ),
                    tuple(function_allocations[name]),
                )
                for name, space in self.function_pool_spaces.items()
            ),
            calls=resolved_calls,
            kernel_calls=tuple(kernel_records),
            result_aliases=result_aliases,
            values=tuple(
                (node.id, self.bindings[node.id])
                for node in nodes
                if node.id in self.bindings and self.bindings[node.id]
            ),
        )

    def _ensure_sharded_view_backing(
        self,
        source_id: str,
        view_node_id: str,
    ) -> tuple[DistributedBufferStorageKind, DistributedType | None]:
        """Select or materialize the coordinate backing for a shard view.

        PyNTT's semantic ``ShardedView`` may reassign physical block axes
        because every owner writes its disjoint canonical coordinates (or an
        idempotent broadcast replica).  A compact-per-owner descriptor cannot
        represent that contract: its MemSpan exposes only one dense local
        component.  Promote the entire physical alias group to canonical
        logical spans, matching nncase ``EnsureChipLocalShardedBacking``.

        A fully broadcast value in a block-scoped pool is different: every
        block already owns a complete logical replica. Preserve that physical
        placement and expose canonical coordinates only inside the replica.
        """

        source = self.descriptors[source_id]
        if source.distributed_type is None:
            # Tensor constants and scalar immediates already expose their
            # complete logical allocation.  Only the distributed view needs
            # the canonical-global annotation.
            return DistributedBufferStorageKind.CANONICAL_GLOBAL, None
        if (
            source.distributed_storage_kind
            is DistributedBufferStorageKind.CANONICAL_GLOBAL
        ):
            return DistributedBufferStorageKind.CANONICAL_GLOBAL, None
        view = self.module.node_map[view_node_id]
        target_type = view.type
        source_space = self.options.memory_space_map.get(
            source.mem_span.buffer.memory_space
        )
        if (
            source_space is not None
            and source_space.sharing_scope is MemorySharingScope.BLOCK
            and source_space.allocation_scope is MemoryAllocationScope.FUNCTION
            and source_space.kind != "shared"
            and all(
                level == "b"
                for level in source.distributed_type.placement.hierarchy_levels
            )
            and is_fully_replicated(source.distributed_type)
            and isinstance(target_type, DistributedType)
            and is_local_shard_subview(source.distributed_type, target_type)
        ):
            return DistributedBufferStorageKind.REPLICATED_LOCAL, None
        if isinstance(target_type, DistributedType) and source.distributed_storage_kind in {
            DistributedBufferStorageKind.COMPACT_LOCAL,
            DistributedBufferStorageKind.COMPACT_PER_OWNER,
        }:
            storage_type = source.storage_distributed_type
            if (
                storage_type is not None
                and is_local_shard_subview(storage_type, target_type)
            ):
                # Preserve the dense component of the coarser source shard.
                # The target ABI maps its local coordinates relative to this
                # explicit parent origin instead of pretending that the
                # smaller target shard begins at byte zero.
                return source.distributed_storage_kind, storage_type
        self._promote_alias_group_to_canonical(source_id, view_node_id)
        return DistributedBufferStorageKind.CANONICAL_GLOBAL, None

    def _promote_alias_group_to_canonical(self, source_id: str, use_node_id: str) -> None:
        """Constrain every producer/view in a MemSpan group before codegen."""
        source = self.descriptors[source_id]
        physical_id = source.physical_id
        aliases = [
            (buffer_id, descriptor)
            for buffer_id, descriptor in self.descriptors.items()
            if descriptor.physical_id == physical_id
        ]
        if not aliases:
            raise IRVerificationError(
                f"Canonical backing source {source_id!r} has no physical alias group.",
                stage=self.module.stage,
                node_id=use_node_id,
            )
        for buffer_id, descriptor in aliases:
            if descriptor.distributed_type is None:
                raise IRVerificationError(
                    f"Canonical backing cannot promote storage shared with non-distributed "
                    f"buffer {buffer_id!r}.",
                    stage=self.module.stage,
                    node_id=use_node_id,
                )

        old_physical = source.mem_span.buffer
        required_size = max(
            descriptor.byte_offset
            + prod(descriptor.shape, start=1) * descriptor.dtype.itemsize
            for _, descriptor in aliases
        )
        promoted_physical = replace(
            old_physical,
            size=max(old_physical.nbytes, required_size),
        )
        self.physical_buffers[physical_id] = promoted_physical
        for buffer_id, descriptor in aliases:
            logical_nbytes = (
                prod(descriptor.shape, start=1) * descriptor.dtype.itemsize
            )
            promoted_span = MemSpan(
                promoted_physical,
                descriptor.mem_span.start,
                logical_nbytes,
            )
            promoted = replace(
                descriptor,
                strides=_dense_strides(descriptor.shape),
                mem_span=promoted_span,
                distributed_storage_kind=(
                    DistributedBufferStorageKind.CANONICAL_GLOBAL
                ),
                distributed_backing_type=None,
            )
            self.descriptors[buffer_id] = promoted
            self.alias_analysis.replace_span(buffer_id, promoted_span)

    def _plan_kernel_workspaces(
        self,
        function: Function,
        node,
        index: int,
        function_end: int,
    ) -> KernelCallBufferBinding | None:
        from triton.flagmega.ir.tir import kernel_dispatch_for_call

        dispatch = kernel_dispatch_for_call(self.module, node)
        if dispatch is None:
            return None
        bindings = []
        for requirement in dispatch.workspaces:
            if requirement.memory_space not in self.function_pool_spaces:
                raise IRVerificationError(
                    f"Kernel call {node.id!r} requests unsupported caller scratch space "
                    f"{requirement.memory_space!r}.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            actuals = self._plan_value(
                function,
                node.id,
                requirement.type,
                index,
                (
                    function_end
                    if requirement.lifetime.value == "function"
                    else index
                ),
                set(),
                prefix=f"{node.id}.workspace.{requirement.name}",
                alignment=requirement.alignment,
                role="kernel_workspace",
                memory_space=self.function_pool_spaces[requirement.memory_space],
            )
            if len(actuals) != 1:
                raise IRVerificationError(
                    f"Kernel call {node.id!r} workspace {requirement.name!r} "
                    "must lower to one physical buffer."
                )
            bindings.append((requirement.name, actuals[0]))
        return KernelCallBufferBinding(
            node.id,
            function.name,
            str(node.attrs["callee"]),
            tuple(bindings),
        )

    def _plan_call(
        self,
        function,
        node,
        index,
        last_use,
        get_item_users,
        output_leaf_ids,
        function_argument_leaves,
    ) -> tuple[CallBufferBinding, list[BufferLifetime]]:
        callee_name = str(node.attrs["callee"])
        callee = self.module.function_map[callee_name]
        callee_plan = self.function_plans[callee_name]
        if len(node.inputs) != len(callee.parameters):
            raise IRVerificationError(
                f"Call {node.id!r} passes {len(node.inputs)} arguments to @{callee_name}, "
                f"which requires {len(callee.parameters)}.",
                stage=self.module.stage,
                node_id=node.id,
            )
        arguments = []
        actual_node_for_formal: dict[str, str] = {}
        for formal_id, actual_id in zip(callee.parameters, node.inputs):
            formals = self.bindings[formal_id]
            actuals = self.bindings[actual_id]
            if len(formals) != len(actuals):
                raise IRVerificationError(
                    f"Call {node.id!r} argument {actual_id!r} does not match formal {formal_id!r}.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            for formal, actual in zip(formals, actuals):
                self._verify_call_argument_physical_abi(
                    node.id, formal, actual)
            arguments.extend(zip(formals, actuals))
            actual_node_for_formal.update((formal, actual_id) for formal in formals)
        formal_to_actual = dict(arguments)
        alias_results = {}
        for result, parameter in callee_plan.result_aliases:
            actual = formal_to_actual[parameter]
            descriptor = self.descriptors[actual]
            actual_node = actual_node_for_formal[parameter]
            identity_passthrough = result == parameter
            if (
                not identity_passthrough
                and descriptor.storage not in {"parameter", "state"}
                and not self._is_function_pool_storage(descriptor.storage)
            ):
                raise IRVerificationError(
                    f"Call {node.id!r} passes non-writable {descriptor.storage} buffer "
                    f"{actual!r} to destructive result/formal alias {result!r}->{parameter!r}.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            if (
                not identity_passthrough
                and descriptor.storage != "state"
                and last_use.get(actual_node) != index
            ):
                raise IRVerificationError(
                    f"Call {node.id!r} cannot overwrite still-live argument {actual_node!r}.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            alias_results[result] = actual

        actual_results = []
        ids = []
        formal_results = [value for _, values in callee_plan.outputs for value in values]
        result_leaves = _leaf_types(node.id, node.type)
        result_top_fields = _top_level_leaf_indices(node.type)
        if len(formal_results) != len(result_leaves):
            raise IRVerificationError(
                f"Call {node.id!r} result arity does not match @{callee_name}.",
                stage=self.module.stage,
                node_id=node.id,
            )
        for ordinal, ((buffer_id, value_type, field), formal_result) in enumerate(
            zip(result_leaves, formal_results)
        ):
            alias = alias_results.get(formal_result)
            field_index = result_top_fields[ordinal]
            live_end = (
                self._field_live_end(node.id, field_index, index, last_use, get_item_users)
                if field_index is not None else last_use[node.id]
            )
            if alias is not None and _descriptor_matches_type(
                self.descriptors[alias], value_type
            ):
                planned = (alias,)
            else:
                planned = self._plan_value(
                    function,
                    node.id,
                    value_type,
                    index,
                    live_end,
                    output_leaf_ids,
                    prefix=buffer_id,
                    field=field,
                    alias=alias,
                    alias_kind=(
                        AliasKind.RESULT if alias is not None else None
                    ),
                    role=(
                        "function_argument"
                        if ordinal in function_argument_leaves.get(node.id, ())
                        else "call_result_alias" if alias is not None
                        else None
                    ),
                    memory_space=self._temporary_memory_space(node),
                    distributed_storage_kind=(
                        self.descriptors[formal_result].distributed_storage_kind
                    ),
                    distributed_backing_type=(
                        self.descriptors[formal_result].distributed_backing_type
                    ),
                )
            ids.extend(planned)
            if len(planned) != 1:
                raise IRVerificationError("Nested call result ABI leaves are not supported.")
            self._verify_call_argument_physical_abi(
                node.id, formal_result, planned[0]
            )
            actual_results.append((formal_result, planned[0]))
        self.bindings[node.id] = tuple(ids)
        lifetimes: dict[str, list[BufferLifetime]] = {
            name: [] for name in self.function_pool_spaces
        }
        memory_pools = []
        for pool in callee_plan.memory_pools:
            if not pool.scope_bytes:
                continue
            if pool.memory_space not in self.function_pool_spaces:
                raise IRVerificationError(
                    f"Call {node.id!r} requires unavailable function memory "
                    f"space {pool.memory_space!r}.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            allocation = (
                f"callframe:{function.name}:{node.id}:{pool.memory_space}"
            )
            lifetimes[pool.memory_space].append(BufferLifetime(
                allocation,
                pool.scope_bytes,
                pool.alignment,
                index,
                index,
                "call_memory_pool",
            ))
            memory_pools.append(CallMemoryPoolBinding(
                pool.memory_space,
                allocation,
                0,
                pool.scope_bytes,
            ))
        return CallBufferBinding(
            node.id,
            function.name,
            callee_name,
            tuple(arguments),
            tuple(actual_results),
            memory_pools=tuple(memory_pools),
        ), lifetimes

    def _verify_call_argument_physical_abi(
        self,
        call_id: str,
        formal_id: str,
        actual_id: str,
    ) -> None:
        """Require caller storage to match the callee's pointer interpretation."""

        formal = self.descriptors[formal_id]
        actual = self.descriptors[actual_id]
        if (
            formal.dtype != actual.dtype
            or formal.shape != actual.shape
            or formal.strides != actual.strides
            or formal.distributed_type != actual.distributed_type
            or formal.distributed_storage_kind
            is not actual.distributed_storage_kind
            or formal.distributed_backing_type
            != actual.distributed_backing_type
        ):
            raise IRVerificationError(
                f"Call {call_id!r} actual buffer {actual_id!r} does not match "
                f"formal buffer {formal_id!r} physical ABI "
                "(dtype/shape/stride/distribution/storage kind/backing).",
                stage=self.module.stage,
                node_id=call_id,
            )

    def _plan_value(
        self, function, source_node, value_type, live_start, live_end, output_leaf_ids,
        *, prefix, field=None, alias=None, alignment=None, role=None,
        reinterpret=False, alias_kind=None,
        distributed_storage_kind=None, distributed_backing_type=None,
        memory_space: MemorySpace | None = None,
    ) -> tuple[str, ...]:
        alignment = max(int(alignment or 1), self._required_alignment(source_node))
        leaves = _leaf_types(prefix, value_type)
        if alias is not None and len(leaves) != 1:
            # Ref aliases are already flattened by _ref_alias.
            if isinstance(alias, tuple) and len(alias) == len(leaves):
                result = []
                for leaf, source in zip(leaves, alias):
                    result.extend(self._plan_value(
                        function, source_node, leaf[1], live_start, live_end, output_leaf_ids,
                        prefix=leaf[0], field=leaf[2], alias=source,
                        alignment=alignment, role=role,
                        reinterpret=reinterpret, alias_kind=alias_kind,
                        distributed_storage_kind=distributed_storage_kind,
                        distributed_backing_type=distributed_backing_type,
                        memory_space=memory_space,
                    ))
                return tuple(result)
            raise IRVerificationError(f"Alias arity mismatch for value {prefix!r}.")
        result = []
        for buffer_id, tensor_type, leaf_field in leaves:
            shape = _maximum_shape(tensor_type, self.module.stage, source_node)
            is_output = buffer_id in output_leaf_ids
            if alias is not None:
                source = self.descriptors[str(alias)]
                if memory_space is not None and source.storage != memory_space.name:
                    raise IRVerificationError(
                        f"Alias {buffer_id!r} requests memory space "
                        f"{memory_space.name!r}, but its source is in "
                        f"{source.storage!r}.",
                        stage=self.module.stage,
                        node_id=source_node,
                    )
                if is_output and self._is_function_pool_storage(source.storage):
                    self._promote_workspace_alias_group_to_result(
                        function, source, buffer_id
                    )
                    source = self.descriptors[str(alias)]
                target_tensor = _tensor_type(tensor_type)
                # A view never changes physical storage. In particular, a
                # sharded coordinate view over a canonical parameter/rdata
                # buffer remains canonical-global; compact-per-owner storage
                # is created only by an allocating producer.
                target_kind = (
                    source.distributed_storage_kind
                    if distributed_storage_kind is None
                    else DistributedBufferStorageKind(distributed_storage_kind)
                )
                target_backing_type = distributed_backing_type
                if (
                    target_backing_type is None
                    and not reinterpret
                    and isinstance(tensor_type, DistributedType)
                    and tensor_type == source.distributed_type
                    and not target_kind.exposes_logical_coordinates
                ):
                    target_backing_type = source.distributed_backing_type
                target_component_shape = _component_shape(
                    tensor_type,
                    shape,
                    target_kind,
                    distributed_backing_type=target_backing_type,
                )
                target_nbytes = _component_nbytes(
                    tensor_type,
                    shape,
                    target_kind,
                    distributed_backing_type=target_backing_type,
                )
                if (
                    source.nbytes != target_nbytes
                    or (
                        not reinterpret
                        and (
                            source.dtype != target_tensor.dtype
                            or source.shape != shape
                            or source.strides != _dense_strides(target_component_shape)
                        )
                    )
                    or (
                        target_kind is DistributedBufferStorageKind.COMPACT_PER_OWNER
                        and source.distributed_storage_kind
                        is not DistributedBufferStorageKind.COMPACT_PER_OWNER
                    )
                ):
                    raise IRVerificationError(
                        f"Alias {buffer_id!r} is physically incompatible with {source.id!r}.",
                        stage=self.module.stage,
                        node_id=source_node,
                    )
                result.append(self._add_descriptor(
                    buffer_id, tensor_type, shape, source.storage, source_node,
                    field=leaf_field if leaf_field is not None else field,
                    alias_of=source.id, physical_id=source.physical_id,
                    byte_offset=source.byte_offset, offset=source.offset,
                    live_start=live_start if self._is_function_pool_storage(source.storage) else None,
                    live_end=live_end if self._is_function_pool_storage(source.storage) else None,
                    function=function.name, role=role or "alias",
                    alignment=alignment,
                    distributed_storage_kind=target_kind,
                    distributed_backing_type=target_backing_type,
                    alias_kind=alias_kind,
                ))
                continue
            selected_space = memory_space or self.workspace_space
            storage = "output" if is_output and function.name == self.module.entry else (
                "return" if is_output else selected_space.name
            )
            physical_id = (
                f"external:{function.name}:result:{buffer_id}"
                if storage in {"output", "return"} else self._new_physical(selected_space.name)
            )
            result.append(self._add_descriptor(
                buffer_id, tensor_type, shape, storage, source_node,
                field=leaf_field if leaf_field is not None else field,
                physical_id=physical_id,
                live_start=live_start if self._is_function_pool_storage(storage) else None,
                live_end=live_end if self._is_function_pool_storage(storage) else None,
                function=function.name,
                role=role or ("function_result" if is_output else "temporary"),
                alignment=alignment,
                distributed_storage_kind=distributed_storage_kind,
                distributed_backing_type=distributed_backing_type,
            ))
        return tuple(result)

    def _promote_workspace_alias_group_to_result(
        self,
        function: Function,
        source: BufferDescriptor,
        result_buffer_id: str,
    ) -> None:
        """Make a view-valued function result caller-owned storage.

        A result can be a zero-copy ``ShardedView``/``BufferView`` of a value
        that was allocated before the planner knew the alias would escape the
        function.  Leaving the physical root in the SAT workspace makes the
        runtime bind the result to the workspace pool and silently ignore the
        caller's output pointer.  Result ownership is a constraint on the
        complete MemSpan alias group: promote every descriptor to one external
        physical root so the producer writes directly into caller storage.
        """

        if (
            source.distributed_type is not None
            and source.distributed_type.partial is None
            and source.distributed_storage_kind
            is not DistributedBufferStorageKind.CANONICAL_GLOBAL
        ):
            # Escaping a function constrains both ownership and coordinates.
            # A local alias may have dense per-owner strides; retaining those
            # after making the allocation caller-owned disagrees with formal
            # parameter ABIs. Refine the complete group so earlier producers
            # write canonical coordinates directly, without a runtime copy or
            # abandoning their in-place alias. Partial components stay distinct.
            self._promote_alias_group_to_canonical(source.id, source.source_node)
            source = self.descriptors[source.id]
        old_physical = source.mem_span.buffer
        if old_physical.memory_space not in self.function_pool_spaces:
            raise IRVerificationError(
                f"Result alias {result_buffer_id!r} cannot promote non-pool "
                f"storage {old_physical.memory_space!r}.",
                stage=self.module.stage,
                node_id=source.source_node,
            )
        aliases = [
            (buffer_id, descriptor)
            for buffer_id, descriptor in self.descriptors.items()
            if descriptor.function == function.name
            and descriptor.physical_id == old_physical.id
        ]
        if not aliases:
            raise IRVerificationError(
                f"Result alias {result_buffer_id!r} has no physical alias group.",
                stage=self.module.stage,
                node_id=source.source_node,
            )
        storage = "output" if function.name == self.module.entry else "return"
        physical = PhysicalBuffer(
            id=f"external:{function.name}:result:{result_buffer_id}",
            memory_space="external",
            size=old_physical.size,
            alignment=old_physical.alignment,
            function=function.name,
            role="function_result",
        )
        if physical.id in self.physical_buffers:
            raise IRVerificationError(
                f"Duplicate result PhysicalBuffer {physical.id!r}.",
                stage=self.module.stage,
                node_id=source.source_node,
            )
        self.physical_buffers.pop(old_physical.id)
        self.physical_buffers[physical.id] = physical
        for buffer_id, descriptor in aliases:
            span = MemSpan(
                physical,
                descriptor.mem_span.byte_offset,
                descriptor.mem_span.size,
            )
            promoted = replace(
                descriptor,
                storage=storage,
                mem_span=span,
                live_start=None,
                live_end=None,
            )
            self.descriptors[buffer_id] = promoted
            self.alias_analysis.replace_span(buffer_id, span)

    def _named_inplace_alias(
        self,
        function,
        node,
        index,
        last_use,
        output_leaf_ids,
        *,
        result_index=None,
    ):
        if (
            function.name == self.module.entry
            and (
                node.id in output_leaf_ids
                or (
                    result_index is not None
                    and f"{node.id}.{result_index}" in output_leaf_ids
                )
            )
        ):
            return None
        if result_index is None and not isinstance(logical_type(node.type), TensorType):
            return None
        requested_space = self._temporary_memory_space(node)
        for input_id in self._named_inplace_inputs(node, result_index):
            values = self.bindings.get(input_id, ())
            if last_use.get(input_id) == index and len(values) == 1:
                source = self.descriptors[values[0]]
                result_type = (node.type.fields[result_index]
                               if result_index is not None else node.type)
                # Named aliases are opportunities, not a promise that every
                # dtype/shape specialization can overwrite this input. A
                # fused output conversion must allocate its own representation.
                if not _descriptor_matches_type(source, result_type):
                    continue
                if (
                    requested_space is not None
                    and source.storage != requested_space.name
                ):
                    continue
                source_space = self.function_pool_spaces.get(source.storage)
                if requested_space is None and source_space is not None:
                    # An unannotated result still requires the default pool's
                    # physical domain. Optional in-place reuse cannot inherit
                    # a private operand's scope: later ShardedViews may expose
                    # this result to other owners. Equal-domain pools remain
                    # reusable; parameter ownership is proved independently.
                    if (
                        source_space.kind != self.workspace_space.kind
                        or source_space.sharing_scope
                        is not self.workspace_space.sharing_scope
                    ):
                        continue
                writable = self._is_function_pool_storage(source.storage) or (
                    function.name != self.module.entry
                    and source.storage == "parameter"
                    and source.mem_span.buffer.id not in self.non_consumable_physical_buffers
                )
                if writable and self._alias_group_is_dead_after(
                    function.name, source, index, last_use
                ):
                    return source.id
        return None

    def _named_inplace_inputs(self, node, result_index) -> tuple[str, ...]:
        """Resolve TIR-owned aliases, with a legacy graph-definition fallback."""

        dispatch = None
        if node.op == "tir.call":
            from triton.flagmega.ir.tir import kernel_dispatch_of

            function = self.module.kernel_callable_map.get(
                str(node.attrs.get("callee", ""))
            )
            dispatch = None if function is None else kernel_dispatch_of(function)
        if dispatch is not None and dispatch.inplace_alias_candidates is not None:
            output_index = 0 if result_index is None else int(result_index)
            if output_index >= len(dispatch.outputs):
                return ()
            output_name = dispatch.outputs[output_index]
            argument_indices = {
                name: index for index, name in enumerate(dispatch.arguments)
            }
            return tuple(
                node.inputs[argument_indices[value.input]]
                for value in dispatch.inplace_alias_candidates
                if value.output == output_name
            )

        semantic_op = self._semantic_op(node)
        if not semantic_op:
            return ()
        from triton.flagmega.ir.ops.core import get_definition

        try:
            definition = get_definition(semantic_op)
        except KeyError:
            return ()
        if result_index is None:
            parameters = definition.inplace_input_parameters
        else:
            output_parameters = definition.inplace_output_parameters
            parameters = (
                ()
                if result_index >= len(output_parameters)
                or output_parameters[result_index] is None
                else (output_parameters[result_index],)
            )
        return tuple(str(parameter.read(node.inputs)) for parameter in parameters)

    def _alias_group_is_dead_after(
        self,
        function_name: str,
        source: BufferDescriptor,
        index: int,
        last_use: Mapping[str, int],
    ) -> bool:
        """Return whether a destructive write can consume ``source``.

        The direct SSA operand may be a short-lived BufferView while another
        logical value backed by the same MemSpan remains live.  In-place
        eligibility therefore belongs to the complete physical alias group,
        not only to ``last_use[input_id]``.  Existing aliases are sufficient:
        any future value has not been defined yet and cannot preserve the old
        contents across this write.
        """

        for descriptor in self.descriptors.values():
            if (
                descriptor.function != function_name
                or not descriptor.mem_span.may_alias(source.mem_span)
            ):
                continue
            live_end = descriptor.live_end
            if live_end is None:
                live_end = last_use.get(descriptor.source_node, index)
            if int(live_end) > index:
                return False
        # Identity-forwarding nodes such as builtin.get_item, tuple assembly,
        # Ref results, and nested-call aliases intentionally reuse an existing
        # buffer id instead of manufacturing another descriptor.  Their SSA
        # liveness is therefore visible only in value->buffer bindings.  A
        # destructive consumer must preserve all of those names as well.
        for value_id, buffer_ids in self.bindings.items():
            if int(last_use.get(value_id, index)) <= index:
                continue
            for buffer_id in buffer_ids:
                descriptor = self.descriptors[buffer_id]
                if (
                    descriptor.function == function_name
                    and descriptor.mem_span.may_alias(source.mem_span)
                ):
                    return False
        return True

    def _semantic_op(self, node):
        if node.op == "tir.kernel":
            return str(node.attrs.get("semantic_op", ""))
        if node.op == "tir.call":
            from triton.flagmega.ir.tir import kernel_dispatch_of

            function = self.module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
            if function is not None:
                dispatch = kernel_dispatch_of(function)
                return "" if dispatch is None else dispatch.semantic_op
        return node.op

    def _ref_alias(self, inputs, field_type):
        if not isinstance(logical_type(field_type), RefType):
            return None
        candidates = [
            self.bindings[input_id] for input_id in inputs
            if isinstance(logical_type(self.module.node_map[input_id].type), RefType)
        ]
        if len(candidates) != 1:
            raise IRVerificationError("A Ref result requires exactly one Ref operand.")
        return candidates[0]

    def _output_leaf_ids(self, function, get_items):
        result = set()

        def add_value(value_id):
            node = self.module.node_map[value_id]
            if node.op == "builtin.tuple":
                for input_id in node.inputs:
                    add_value(input_id)
                return
            if value_id in get_items:
                source, index = get_items[value_id]
                source_node = self.module.node_map[source]
                if source_node.op == "builtin.tuple":
                    add_value(source_node.inputs[index])
                    return
                result.update(value[0] for value in _leaf_types(
                    f"{source}.{index}", logical_type(source_node.type).fields[index]
                ))
                return
            result.update(value[0] for value in _leaf_types(value_id, node.type))

        for output_id in function.outputs:
            add_value(output_id)
        return result

    @staticmethod
    def _field_live_end(node_id, field_index, default, last_use, get_item_users):
        if field_index is None:
            return last_use[node_id]
        return max(
            (last_use[value] for value in get_item_users.get((node_id, field_index), ())),
            default=default,
        )

    def _add_descriptor(
        self, buffer_id, value_type, shape, storage, source_node, *, field=None,
        alias_of=None, physical_id=None, byte_offset=0, offset=0, weight_key=None,
        rdata_group=None, group_index=None, group_count=None, live_start=None,
        live_end=None, function=None, role="value",
        alignment=None, distributed_storage_kind=None,
        distributed_backing_type=None, alias_kind=None,
    ):
        if buffer_id in self.descriptors:
            raise IRVerificationError(f"Duplicate logical buffer id {buffer_id!r}.")
        if physical_id is None:
            raise IRVerificationError(f"Logical buffer {buffer_id!r} has no PhysicalBuffer identity.")
        tensor = _tensor_type(value_type)
        distributed = value_type if isinstance(value_type, DistributedType) else None
        storage_space = self.options.memory_space_map.get(storage)
        storage_kind = (
            _distributed_storage_kind(
                value_type,
                storage,
                role,
                function_pool_sharing_scope=(
                    None if storage_space is None else storage_space.sharing_scope
                ),
                function_pool=(storage in self.function_pool_spaces),
            )
            if distributed_storage_kind is None
            else DistributedBufferStorageKind(distributed_storage_kind)
        )
        component_shape = _component_shape(
            value_type,
            shape,
            storage_kind,
            distributed_backing_type=distributed_backing_type,
        )
        nbytes = _component_nbytes(
            value_type,
            shape,
            storage_kind,
            distributed_backing_type=distributed_backing_type,
        )
        physical_nbytes = _physical_nbytes(
            value_type,
            shape,
            storage,
            role,
            storage_kind=storage_kind,
            distributed_backing_type=distributed_backing_type,
        )
        physical_id = str(physical_id)
        if alias_of is not None:
            source = self.descriptors[str(alias_of)]
            physical = source.mem_span.buffer
            resolved_alias_kind = (
                AliasKind(alias_kind)
                if alias_kind is not None
                else AliasKind.VIEW
                if byte_offset != source.mem_span.byte_offset
                else AliasKind.INPLACE
            )
            alias = AliasInfo(source.id, resolved_alias_kind)
        else:
            physical = self.physical_buffers.get(physical_id)
            if physical is None:
                memory_space = (
                    storage if storage in self.function_pool_spaces
                    else self.rdata_space.name if storage == "rdata"
                    else "external"
                )
                physical_space = self.options.memory_space_map[memory_space]
                required_alignment = max(
                    physical_space.granularity,
                    int(alignment or 1),
                )
                physical = PhysicalBuffer(
                    physical_id,
                    memory_space,
                    max(physical_nbytes + byte_offset, 0),
                    required_alignment,
                    offset,
                    function,
                    live_start,
                    live_end,
                    role,
                )
                self.physical_buffers[physical_id] = physical
            alias = None
        descriptor = BufferDescriptor(
            id=buffer_id,
            dtype=tensor.dtype,
            shape=shape,
            strides=_dense_strides(component_shape),
            storage=storage,
            alignment=max(
                (
                    storage_space.granularity
                    if storage_space is not None
                    else self.workspace_space.granularity
                ),
                int(alignment or 1),
            ),
            mem_span=MemSpan(physical, byte_offset, nbytes),
            source_node=source_node,
            field=field,
            alias=alias,
            weight_key=weight_key,
            rdata_group=rdata_group,
            group_index=group_index,
            group_count=group_count,
            live_start=live_start,
            live_end=live_end,
            function=function,
            role=role,
            distributed_type=distributed,
            distributed_storage_kind=storage_kind,
            distributed_backing_type=distributed_backing_type,
        )
        if alias_of is None:
            self.alias_analysis.define_span(buffer_id, descriptor.mem_span)
        else:
            view = self.alias_analysis.add_alias(
                buffer_id,
                alias_of,
                byte_offset=byte_offset - self.alias_analysis.view(alias_of).byte_offset,
                nbytes=descriptor.nbytes,
                kind=descriptor.alias.kind,
            )
            if not view.mem_span.must_alias(descriptor.mem_span):
                raise IRVerificationError(
                    f"Alias {buffer_id!r} was assigned an inconsistent MemSpan."
                )
        self.descriptors[buffer_id] = descriptor
        return buffer_id

    def _allocate_rdata_physical(self, physical_id, nbytes, *, alignment=None):
        alignment = max(self.rdata_space.granularity, int(alignment or 1))
        self.rdata_offset = _align_up(self.rdata_offset, alignment)
        allocation = PhysicalBuffer(
            physical_id,
            self.rdata_space.name,
            nbytes,
            alignment,
            self.rdata_offset,
            None,
            None,
            None,
            "readonly_data",
        )
        self.physical_buffers[physical_id] = allocation
        self.rdata_offset += nbytes

    def _required_alignment(self, node_id) -> int:
        return int(self.required_alignments.get(str(node_id), 1))

    def _allocation(self, physical_id):
        return self.physical_buffers[physical_id]

    def _new_physical(self, prefix):
        value = f"{prefix}:{self._physical_serial}"
        self._physical_serial += 1
        return value

    @staticmethod
    def _rdata_key(node):
        return node.id if node.op == "builtin.const_asset" else str(
            node.attrs.get("key", node.attrs.get("weight_key", node.id))
        )


def plan_buffers(
    module: IRModule,
    *,
    alignment: int = 256,
    options: BufferizationOptions | None = None,
) -> BufferPlan:
    from .reuse_preferences import collect_reuse_preferences, dominates_memory_schedule

    resolved_options = options or BufferizationOptions.generic(alignment=alignment)
    baseline = BufferPlanner(module, resolved_options).run()
    preferences = collect_reuse_preferences(module, baseline)
    if not preferences:
        return baseline
    candidate = BufferPlanner(module, resolved_options, reuse_preferences=preferences).run()
    # Both alternatives are ordinary verified SAT placements. Select only a
    # Pareto improvement in synchronization and pool size; solver failures
    # remain errors, and all surviving hazards are materialized normally.
    return candidate if dominates_memory_schedule(module, candidate, baseline) else baseline


def _function_argument_leaf_requirements(nodes, node_map, function_names):
    """Trace reusable-call ABI storage requirements to allocating producers.

    ``builtin.tuple``, ``builtin.get_item`` and buffer views do not allocate
    storage.  A call argument reached through one of those nodes therefore
    constrains the corresponding producer leaf, not merely the final SSA
    forwarding node.  The result maps each allocating value to the flattened
    leaf ordinals that must use the canonical function-call ABI.
    """

    pending: list[tuple[str, frozenset[int]]] = []
    for node in nodes:
        is_call = node.op == "builtin.call" or (
            node.op == "tir.call"
            and str(node.attrs.get("callee", "")) in function_names
        )
        if not is_call:
            continue
        for input_id in node.inputs:
            leaf_count = len(_leaf_types("argument", node_map[input_id].type))
            pending.append((input_id, frozenset(range(leaf_count))))

    visited: dict[str, set[int]] = {}
    required: dict[str, set[int]] = {}
    while pending:
        value_id, ordinals = pending.pop()
        unseen = set(ordinals) - visited.setdefault(value_id, set())
        if not unseen:
            continue
        visited[value_id].update(unseen)
        node = node_map[value_id]

        if node.op == "builtin.get_item":
            source_id = node.inputs[0]
            source_type = node_map[source_id].type
            if not isinstance(source_type, TupleType):
                raise IRVerificationError(
                    "builtin.get_item source must have TupleType.",
                    node_id=node.id,
                )
            field_index = int(node.attrs["index"])
            field_counts = [
                len(_leaf_types("field", field)) for field in source_type.fields
            ]
            if field_index < 0 or field_index >= len(field_counts):
                raise IRVerificationError(
                    f"builtin.get_item index {field_index} is out of range.",
                    node_id=node.id,
                )
            if unseen and max(unseen) >= field_counts[field_index]:
                raise IRVerificationError(
                    f"builtin.get_item {node.id!r} has an inconsistent leaf ABI.",
                    node_id=node.id,
                )
            start = sum(field_counts[:field_index])
            pending.append((
                source_id,
                frozenset(start + ordinal for ordinal in unseen),
            ))
            continue

        if node.op == "builtin.tuple":
            offset = 0
            for input_id in node.inputs:
                count = len(_leaf_types("field", node_map[input_id].type))
                selected = frozenset(
                    ordinal - offset
                    for ordinal in unseen
                    if offset <= ordinal < offset + count
                )
                if selected:
                    pending.append((input_id, selected))
                offset += count
            if unseen and max(unseen) >= offset:
                raise IRVerificationError(
                    f"builtin.tuple {node.id!r} has an inconsistent leaf ABI.",
                    node_id=node.id,
                )
            continue

        if node.op in {"distributed.sharded_view", "tir.buffer_view"}:
            if len(node.inputs) != 1:
                raise IRVerificationError(
                    f"Buffer view {node.id!r} must have one input.",
                    node_id=node.id,
                )
            source_id = node.inputs[0]
            source_count = len(_leaf_types("source", node_map[source_id].type))
            result_count = len(_leaf_types("result", node.type))
            if source_count != result_count:
                raise IRVerificationError(
                    f"Buffer view {node.id!r} changes physical leaf arity.",
                    node_id=node.id,
                )
            pending.append((source_id, frozenset(unseen)))
            continue

        required.setdefault(value_id, set()).update(unseen)

    return {
        value_id: frozenset(sorted(ordinals))
        for value_id, ordinals in required.items()
    }


def _metadata_only_structural_values(module, nodes, function):
    """Return structural values reached only by metadata kernel operands."""

    physical_roots = set(function.outputs)
    metadata_roots = set()
    function_names = frozenset(module.function_map)
    primitive_map = module.kernel_callable_map
    for node in nodes:
        primitive = (
            primitive_map.get(str(node.attrs.get("callee", "")))
            if node.op == "tir.call"
            else None
        )
        if primitive is not None:
            if len(primitive.runtime_parameters) != len(node.inputs):
                raise IRVerificationError(
                    f"Kernel call {node.id!r} argument arity differs from "
                    f"@{primitive.name}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            for parameter, input_id in zip(
                primitive.runtime_parameters, node.inputs, strict=True
            ):
                (
                    metadata_roots
                    if parameter.role.value == "metadata"
                    else physical_roots
                ).add(input_id)
            continue
        if node.op == "builtin.call" or (
            node.op == "tir.call"
            and str(node.attrs.get("callee", "")) in function_names
        ):
            physical_roots.update(node.inputs)

    structural = {
        "builtin.get_item",
        "builtin.tuple",
        "distributed.sharded_view",
        "tir.buffer_view",
    }
    def closure(roots):
        result = set(roots)
        pending = list(roots)
        while pending:
            value_id = pending.pop()
            node = module.node_map.get(value_id)
            if node is None or node.op not in structural:
                continue
            for input_id in node.inputs:
                if input_id not in result:
                    result.add(input_id)
                    pending.append(input_id)
        return result

    physical = closure(physical_roots)
    metadata = closure(metadata_roots)
    return frozenset(
        value_id
        for value_id in metadata - physical
        if module.node_map[value_id].op in structural
    )


def _leaf_types(prefix: str, value_type: IRType):
    if isinstance(value_type, NoneType):
        return ()
    if isinstance(value_type, (TensorType, DistributedType)):
        return ((prefix, value_type, None),)
    if isinstance(value_type, RefType):
        result = []
        for name, field_type in value_type.fields:
            result.extend(_leaf_types(f"{prefix}.{name}", field_type))
        return tuple(result)
    if isinstance(value_type, TupleType):
        result = []
        for index, field_type in enumerate(value_type.fields):
            result.extend(_leaf_types(f"{prefix}.{index}", field_type))
        return tuple(result)
    raise IRVerificationError(f"Cannot bufferize {type(value_type).__name__}.")


def _field_binding(binding, value_type, field_index):
    if not isinstance(value_type, TupleType):
        raise IRVerificationError("builtin.get_item source must have TupleType.")
    counts = [len(_leaf_types("field", field)) for field in value_type.fields]
    start = sum(counts[:field_index])
    return tuple(binding[start:start + counts[field_index]])


def _top_level_leaf_indices(value_type):
    if not isinstance(value_type, TupleType):
        return (None,) * len(_leaf_types("result", value_type))
    result = []
    for index, field in enumerate(value_type.fields):
        result.extend((index,) * len(_leaf_types("field", field)))
    return tuple(result)


def _maximum_shape(value_type, stage, node_id):
    value_type = _tensor_type(value_type)
    shape = []
    for dimension in value_type.shape:
        value = dimension.value if dimension.value is not None else dimension.maximum
        if value is None:
            raise IRVerificationError(
                f"Bufferization requires a finite upper bound for dimension {dimension}.",
                stage=stage,
                node_id=node_id,
            )
        if value < 0:
            raise IRVerificationError("Buffer dimensions cannot be negative.", stage=stage, node_id=node_id)
        shape.append(int(value))
    return tuple(shape)


def _finite_dimension(value, owner: str) -> int:
    result = value.fixed_value if value.is_fixed else value.maximum
    if result is None or result < 0:
        raise IRVerificationError(f"{owner} requires a finite non-negative bound.")
    return int(result)


def _tensor_type(value_type):
    if isinstance(value_type, DistributedType):
        return value_type.tensor
    if isinstance(value_type, TensorType):
        return value_type
    raise IRVerificationError(f"Expected tensor buffer leaf, got {type(value_type).__name__}.")


def _descriptor_matches_type(
    descriptor: BufferDescriptor,
    value_type: IRType,
) -> bool:
    tensor = _tensor_type(value_type)
    shape = tuple(
        dimension.fixed_value if dimension.is_fixed else dimension.maximum
        for dimension in tensor.shape
    )
    return (
        None not in shape
        and descriptor.dtype == tensor.dtype
        and descriptor.shape == tuple(shape)
        and descriptor.distributed_type
        == (value_type if isinstance(value_type, DistributedType) else None)
    )


def _distributed_storage_kind(
    value_type,
    storage,
    role,
    *,
    function_pool_sharing_scope=None,
    function_pool=False,
):
    if not isinstance(value_type, DistributedType):
        return DistributedBufferStorageKind.COMPACT_LOCAL
    if value_type.partial is not None:
        # Partial values contain distinct owner components, even when their
        # logical tensor is broadcast. A function boundary changes ownership
        # of the allocation, not its representation. Canonical-global storage
        # would alias all contributions and lose the later reduction.
        if (
            function_pool
            and function_pool_sharing_scope is MemorySharingScope.BLOCK
            and role != "function_argument"
        ):
            return DistributedBufferStorageKind.COMPACT_LOCAL
        return DistributedBufferStorageKind.COMPACT_PER_OWNER
    if (
        storage in {"rdata", "input", "output", "state", "external"}
        or role == "function_argument"
        or role in {
            "readonly_data",
            "readonly_data_group_member",
            "parameter",
            "function_result",
        } and not function_pool
    ):
        return DistributedBufferStorageKind.CANONICAL_GLOBAL
    if (
        function_pool
        and function_pool_sharing_scope is MemorySharingScope.BLOCK
    ):
        # One physical arena exists per block, so it stores exactly one local
        # component. The runtime pool stride, rather than the PhysicalBuffer,
        # selects the current owner instance.
        return DistributedBufferStorageKind.COMPACT_LOCAL
    if is_fully_sharded_across_placement(value_type):
        return DistributedBufferStorageKind.COMPACT_PER_OWNER
    return DistributedBufferStorageKind.CANONICAL_GLOBAL


def _component_shape(
    value_type,
    global_shape,
    storage_kind,
    *,
    distributed_backing_type=None,
):
    if (
        isinstance(value_type, DistributedType)
        and not storage_kind.exposes_logical_coordinates
    ):
        storage_type = distributed_backing_type or value_type
        result = []
        for dimension in local_shape(storage_type):
            extent = dimension.fixed_value if dimension.is_fixed else dimension.maximum
            if extent is None:
                raise IRVerificationError("Distributed buffer component shape requires finite bounds.")
            result.append(int(extent))
        return tuple(result)
    return tuple(global_shape)


def _component_nbytes(
    value_type,
    global_shape,
    storage_kind,
    *,
    distributed_backing_type=None,
):
    shape = _component_shape(
        value_type,
        global_shape,
        storage_kind,
        distributed_backing_type=distributed_backing_type,
    )
    return prod(shape, start=1) * _tensor_type(value_type).dtype.itemsize


def _physical_nbytes(
    value_type,
    global_shape,
    storage,
    role,
    *,
    storage_kind=None,
    distributed_backing_type=None,
):
    kind = (
        _distributed_storage_kind(value_type, storage, role)
        if storage_kind is None
        else DistributedBufferStorageKind(storage_kind)
    )
    component = _component_nbytes(
        value_type,
        global_shape,
        kind,
        distributed_backing_type=distributed_backing_type,
    )
    if isinstance(value_type, DistributedType) and kind is DistributedBufferStorageKind.COMPACT_PER_OWNER:
        return component * placement_owner_count(value_type)
    return component


def _dense_strides(shape: Sequence[int]):
    result = []
    running = 1
    for value in reversed(shape):
        result.append(running)
        running *= value
    return tuple(reversed(result))


def _is_scalar_constant(op, value_type):
    value_type = logical_type(value_type)
    return op in {"builtin.scalar_const", "tir.scalar_const"} and (
        isinstance(value_type, TensorType) and value_type.rank == 0
    )


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


__all__ = ["BufferPlanner", "BufferizationOptions", "plan_buffers"]
