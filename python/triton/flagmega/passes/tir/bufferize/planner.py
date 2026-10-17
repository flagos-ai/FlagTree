# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Alias-aware, per-function physical bufferization planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import gcd, isfinite, prod

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import (
    AliasInfo,
    AliasKind,
    AllocationPolicy,
    AllocationStrategy,
    BufferDescriptor,
    BufferPlan,
    CallBufferBinding,
    CallMemoryPoolBinding,
    FunctionBufferPlan,
    FunctionMemoryPool,
    KernelCallBufferBinding,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
    MemSpan,
    PhysicalBuffer,
)
from triton.flagmega.ir.bufferization.allocation_record import AllocationRecord
from triton.flagmega.ir.bufferization.leaves import buffer_leaves
from triton.flagmega.ir.dim_expr import DimVar, dim
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import (
    SBPSplit,
    block_owner_count,
    is_fully_replicated,
    is_fully_sharded_across_placement,
    is_local_shard_subview,
    launch_chip_count,
    local_shape,
    placement_owner_count,
)
from triton.flagmega.ir.model import (
    DistributedType,
    Function,
    IRModule,
    IRType,
    RefType,
    TensorType,
    TupleType,
    VectorType,
    logical_type,
)
from triton.flagmega.passes.tir.bufferize.alias_analysis import AliasAnalysis
from triton.flagmega.passes.tir.bufferize.alignment import (
    effective_alignment,
    transfer_source_alignment_requirements,
)
from triton.flagmega.passes.tir.bufferize.allocation import BufferLifetime
from triton.flagmega.passes.tir.bufferize.allocation_session import AllocationSession
from triton.flagmega.passes.tir.bufferize.call_input_ownership import non_consumable_parameters
from triton.flagmega.passes.tir.bufferize.graph import callee_first_functions, function_nodes
from triton.flagmega.passes.tir.bufferize.readonly_groups import ReadonlyGroupResolver
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
    optimization_level: str = "optimized"
    barrier_bytes_budget: int = 0
    barrier_fixpoint_rounds: int = 1
    # Target launch meshes with c/d axes load module readonly data by chip.
    # This is supplied before bufferization, when launch_contract metadata has
    # not yet been finalized.
    rdata_chip_sharding: bool = False

    def __post_init__(self):
        if self.optimization_level not in {"fast", "optimized"}:
            raise ValueError("Bufferization optimization_level must be 'fast' or 'optimized'.")
        if not isfinite(self.solver_time_seconds) or self.solver_time_seconds <= 0:
            raise ValueError("Bufferization solver_time_seconds must be finite and positive.")
        if isinstance(self.barrier_bytes_budget, bool) or not isinstance(self.barrier_bytes_budget, int) \
                or self.barrier_bytes_budget < 0:
            raise ValueError("Bufferization barrier_bytes_budget must be a non-negative integer.")
        if isinstance(self.barrier_fixpoint_rounds, bool) or not isinstance(self.barrier_fixpoint_rounds, int) \
                or self.barrier_fixpoint_rounds < 1:
            raise ValueError("Bufferization barrier_fixpoint_rounds must be a positive integer.")

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
                AllocationStrategy.REUSE, AllocationPolicy.GRANULARITY_ALIGNED,
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

    def __init__(self, module: IRModule, options: BufferizationOptions, *, reuse_preferences=None,
                 allocation_session=None) -> None:
        self.module = module
        self.options = options
        self.reuse_preferences = reuse_preferences or {}
        spaces = options.memory_space_map
        try:
            self.workspace_space = spaces[options.workspace]
            self.rdata_space = spaces[options.readonly_data]
        except KeyError as error:
            raise IRVerificationError(f"Missing required buffer memory space {error.args[0]!r}.") from error
        if not self.workspace_space.supports_lifetime_reuse:
            raise IRVerificationError("Workspace memory must support lifetime reuse.")
        if self.rdata_space.strategy is not AllocationStrategy.LINEAR:
            raise IRVerificationError("Readonly data memory must use linear allocation.")
        self.function_pool_spaces = {
            value.name: value
            for value in options.memory_spaces
            if value.supports_lifetime_reuse
            and value.allocation_scope is MemoryAllocationScope.FUNCTION
            and value.kind != "shared"
        }
        if self.workspace_space.name not in self.function_pool_spaces:
            raise IRVerificationError(
                "Default workspace must be a non-shared function-scoped lifetime pool."
            )
        self.allocator = allocation_session or AllocationSession(options)
        if (self.allocator.optimization_level != options.optimization_level
                or self.allocator.solver_time_seconds != options.solver_time_seconds):
            raise IRVerificationError("Allocation session does not match bufferization options.")
        self.allocation_records = []
        self.shared_space = spaces.get("shared")
        self.alias_analysis = AliasAnalysis()
        self.non_consumable_parameters = non_consumable_parameters(module)
        self.non_consumable_physical_buffers: set[str] = set()
        self.required_alignments = transfer_source_alignment_requirements(module)
        self.descriptors: dict[str, BufferDescriptor] = {}
        self.bindings: dict[str, tuple[str, ...]] = {}
        self.physical_buffers: dict[str, PhysicalBuffer] = {}
        self.function_plans: dict[str, FunctionBufferPlan] = {}
        self.readonly_result_views: dict[str, frozenset[tuple[str, str]]] = {}
        self.rdata_offset = 0
        self._physical_serial = 0
        # Tensor-descriptor transfers address a module-global tensor map.  A
        # chip-local component cannot be passed to those kernels because the
        # descriptor's base/stride contract is canonical.  Compute this from
        # selected-kernel capabilities and propagate it through graph-call
        # boundaries so both the formal parameter and its rdata source keep
        # the same physical representation.
        self.canonical_global_values = _canonical_global_values(module)
        # A module-scoped readonly pool is loaded once per process.  When the
        # launch mesh has device/node axes, rdata descriptors use one segment
        # per chip and retain only block-owner components in each segment.
        self.rdata_chip_sharding = bool(options.rdata_chip_sharding or launch_chip_count(module) > 1)

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
            optimization_level=self.options.optimization_level,
            allocation_records=tuple(self.allocation_records),
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
        if not self.shared_space.supports_lifetime_reuse:
            raise IRVerificationError("Shared memory must support lifetime allocation.")

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
            self._record_allocation(function.name, self.shared_space.name, allocated)
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
        # A parameter bank is readonly storage whose rows repeat one schema,
        # which is exactly the layout an rdata group describes. Expand each
        # field into per-row members so a row access becomes a byte offset.
        bank_nodes = self._parameter_bank_rdata_nodes()
        nodes = (*nodes, *bank_nodes)
        # Members are synthesized after canonical-global analysis ran, so
        # inherit the requirement from the row projection that consumes the
        # field: a descriptor kernel needs a module-global base address.
        for member in bank_nodes:
            if str(member.metadata.get("canonical_global", "")) == "true":
                self.canonical_global_values = self.canonical_global_values | {member.id}
        grouped: dict[str, list[tuple[object, Mapping[str, object]]]] = {}
        group_order: list[str] = []
        ungrouped = []
        group_resolver = ReadonlyGroupResolver(self.module)
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
            name = group_resolver.name_for(node, str(hint["name"]))
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
                required_alignment = self._required_alignment(node.id, leaf=buffer_id)
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
                    storage_kind = (
                        DistributedBufferStorageKind.CANONICAL_GLOBAL
                        if self._requires_canonical_global(node, value_type)
                        else _distributed_storage_kind(
                            value_type,
                            "rdata",
                            "readonly_data",
                            rdata_chip_sharding=self.rdata_chip_sharding,
                        )
                    )
                    self._allocate_rdata_physical(
                        physical_id,
                        _physical_nbytes(
                            value_type,
                            shape,
                            "rdata",
                            "readonly_data",
                            storage_kind=storage_kind,
                        ),
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
                for leaf, (buffer_id, value_type, field) in enumerate(
                    _leaf_types(node.id, node.type)
                ):
                    shape = _maximum_shape(value_type, self.module.stage, node.id)
                    required_alignment = self._required_alignment(node.id, leaf=buffer_id)
                    # Group members keep independent physical identities so
                    # layout equivalence can never deduplicate distinct layer
                    # values; their offsets stay contiguous.
                    physical_id = self._new_physical("rdata")
                    storage_kind = (
                        DistributedBufferStorageKind.CANONICAL_GLOBAL
                        if self._requires_canonical_global(node, value_type)
                        else _distributed_storage_kind(
                            value_type,
                            "rdata",
                            "readonly_data_group_member",
                            rdata_chip_sharding=self.rdata_chip_sharding,
                        )
                    )
                    member_nbytes = _physical_nbytes(
                        value_type,
                        shape,
                        "rdata",
                        "readonly_data_group_member",
                        storage_kind=storage_kind,
                    )
                    self._allocate_rdata_physical(
                        physical_id, member_nbytes, alignment=required_alignment,
                    )
                    member_offset = 0
                    ids.append(self._add_descriptor(
                        buffer_id, value_type, shape, "rdata", node.id,
                        field=field, physical_id=physical_id, weight_key=self._rdata_key(node),
                        offset=self._allocation(physical_id).offset,
                        rdata_group=group_name, group_index=index, group_count=count,
                        role="readonly_data_group_member",
                        alignment=required_alignment,
                        distributed_storage_kind=storage_kind,
                        fused_weight_keys=node.attrs.get("fused_keys", ()),
                    ))
                self.bindings[node.id] = tuple(ids)

        # Index each bank field's row-0 member so a row access can bind to it
        # and let codegen stride by the runtime row index.
        self.bank_group_members = {}
        for node in bank_nodes:
            if int(node.metadata.get("bank_row", -1)) != 0:
                continue
            bank_id = str(node.metadata["parameter_bank"])
            field_index = int(node.metadata["bank_field_index"])
            leaves = self.bindings.get(node.id, ())
            self.bank_group_members.setdefault(bank_id, {}).setdefault(
                field_index, []).extend(leaves)
        self.bank_group_members = {
            bank: {index: tuple(values) for index, values in fields.items()}
            for bank, fields in self.bank_group_members.items()
        }

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
        # A view op aliases its source's storage rather than copying it, so the
        # source is still being read wherever the view is read.  Propagate each
        # view's last use back to its source until it settles; without this a
        # row access dies at its own get_items while the call consuming those
        # views runs later, and that call's pool frame may be placed over the
        # bytes it is reading.
        aliasing = {"builtin.get_item", "builtin.tuple"}
        for _ in range(len(nodes)):
            changed = False
            for node in nodes:
                if node.op not in aliasing:
                    continue
                reach = last_use.get(node.id)
                if reach is None:
                    continue
                for input_id in node.inputs:
                    if input_id in local_index and last_use[input_id] < reach:
                        last_use[input_id] = reach
                        changed = True
            if not changed:
                break
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
            if _is_parameter_bank(logical_type(node.type)):
                # Readonly bank storage is addressed by row access inside the
                # callee, exactly as a weight root is in its own function, so
                # it owns no planned buffer on either side of the call.
                continue
            ids = []
            for buffer_id, value_type, field in _leaf_types(node.id, node.type):
                shape = _maximum_shape(value_type, self.module.stage, node.id)
                # A rank-zero tensor of vectors still owns all element lanes
                # in memory; it cannot be passed as one scalar SSA value.
                scalar = not shape and not isinstance(logical_type(value_type).dtype, VectorType)
                storage = "scalar" if scalar else (
                    "state" if isinstance(logical_type(node.type), RefType) and function.name == self.module.entry
                    else "input" if function.name == self.module.entry
                    else "parameter"
                )
                ids.append(self._add_descriptor(
                    buffer_id, value_type, shape, storage, node.id,
                    field=field, physical_id=f"external:{function.name}:{buffer_id}",
                    function=function.name, role="scalar_parameter" if scalar else "parameter",
                    alignment=self._required_alignment(node.id, leaf=buffer_id),
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
                "builtin.weight", "builtin.const_asset", "builtin.parameter_bank",
                "tir.parameter_bank", "tir.buffer"
            }:
                # A parameter bank is a readonly storage root, like a weight:
                # its Ref value names stored rows rather than aliasing another
                # reference, so it owns no planned buffer of its own.
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
            if node.op in {"control.bank_get", "tir.bank_get"}:
                # A row access is a readonly view selected at runtime: bind it
                # to its fields' rdata group members so the generated code
                # addresses them as "group base + index * per-row stride"
                # rather than copying a row into workspace.
                self.bindings[node.id] = self._plan_bank_row(
                    function, node, index, last_use[node.id]
                )
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
            if node.op == "tir.ref_slice":
                self._plan_ref_slice(function, node)
                continue
            if node.op == "tir.buffer_subspan":
                self._plan_tensor_subspan(function, node, index, last_use[node.id])
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
                    node.op in {"tir.call", "control.for"}
                    and str(node.attrs.get("callee", "")) in self.module.function_map
                )
            ):
                # A structured loop calls its body region once per iteration,
                # so it needs the same argument/result buffer binding as an
                # ordinary call. Without it the loop has nothing to render.
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
                bytes_budget=self.options.barrier_bytes_budget,
            )
            for name, space in self.function_pool_spaces.items()
        }

        for name, result in allocation_results.items():
            self._record_allocation(function.name, name, result)
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
        # A readonly parameter bank contributes no buffer, so it does not
        # appear in the physical parameter list at all.
        parameters = tuple(
            (node_id, self.bindings[node_id])
            for node_id in function.parameters
            if node_id in self.bindings
        )
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
        self.readonly_result_views[function.name] = frozenset(
            (result, parameter) for result, parameter in result_aliases
            if result != parameter and all(node.effect.is_pure for node in nodes)
            and self._readonly_view_path(result, parameter)
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
                if self.bindings.get(node.id)
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
            DistributedBufferStorageKind.COMPACT_PER_CHIP,
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
        aliases_by_id = dict(aliases)
        canonical_offsets = {}

        def canonical_offset(buffer_id):
            if buffer_id in canonical_offsets:
                return canonical_offsets[buffer_id]
            descriptor = aliases_by_id[buffer_id]
            parent_id = descriptor.alias_of
            if parent_id in aliases_by_id:
                offset = canonical_offset(parent_id)
                node = self.module.node_map.get(descriptor.source_node)
                if node is not None and node.op == "tir.buffer_subspan":
                    from triton.flagmega.ir.ops.tir.buffer_subspan import dense_subspan_offset
                    offset += dense_subspan_offset(aliases_by_id[parent_id].shape, descriptor.shape,
                                                    node.attrs["offsets"], descriptor.dtype.itemsize)
                elif descriptor.mem_span.start != aliases_by_id[parent_id].mem_span.start:
                    raise IRVerificationError("Canonical promotion requires a typed subspan offset.", node_id=use_node_id)
            else:
                offset = descriptor.byte_offset
            canonical_offsets[buffer_id] = offset
            return offset

        required_size = max(
            canonical_offset(buffer_id)
            + prod(descriptor.shape, start=1) * descriptor.dtype.itemsize
            for buffer_id, descriptor in aliases
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
                canonical_offsets[buffer_id],
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
                owner_stride_bytes=None,
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

    def _readonly_view_path(self, result: str, parameter: str) -> bool:
        """Reuse a callee's verified full-span view proof at the call boundary."""
        current, visited = result, set()
        while current != parameter:
            if current in visited:
                return False
            visited.add(current)
            alias = self.descriptors[current].alias
            if alias is None or alias.kind not in {AliasKind.VIEW, AliasKind.IDENTITY}:
                return False
            current = alias.source
        return self.descriptors[result].mem_span.must_alias(self.descriptors[parameter].mem_span)

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
        # A For's first three operands are its bounds and its body's first
        # parameter is the induction value, which the loop supplies rather
        # than the caller. Everything after that binds positionally.
        is_loop = node.op == "control.for"
        # The loop's own induction value binds the body's first formal, so it
        # is planned as a scalar the loop produces rather than a caller edge.
        induction = callee.parameters[0] if is_loop else None
        formals = tuple(callee.parameters[1:]) if is_loop else tuple(callee.parameters)
        actuals = tuple(node.inputs[3:]) if is_loop else tuple(node.inputs)
        if len(actuals) != len(formals):
            raise IRVerificationError(
                f"Call {node.id!r} passes {len(actuals)} arguments to @{callee_name}, "
                f"which requires {len(formals)}.",
                stage=self.module.stage,
                node_id=node.id,
            )
        arguments = []
        actual_node_for_formal: dict[str, str] = {}
        if induction is not None:
            # The induction value is produced by the loop, so its formal is
            # its own actual: one scalar slot the loop writes each iteration.
            for formal in self.bindings.get(induction, ()):
                arguments.append((formal, formal))
                actual_node_for_formal[formal] = induction
        for formal_id, actual_id in zip(formals, actuals):
            if formal_id not in self.bindings or actual_id not in self.bindings:
                # Readonly parameter-bank storage is addressed by row access
                # inside the callee, so neither side owns a planned buffer and
                # there is nothing to bind for this argument position.
                continue
            # Keep distinct names: the pair being iterated is the whole
            # argument list, so rebinding it here would truncate the walk.
            formal_leaves = self.bindings[formal_id]
            actual_leaves = self.bindings[actual_id]
            if len(formal_leaves) != len(actual_leaves):
                raise IRVerificationError(
                    f"Call {node.id!r} argument {actual_id!r} does not match formal {formal_id!r}.",
                    stage=self.module.stage,
                    node_id=node.id,
                )
            for formal, actual in zip(formal_leaves, actual_leaves):
                self._reconcile_call_argument_storage_abi(formal, actual)
                self._verify_call_argument_physical_abi(
                    node.id, formal, actual)
            arguments.extend(zip(formal_leaves, actual_leaves))
            actual_node_for_formal.update((formal, actual_id) for formal in formal_leaves)
        formal_to_actual = dict(arguments)
        alias_results = {}
        readonly_views = self.readonly_result_views[callee_name]
        view_results = set()
        for result, parameter in callee_plan.result_aliases:
            actual = formal_to_actual[parameter]
            descriptor = self.descriptors[actual]
            actual_node = actual_node_for_formal[parameter]
            readonly_view = (result, parameter) in readonly_views
            if readonly_view:
                view_results.add(result)
            identity_passthrough = result == parameter or readonly_view
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
                    reinterpret=formal_result in view_results,
                    alias_kind=(
                        AliasKind.VIEW if formal_result in view_results
                        else AliasKind.RESULT if alias is not None else None
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
        # A callee's pool frame holds results the caller keeps reading after the
        # call returns.  Reserving it for the call step alone lets the caller
        # place its own buffers inside the frame, on top of data a later step
        # still reads, so the frame stays live until the last use of anything
        # the call produced.
        frame_live_end = max(
            [index, last_use.get(node.id, index)]
            + [
                last_use.get(user, index)
                for user in get_item_users.get(node.id, ())
            ]
        )
        memory_pools = []
        for pool in callee_plan.memory_pools:
            if not pool.requires_binding:
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
                frame_live_end,
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
            or formal.component_stride_bytes != actual.component_stride_bytes
        ):
            raise IRVerificationError(
                f"Call {call_id!r} actual buffer {actual_id!r} does not match "
                f"formal buffer {formal_id!r} physical ABI "
                "(dtype/shape/stride/distribution/storage kind/backing).",
                stage=self.module.stage,
                node_id=call_id,
            )

    def _reconcile_call_argument_storage_abi(
        self,
        formal_id: str,
        actual_id: str,
    ) -> None:
        """Promote a chip-local boundary value when the callee needs global coordinates.

        A callee parameter can become canonical after its body exposes a
        sharded view that changes the physical owner axes.  The caller's
        boundary reshard is planned before the call itself and would
        otherwise retain the compact chip-local component selected from its
        role.  Promote that caller alias group before ABI verification so the
        boundary allocation and the finalized formal use one physical
        representation.  This is driven by the formal storage contract and
        applies to every reusable function, independent of model names or
        layer indices.
        """
        formal = self.descriptors[formal_id]
        actual = self.descriptors[actual_id]
        if (
            formal.distributed_storage_kind
            is not DistributedBufferStorageKind.CANONICAL_GLOBAL
            or actual.distributed_storage_kind
            not in {
                DistributedBufferStorageKind.COMPACT_PER_OWNER,
                DistributedBufferStorageKind.COMPACT_PER_CHIP,
            }
            or formal.distributed_type != actual.distributed_type
        ):
            return
        self._promote_alias_group_to_canonical(actual.id, actual.source_node)

    def _plan_value(
        self,
        function,
        source_node,
        value_type,
        live_start,
        live_end,
        output_leaf_ids,
        *,
        prefix,
        field=None,
        alias=None,
        alignment=None,
        role=None,
        reinterpret=False,
        alias_kind=None,
        distributed_storage_kind=None,
        distributed_backing_type=None,
        owner_stride_bytes=None,
        memory_space: MemorySpace | None = None,
    ) -> tuple[str, ...]:
        # Each leaf of an aggregate is its own buffer with its own declared
        # contract. Resolving the requirement per leaf below keeps a strict
        # field from padding the per-owner stride of its siblings.
        requested_alignment = int(alignment or 1)
        leaves = _leaf_types(prefix, value_type)
        if alias is not None and len(leaves) != 1:
            # Ref aliases are already flattened by _ref_alias.
            if isinstance(alias, tuple) and len(alias) == len(leaves):
                result = []
                for leaf, source in zip(leaves, alias):
                    result.extend(
                        self._plan_value(
                            function,
                            source_node,
                            leaf[1],
                            live_start,
                            live_end,
                            output_leaf_ids,
                            prefix=leaf[0],
                            field=leaf[2],
                            alias=source,
                            alignment=requested_alignment,
                            role=role,
                            reinterpret=reinterpret,
                            alias_kind=alias_kind,
                            distributed_storage_kind=distributed_storage_kind,
                            distributed_backing_type=distributed_backing_type,
                            owner_stride_bytes=owner_stride_bytes,
                            memory_space=memory_space,
                        ))
                return tuple(result)
            raise IRVerificationError(f"Alias arity mismatch for value {prefix!r}.")
        result = []
        for buffer_id, tensor_type, leaf_field in leaves:
            alignment = max(
                requested_alignment,
                self._required_alignment(source_node, leaf=buffer_id),
            )
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
                    self._promote_workspace_alias_group_to_result(function, source, buffer_id)
                    source = self.descriptors[str(alias)]
                target_tensor = _tensor_type(tensor_type)
                if reinterpret and source.distributed_backing_type is not None:
                    # A coarser shard backing names axes of the source type.
                    # Reshape/Bitcast may change those axes and element bytes;
                    # expose canonical coordinates for the whole alias group
                    # rather than dropping the backing and shrinking the span.
                    self._promote_alias_group_to_canonical(source.id, source_node)
                    source = self.descriptors[str(alias)]
                    distributed_storage_kind = DistributedBufferStorageKind.CANONICAL_GLOBAL
                    distributed_backing_type = None
                # A view never changes physical storage. In particular, a
                # sharded coordinate view over a canonical parameter/rdata
                # buffer remains canonical-global; compact-per-owner storage
                # is created only by an allocating producer.
                target_kind = (source.distributed_storage_kind if distributed_storage_kind is None else
                               DistributedBufferStorageKind(distributed_storage_kind))
                target_backing_type = distributed_backing_type
                if (target_backing_type is None and not reinterpret and isinstance(tensor_type, DistributedType)
                        and tensor_type == source.distributed_type and not target_kind.exposes_logical_coordinates):
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
                        target_kind in {
                            DistributedBufferStorageKind.COMPACT_PER_OWNER,
                            DistributedBufferStorageKind.COMPACT_PER_CHIP,
                        }
                        and source.distributed_storage_kind
                        not in {
                            DistributedBufferStorageKind.COMPACT_PER_OWNER,
                            DistributedBufferStorageKind.COMPACT_PER_CHIP,
                        }
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
                    # A runtime-indexed parent (a parameter-bank row) has no
                    # fixed byte offset; carry its symbolic start instead.
                    byte_offset=(
                        source.mem_span.start if not source.mem_span.start.is_fixed
                        else source.byte_offset
                    ),
                    offset=(0 if not source.mem_span.start.is_fixed else source.offset),
                    live_start=live_start if self._is_function_pool_storage(source.storage) else None,
                    live_end=live_end if self._is_function_pool_storage(source.storage) else None,
                    function=function.name, role=role or "alias",
                    alignment=alignment,
                    distributed_storage_kind=target_kind,
                    distributed_backing_type=target_backing_type,
                    owner_stride_bytes=source.owner_stride_bytes,
                    offset_bindings=source.offset_bindings,
                    alias_kind=alias_kind,
                ))
                continue
            selected_space = memory_space or self.workspace_space
            storage = "output" if is_output and function.name == self.module.entry else (
                "return" if is_output else selected_space.name)
            physical_id = (f"external:{function.name}:result:{buffer_id}"
                           if storage in {"output", "return"} else self._new_physical(selected_space.name))
            result.append(
                self._add_descriptor(
                    buffer_id,
                    tensor_type,
                    shape,
                    storage,
                    source_node,
                    field=leaf_field if leaf_field is not None else field,
                    physical_id=physical_id,
                    live_start=live_start if self._is_function_pool_storage(storage) else None,
                    live_end=live_end if self._is_function_pool_storage(storage) else None,
                    function=function.name,
                    role=role or ("function_result" if is_output else "temporary"),
                    alignment=alignment,
                    distributed_storage_kind=distributed_storage_kind,
                    distributed_backing_type=distributed_backing_type,
                    owner_stride_bytes=owner_stride_bytes,
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
                if (node.id in output_leaf_ids or (result_index is not None
                                                   and f"{node.id}.{result_index}" in output_leaf_ids)) and any(
                    other.physical_id == source.physical_id and not other.mem_span.must_alias(source.mem_span)
                    for other in self.descriptors.values()
                ):
                    # A borrowed interval cannot become an owning result by
                    # promoting its larger parent allocation to a small ABI.
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
            input_id for input_id in inputs if isinstance(logical_type(self.module.node_map[input_id].type), RefType)
        ]
        if len(candidates) > 1:
            # A loop may capture readonly parameter-bank storage alongside the
            # state it carries. Only a Ref of the same kind can be the identity
            # source, so a captured bank never competes with the carried state.
            matching = [
                input_id for input_id in candidates
                if logical_type(self.module.node_map[input_id].type) == logical_type(field_type)
            ]
            if len(matching) == 1:
                candidates = matching
        if len(candidates) != 1:
            raise IRVerificationError("A Ref result requires exactly one Ref operand.")
        source = candidates[0]
        if logical_type(self.module.node_map[source].type) != logical_type(field_type):
            raise IRVerificationError(
                "A Ref identity result must preserve its input type; use an explicit reference view.")
        return self.bindings[source]

    def _plan_ref_slice(self, function, node):
        from math import gcd

        from triton.flagmega.ir.dim_expr import DimVar, dim

        source_id, index_id = node.inputs
        source_type = self.module.node_map[source_id].type
        index = self.module.node_map[index_id]
        length = int(node.attrs["length"])
        extent = source_type.fields[0][1].shape[0].fixed_value
        offset_bindings = ()
        if index.op in {"builtin.scalar_const", "tir.scalar_const"}:
            coordinate = dim(int(index.attrs["value"]))
        else:
            bound = self.bindings[index_id]
            if len(bound) != 1 or self.descriptors[bound[0]].storage != "scalar":
                raise IRVerificationError("RefSlice index requires a scalar buffer binding.", node_id=node.id)
            symbol = "ref_index_" + bound[0].encode("utf-8").hex()
            coordinate = DimVar(symbol, 0, extent - length)
            offset_bindings = ((symbol, bound[0]), )
        views = []
        leaves = _leaf_types(node.id, node.type)
        for source_buffer_id, (buffer_id, value_type, field) in zip(self.bindings[source_id], leaves, strict=True):
            source = self.descriptors[source_buffer_id]
            shape = _maximum_shape(value_type, self.module.stage, node.id)
            row_bytes = source.strides[0] * source.dtype.itemsize
            view = source.subview(buffer_id, dtype=source.dtype, shape=shape, strides=source.strides,
                                  byte_offset=coordinate * row_bytes, byte_size=length * row_bytes,
                                  alignment=gcd(source.alignment, row_bytes), source_node=node.id, field=field,
                                  role="reference_view", offset_bindings=offset_bindings)
            self.descriptors[buffer_id] = view
            self.alias_analysis.add_alias(buffer_id, source_buffer_id, byte_offset=coordinate * row_bytes,
                                          nbytes=length * row_bytes, kind=AliasKind.VIEW)
            views.append(buffer_id)
        self.bindings[node.id] = tuple(views)

    def _plan_tensor_subspan(self, function, node, index, live_end):
        from math import gcd

        from triton.flagmega.ir.ops.tir.buffer_subspan import dense_subspan_offset

        [source_id] = self.bindings[node.inputs[0]]
        source = self.descriptors[source_id]
        shape = _maximum_shape(node.type, self.module.stage, node.id)
        backing = source.distributed_backing_type
        if backing is not None:
            backing = replace(backing, tensor=_tensor_type(node.type))
        component = _component_shape(node.type, shape, source.distributed_storage_kind,
                                     distributed_backing_type=backing)
        if source.strides != _dense_strides(source.component_shape):
            raise IRVerificationError("Tensor subspan requires contiguous source storage.", node_id=node.id)
        offset = dense_subspan_offset(source.component_shape, component, node.attrs["offsets"], source.dtype.itemsize)
        size = prod(component) * source.dtype.itemsize
        alignment = gcd(source.alignment, offset)
        if alignment < self._required_alignment(node.id):
            raise IRVerificationError("Tensor subspan does not satisfy consumer alignment.", node_id=node.id)
        view = source.subview(node.id, dtype=source.dtype, shape=shape, strides=_dense_strides(component),
                              byte_offset=offset, byte_size=size, alignment=alignment, source_node=node.id,
                              role="tensor_subspan")
        view = replace(view, distributed_type=node.type if isinstance(node.type, DistributedType) else None,
                       distributed_storage_kind=source.distributed_storage_kind, distributed_backing_type=backing,
                       owner_stride_bytes=(source.component_stride_bytes if source.distributed_storage_kind
                                           in {
                                               DistributedBufferStorageKind.COMPACT_PER_OWNER,
                                               DistributedBufferStorageKind.COMPACT_PER_CHIP,
                                           } else None),
                       live_start=index if self._is_function_pool_storage(source.storage) else None,
                       live_end=live_end if self._is_function_pool_storage(source.storage) else None)
        self.descriptors[node.id] = view
        self.alias_analysis.add_alias(node.id, source_id, byte_offset=offset, nbytes=size, kind=AliasKind.VIEW)
        self.bindings[node.id] = (node.id,)

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
        alignment=None, distributed_storage_kind=None, fused_weight_keys=(),
        offset_bindings=(),
        distributed_backing_type=None, alias_kind=None,
        owner_stride_bytes=None,
    ):
        if buffer_id in self.descriptors:
            raise IRVerificationError(f"Duplicate logical buffer id {buffer_id!r}.")
        if physical_id is None:
            raise IRVerificationError(f"Logical buffer {buffer_id!r} has no PhysicalBuffer identity.")
        tensor = _tensor_type(value_type)
        distributed = value_type if isinstance(value_type, DistributedType) else None
        storage_space = self.options.memory_space_map.get(storage)
        storage_kind = (
            DistributedBufferStorageKind.CANONICAL_GLOBAL
            if distributed_storage_kind is None
            and source_node in self.canonical_global_values
            and isinstance(value_type, DistributedType)
            else (
                _distributed_storage_kind(
                    value_type,
                    storage,
                    role,
                    function_pool_sharing_scope=(
                        None if storage_space is None else storage_space.sharing_scope
                    ),
                    function_pool=(storage in self.function_pool_spaces),
                    rdata_chip_sharding=self.rdata_chip_sharding,
                )
                if distributed_storage_kind is None
                else DistributedBufferStorageKind(distributed_storage_kind)
            )
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
        if storage_kind in {
            DistributedBufferStorageKind.COMPACT_PER_OWNER,
            DistributedBufferStorageKind.COMPACT_PER_CHIP,
        }:
            required_alignment = int(alignment or 1)
            if alias_of is None and owner_stride_bytes is None:
                stride = _align_up(nbytes, required_alignment)
                if stride != nbytes:
                    owner_stride_bytes = stride
            stride = nbytes if owner_stride_bytes is None else owner_stride_bytes
            if stride % required_alignment:
                raise IRVerificationError(f"Buffer {buffer_id!r} owner stride violates its alignment contract.")
            owner_count = (
                placement_owner_count(distributed)
                if storage_kind is DistributedBufferStorageKind.COMPACT_PER_OWNER
                else block_owner_count(distributed)
            )
            physical_nbytes = max(physical_nbytes, stride * (owner_count - 1) + nbytes)
        physical_id = str(physical_id)
        if alias_of is not None:
            source = self.descriptors[str(alias_of)]
            physical = source.mem_span.buffer
            symbolic_start = not getattr(byte_offset, "is_fixed", True) or \
                not source.mem_span.start.is_fixed
            if symbolic_start:
                # A runtime-selected parameter-bank row moves in whole-member
                # strides, so every row keeps the member's own alignment.
                descriptor_alignment = source.alignment
            else:
                descriptor_alignment = gcd(source.alignment, byte_offset - source.byte_offset)
            if descriptor_alignment < int(alignment or 1):
                raise IRVerificationError(f"Alias {buffer_id!r} does not satisfy its required alignment.")
            resolved_alias_kind = (
                AliasKind(alias_kind)
                if alias_kind is not None
                else AliasKind.VIEW
                if symbolic_start or byte_offset != source.mem_span.byte_offset
                else AliasKind.INPLACE
            )
            alias = AliasInfo(source.id, resolved_alias_kind)
        else:
            descriptor_alignment = max(
                storage_space.granularity if storage_space is not None else self.workspace_space.granularity,
                int(alignment or 1),
            )
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
            alignment=(gcd(descriptor_alignment, nbytes if owner_stride_bytes is None else owner_stride_bytes)
                       if storage_kind in {
                           DistributedBufferStorageKind.COMPACT_PER_OWNER,
                           DistributedBufferStorageKind.COMPACT_PER_CHIP,
                       } and nbytes
                       else descriptor_alignment),
            mem_span=MemSpan(physical, byte_offset, nbytes),
            offset_bindings=tuple(offset_bindings),
            source_node=source_node,
            field=field,
            alias=alias,
            # A zero-copy view of readonly data denotes the same stored bytes,
            # so it names the same checkpoint tensor as its source. Without
            # this the packer sees an rdata buffer with no tensor behind it.
            weight_key=(
                weight_key
                if weight_key is not None or alias_of is None
                else self.descriptors[alias_of].weight_key
            ),
            fused_weight_keys=(
                tuple(fused_weight_keys)
                if fused_weight_keys or alias_of is None
                else self.descriptors[alias_of].fused_weight_keys
            ),
            rdata_group=(
                rdata_group
                if rdata_group is not None or alias_of is None
                else self.descriptors[alias_of].rdata_group
            ),
            group_index=(
                group_index
                if group_index is not None or alias_of is None
                else self.descriptors[alias_of].group_index
            ),
            group_count=(
                group_count
                if group_count is not None or alias_of is None
                else self.descriptors[alias_of].group_count
            ),
            live_start=live_start,
            live_end=live_end,
            function=function,
            role=role,
            distributed_type=distributed,
            distributed_storage_kind=storage_kind,
            distributed_backing_type=distributed_backing_type,
            owner_stride_bytes=owner_stride_bytes,
        )
        if alias_of is None:
            self.alias_analysis.define_span(buffer_id, descriptor.mem_span)
        else:
            parent = self.alias_analysis.view(alias_of)
            # A runtime-indexed parent carries a symbolic start, so the child's
            # offset relative to it is what matters; the absolute address is
            # resolved at run time from the same binding.
            relative = (
                dim(0)
                if not parent.mem_span.start.is_fixed
                else byte_offset - parent.byte_offset
            )
            view = self.alias_analysis.add_alias(
                buffer_id,
                alias_of,
                byte_offset=relative,
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

    def _required_alignment(self, node_id, *, leaf: str | None = None) -> int:
        """Declared contract for a value, or for one of its buffer leaves."""
        return effective_alignment(self.required_alignments, node_id, leaf)

    def _allocation(self, physical_id):
        return self.physical_buffers[physical_id]

    def _new_physical(self, prefix):
        value = f"{prefix}:{self._physical_serial}"
        self._physical_serial += 1
        return value

    def _record_allocation(self, function, space, result):
        self.allocation_records.append(AllocationRecord(
            function, space, self.allocator.name, result.status,
            tuple((objective.name, objective.status, objective.value, objective.best_bound)
                  for objective in result.objectives),
        ))

    def _plan_bank_row(self, function, node, live_start, live_end):
        """Bind one bank row to the index-0 member of each field's group.

        The group members are contiguous and identically typed, so a runtime
        row index is a byte stride away. Codegen reads the row index from the
        loop and applies that stride; recording the group identity here is
        what lets it do so.
        """
        bank_id = self._bank_of_value(node.inputs[0])
        if bank_id is None:
            raise IRVerificationError(
                f"Bank row {node.id!r} does not resolve to parameter-bank storage.",
                stage=self.module.stage, node_id=node.id,
            )
        members = self.bank_group_members.get(bank_id, {})
        schema = self.module.node_map[bank_id].attrs["schema"]
        row_types = {
            index: value for index, value in enumerate(
                node.type.fields if hasattr(node.type, "fields") else ()
            )
        }
        ids: list[str] = []
        for index, field in enumerate(schema.fields):
            leaves = members.get(index)
            if leaves is None:
                # A literal field, or one whose transform was absorbed into a
                # fused field, has no stored rows. Plan it at the type the row
                # declares so the value binding still matches its IR type.
                declared = row_types.get(index, field.type)
                # A row view is consumed by a later call, so its storage must
                # stay live until that last use.  Planning it at [0, 0] lets a
                # callee's pool frame overlap bytes the call itself still
                # reads, which corrupts the row during the call.
                ids.extend(self._plan_value(
                    function, f"{node.id}.{index}", declared,
                    live_start, live_end, frozenset(),
                    prefix=f"{node.id}.{index}",
                ))
                continue
            ids.extend(leaves)
        return tuple(ids)

    def _bank_of_value(self, value_id):
        node = self.module.node_map.get(value_id)
        if node is None:
            return None
        if node.op in {"builtin.parameter_bank", "tir.parameter_bank"}:
            return node.id
        for owner in self.module.nodes:
            if owner.op != "control.for":
                continue
            region = self.module.function_map.get(str(owner.attrs.get("callee", "")))
            if region is None:
                continue
            for parameter, actual in zip(region.parameters[1:], owner.inputs[3:]):
                if parameter == value_id:
                    return self._bank_of_value(actual)
        return None

    def _parameter_bank_rdata_nodes(self):
        """Synthesize one readonly member per bank row and field."""
        from triton.flagmega.ir.model import Node

        result = []
        for node in self.module.nodes:
            if node.op not in {"builtin.parameter_bank", "tir.parameter_bank"}:
                continue
            schema = node.attrs["schema"]
            rows = node.attrs["rows"]
            source = str(node.attrs["source"])
            # Only a field some row projection actually reads needs storage.
            # A field whose transform was absorbed into a fused one is dead,
            # and giving it rdata would claim the same checkpoint tensor twice.
            consumed = self._consumed_bank_field_indices(node.id)
            canonical_fields = self._canonical_bank_field_indices(node.id)
            consumed_types = self._consumed_bank_field_types(node.id)
            for index, field in enumerate(schema.fields):
                if field.transform != "checkpoint" or index not in consumed:
                    continue
                for row_index, row in enumerate(rows):
                    keys = row[index] if isinstance(row[index], tuple) else (row[index],)
                    # A fused field is one physical buffer per row: the
                    # program concatenates its parts offline, so storage holds
                    # the fused result, not the individual sources.
                    entries = (
                        tuple(zip(keys, field.stored_types, strict=True))
                        if len(keys) == 1
                        else ((keys, field.type),)
                    )
                    for part, (key, stored) in enumerate(entries):
                        suffix = ""
                        # One stored source keeps the type its row projection
                        # is consumed at, including any distribution, so the
                        # projection aliases the member directly; a fused
                        # field keeps each part's own stored type.
                        member_type = consumed_types.get(index, stored)
                        result.append(Node(
                            f"{node.id}.{field.name}{suffix}.row{row_index}",
                            "builtin.weight", (), member_type,
                            attrs={
                                "name": key if isinstance(key, str) else key[0],
                                "source": source,
                                "key": key if isinstance(key, str) else key[0],
                                **({} if isinstance(key, str) else {"fused_keys": tuple(key)}),
                            },
                            metadata={
                                "parameter_bank": node.id,
                                "bank_field": field.name,
                                "bank_field_index": index,
                                "bank_row": row_index,
                                "canonical_global": (
                                    "true" if index in canonical_fields else "false"
                                ),
                                "rdata_group": {
                                    "name": f"{node.id}.{field.name}{suffix}",
                                    "index": row_index,
                                    "count": len(rows),
                                },
                            },
                        ))
        return tuple(result)

    def _requires_canonical_global(self, node, value_type) -> bool:
        """Whether an rdata root must expose module-global coordinates.

        A parameter-bank row is selected at runtime, so every consumer
        addresses it from the module-global base instead of a shard-local
        origin. That holds whether or not the stored member itself carries a
        distribution, which is why this is checked before the distributed
        storage-kind rules.
        """
        if not isinstance(value_type, DistributedType):
            # An undistributed root already exposes its whole allocation;
            # canonical-global is only representable for distributed storage.
            return False
        return (
            node.id in self.canonical_global_values
            or str(node.metadata.get("canonical_global", "")) == "true"
        )

    def _consumed_bank_field_types(self, bank_id: str) -> dict:
        """Return the type each field's row projection is consumed at.

        AutoDistributed gives a projection its own distributed type while the
        schema records the logical one. Storage must match the consumer so a
        row access can alias the member without a physical conversion.
        """
        banks = {bank_id}
        for node in self.module.nodes:
            if node.op != "control.for":
                continue
            region = self.module.function_map.get(str(node.attrs.get("callee", "")))
            if region is None:
                continue
            for parameter, actual in zip(region.parameters[1:], node.inputs[3:]):
                if actual in banks:
                    banks.add(parameter)
        rows = {
            node.id for node in self.module.nodes
            if node.op in {"control.bank_get", "tir.bank_get"} and node.inputs[0] in banks
        }
        return {
            int(node.attrs["index"]): node.type
            for node in self.module.nodes
            if node.op == "builtin.get_item" and node.inputs and node.inputs[0] in rows
        }

    def _canonical_bank_field_indices(self, bank_id: str) -> frozenset[int]:
        """Return the field indices whose row projection needs a global base.

        A row is selected at runtime, so every consumer addresses it from the
        module-global rdata base rather than from a shard-local origin. Views
        built on such a row therefore require canonical coordinates, exactly
        as an undistributed rdata root does.
        """
        return self._consumed_bank_field_indices(bank_id)

    def _consumed_bank_field_indices(self, bank_id: str) -> frozenset[int]:
        """Return the field indices read through this bank's row accesses."""
        banks = {bank_id}
        for node in self.module.nodes:
            if node.op != "control.for":
                continue
            region = self.module.function_map.get(str(node.attrs.get("callee", "")))
            if region is None:
                continue
            for parameter, actual in zip(region.parameters[1:], node.inputs[3:]):
                if actual in banks:
                    banks.add(parameter)
        rows = {
            node.id for node in self.module.nodes
            if node.op in {"control.bank_get", "tir.bank_get"} and node.inputs[0] in banks
        }
        return frozenset(
            int(node.attrs["index"]) for node in self.module.nodes
            if node.op == "builtin.get_item" and node.inputs and node.inputs[0] in rows
        )

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
    allocation_session: AllocationSession | None = None,
) -> BufferPlan:
    from .reuse_preferences import collect_reuse_preferences, dominates_memory_schedule

    resolved_options = options or BufferizationOptions.generic(alignment=alignment)
    session = allocation_session or AllocationSession(resolved_options)
    baseline = BufferPlanner(module, resolved_options, allocation_session=session).run()
    if resolved_options.optimization_level == "fast":
        return baseline
    preferences = collect_reuse_preferences(module, baseline)
    if not preferences:
        return baseline
    # Separating one reuse pair can expose another; within the byte budget the
    # fixed point collects every round's newly exposed pairs. Each round must
    # stay a strict improvement over the previous best, or it is discarded.
    best = baseline
    seen = frozenset()
    for _ in range(max(1, resolved_options.barrier_fixpoint_rounds)):
        if preferences == seen or not preferences:
            break
        seen = frozenset(preferences)
        candidate = BufferPlanner(
            module, resolved_options, reuse_preferences=preferences, allocation_session=session,
        ).run()
        # Both alternatives are ordinary verified SAT placements. Select only
        # a Pareto improvement in synchronization and pool size; solver
        # failures remain errors, and all surviving hazards are materialized
        # normally.
        if not dominates_memory_schedule(module, candidate, best,
                                         bytes_budget=resolved_options.barrier_bytes_budget):
            break
        best = candidate
        preferences = collect_reuse_preferences(module, best) | preferences
    return best


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

        if node.op in {"distributed.sharded_view", "tir.buffer_view", "tir.buffer_subspan"}:
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
        "tir.buffer_subspan",
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
    return buffer_leaves(prefix, value_type)


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
    rdata_chip_sharding=False,
):
    if not isinstance(value_type, DistributedType):
        return DistributedBufferStorageKind.COMPACT_LOCAL
    if value_type.exclusive is not None:
        if any(not value_type.placement.is_physical_block_axis(axis) for axis in value_type.exclusive.axes):
            raise IRVerificationError("Exclusive SBP requires physical block placement axes.")
        if function_pool and function_pool_sharing_scope is MemorySharingScope.BLOCK:
            return DistributedBufferStorageKind.EXCLUSIVE_LOCAL
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
    chip_split = any(
        isinstance(policy, SBPSplit)
        and any(
            axis in {index for index, level in enumerate(value_type.placement.hierarchy_levels) if level in {"c", "d"}}
            for stage in policy.stages
            for axis in stage.hierarchy_axes
        )
        for policy in value_type.axis_policies
    )
    if (
        rdata_chip_sharding
        and chip_split
        and value_type.placement.physical_level_size("c")
        * value_type.placement.physical_level_size("d") > 1
        and (storage == "parameter" or role in {"parameter", "function_argument", "function_result"})
    ):
        # Function boundaries in a chip-sharded launch carry the same local
        # component as the caller.  Keeping canonical storage here would make
        # an rdata-backed boundary fail the physical ABI equality check.
        return DistributedBufferStorageKind.COMPACT_PER_CHIP
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
        if (
            storage == "rdata"
            and rdata_chip_sharding
            and chip_split
            and value_type.placement.physical_level_size("c")
            * value_type.placement.physical_level_size("d") > 1
        ):
            return DistributedBufferStorageKind.COMPACT_PER_CHIP
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


def _canonical_global_values(module: IRModule) -> frozenset[str]:
    """Find values consumed by descriptor kernels and their call arguments.

    Tensor descriptors encode a canonical module-global base address.  The
    requirement belongs to the selected kernel capability, rather than to a
    model weight name or a particular graph function.  Start at descriptor
    primitive calls, then propagate through graph-function actual/formal
    bindings until reaching the rdata-producing value.
    """

    required: set[str] = set()
    function_map = module.function_map
    primitive_map = module.kernel_callable_map
    from triton.flagmega.ir.tir import kernel_dispatch_of

    for function in module.functions:
        for node in function_nodes(module, function):
            if node.op not in {"builtin.call", "tir.call"}:
                continue
            primitive = primitive_map.get(str(node.attrs.get("callee", "")))
            if primitive is None:
                continue
            dispatch = kernel_dispatch_of(primitive)
            if dispatch is None or dispatch.microkernel is None:
                continue
            descriptor_arguments = set(dispatch.microkernel.facts.get("canonical_global_arguments", ()))
            unknown = descriptor_arguments - set(dispatch.arguments)
            if unknown:
                raise IRVerificationError(
                    f"Kernel {primitive.name!r} declares unknown canonical arguments {sorted(unknown)}."
                )
            indices = {
                index for index, argument in enumerate(dispatch.arguments)
                if argument in descriptor_arguments
            }
            required.update(
                node.inputs[index]
                for index in indices
                if index < len(node.inputs)
            )

    # A descriptor operand is often a graph-function formal parameter.  Carry
    # the physical-storage requirement to each caller's actual value, which
    # may be a builtin.const_asset/tir.buffer rdata root.
    changed = True
    while changed:
        changed = False
        for function in module.functions:
            for node in function_nodes(module, function):
                if node.id in required and node.op in {
                    "builtin.get_item", "builtin.tuple", "distributed.sharded_view",
                    "tir.buffer_view", "tir.buffer_subspan",
                    # A bank row access carries the requirement to the bank,
                    # whose rows are the rdata roots behind the descriptor.
                    "control.bank_get", "tir.bank_get",
                }:
                    for input_id in node.inputs:
                        if input_id not in required:
                            required.add(input_id)
                            changed = True
                if node.op not in {"builtin.call", "tir.call", "control.for"}:
                    continue
                callee = function_map.get(str(node.attrs.get("callee", "")))
                if callee is None:
                    continue
                # A For's first three operands are its bounds and its body's
                # first formal is the loop-supplied induction value.
                if node.op == "control.for":
                    pairs = zip(callee.parameters[1:], node.inputs[3:])
                else:
                    pairs = zip(callee.parameters, node.inputs, strict=False)
                for formal, actual in pairs:
                    if formal in required and actual not in required:
                        required.add(actual)
                        changed = True
    return frozenset(required)


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
    if isinstance(value_type, DistributedType):
        if kind is DistributedBufferStorageKind.COMPACT_PER_OWNER:
            return component * placement_owner_count(value_type)
        if kind is DistributedBufferStorageKind.COMPACT_PER_CHIP:
            return component * block_owner_count(value_type)
    return component


def _dense_strides(shape: Sequence[int]):
    result = []
    running = 1
    for value in reversed(shape):
        result.append(running)
        running *= value
    return tuple(reversed(result))


def _is_scalar_constant(op, value_type):
    """Whether a value is a rank-zero scalar carried as an immediate.

    A dimension/logical expression is computed where it is used, exactly like
    a scalar constant, so it needs no storage and no pointer formal.
    """
    value_type = logical_type(value_type)
    return op in {
        "builtin.scalar_const", "tir.scalar_const", "shapes.dim_as_tensor",
    } and (isinstance(value_type, TensorType) and value_type.rank == 0)


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


__all__ = ["BufferPlanner", "BufferizationOptions", "plan_buffers"]


def _is_parameter_bank(value_type) -> bool:
    """Return whether a type is readonly parameter-bank storage."""
    return isinstance(value_type, RefType) and value_type.name == "parameter_bank"
