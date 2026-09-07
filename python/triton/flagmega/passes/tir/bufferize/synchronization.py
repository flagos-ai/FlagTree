# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Plan synchronization from concrete post-bufferization byte hazards."""

from __future__ import annotations

from dataclasses import dataclass, field

from triton.flagmega.ir.bufferization import BufferPlan, MemorySharingScope
from triton.flagmega.ir.bufferization.synchronization import (
    MemoryRange,
    MemorySynchronizationPlan,
    SynchronizationEvent,
)
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import ContiguousSplit, SBPSplit
from triton.flagmega.ir.memory_effect import (
    MemoryAccessDomain,
    MemoryAccessMode,
    MemoryAccessPartitionKind,
    MemoryAccessScope,
    MemoryEffect,
    MemoryOwnerAccess,
)
from triton.flagmega.ir.model import DistributedType, IRModule, RefType, logical_type
from triton.flagmega.ir.tir.execution_kind import (
    KernelExecutionKind,
    kernel_execution_kind_for_call,
)
from triton.flagmega.ir.tir.kernel_dispatch import kernel_dispatch_for_call
from triton.flagmega.passes.tir.bufferize.graph import function_nodes
from triton.flagmega.passes.tir.bufferize.barrier_coverage import BarrierCoverage


@dataclass(frozen=True)
class _Access:
    node: str
    buffer: str
    storage: str
    physical_id: str
    offset: int
    nbytes: int
    mode: str
    distributed_type: DistributedType | None
    distributed_storage_kind: DistributedBufferStorageKind
    effect: MemoryEffect
    access_partition: tuple[str, object | None]
    is_reference: bool
    requires_full_chip: bool
    sharing_scope: MemorySharingScope


@dataclass
class _PendingAccess:
    access: _Access
    coverage: BarrierCoverage = field(default_factory=BarrierCoverage)


def plan_memory_synchronization(
    module: IRModule,
    plan: BufferPlan,
) -> MemorySynchronizationPlan:
    """Plan the weakest sufficient barriers for all concrete byte hazards."""

    return _plan_memory_synchronization(module, plan)


def _plan_memory_synchronization(
    module: IRModule,
    plan: BufferPlan,
    *,
    reuse_preferences=None,
) -> MemorySynchronizationPlan:
    """Insert the weakest sufficient barrier before every unresolved hazard.

    Workspace reuse is compared by concrete byte range rather than logical
    buffer id, so WAW/WAR hazards introduced by SAT placement are covered too.
    Like nncase, a pure RAW may use a block or placement-axis group barrier;
    an unprovable owner relation is conservatively promoted to a full grid
    barrier.  The next region starts with the current kernel's accesses.
    """

    events = []
    for function in module.functions:
        abi = plan.function_map[function.name]
        bindings = dict(abi.values)
        history: list[_PendingAccess] = []
        for node in function_nodes(module, function):
            if node.op not in {"tir.kernel", "tir.call"}:
                continue
            current = _node_accesses(module, plan, function.name, node, bindings)
            conflicts = [
                (previous.access, access)
                for previous in history
                for access in current
                if _conflicts(previous.access, access)
                and not previous.coverage.covers(_hazard_requirement(previous.access, access))
            ]
            if conflicts:
                after = max(
                    (previous.node for previous, _ in conflicts),
                    key=lambda value: _node_ordinal(module, value),
                )
                ranges = []
                hazards = set()
                seen = set()
                for previous, access in conflicts:
                    hazards.add(f"{previous.mode.upper()}->{access.mode.upper()}")
                    start = max(previous.offset, access.offset)
                    end = min(previous.offset + previous.nbytes, access.offset + access.nbytes)
                    key = (access.storage, access.physical_id, start, end - start, access.mode)
                    if key in seen:
                        continue
                    seen.add(key)
                    ranges.append(MemoryRange(*key))
                scope, axis_group_axes = _conflict_requirement(conflicts)
                if reuse_preferences is not None and scope == "grid" and not axis_group_axes:
                    from .reuse_preferences import record_reuse_preferences

                    record_reuse_preferences(plan, function.name, conflicts, reuse_preferences)
                events.append(SynchronizationEvent(
                    function.name,
                    after,
                    node.id,
                    scope,
                    tuple(sorted(hazards)),
                    tuple(ranges),
                    axis_group_axes,
                ))
                placement = next((
                    requirement[2]
                    for previous, access in conflicts
                    if (requirement := _hazard_requirement(previous, access))[0] == "grid"
                ), None)
                requirement = (scope, axis_group_axes, placement)
                for pending in history:
                    pending.coverage.apply(requirement, pending.access.distributed_type)
                # A block or axis-group barrier does not discharge outstanding
                # chip-visible accesses. Preserve them, with coverage, for a
                # later consumer whose owner relation may be wider.
                history = [pending for pending in history if not (
                    pending.coverage.full_chip_synchronized
                    or (
                        _resolved_scope(pending.access) is MemoryAccessScope.BLOCK
                        and pending.coverage.block_synchronized
                    )
                )]
            history.extend(_PendingAccess(access) for access in current)
    return MemorySynchronizationPlan(tuple(events))


def _node_accesses(module, plan, function_name, node, bindings):
    result: dict[tuple[object, ...], _Access] = {}
    # A cooperative/synchronized-local kernel may execute an internal grid
    # barrier without publishing a value to every placement owner.  Its
    # inter-kernel visibility is still described by typed memory effects and
    # DistributedType owner maps (for example, paged-attention merge only
    # rendezvous over its partial axis).  Only a true collective changes the
    # owner set and therefore forces the conservative full-chip fallback.
    publishes_across_chip = (
        kernel_execution_kind_for_call(module, node)
        is KernelExecutionKind.COLLECTIVE
    )
    dispatch = kernel_dispatch_for_call(module, node)
    effect_map = {} if dispatch is None else dict(dispatch.memory_effect_map)
    has_typed_effects = dispatch is not None and bool(dispatch.memory_effects)

    def add(
        buffer_id,
        effect: MemoryEffect,
        *,
        reference_access=False,
        publishes=False,
    ):
        descriptor = plan.buffer_map[buffer_id]
        if descriptor.storage in {"scalar", "rdata"} or descriptor.nbytes == 0:
            return
        # All workspace allocations of one function are ranges in the same
        # pool even when SAT assigns them distinct allocation ids.
        memory_space = plan.memory_space_map[
            descriptor.mem_span.buffer.memory_space
        ]
        physical_id = (
            f"{memory_space.name}:@{function_name}"
            if memory_space.strategy.value == "sat"
            and memory_space.allocation_scope.value == "function"
            and memory_space.kind != "shared"
            else str(descriptor.physical_id)
        )
        access_partition = _resolve_access_partition(effect, node, module)
        # The ABI's MemSpan exposes one owner component. Cross-kernel hazards
        # in a shared pool must include all components, including SAT reuse
        # intersecting only a nonzero owner. Keep owner maps for scope proofs;
        # expanding the byte footprint does not itself require a grid barrier.
        access_span = descriptor.physical_access_span
        key = (
            descriptor.storage,
            physical_id,
            access_span.offset,
            access_span.nbytes,
            access_partition,
        )
        previous = result.get(key)
        mode = effect.physical_mode
        previous_mode = (
            MemoryAccessMode.NONE
            if previous is None
            else previous.effect.physical_mode
        )
        merged_mode = mode | previous_mode
        merged_effect = MemoryEffect(
            merged_mode,
            _merge_access_scope(
                effect.scope,
                MemoryAccessScope.INFERRED if previous is None else previous.effect.scope,
            ),
            effect.kind if previous is None or effect.kind == previous.effect.kind else "direct",
            effect.access_domain if previous is None or effect.access_domain == previous.effect.access_domain else MemoryAccessDomain(),
            effect.access_partition if previous is None or effect.access_partition == previous.effect.access_partition else {},
            effect.owner_access if previous is None or effect.owner_access == previous.effect.owner_access else MemoryOwnerAccess.PARTIAL_GROUP,
        )
        result[key] = _Access(
            node.id, buffer_id, descriptor.storage, physical_id,
            access_span.offset, access_span.nbytes, _mode_name(merged_mode),
            descriptor.distributed_type,
            descriptor.distributed_storage_kind,
            merged_effect,
            access_partition,
            reference_access or bool(previous and previous.is_reference),
            (
                effect.scope is MemoryAccessScope.CHIP
                or publishes
                or bool(previous and previous.requires_full_chip)
            ),
            plan.memory_space_map[descriptor.mem_span.buffer.memory_space].sharing_scope,
        )

    argument_names = (
        tuple(dispatch.arguments)
        if dispatch is not None and len(dispatch.arguments) == len(node.inputs)
        else tuple(f"argument_{index}" for index in range(len(node.inputs)))
    )
    for input_id, argument_name in zip(node.inputs, argument_names, strict=True):
        is_reference = isinstance(
            logical_type(module.node_map[input_id].type), RefType
        )
        effect = effect_map.get(
            argument_name,
            MemoryEffect.NONE if has_typed_effects else MemoryEffect.READ,
        )
        for buffer_id in bindings.get(input_id, ()):
            if effect.physical_mode is not MemoryAccessMode.NONE:
                add(
                    buffer_id,
                    effect,
                    reference_access=is_reference,
                    publishes=publishes_across_chip,
                )
    output_buffers = tuple(bindings.get(node.id, ()))
    output_effects = _expanded_output_effects(
        module,
        node,
        dispatch,
        effect_map,
        has_typed_effects,
        len(output_buffers),
    )
    for buffer_id, output_effect in zip(
        output_buffers, output_effects, strict=True
    ):
        if output_effect.physical_mode is not MemoryAccessMode.NONE:
            add(buffer_id, output_effect, publishes=publishes_across_chip)
    return tuple(result.values())


def _expanded_output_effects(
    module: IRModule,
    node,
    dispatch,
    effect_map,
    has_typed_effects: bool,
    output_buffer_count: int,
) -> tuple[MemoryEffect, ...]:
    if dispatch is None:
        return (MemoryEffect.WRITE,) * output_buffer_count
    default = MemoryEffect.NONE if has_typed_effects else MemoryEffect.WRITE
    effects = tuple(effect_map.get(name, default) for name in dispatch.outputs)
    primitive = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
    if primitive is None or len(primitive.output_parameters) != len(effects):
        return (_merge_effects(effects),) * output_buffer_count
    expanded = tuple(
        effect
        for parameter, effect in zip(
            primitive.output_parameters, effects, strict=True
        )
        for _ in parameter.buffers
    )
    return (
        expanded
        if len(expanded) == output_buffer_count
        else (_merge_effects(effects),) * output_buffer_count
    )


def _conflicts(previous: _Access, current: _Access) -> bool:
    producer_reads = bool(previous.effect.physical_mode & MemoryAccessMode.READ)
    producer_writes = bool(previous.effect.physical_mode & MemoryAccessMode.WRITE)
    consumer_reads = bool(current.effect.physical_mode & MemoryAccessMode.READ)
    consumer_writes = bool(current.effect.physical_mode & MemoryAccessMode.WRITE)
    requires_synchronization = (
        (producer_writes and consumer_reads)
        or (producer_reads and consumer_writes)
        or (
            producer_writes
            and consumer_writes
            and previous.buffer != current.buffer
        )
    )
    if not requires_synchronization or not _partitions_may_alias(
        previous.access_partition, current.access_partition
    ):
        return False
    return (
        previous.storage == current.storage
        and previous.physical_id == current.physical_id
        and previous.offset < current.offset + current.nbytes
        and current.offset < previous.offset + previous.nbytes
    )


def _conflict_requirement(
    conflicts: list[tuple[_Access, _Access]],
) -> tuple[str, tuple[int, ...]]:
    """Infer the weakest barrier and owner axis group for all byte hazards.

    This is the executable subset of nncase's ``MemoryEffectAnalyzer`` scope
    rule.  Ordinary storage and a RAW whose producer/consumer retain the exact
    same distributed owner map are block-local.  Reference effects, explicit
    collective/synchronized implementations, and a changed/unknown owner map
    require chip synchronization.  A future axis-group representation can
    refine the latter without weakening this contract.
    """

    requirements = tuple(
        _hazard_requirement(previous, current) for previous, current in conflicts
    )
    if all(scope == "block" for scope, _, _ in requirements):
        return "block", ()
    chip = tuple(value for value in requirements if value[0] == "grid")
    placements = {value[2] for value in chip}
    if any(not axes for _, axes, _ in chip) or len(placements) != 1:
        return "grid", ()
    placement = next(iter(placements))
    if placement is None:
        return "grid", ()
    axes = tuple(sorted({axis for _, values, _ in chip for axis in values}))
    block_axes = _block_axes(placement)
    return ("grid", ()) if axes == block_axes else ("grid", axes)


def _hazard_requirement(
    previous: _Access,
    current: _Access,
) -> tuple[str, tuple[int, ...], object | None]:
    producer = previous.distributed_type
    consumer = current.distributed_type
    placement = (
        producer.placement
        if producer is not None and consumer is not None and producer.placement == consumer.placement
        else None
    )
    if previous.effect.access_domain.is_same_fixed_block(
        current.effect.access_domain
    ) and (
        None if producer is None else producer.placement
    ) == (
        None if consumer is None else consumer.placement
    ):
        return "block", (), placement
    if (
        _resolved_scope(previous) is MemoryAccessScope.BLOCK
        and _resolved_scope(current) is MemoryAccessScope.BLOCK
    ):
        # Match nncase's ordering: a physically replicated block-local arena
        # cannot create a cross-block byte hazard, even when the producing op
        # also has collective semantics for another operand or result.
        return "block", (), placement
    if previous.requires_full_chip or current.requires_full_chip:
        return "grid", (), placement
    producer_reads = bool(previous.effect.physical_mode & MemoryAccessMode.READ)
    producer_writes = bool(previous.effect.physical_mode & MemoryAccessMode.WRITE)
    consumer_reads = bool(current.effect.physical_mode & MemoryAccessMode.READ)
    consumer_writes = bool(current.effect.physical_mode & MemoryAccessMode.WRITE)
    has_raw = producer_writes and consumer_reads
    has_war = producer_reads and consumer_writes
    has_waw = (
        producer_writes
        and consumer_writes
        and previous.buffer != current.buffer
    )
    if has_raw and not has_war and not has_waw:
        axes = _infer_raw_axis_group(previous, current)
        if axes is not None:
            if not axes:
                return "block", (), placement
            return "grid", axes, placement
    if has_war and not has_raw and not has_waw:
        # A WAR is the ownership dual of a RAW: every old reader must finish
        # before a new writer may reuse the byte range.  Reverse the owner-map
        # relation so a broadcast reader followed by an axis-sharded writer,
        # for example, rendezvous only on the newly introduced owner axis.
        axes = _infer_raw_axis_group(current, previous)
        if axes is not None:
            if not axes:
                return "block", (), placement
            return "grid", axes, placement
    return "grid", (), placement


def _resolved_scope(access: _Access) -> MemoryAccessScope:
    if access.effect.scope is not MemoryAccessScope.INFERRED:
        return access.effect.scope
    if access.sharing_scope is MemorySharingScope.BLOCK:
        return MemoryAccessScope.BLOCK
    return (
        MemoryAccessScope.CHIP
        if access.sharing_scope in {
            MemorySharingScope.DIE,
            MemorySharingScope.CHIP,
        }
        or access.is_reference
        # The current NVIDIA workspace is explicitly chip-scoped. Distributed
        # views in that arena are therefore chip-visible unless pure-RAW
        # owner-map inference below proves a block or placement-axis group.
        or access.distributed_type is not None
        or access.distributed_storage_kind
        is DistributedBufferStorageKind.COMPACT_PER_OWNER
        else MemoryAccessScope.BLOCK
    )


def _resolve_access_partition(
    effect: MemoryEffect,
    node,
    module: IRModule,
) -> tuple[str, object | None]:
    partition = effect.access_partition
    if partition.kind is MemoryAccessPartitionKind.WHOLE_RESOURCE:
        return ("whole", None)
    assert partition.argument_index is not None
    if partition.argument_index >= len(node.inputs):
        # Verification of a selected kernel must reject this eventually.  At
        # planning time an invalid dynamic provider is conservatively treated
        # as the whole resource rather than proving a false disjointness.
        return ("whole", None)
    selector = module.node_map[node.inputs[partition.argument_index]]
    if selector.op in {"builtin.scalar_const", "tir.scalar_const"}:
        value = selector.attrs.get("value")
        if isinstance(value, int) and not isinstance(value, bool):
            return ("static", value)
    return ("symbolic", selector.id)


def _partitions_may_alias(
    lhs: tuple[str, object | None],
    rhs: tuple[str, object | None],
) -> bool:
    # This deliberately mirrors nncase's ResolvedMemoryAccessPartition:
    # distinct compile-time integers prove disjointness; whole-resource and
    # symbolic selectors remain conservative, even when their SSA ids differ.
    return not (
        lhs[0] == rhs[0] == "static"
        and lhs[1] != rhs[1]
    )


def _infer_raw_axis_group(
    previous: _Access,
    current: _Access,
) -> tuple[int, ...] | None:
    producer = previous.distributed_type
    consumer = current.distributed_type
    if (
        producer is None
        or consumer is None
        or producer.placement != consumer.placement
        or producer.partial != consumer.partial
    ):
        return None
    # SBP describes ownership of logical coordinates, not ownership of an
    # arbitrary reused arena address. A shifted allocation can make writer
    # owner 0 alias reader owner 1 despite identical SBP. Likewise a compact
    # owner-major array and canonical cyclic storage have different byte maps.
    # The logical owner-map proof below applies only at a common physical
    # origin and in the same coordinate system/owner stride.
    if previous.offset != current.offset:
        return None
    producer_storage = previous.distributed_storage_kind
    consumer_storage = current.distributed_storage_kind
    if producer_storage != consumer_storage and not (
        producer_storage.exposes_logical_coordinates
        and consumer_storage.exposes_logical_coordinates
    ):
        return None
    if (
        producer_storage is DistributedBufferStorageKind.COMPACT_PER_OWNER
        and previous.nbytes != current.nbytes
    ):
        return None
    producer_split = _split_assignments(producer)
    consumer_split = _split_assignments(consumer)
    if producer_split is None or consumer_split is None:
        return None
    producer_assignments, producer_orders = producer_split
    consumer_assignments, consumer_orders = consumer_split
    for tensor_axis in range(max(len(producer.axis_policies), len(consumer.axis_policies))):
        producer_order = producer_orders.get(tensor_axis, ())
        consumer_order = consumer_orders.get(tensor_axis, ())
        common = tuple(axis for axis in producer_order if axis in consumer_order)
        if common != tuple(axis for axis in consumer_order if axis in producer_order):
            return None
        removes_axis = any(axis not in consumer_order for axis in producer_order)
        if removes_axis and (
            len(consumer_order) > len(producer_order)
            or producer_order[:len(consumer_order)] != consumer_order
        ):
            return None
        if removes_axis and consumer_order and not _is_prefix_coarsening(
            producer.axis_policies[tensor_axis],
            consumer.axis_policies[tensor_axis],
            producer,
            consumer,
            tensor_axis,
        ):
            return None
    required: list[int] = []
    for access, distributed in ((previous, producer), (current, consumer)):
        if (
            access.effect.owner_access is MemoryOwnerAccess.PARTIAL_GROUP
            and distributed.partial is not None
        ):
            required.extend(distributed.partial.axes)
    for access, assignments in (
        (previous, producer_assignments),
        (current, consumer_assignments),
    ):
        if (
            access.distributed_storage_kind
            is not DistributedBufferStorageKind.COMPACT_PER_OWNER
            and access.sharing_scope is not MemorySharingScope.BLOCK
        ):
            # The current FlagMega workspace has no nncase-style
            # BlockLocalData pool stride. A canonical/global component is
            # physically shared by owners on every unassigned (broadcast)
            # mesh axis, so those owners must rendezvous before a RAW. Fully
            # sharded or compact-per-owner components remain block-local.
            required.extend(
                axis
                for axis in _block_axes(producer.placement)
                if axis not in assignments
            )
    for axis in sorted(set(producer_assignments) | set(consumer_assignments)):
        producer_tensor_axis = producer_assignments.get(axis)
        consumer_tensor_axis = consumer_assignments.get(axis)
        if (
            producer_tensor_axis is not None
            and consumer_tensor_axis is not None
            and producer_tensor_axis != consumer_tensor_axis
        ):
            return None
        if producer_tensor_axis is not None and consumer_tensor_axis is None:
            required.append(axis)
    for tensor_axis in range(min(len(producer.axis_policies), len(consumer.axis_policies))):
        producer_order = producer_orders.get(tensor_axis, ())
        consumer_order = consumer_orders.get(tensor_axis, ())
        if (
            producer_order
            and producer_order == consumer_order
            and not _equivalent_split_byte_units(
                producer.axis_policies[tensor_axis],
                consumer.axis_policies[tensor_axis],
                producer,
                consumer,
                tensor_axis,
            )
        ):
            required.extend(producer_order)
    axes = tuple(sorted(set(required)))
    block_axes = _block_axes(producer.placement)
    if any(axis not in block_axes for axis in axes):
        return None
    return axes


def _split_assignments(distributed_type: DistributedType):
    assignments: dict[int, int] = {}
    orders: dict[int, tuple[int, ...]] = {}
    for tensor_axis, policy in enumerate(distributed_type.axis_policies):
        if not isinstance(policy, SBPSplit):
            continue
        axes = tuple(policy.hierarchy_axes)
        if (
            len(set(axes)) != len(axes)
            or any(
                axis >= distributed_type.placement.rank
                or not distributed_type.placement.is_physical_block_axis(axis)
                for axis in axes
            )
        ):
            return None
        orders[tensor_axis] = axes
        for axis in axes:
            if axis in assignments:
                return None
            assignments[axis] = tensor_axis
    return assignments, orders


def _is_prefix_coarsening(
    producer_policy,
    consumer_policy,
    producer_type: DistributedType,
    consumer_type: DistributedType,
    tensor_axis: int,
) -> bool:
    if not isinstance(producer_policy, SBPSplit) or not isinstance(consumer_policy, SBPSplit):
        return False
    producer_index = 0
    for consumer_index, consumer_stage in enumerate(consumer_policy.stages):
        if producer_index >= len(producer_policy.stages):
            return False
        producer_stage = producer_policy.stages[producer_index]
        producer_index += 1
        if producer_stage == consumer_stage:
            continue
        if (
            consumer_index != len(consumer_policy.stages) - 1
            or producer_stage.hierarchy_axes[:len(consumer_stage.hierarchy_axes)]
            != consumer_stage.hierarchy_axes
            or not isinstance(producer_stage.distribution, ContiguousSplit)
            or not isinstance(consumer_stage.distribution, ContiguousSplit)
        ):
            return False
        producer_granularity = producer_stage.distribution.granularity
        consumer_granularity = consumer_stage.distribution.granularity
        if producer_granularity is None or consumer_granularity is None:
            return producer_granularity is consumer_granularity
        if (
            not producer_granularity.is_fixed
            or not consumer_granularity.is_fixed
            or not _same_fixed_axis_byte_extent(
                producer_type, consumer_type, tensor_axis
            )
        ):
            return False
        removed_axes = producer_stage.hierarchy_axes[
            len(consumer_stage.hierarchy_axes):
        ]
        removed_owner_count = 1
        for axis in removed_axes:
            removed_owner_count *= producer_type.placement.hierarchy[axis]
        producer_bytes = (
            producer_granularity.fixed_value
            * producer_type.tensor.dtype.itemsize
            * removed_owner_count
        )
        consumer_bytes = (
            consumer_granularity.fixed_value
            * consumer_type.tensor.dtype.itemsize
        )
        return producer_bytes == consumer_bytes
    return True


def _same_fixed_axis_byte_extent(
    lhs: DistributedType,
    rhs: DistributedType,
    tensor_axis: int,
) -> bool:
    """Compare split-axis units after vector pack/unpack reinterpretation."""

    if (
        tensor_axis >= len(lhs.tensor.shape)
        or tensor_axis >= len(rhs.tensor.shape)
    ):
        return False
    lhs_extent = lhs.tensor.shape[tensor_axis]
    rhs_extent = rhs.tensor.shape[tensor_axis]
    return (
        lhs_extent.is_fixed
        and rhs_extent.is_fixed
        and lhs_extent.fixed_value * lhs.tensor.dtype.itemsize
        == rhs_extent.fixed_value * rhs.tensor.dtype.itemsize
    )


def _equivalent_split_byte_units(
    lhs_policy,
    rhs_policy,
    lhs_type: DistributedType,
    rhs_type: DistributedType,
    tensor_axis: int,
) -> bool:
    """Compare an unchanged split owner map in physical byte coordinates."""

    if lhs_policy == rhs_policy:
        return True
    if (
        not isinstance(lhs_policy, SBPSplit)
        or not isinstance(rhs_policy, SBPSplit)
        or len(lhs_policy.stages) != len(rhs_policy.stages)
        or not _same_fixed_axis_byte_extent(lhs_type, rhs_type, tensor_axis)
    ):
        return False
    for lhs_stage, rhs_stage in zip(
        lhs_policy.stages, rhs_policy.stages, strict=True
    ):
        if lhs_stage.hierarchy_axes != rhs_stage.hierarchy_axes:
            return False
        lhs_distribution = lhs_stage.distribution
        rhs_distribution = rhs_stage.distribution
        if not isinstance(lhs_distribution, ContiguousSplit) or not isinstance(
            rhs_distribution, ContiguousSplit
        ):
            return False
        lhs_granularity = lhs_distribution.granularity
        rhs_granularity = rhs_distribution.granularity
        if lhs_granularity is None or rhs_granularity is None:
            if lhs_granularity is not rhs_granularity:
                return False
            continue
        if not lhs_granularity.is_fixed or not rhs_granularity.is_fixed:
            return False
        if (
            lhs_granularity.fixed_value * lhs_type.tensor.dtype.itemsize
            != rhs_granularity.fixed_value * rhs_type.tensor.dtype.itemsize
        ):
            return False
    return True


def _block_axes(placement) -> tuple[int, ...]:
    return tuple(
        axis for axis in range(placement.rank)
        if placement.is_physical_block_axis(axis)
    )


def _merge_access_scope(
    lhs: MemoryAccessScope,
    rhs: MemoryAccessScope,
) -> MemoryAccessScope:
    if MemoryAccessScope.CHIP in {lhs, rhs}:
        return MemoryAccessScope.CHIP
    if MemoryAccessScope.BLOCK in {lhs, rhs}:
        return MemoryAccessScope.BLOCK
    return MemoryAccessScope.INFERRED


def _merge_effects(effects: tuple[MemoryEffect, ...]) -> MemoryEffect:
    if not effects:
        return MemoryEffect.WRITE
    result = effects[0]
    for value in effects[1:]:
        result = MemoryEffect(
            result.mode | value.mode,
            _merge_access_scope(result.scope, value.scope),
            result.kind if result.kind == value.kind else "direct",
            result.access_domain if result.access_domain == value.access_domain else MemoryAccessDomain(),
            result.access_partition if result.access_partition == value.access_partition else {},
            result.owner_access if result.owner_access == value.owner_access else MemoryOwnerAccess.PARTIAL_GROUP,
        )
    return result


def _mode_name(mode: MemoryAccessMode) -> str:
    if mode == MemoryAccessMode.READ:
        return "read"
    if mode == MemoryAccessMode.WRITE:
        return "write"
    return "read_write"


def _node_ordinal(module, node_id):
    return next(index for index, node in enumerate(module.nodes) if node.id == node_id)


__all__ = ["plan_memory_synchronization"]
