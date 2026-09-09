# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove function-temporary placement in target-declared memory spaces."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    DistributedType,
    IRModule,
    MemoryAccessMode,
    MemoryAccessScope,
    MemoryEffectKind,
    MemoryOwnerAccess,
    kernel_dispatch_for_call,
    is_local_shard_subview,
    TupleType,
    dim,
    local_shape,
    simplify_dim,
)
from triton.flagmega.passes.functions.graph import function_nodes

if TYPE_CHECKING:
    from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions


MEMORY_SPACE_METADATA = "bufferization.memory_space"
FUNCTION_MEMORY_PLACEMENT_SCHEMA = "flagmega.function-memory-placement/v1"


def plan_function_memory(
    module: IRModule,
    options: BufferizationOptions,
) -> IRModule:
    """Place owner-private kernel edges in a block-scoped lifetime pool.

    The proof is intentionally expressed in graph/effect terms, not kernel or
    model names. A candidate must be a distributed, non-partial value whose
    producer writes only its own owner region and every direct consumer reads
    that region independently in the same block owner domain. A producer may
    itself gather/reduce remote inputs: its execution kind does not widen an
    output whose explicit memory effect is owner-local. Views, calls, escapes
    and chip/partial output effects remain in the default pool.
    """

    block_local = options.block_local
    spaces = options.memory_space_map
    if block_local is None or block_local == options.workspace:
        return _record_placement(module, options, ())
    try:
        block_space = spaces[block_local]
    except KeyError as error:
        raise IRVerificationError(
            f"Block-local memory space {block_local!r} is not declared.",
            stage=module.stage,
        ) from error
    if (
        not block_space.supports_lifetime_reuse
        or block_space.allocation_scope.value != "function"
        or block_space.sharing_scope.value != "block"
        or block_space.kind == "shared"
    ):
        raise IRVerificationError(
            f"Block-local memory space {block_local!r} must be a non-shared "
            "function-scoped block-sharing lifetime pool.",
            stage=module.stage,
        )

    promoted: list[str] = []
    nodes = {node.id: node for node in module.nodes}
    for function in module.functions:
        local_nodes = function_nodes(module, function)
        local_ids = {node.id for node in local_nodes}
        users: dict[str, list[tuple[object, int]]] = {}
        for user in local_nodes:
            for index, input_id in enumerate(user.inputs):
                if input_id in local_ids:
                    users.setdefault(input_id, []).append((user, index))
        outputs = set(function.outputs)
        for node in local_nodes:
            if MEMORY_SPACE_METADATA in node.metadata:
                continue
            if _can_place_block_local(
                module,
                node,
                users,
                outputs,
            ):
                promoted.append(node.id)

    promoted_set = set(promoted)
    rewritten = tuple(
        replace(
            node,
            metadata={**dict(node.metadata), MEMORY_SPACE_METADATA: block_local},
        )
        if node.id in promoted_set
        else node
        for node in module.nodes
    )
    return _record_placement(
        replace(module, nodes=rewritten), options, tuple(promoted)
    )


def _can_place_block_local(module, node, users_by_value, function_outputs) -> bool:
    value_type = node.type
    if (
        node.op in {
            "builtin.get_item",
            "builtin.tuple",
            "distributed.sharded_view",
            "tir.buffer_view",
        }
        or not _all_owner_local_distributed_leaves(value_type)
        or node.id in function_outputs
        or not users_by_value.get(node.id)
        or _may_alias_function_output(
            module, node.id, users_by_value, function_outputs
        )
    ):
        return False
    producer = kernel_dispatch_for_call(module, node)
    if (
        producer is None
        or not _effects_are_owner_local(
            producer, producer.outputs, require_write=True
        )
    ):
        return False
    users = _terminal_consumers_through_local_views(
        module,
        node,
        users_by_value,
        function_outputs,
    )
    if not users:
        return False
    for user, input_index in users:
        consumer = kernel_dispatch_for_call(module, user)
        if (
            consumer is None
            or input_index >= len(consumer.arguments)
            or not _effects_are_owner_local(
                consumer, (consumer.arguments[input_index],), require_write=False
            )
        ):
            return False
    return True


def _may_alias_function_output(
    module,
    source_id: str,
    users_by_value,
    function_outputs,
) -> bool:
    """Trace explicit view and kernel in-place opportunities to an escape.

    Bufferization is allowed to consume a dead operand through a typed
    ``InplaceAliasCandidate``. A value on such a path cannot be committed to a
    block pool when the eventual result is caller-owned: the complete alias
    group must be promoted to the function result allocation instead.
    """

    pending = [source_id]
    visited: set[str] = set()
    while pending:
        current_id = pending.pop()
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in function_outputs:
            return True
        for user, input_index in users_by_value.get(current_id, ()):
            if user.op == "builtin.get_item":
                if input_index == 0 and len(user.inputs) == 1:
                    pending.append(user.id)
                continue
            if user.op in {
                "distributed.sharded_view",
                "tir.buffer_view",
            }:
                pending.append(user.id)
                continue
            if user.op == "builtin.tuple":
                if user.id in function_outputs:
                    return True
                for projection, _ in users_by_value.get(user.id, ()):
                    if (
                        projection.op == "builtin.get_item"
                        and int(projection.attrs.get("index", -1)) == input_index
                    ):
                        pending.append(projection.id)
                continue
            dispatch = kernel_dispatch_for_call(module, user)
            if dispatch is None or input_index >= len(dispatch.arguments):
                continue
            aliases = dispatch.inplace_alias_candidates or ()
            argument = dispatch.arguments[input_index]
            output_indices = {
                index
                for index, output in enumerate(dispatch.outputs)
                if any(
                    candidate.input == argument
                    and candidate.output == output
                    for candidate in aliases
                )
            }
            if not output_indices:
                continue
            if user.id in function_outputs:
                return True
            if len(dispatch.outputs) == 1:
                pending.append(user.id)
                continue
            for projection, _ in users_by_value.get(user.id, ()):
                if (
                    projection.op == "builtin.get_item"
                    and int(projection.attrs.get("index", -1))
                    in output_indices
                ):
                    pending.append(projection.id)
    return False


def _terminal_consumers_through_local_views(
    module,
    source,
    users_by_value,
    function_outputs,
):
    """Return kernel consumers reached through storage-preserving TIR views.

    ``tir.buffer_view`` is an alias-only TIR operation and therefore does not
    change the physical owner scope when its placement and local byte extent
    remain identical. A narrowing ``distributed.sharded_view`` is also local
    when every target owner shard is provably contained in the source shard at
    the same block coordinate. Bufferization retains the coarser source
    distribution as the view's explicit physical backing.
    """

    terminal = []
    pending = [(source, source.type)]
    visited: set[str] = set()
    while pending:
        current, current_type = pending.pop()
        if current.id in visited:
            continue
        visited.add(current.id)
        current_users = tuple(users_by_value.get(current.id, ()))
        if not current_users:
            continue
        for user, input_index in current_users:
            if user.op == "builtin.get_item":
                if (
                    input_index != 0
                    or len(user.inputs) != 1
                    or user.id in function_outputs
                    or not isinstance(current_type, TupleType)
                ):
                    return ()
                field_index = int(user.attrs.get("index", -1))
                if field_index < 0 or field_index >= len(current_type.fields):
                    return ()
                field_type = current_type.fields[field_index]
                if not _all_owner_local_distributed_leaves(field_type):
                    return ()
                pending.append((user, field_type))
                continue
            if user.op == "distributed.sharded_view":
                if (
                    input_index == 0
                    and len(user.inputs) == 1
                    and user.id not in function_outputs
                    and _is_same_owner_local_narrowing_view(
                        current_type, user.type
                    )
                ):
                    pending.append((user, user.type))
                    continue
                # Other ShardedViews remain canonical-global backing
                # boundaries. They are irrelevant only when every downstream
                # use is a metadata-only PrimFunction operand.
                if _branch_has_physical_use(
                    module,
                    user,
                    users_by_value,
                    function_outputs,
                    set(),
                ):
                    return ()
                continue
            if user.op != "tir.buffer_view":
                if not _consumer_reads_physical_input(
                    module, user, input_index
                ):
                    continue
                terminal.append((user, input_index))
                continue
            if (
                input_index != 0
                or len(user.inputs) != 1
                or user.id in function_outputs
                or not _same_owner_local_storage(current_type, user.type)
            ):
                return ()
            pending.append((user, user.type))
    return tuple(terminal)


def _is_same_owner_local_narrowing_view(source_type, result_type) -> bool:
    return (
        isinstance(source_type, DistributedType)
        and isinstance(result_type, DistributedType)
        and all(
            level == "b"
            for level in source_type.placement.hierarchy_levels
        )
        and is_local_shard_subview(source_type, result_type)
    )


def _consumer_reads_physical_input(module, user, input_index: int) -> bool:
    """Whether one logical call operand has a physical buffer use.

    Complete typed dispatch effects distinguish an explicit metadata-only
    operand from an unknown legacy ABI.  Unknown/non-kernel consumers remain
    conservative physical uses.
    """

    consumer = kernel_dispatch_for_call(module, user)
    if consumer is None or input_index >= len(consumer.arguments):
        return True
    primitive = module.kernel_callable_map.get(str(user.attrs.get("callee", "")))
    if primitive is not None and input_index < len(primitive.runtime_parameters):
        return primitive.runtime_parameters[input_index].role.value != "metadata"
    if not consumer.memory_effects:
        return True
    effect = consumer.memory_effect_map.get(consumer.arguments[input_index])
    return effect is None or effect.physical_mode is not MemoryAccessMode.NONE


def _branch_has_physical_use(
    module,
    source,
    users_by_value,
    function_outputs,
    visited,
) -> bool:
    """Trace structural aliases until a physical consumer or escape."""

    if source.id in visited:
        return False
    visited.add(source.id)
    if source.id in function_outputs:
        return True
    for user, input_index in users_by_value.get(source.id, ()):
        if user.op in {
            "builtin.get_item",
            "builtin.tuple",
            "tir.buffer_view",
            "distributed.sharded_view",
        }:
            if _branch_has_physical_use(
                module,
                user,
                users_by_value,
                function_outputs,
                visited,
            ):
                return True
            continue
        if _consumer_reads_physical_input(module, user, input_index):
            return True
    return False


def _all_owner_local_distributed_leaves(value_type) -> bool:
    """Return whether every physical result leaf has a block-owner component."""

    if isinstance(value_type, DistributedType):
        return (
            value_type.partial is None
            and all(level == "b" for level in value_type.placement.hierarchy_levels)
        )
    if isinstance(value_type, TupleType):
        return bool(value_type.fields) and all(
            _all_owner_local_distributed_leaves(field)
            for field in value_type.fields
        )
    return False


def _same_owner_local_storage(source_type, result_type) -> bool:
    if (
        not isinstance(source_type, DistributedType)
        or not isinstance(result_type, DistributedType)
        or source_type.partial is not None
        or result_type.partial is not None
        or source_type.placement != result_type.placement
    ):
        return False
    return _local_storage_nbytes(source_type) == _local_storage_nbytes(result_type)


def _local_storage_nbytes(value: DistributedType):
    result = dim(value.tensor.dtype.itemsize)
    for extent in local_shape(value):
        result = simplify_dim(result * extent)
    return result


def _effects_are_owner_local(dispatch, names, *, require_write: bool) -> bool:
    effects = dispatch.memory_effect_map
    for name in names:
        effect = effects.get(name)
        if effect is None:
            return False
        if (
            effect.scope is MemoryAccessScope.CHIP
            or effect.owner_access is not MemoryOwnerAccess.LOCAL
            or effect.kind is not MemoryEffectKind.DIRECT
            or (
                require_write
                and not effect.physical_mode & MemoryAccessMode.WRITE
            )
        ):
            return False
    return True


def _record_placement(module, options, promoted) -> IRModule:
    return replace(module, metadata={
        **dict(module.metadata),
        "function_memory_placement": {
            "schema": FUNCTION_MEMORY_PLACEMENT_SCHEMA,
            "default": options.workspace,
            "block_local": options.block_local,
            "promoted": list(promoted),
        },
    })


__all__ = [
    "FUNCTION_MEMORY_PLACEMENT_SCHEMA",
    "MEMORY_SPACE_METADATA",
    "plan_function_memory",
]
