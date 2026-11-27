# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TIR candidates for executable distributed layout transitions."""

from math import prod

from triton.flagmega.codegen.triton.candidates.core import (
    TritonCandidateContext,
    TritonCandidateProposal,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import (
    DistributedType,
    TensorType,
    TupleType,
    VectorType,
    boxing_reads_peer_chips,
    is_local_shard_subview,
    local_shape,
)


class DistributedBoxingCandidateProvider:
    """Map boxing leaves to nncase's three explicit NTT transfer forms.

    Applicability is purely semantic: the logical tensor is unchanged.  The
    target implementation model decides which physical transfer forms exist
    and how they are implemented on a machine.
    """

    op_names = frozenset({"distributed.boxing", "distributed.force_boxing"})

    def propose(
        self,
        node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        if len(node.inputs) != 1:
            return None
        source = context.module.node_map[node.inputs[0]].type
        # The transfer owns the original tensor/ownership transition. Its
        # numeric PostOps writes the physical result type after that copy.
        target = node.attrs["new_type"] if node.attrs.get("post_ops") else node.type
        if source == target:
            return None
        transitions = _leaf_transitions(source, target)
        executable = frozenset(value for value in transitions if value != "identity")
        if len(executable) != 1:
            return None
        transition = next(iter(executable))
        tuple_boxing = isinstance(source, TupleType)
        semantics = transition.replace("_", "-")
        if tuple_boxing:
            semantics = f"tuple-{semantics}"
        implementations = context.implementations(
            "distributed_boxing", transition=transition
        )
        applicable = []
        for implementation in implementations:
            for name, predicate in (("peer_gather", _peer_gather), ("peer_partial_reduce", _peer_partial_reduce)):
                if name in implementation.contract:
                    required = implementation.contract[name]
                    if type(required) is not bool:
                        raise CodegenError(f"A Boxing {name} contract must be boolean.")
                    if required != predicate(source, target, int(implementation.parameters["tile"])):
                        break
            else:
                applicable.append(implementation)
        implementations = tuple(applicable)
        facts = {
            "collective_semantics": semantics,
            "tuple_field_count": len(transitions),
        }
        if _reads_peer_chips(source, target):
            # The leaf has to read the peer chips' instances of the source
            # bytes, so the hazard planner must bracket it with device-scope
            # rendezvous instead of chip-local grid barriers.
            facts["source_memory_scope"] = "device"
        elif _reads_peer_blocks(source, target):
            # Destination-driven relayout reads other CTA components. Keep
            # the producer in a chip-visible arena before bufferization.
            facts["source_memory_scope"] = "chip"
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={
                    "source_type": source,
                    "target_type": target,
                    "leaf_transitions": transitions,
                },
                facts=dict(facts),
            )
            for implementation in implementations
        )
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates,
            context.choose_default("distributed_boxing", candidates),
        )


def _reads_peer_chips(source, target) -> bool:
    """Whether any leaf of this boxing reads a peer chip's source instance."""

    pending = [(source, target)]
    while pending:
        source, target = pending.pop()
        if isinstance(source, TupleType) and isinstance(target, TupleType):
            pending.extend(zip(source.fields, target.fields))
            continue
        if boxing_reads_peer_chips(source, target):
            return True
    return False


def _reads_peer_blocks(source, target) -> bool:
    pending = [(source, target)]
    while pending:
        before, after = pending.pop()
        if isinstance(before, TupleType) and isinstance(after, TupleType):
            pending.extend(zip(before.fields, after.fields, strict=True))
        elif (isinstance(before, DistributedType) and isinstance(after, DistributedType)
              and before.placement == after.placement and before.partial is None
              and before.exclusive is None and after.partial is None
              and any(level == "b" and extent > 1 for level, extent in zip(
                  before.placement.hierarchy_levels, before.placement.hierarchy, strict=True))
              and not is_local_shard_subview(before, after)):
            return True
    return False


def _peer_gather(source, target, tile) -> bool:
    """Whether all changed leaves admit one chip-shared copy split among CTAs."""
    pending = [(source, target)]
    changed = False
    while pending:
        before, after = pending.pop()
        if before == after:
            continue
        if isinstance(before, TupleType) and isinstance(after, TupleType):
            if len(before.fields) != len(after.fields):
                return False
            pending.extend(zip(before.fields, after.fields))
            continue
        if (not isinstance(before, DistributedType) or before.partial is not None
                or before.exclusive is not None or not boxing_reads_peer_chips(before, after)):
            return False
        mesh = before.placement
        if not _cooperative_output_tiles(mesh, after, tile):
            return False
        changed = True
    return changed


def _peer_partial_reduce(source, target, tile) -> bool:
    """Chip partials may retain their sum tree while CTAs split output tiles."""
    pending = [(source, target)]
    changed = False
    while pending:
        before, after = pending.pop()
        if before == after:
            continue
        if isinstance(before, TupleType) and isinstance(after, TupleType):
            if len(before.fields) != len(after.fields):
                return False
            pending.extend(zip(before.fields, after.fields))
            continue
        if (not isinstance(before, DistributedType) or before.partial is None or before.exclusive is not None
                or not boxing_reads_peer_chips(before, after)):
            return False
        mesh = before.placement
        if any(mesh.hierarchy_levels[axis] != "c" for axis in before.partial.axes):
            return False
        if not _cooperative_output_tiles(mesh, after, tile):
            return False
        changed = True
    return changed


def _cooperative_output_tiles(mesh, result, tile) -> bool:
    block_axes = {axis for axis, level in enumerate(mesh.hierarchy_levels) if level == "b"}
    if prod(mesh.hierarchy[axis] for axis in block_axes) <= 1:
        return False
    if isinstance(result, DistributedType):
        if result.placement != mesh or result.partial is not None or result.exclusive is not None:
            return False
        used = {axis for policy in result.axis_policies for stage in getattr(policy, "stages", ())
                for axis in stage.hierarchy_axes}
        if used & block_axes:
            return False
        shape, dtype = local_shape(result), result.tensor.dtype
    elif isinstance(result, TensorType):
        shape, dtype = result.shape, result.dtype
    else:
        return False
    if any(not extent.is_fixed for extent in shape):
        return False
    capacity = prod(extent.fixed_value for extent in shape)
    capacity *= prod(dtype.lanes) if isinstance(dtype, VectorType) else 1
    return capacity > tile


def _leaf_transitions(source, target) -> tuple[str, ...]:
    """Classify a structurally identical tensor/tuple boxing recursively."""

    pending = [(source, target)]
    leaves = []
    while pending:
        source, target = pending.pop()
        # Expand even an unchanged subtree: its flattened ABI still has one
        # buffer per tensor. Validate structure before classifying leaves.
        if isinstance(source, TupleType) or isinstance(target, TupleType):
            if (
                not isinstance(source, TupleType)
                or not isinstance(target, TupleType)
                or len(source.fields) != len(target.fields)
            ):
                return ()
            pending.extend(reversed(tuple(zip(source.fields, target.fields))))
            continue
        source_tensor = source.tensor if isinstance(source, DistributedType) else source
        target_tensor = target.tensor if isinstance(target, DistributedType) else target
        if not isinstance(source_tensor, TensorType) or source_tensor != target_tensor:
            return ()
        if source == target:
            leaves.append("identity")
        elif isinstance(source, TensorType) and isinstance(target, DistributedType):
            leaves.append("tensor_load")
        elif isinstance(source, DistributedType) and isinstance(target, TensorType):
            # A plain destination does not remove the source's partial sum.
            leaves.append("gather_reduce_scatter" if source.partial is not None else "tensor_store")
        elif isinstance(source, DistributedType) and isinstance(target, DistributedType):
            leaves.append("gather_reduce_scatter")
        else:
            return ()
    return tuple(leaves)


__all__ = ["DistributedBoxingCandidateProvider"]
