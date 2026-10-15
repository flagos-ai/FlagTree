# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TIR candidates for executable distributed layout transitions."""

from triton.flagmega.ir import (
    DistributedType,
    TensorType,
    TupleType,
    boxing_reads_peer_chips,
)
from triton.flagmega.codegen.triton.candidates.core import (
    TritonCandidateContext,
    TritonCandidateProposal,
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
        target = node.type
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
        facts = {
            "collective_semantics": semantics,
            "tuple_field_count": len(transitions),
        }
        if _reads_peer_chips(source, target):
            # The leaf has to read the peer chips' instances of the source
            # bytes, so the hazard planner must bracket it with device-scope
            # rendezvous instead of chip-local grid barriers.
            facts["source_memory_scope"] = "device"
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
