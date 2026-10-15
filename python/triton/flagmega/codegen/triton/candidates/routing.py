# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Single-tile staged routing: the tile must cover the whole router extent."""

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.distributed_inference import tensor_of

from .core import TritonCandidateProposal


class _NamedFamilyCandidateProvider:
    """Offer every owner-local implementation of the family named by the op.

    Operations whose semantic name is already the kernel family need no routing
    decision: the target's implementation model decides which variants exist,
    and the portable ``tir.<family>.local`` is the default.
    """

    op_names = frozenset()

    def propose(self, node, context):
        family = node.op.removeprefix("nn.")
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family, indexing="local"))
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates, context.choose_default(family, candidates, portable_fallback=f"tir.{family}.local"))


class KdaCandidateProvider(_NamedFamilyCandidateProvider):
    op_names = frozenset({"nn.kda_convolution", "nn.kda_recurrent_core",
                          "nn.kda_gated_recurrent_core"})


class MhcCandidateProvider(_NamedFamilyCandidateProvider):
    """The hyper-connection stages: one replicated kernel family each."""

    op_names = frozenset({"nn.mhc_pre", "nn.mhc_post"})


class StagedRoutingCandidateProvider:
    op_names = frozenset({"nn.staged_routing"})

    def propose(self, node, context):
        source = context.module.node_map[node.inputs[0]]
        value_type = tensor_of(source.type)
        extent = value_type.shape[-1].fixed_value
        implementations = context.implementations("staged_routing", indexing="local")
        candidates = []
        for implementation in implementations:
            tile = int(implementation.parameters["elements_per_program"])
            if tile >= extent:
                candidates.append(context.configure_implementation(implementation))
        if not candidates:
            available = tuple(int(implementation.parameters["elements_per_program"])
                             for implementation in implementations)
            raise CodegenError(
                "StagedRouting has no complete local tile for router extent "
                f"{extent}; available single-tile extents are {available}."
            )
        candidates.sort(key=lambda candidate: int(candidate.parameters["elements_per_program"]))
        return TritonCandidateProposal(
            candidates, context.choose_default("staged_routing", candidates,
                                               portable_fallback=candidates[0].id))
