# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Single-tile staged routing: the tile must cover the whole router extent."""

from typing import ClassVar

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
    family_aliases: ClassVar[dict[str, str]] = {}

    def propose(self, node, context):
        family = self.family_aliases.get(node.op, node.op.removeprefix("nn."))
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family, indexing="local"))
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates, context.choose_default(family, candidates, portable_fallback=f"tir.{family}.local"))


class KdaCandidateProvider(_NamedFamilyCandidateProvider):
    family_aliases: ClassVar[dict[str, str]] = {"nn.kda_gated_step": "kda_gated_recurrent_core"}
    op_names = frozenset({"nn.kda_convolution", "nn.kda_recurrent_core",
                          "nn.kda_gated_recurrent_core", "nn.kda_gated_step"})

    def propose(self, node, context):
        if ((node.op == "nn.kda_gated_step"
             or node.op == "nn.kda_gated_recurrent_core" and node.attrs.get("arithmetic") == "chunk_bf16")
                and getattr(context.target, "codegen_platform", None) != "nvidia"):
            raise CodegenError(
                "KDA chunk_bf16 requires the implemented CUDA PTX fast-math contract."
            )
        return super().propose(node, context)


class MhcCandidateProvider(_NamedFamilyCandidateProvider):
    """The hyper-connection stages: one replicated kernel family each."""

    op_names = frozenset({"nn.mhc_pre", "nn.mhc_post"})

    def propose(self, node, context):
        family = node.op.removeprefix("nn.")
        reduction = node.attrs.get("projection_reduction", "parallel") if node.op == "nn.mhc_pre" else "parallel"
        if reduction == "contiguous_tf32" and (
                getattr(context.target, "codegen_platform", None) != "nvidia"
                or getattr(context.target, "codegen_architecture", None) != "sm90"):
            raise CodegenError("mHC contiguous_tf32 requires the implemented SM90 WGMMA numerical contract.")
        if reduction == "stream_fma_128":
            capability = getattr(getattr(context.target, "machine", None), "capability", None)
            if (getattr(context.target, "codegen_platform", None) != "nvidia"
                    or getattr(capability, "warp_size", None) != 32):
                raise CodegenError("mHC stream_fma_128 requires the implemented CUDA 32-lane numerical contract.")
        candidates = tuple(context.configure_implementation(implementation)
                           for implementation in context.implementations(family, indexing="local")
                           if implementation.contract.get("projection_reduction", "parallel") == reduction)
        if not candidates:
            return None
        variant = {"parallel": "local", "stream_fma_128": "stream_fma", "contiguous_tf32": "contiguous_tf32"}[reduction]
        fallback = f"tir.{family}.{variant}"
        return TritonCandidateProposal(candidates, context.choose_default(family, candidates, portable_fallback=fallback))


class StagedRoutingCandidateProvider:
    op_names = frozenset({"nn.staged_routing"})

    def propose(self, node, context):
        if node.attrs.get("arithmetic", "standard") == "tanh_scaled_reciprocal":
            capability = getattr(getattr(context.target, "machine", None), "capability", None)
            if (getattr(context.target, "codegen_platform", None) != "nvidia"
                    or getattr(capability, "warp_size", None) != 32):
                raise CodegenError("StagedRouting tanh_scaled_reciprocal requires the CUDA 32-lane numerical contract.")
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
