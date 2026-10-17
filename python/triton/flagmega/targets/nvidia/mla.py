# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""BF16 per-head latent-attention projection implementations."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def mla_projection_implementations():
    return (
        TritonImplementation(
            "tir.mla_projection.mma",
            "mla_projection",
            "mma",
            {"compute_num_warps": 8, "block_k": 64, "tile_n": 64},
            {"mode": "decode"},
            facts={"portable_triton": False, "canonical_global_arguments": ("weight", "scale")},
        ),
    )


class NvidiaMlaProjectionSemanticTIRCandidateProvider:
    """Expose the implemented shape/target contract without a fallback."""

    op_names = frozenset({"nn.mla_projection"})

    def propose(self, node, context):
        from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal
        from triton.flagmega.errors import CodegenError
        from triton.flagmega.ir import Candidate
        from triton.flagmega.ir.distributed_inference import tensor_of

        if node.op not in self.op_names:
            return None
        if (
            getattr(context.target, "codegen_platform", None) != "nvidia"
            or getattr(context.target, "codegen_architecture", None) != "sm90"
        ):
            raise CodegenError("MLA projection currently requires the implemented NVIDIA SM90 BF16 MMA kernel.")
        source = tensor_of(context.module.node_map[node.inputs[0]].type)
        if not source.shape[0].is_fixed or source.shape[0].fixed_value != 1:
            raise CodegenError("MLA projection MMA currently supports single-token chunks.")
        candidate = Candidate("semantic.nn.mla_projection", {}, {})
        return TritonCandidateProposal((candidate,), candidate.id, selection_kind="semantic_tir")
