# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small semantic candidate providers over a target-owned implementation catalog."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node
from triton.flagmega.codegen.triton.vectorization import (
    configured_vector_schedule,
    vectorization_contract,
)

from .core import TritonCandidateContext, TritonCandidateProposal


def _proposal(
    candidates: tuple[Candidate, ...],
    default: str | None = None,
) -> TritonCandidateProposal:
    return TritonCandidateProposal(candidates, default or candidates[0].id)


class ElementwiseCandidateProvider:
    op_names = frozenset({
        "math.add",
        "math.mul",
        "math.div",
        "math.sigmoid",
        "math.silu",
        "math.vectorized_binary",
        "math.vectorized_unary",
        "ntt.vectorized_cast",
        "tensors.cast",
    })

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        contract = vectorization_contract(node)
        semantic_op = _elementwise_semantic_op(node)
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={
                    "vector_schedule": _elementwise_vector_schedule(
                        contract, implementation.parameters
                    ),
                },
            )
            for implementation in context.implementations(
                "elementwise",
                semantic_op=semantic_op,
                vectorization_kind=contract["kind"],
            )
        )
        return None if not candidates else _proposal(candidates)


def _elementwise_semantic_op(node: Node) -> str:
    if node.op == "math.vectorized_binary":
        return f"math.{node.attrs['binary_op']}"
    if node.op == "math.vectorized_unary":
        return f"math.{node.attrs['unary_op']}"
    if node.op == "ntt.vectorized_cast":
        return "tensors.cast"
    return node.op


class BlockFp8CandidateProvider:
    op_names = frozenset({
        "math.block_scaled_matmul",
        "math.packed_block_scaled_matmul",
    })

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        del node
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations("block_fp8")
        )
        if not candidates:
            return None
        return _proposal(candidates, context.choose_default("block_fp8", candidates))


class MatmulGluCandidateProvider:
    op_names = frozenset({"nn.matmul_glu", "nn.packed_matmul_glu"})

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        del node
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations("matmul_glu")
        )
        if not candidates:
            return None
        return _proposal(candidates, context.choose_default("matmul_glu", candidates))


class EmbeddingCandidateProvider:
    op_names = frozenset({"nn.embedding"})

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        del node
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations("embedding", mode="decode")
        )
        return None if not candidates else _proposal(candidates)


class GreedySampleCandidateProvider:
    op_names = frozenset({"nn.greedy_sample"})

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        del node
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={"tie_break": "lowest_index"},
            )
            for implementation in context.implementations(
                "greedy_sample", tie_break="lowest_index"
            )
        )
        if not candidates:
            return None
        return _proposal(
            candidates, context.choose_default("greedy_sample", candidates)
        )


class RmsNormCandidateProvider:
    op_names = frozenset({"nn.rms_norm"})

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        contract = vectorization_contract(node)
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={
                    "vector_schedule": _rms_vector_schedule(
                        contract, implementation.parameters
                    ),
                },
                facts={"local_shard_reduction": True},
            )
            for implementation in context.implementations(
                "rms_norm",
                schedule="local",
                fuse_consumer=False,
            )
        )
        return None if not candidates else _proposal(candidates)


def _elementwise_vector_schedule(contract, parameters):
    if contract["kind"] == "scalar":
        elements = int(parameters["elements_per_program"])
    else:
        elements = int(contract["lane_count"]) * int(parameters["vector_groups"])
    return configured_vector_schedule(
        contract,
        lowering="packed_axes",
        elements_per_program=elements,
    )


def _rms_vector_schedule(contract, parameters):
    return configured_vector_schedule(
        contract,
        lowering="local_reduction",
        block_size=int(parameters["block_size"]),
    )


class GdnCandidateProvider:
    op_names = frozenset({
        "nn.gdn_convolution",
        "nn.gdn_recurrent_core",
    })

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        family = {
            "nn.gdn_convolution": "gdn_convolution",
            "nn.gdn_recurrent_core": "gdn_recurrent",
        }[node.op]
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family)
        )
        if not candidates:
            return None
        return _proposal(candidates, context.choose_default(family, candidates))


__all__ = [
    "BlockFp8CandidateProvider",
    "ElementwiseCandidateProvider",
    "EmbeddingCandidateProvider",
    "GdnCandidateProvider",
    "GreedySampleCandidateProvider",
    "MatmulGluCandidateProvider",
    "RmsNormCandidateProvider",
]
