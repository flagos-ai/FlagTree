# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small semantic candidate providers over a target-owned implementation catalog."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node, DistributedType, TensorType, VectorType, DType
from triton.flagmega.ir.distributed_type import is_fully_sharded_across_placement
from triton.flagmega.ir.distributed_inference import tensor_of
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
            if _gdn_applicable(node, context, implementation)
        )
        if not candidates:
            return None
        return _proposal(candidates, context.choose_default(family, candidates))


def _gdn_applicable(node, context, implementation):
    if (node.op == "nn.gdn_recurrent_core"
            and int(node.attrs["key_head_dim"]) > implementation.parameters["tile_state"][0]):
        return False
    if not implementation.contract.get("owner_row_state_snapshot"):
        return True
    output = node.type.fields[0]
    if (not isinstance(output, DistributedType) or output.partial is not None or output.exclusive is not None
            or not is_fully_sharded_across_placement(output)
            or not isinstance(output.tensor.dtype, DType)
            or output.tensor.dtype.value != implementation.contract["required_activation_dtype"]
            or not output.tensor.shape[0].is_fixed or output.tensor.shape[0].fixed_value != 1):
        return False
    if context.module.node_map[node.inputs[2]].type != output:
        return False
    for index in (1, 2, 3, 4, 5):
        tensor = tensor_of(context.module.node_map[node.inputs[index]].type)
        if (not isinstance(tensor.dtype, DType)
                or tensor.dtype.value != implementation.contract["required_activation_dtype"]):
            return False
    key_dim = int(node.attrs["key_head_dim"])
    if key_dim <= 0 or key_dim % 4:
        return False
    state = context.module.node_map[node.inputs[0]].type
    partition = implementation.transfer_pipeline.channels[0].inplace_partition
    _, leaf = partition.source_leaf(state)
    shape = leaf.shape
    return (isinstance(leaf, TensorType) and leaf.dtype == VectorType(DType.FLOAT32, (4,))
            and len(shape) == 4 and all(value.is_fixed for value in shape)
            and tuple(value.fixed_value for value in shape) == (
                1, int(node.attrs["num_value_heads"]), int(node.attrs["value_head_dim"]),
                key_dim // 4))


__all__ = [
    "BlockFp8CandidateProvider",
    "ElementwiseCandidateProvider",
    "EmbeddingCandidateProvider",
    "GdnCandidateProvider",
    "GreedySampleCandidateProvider",
    "MatmulGluCandidateProvider",
    "RmsNormCandidateProvider",
]
