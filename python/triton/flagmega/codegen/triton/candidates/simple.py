# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small semantic candidate providers over a target-owned implementation catalog."""

from __future__ import annotations

from triton.flagmega.codegen.triton.vectorization import (
    ELEMENTWISE_OP_NAMES,
    configured_vector_schedule,
    vectorization_contract,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Candidate, DistributedType, DType, Node, TensorType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import is_fully_sharded_across_placement
from triton.flagmega.ir.ops.math._block_scaled import (
    block_scaled_mma_supports_precision,
    block_scaled_precision_supports_target,
)

from .core import TritonCandidateContext, TritonCandidateProposal


def _proposal(
    candidates: tuple[Candidate, ...],
    default: str | None = None,
) -> TritonCandidateProposal:
    return TritonCandidateProposal(candidates, default or candidates[0].id)


class ElementwiseCandidateProvider:
    op_names = ELEMENTWISE_OP_NAMES

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        contract = vectorization_contract(node)
        semantic_op = _elementwise_semantic_op(node)
        from triton.flagmega.ir.execution_mesh import effective_execution_mesh, ExecutionMeshKind
        mesh = effective_execution_mesh(context.module, node)
        dynamic = mesh is not None and mesh.kind is ExecutionMeshKind.DYNAMIC
        candidates = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters={
                    "vector_schedule": _elementwise_vector_schedule(
                        contract, implementation.parameters
                    ),
                    **({'task_mapping': {
                        'schema': 'flagmega.independent-linear-tiles/v1',
                        'program_kind': node.metadata.get('kernel_program_kind', 'ordinary'),
                        'max_ctas': node.metadata.get('kernel_max_ctas'),
                        'ctas_per_sm': node.metadata.get('kernel_ctas_per_sm', 1),
                    }} if dynamic else {}),
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
        native_quant = node.attrs.get("weight_encoding")
        if not block_scaled_precision_supports_target(
                node.attrs, getattr(context.target, "codegen_platform", None),
                getattr(context.target, "codegen_architecture", None)):
            raise CodegenError("SM90 reciprocal activation quantization requires the NVIDIA SM90 target.")
        value_dtype = tensor_of(context.module.node_map[node.inputs[0]].type).dtype
        mma_legal = block_scaled_mma_supports_precision(node.attrs, value_dtype,
                                                       block_k=int(node.attrs["weight_block_k"]))
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations("block_fp8")
            if (native_quant is None or implementation.variant == "simt")
            and (implementation.variant != "mma" or mma_legal)
            and (not implementation.contract.get("requires_chip_projection_blocks", False)
                 or _supports_chip_projection_blocks(node, context)))
        if not candidates:
            return None
        return _proposal(candidates, context.choose_default("block_fp8", candidates))


def _supports_chip_projection_blocks(node, context):
    from triton.flagmega.codegen.triton.block_fp8_geometry import cooperative_block_geometry

    return cooperative_block_geometry(dict(zip(
        ("value", "weight", "weight_scale"),
        (context.module.node_map[value].type for value in node.inputs), strict=True,
    )), node.type, node.attrs) is not None


class MatmulGluCandidateProvider:
    op_names = frozenset({"nn.matmul_glu", "nn.packed_matmul_glu"})

    def propose(self, node: Node, context: TritonCandidateContext) -> TritonCandidateProposal | None:
        if not block_scaled_precision_supports_target(
                node.attrs, getattr(context.target, "codegen_platform", None),
                getattr(context.target, "codegen_architecture", None)):
            raise CodegenError("SM90 reciprocal activation quantization requires the NVIDIA SM90 target.")
        native_quant = node.attrs.get("gate_weight_encoding") or node.attrs.get("up_weight_encoding")
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations("matmul_glu")
            if (native_quant is None or implementation.variant == "simt")
            and (not implementation.contract.get("requires_chip_glu_blocks", False)
                 or _supports_chip_glu_blocks(node, context)))
        if not candidates:
            return None
        return _proposal(candidates, context.choose_default("matmul_glu", candidates))


def _supports_chip_glu_blocks(node, context):
    from triton.flagmega.codegen.triton.matmul_glu_geometry import cooperative_glu_geometry
    from triton.flagmega.ir.ops.core import get_definition

    return cooperative_glu_geometry({
        parameter.name: context.module.node_map[node.inputs[index]].type
        for index, parameter in enumerate(get_definition(node.op).input_parameters)
    }, node.type, node.attrs) is not None


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
        # Triton tensor ranges require a power-of-two scalar tile. The lane
        # representation itself need not have a power-of-two payload; address
        # each scalar using its full lane coordinate and mask the final tile.
        elements = 1 << (elements - 1).bit_length()
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
    if implementation.contract.get("scalar_head_gates"):
        from .gdn_head_gates import applicable
        return applicable(node, context)
    if node.op == "nn.gdn_recurrent_core" and node.attrs.get("b_weight_encoding") is not None:
        # The native packed projection helper is implemented by the persistent
        # recurrent kernel; the state-pipeline variant still assumes dense rows.
        if implementation.variant != "persistent":
            return False
    if (node.op == "nn.gdn_recurrent_core"
            and int(node.attrs["key_head_dim"]) > implementation.parameters["tile_state"][0]):
        return False
    if not implementation.contract.get("owner_row_state_snapshot"):
        return True
    output = node.type.fields[0]
    if (not isinstance(output, DistributedType) or output.partial is not None or output.exclusive is not None
            or not is_fully_sharded_across_placement(output) or not isinstance(output.tensor.dtype, DType)
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
    return (isinstance(leaf, TensorType) and leaf.dtype == VectorType(DType.FLOAT32, (4, )) and len(shape) == 4
            and all(value.is_fixed for value in shape) and tuple(value.fixed_value for value in shape)
            == (1, int(node.attrs["num_value_heads"]), int(node.attrs["value_head_dim"]), key_dim // 4))


__all__ = [
    "BlockFp8CandidateProvider",
    "ElementwiseCandidateProvider",
    "EmbeddingCandidateProvider",
    "GdnCandidateProvider",
    "GreedySampleCandidateProvider",
    "MatmulGluCandidateProvider",
    "RmsNormCandidateProvider",
]
