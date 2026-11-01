# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete-chip six-projection kernels; no weight concatenation or quantization."""

from dataclasses import replace

from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.tir.shared_workspace_descriptor import TIRSharedWorkspaceDescriptor
from triton.flagmega.ir.tir.transfer_pipeline_channel import TIRTransferPipelineChannel
from triton.flagmega.ir.tir.transfer_pipeline_contract import TIRTransferPipelineContract
from triton.flagmega.targets.nvidia.shared_layout import align_shared_workspaces


def kda_input_projection_implementations():
    facts = {"portable_triton": False, "source_memory_scope": "chip", "result_memory_scope": "chip"}
    implementations = (
        TritonImplementation(
            "tir.kda_input_projection.mma", "kda_input_projection", "mma",
            {"tile_n": 64, "block_k": 32, "compute_num_warps": 4},
            requires=("cooperative_grid",), facts=facts,
        ),
        TritonImplementation(
            "tir.kda_input_projection.wgmma", "kda_input_projection", "wgmma",
            {"tile_n": 64, "block_k": 512, "num_stages": 2, "consumer_warps": 8,
             "producer_warps": 4, "producer_registers": 64},
            requires=("cooperative_grid", "mma_v3", "tma", "warp_specialize"),
            facts={**facts, "host_tensor_descriptor": True, "transfer_pipeline": True},
            shared_workspaces=align_shared_workspaces((
                TIRSharedWorkspaceDescriptor("weight", tensor_type("bfloat16", (2, 64, 512)), 128,
                                             matrix_compatible=True),
                TIRSharedWorkspaceDescriptor("activation", tensor_type("bfloat16", (2, 8, 512)), 128,
                                             matrix_compatible=True),
            )),
            transfer_pipeline=TIRTransferPipelineContract(
                (TIRTransferPipelineChannel("projection", tuple(range(7)), (0, 1), 16),), capacity=2,
            ),
        ),
    )
    # The local asynchronous schedule preserves the entry's compute warp count.
    # A global producer partition expands every neighboring kernel's CTA too.
    return (*implementations, replace(
        implementations[1], id="tir.kda_input_projection.wgmma_local", variant="wgmma_local",
        parameters={"tile_n": 64, "block_k": 512, "num_stages": 2, "compute_num_warps": 8},
        requires=("cooperative_grid", "mma_v3", "tma"), transfer_pipeline=None,
        facts={**facts, "host_tensor_descriptor": True},
    ))


class KdaInputProjectionCandidateProvider:
    op_names = frozenset({"nn.kda_input_projection"})

    def propose(self, node, context):
        from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal
        from triton.flagmega.errors import CodegenError
        from triton.flagmega.ir.distributed_inference import tensor_of

        if (getattr(context.target, "codegen_platform", None) != "nvidia"
                or getattr(context.target, "codegen_architecture", None) != "sm90"):
            raise CodegenError("KDA input projection currently requires NVIDIA SM90.")
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        reduction = value.shape[-1].fixed_value * getattr(value.dtype, "lane_count", 1)
        candidates = tuple(context.configure_implementation(implementation)
                           for implementation in context.implementations("kda_input_projection")
                           if implementation.variant not in {"wgmma", "wgmma_local"} or reduction % 512 == 0)
        if not candidates:
            return None
        preferred = next((value.id for value in candidates if value.id.endswith(".wgmma_local")), candidates[0].id)
        return TritonCandidateProposal(candidates, preferred)
