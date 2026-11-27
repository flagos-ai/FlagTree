# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""BF16 partitioned row projections with chip-owned output tiles."""

from dataclasses import replace

from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.targets.nvidia.kda_input_projection import kda_input_projection_implementations


def partitioned_linear_implementations():
    # Both families have the same two local asynchronous TMA buffers. There
    # is no global workspace and no inter-CTA barrier inside this operation.
    legacy = tuple(replace(implementation, id=f"tir.partitioned_linear.{implementation.variant}",
                         family="partitioned_linear",
                         shared_workspaces=tuple(replace(workspace, name=f"{workspace.name}_tiles")
                                                 for workspace in implementation.shared_workspaces))
                 for implementation in kda_input_projection_implementations()
                 if implementation.variant in {"mma", "wgmma_local"})
    return (*legacy, TritonImplementation(
        'tir.partitioned_linear.simt', 'partitioned_linear', 'simt',
        {'tile_n': 8, 'block_k': 512, 'compute_num_warps': 4},
        facts={'portable_triton': False, 'source_memory_scope': 'chip', 'result_memory_scope': 'chip'},
    ))


class PartitionedLinearCandidateProvider:
    op_names = frozenset({"nn.partitioned_linear"})

    def propose(self, node, context):
        from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal
        from triton.flagmega.errors import CodegenError
        from triton.flagmega.ir import DistributedType, SBPSplit, local_tensor_type
        from triton.flagmega.ir.execution_mesh import ExecutionMeshKind, effective_execution_mesh
        from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe

        if (getattr(context.target, "codegen_platform", None) != "nvidia"
                or getattr(context.target, "codegen_architecture", None) != "sm90"):
            raise CodegenError("PartitionedLinear currently requires NVIDIA SM90.")
        weight = context.module.node_map[node.inputs[1]].type
        local = local_tensor_type(weight) if isinstance(weight, DistributedType) else weight
        parts = 1 if isinstance(weight, DistributedType) and isinstance(weight.axis_policies[1], SBPSplit) else node.attrs["partitions"]
        span = local.shape[1].fixed_value // parts // node.attrs.get("accumulation_splits", 1)
        mesh = effective_execution_mesh(context.module, node)
        dynamic = mesh is not None and mesh.kind is ExecutionMeshKind.DYNAMIC
        def mapping(value):
            return {'task_mapping': TaskLaunchRecipe(
                local.shape[0].fixed_value, value.parameters['tile_n'],
                node.metadata.get('kernel_program_kind', context.module.metadata.get('kernel_program_kind', 'ordinary')),
                node.metadata.get('kernel_max_ctas'), node.metadata.get('kernel_ctas_per_sm', 1)).to_data()}
        candidates = tuple(context.configure_implementation(value,
                            semantic_parameters=mapping(value) if dynamic else {})
                           for value in context.implementations("partitioned_linear")
                           if (not dynamic or value.variant == 'simt')
                           and (value.variant != 'wgmma_local' or span % 512 == 0))
        if not candidates:
            return None
        preferred = next((value.id for value in candidates if value.id.endswith('.wgmma_local')), candidates[0].id)
        return TritonCandidateProposal(candidates, preferred)


__all__ = ["PartitionedLinearCandidateProvider", "partitioned_linear_implementations"]
