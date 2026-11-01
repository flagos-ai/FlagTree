# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete chip-owned FP8 K partitions without intermediate projection scratch."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation


def partitioned_block_scaled_linear_implementations():
    return (TritonImplementation(
        "tir.partitioned_block_scaled_linear.local", "partitioned_block_scaled_linear", "local",
        {"tile_n": 64, "block_k": 128, "compute_num_warps": 8,
         "accumulator_layout": "vector", "mma_columns": 16, "activation_layout": "auto"},
        requires=("cooperative_grid", "mma_v3", "fp8"),
        facts={"portable_triton": False, "source_memory_scope": "chip", "result_memory_scope": "chip"},
    ),)


class PartitionedBlockScaledLinearCandidateProvider:
    op_names = frozenset({"nn.partitioned_block_scaled_linear"})

    def propose(self, node, context):
        from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal
        from triton.flagmega.errors import CodegenError
        from triton.flagmega.ir import DistributedType, local_tensor_type
        from triton.flagmega.ir.execution_mesh import ExecutionMeshKind, effective_execution_mesh
        from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe

        if (getattr(context.target, "codegen_platform", None) != "nvidia"
                or getattr(context.target, "codegen_architecture", None) != "sm90"
                or node.attrs["weight_block_k"] != 128):
            raise CodegenError("PartitionedBlockScaledLinear kernels require NVIDIA SM90 and K128 quantization blocks.")
        mesh = effective_execution_mesh(context.module, node)
        dynamic = mesh is not None and mesh.kind is ExecutionMeshKind.DYNAMIC
        result = local_tensor_type(node.type) if isinstance(node.type, DistributedType) else node.type
        parts, tokens, row_groups = (d.fixed_value for d in result.shape)
        rows = row_groups * getattr(result.dtype, 'lane_count', 1)

        def mapping(implementation):
            tile = implementation.parameters['tile_n']
            tasks = parts * tokens * ((rows + tile - 1) // tile)
            # A task is a complete (P, T, N tile), including a masked N tail.
            # Flattening unpadded scalar rows would merge tails across P/T.
            return {'task_mapping': TaskLaunchRecipe(tasks, 1,
                node.metadata.get('kernel_program_kind', context.module.metadata.get('kernel_program_kind', 'ordinary')),
                node.metadata.get('kernel_max_ctas'), node.metadata.get('kernel_ctas_per_sm', 1)).to_data()}

        choices = tuple(context.configure_implementation(value,
                        semantic_parameters=mapping(value) if dynamic else {})
                        for value in context.implementations("partitioned_block_scaled_linear"))
        return TritonCandidateProposal(choices, choices[0].id) if choices else None


__all__ = ["PartitionedBlockScaledLinearCandidateProvider", "partitioned_block_scaled_linear_implementations"]
