"""
GCU400 MLIR passes with typed Python API.

This module provides type-safe wrapper functions for GCU400 passes,
avoiding string-based pass names and enabling compile-time checks.
"""

__all__ = [
    'add_gcu64_type_verifier',
    'add_gcu_convert_triton_to_tritongpu',
    'add_tle_to_triton_gcu',
    'add_triton_gpu_to_triton_gcu',
    'add_convert_tensor_pointer',
    'add_convert_triton_load_store_to_gcu_dma',
    'add_tritongcu_accelerate_matmul',
    'add_gcu_warp_specialization',
    'add_triton_gcu_allocate_warp_groups',
    'add_triton_wgdot_to_gcu',
    'add_tritongpu_remove_layout_conversions',
    'add_triton_gcu_data_layout_optimize',
    'add_gcu_combine_ops',
    'add_gcu_triton_fusion',
    'add_flatten_triton_func',
    'add_annotate_dot_acc_reuse',
    'add_triton_gcu_local_mem_optimize',
    'add_convert_triton_to_gcu',
]


def add_gcu64_type_verifier(pipeline):
    """Add GCU64 type verifier pass."""
    pipeline.add_pass('gcu64-type-verifier')


def add_gcu_convert_triton_to_tritongpu(pipeline, num_warps: int, threads_per_warp: int, num_ctas: int, target: str):
    """Convert Triton IR to TritonGPU IR with GCU-aware layout.

    Args:
        pipeline: MLIR pass pipeline
        num_warps: Number of warps
        threads_per_warp: Number of threads per warp
        num_ctas: Number of CTAs in a CGA
        target: Target triple, e.g. gcu:gcu400
    """
    options = f'num-warps={num_warps} threads-per-warp={threads_per_warp} num-ctas={num_ctas} target={target}'
    pipeline.add_pass('gcu-convert-triton-to-tritongpu', options)


def add_tle_to_triton_gcu(pipeline, cluster_dims=(1, 1, 1)):
    """Convert TLE dialect to TritonGCU dialect."""
    options = (f'cluster-dim-x={cluster_dims[0]} '
               f'cluster-dim-y={cluster_dims[1]} '
               f'cluster-dim-z={cluster_dims[2]}')
    pipeline.add_pass('tle-to-triton-gcu', options)


def add_triton_gpu_to_triton_gcu(pipeline):
    """Convert TritonGPU dialect to TritonGCU dialect."""
    pipeline.add_pass('triton-gpu-to-triton-gcu')


def add_convert_tensor_pointer(pipeline):
    """Convert tensor pointer operations."""
    pipeline.add_pass('convert-tensor-pointer')


def add_convert_triton_load_store_to_gcu_dma(pipeline, support_stride0: bool = False, redundant_sip: bool = False):
    """Convert Triton load/store operations to GCU DMA operations.

    Args:
        pipeline: MLIR pass pipeline
        support_stride0: Enable stride-0 broadcast support
        redundant_sip: Enable redundant SIP optimization
    """
    parts = []
    if support_stride0:
        parts.append('support_stride0=true')
    if redundant_sip:
        parts.append('skip_dma=true')
    options = ' '.join(parts)
    pipeline.add_pass('convert-triton-load-store-to-gcu-dma', options)


def add_gcu_tle_lower_async_load(pipeline):
    """Lower GCU TLE async load operations."""
    pipeline.add_pass('gcu-tle-lower-async-load')


def add_tle_convert_arg_to_memdesc(pipeline):
    """Convert TLE arguments to memory descriptors."""
    pipeline.add_pass('tle-convert-arg-to-memdesc')


def add_tle_remove_redundant_copy(pipeline):
    """Remove redundant copy operations in TLE."""
    pipeline.add_pass('tle-remove-redundant-copy')


def add_tle_dslregion_inline(pipeline):
    """Inline TLE DSL region operations."""
    pipeline.add_pass('tle-dslregion-inline')


def add_tritongcu_accelerate_matmul(pipeline):
    """Accelerate matrix multiplication operations for GCU."""
    pipeline.add_pass('tritongcu-accelerate-matmul')


def add_gcu_warp_specialization(pipeline, num_stages: int, dump_enabled: bool, ws_inner_barrier_enabled: bool):
    """Warp specialization for GCU.

    Args:
        pipeline: MLIR pass pipeline
        num_stages: Number of stages
        dump_enabled: Dump intermediate steps
        ws_inner_barrier_enabled: Insert inner barrier
    """
    options = f'num-stages={num_stages} dump-intermediate-steps={dump_enabled} inner-barrier={ws_inner_barrier_enabled}'
    pipeline.add_pass('gcu-warp-specialization', options)


def add_triton_gcu_allocate_warp_groups(pipeline):
    """Allocate warp groups for GCU."""
    pipeline.add_pass('triton-gcu-allocate-warp-groups')


def add_triton_wgdot_to_gcu(pipeline):
    """Convert workgroup dot operations to GCU operations."""
    pipeline.add_pass('triton-wgdot-to-gcu')


def add_tritongpu_remove_layout_conversions(pipeline):
    """Remove unnecessary layout conversions in TritonGPU."""
    pipeline.add_pass('tritongpu-remove-layout-conversions')


def add_triton_gcu_data_layout_optimize(pipeline):
    """Optimize data layouts for GCU."""
    pipeline.add_pass('triton-gcu-data-layout-optimize')


def add_gcu_combine_ops(pipeline):
    """Combine adjacent GCU operations."""
    pipeline.add_pass('gcu-combine-ops')


def add_gcu_triton_fusion(pipeline, arch: str):
    """Fuse Triton operations for GCU.

    Args:
        pipeline: MLIR pass pipeline
        arch: Target architecture (e.g., "gcu400", "gcu410")
    """
    options = f'arch={arch}'
    pipeline.add_pass('gcu-triton-fusion', options)


def add_flatten_triton_func(pipeline):
    """Flatten Triton function operations."""
    pipeline.add_pass('flatten-triton-func')


def add_annotate_dot_acc_reuse(pipeline):
    """Annotate dot accumulator reuse opportunities."""
    pipeline.add_pass('annotate-dot-acc-reuse')


def add_triton_gcu_local_mem_optimize(pipeline):
    """Optimize local memory usage for GCU."""
    pipeline.add_pass('triton-gcu-local-mem-optimize')


def add_convert_triton_to_gcu(pipeline):
    """Convert Triton operations to GCU operations."""
    pipeline.add_pass('convert-triton-to-gcu')
