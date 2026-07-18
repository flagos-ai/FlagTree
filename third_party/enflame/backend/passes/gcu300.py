"""
GCU300 MLIR passes with typed Python API.

This module provides type-safe wrapper functions for GCU300 passes,
avoiding string-based pass names and enabling compile-time checks.
"""

__all__ = [
    'add_gcu64_type_verifier',
    'add_gcu_convert_triton_to_tritongpu',
    'add_tritongpu_remove_layout_conversions',
    'add_triton_gpu_to_triton_gcu',
    'add_convert_tensor_pointer',
    'add_triton_gcu_dot_layout_optimize',
    'add_convert_triton_load_store_to_gcu_dma',
    'add_gcu_combine_ops',
    'add_gcu_triton_fusion',
    'add_triton_gcu_data_layout_optimize',
    'add_triton_gcu_pingpong',
    'add_flatten_triton_func',
    'add_convert_triton_to_gcu',
]


def add_gcu64_type_verifier(pipeline):
    """Add GCU64 type verifier pass."""
    pipeline.add_pass('gcu64-type-verifier')


def add_gcu_convert_triton_to_tritongpu(pipeline, num_warps: int, threads_per_warp: int, num_ctas: int, target: str):
    """Add Triton to TritonGPU conversion pass for GCU.

    Args:
        pipeline: MLIR pass pipeline
        num_warps: Number of warps
        threads_per_warp: Number of threads per warp
        num_ctas: Number of CTAs
        target: Target architecture (e.g., "gcu:gcu300")
    """
    options = f'num-warps={num_warps} threads-per-warp={threads_per_warp} num-ctas={num_ctas} target={target}'
    pipeline.add_pass('gcu-convert-triton-to-tritongpu', options)


def add_tritongpu_remove_layout_conversions(pipeline):
    """Remove unnecessary layout conversions in TritonGPU."""
    pipeline.add_pass('tritongpu-remove-layout-conversions')


def add_triton_gpu_to_triton_gcu(pipeline):
    """Convert TritonGPU dialect to TritonGCU dialect."""
    pipeline.add_pass('triton-gpu-to-triton-gcu')


def add_convert_tensor_pointer(pipeline):
    """Convert tensor pointer operations."""
    pipeline.add_pass('convert-tensor-pointer')


def add_triton_gcu_dot_layout_optimize(pipeline):
    """Optimize dot operation layouts for GCU."""
    pipeline.add_pass('triton-gcu-dot-layout-optimize')


def add_convert_triton_load_store_to_gcu_dma(pipeline, support_stride0: bool = False):
    """Convert Triton load/store operations to GCU DMA operations.

    Args:
        pipeline: MLIR pass pipeline
        support_stride0: Enable stride-0 broadcast support
    """
    options = 'support_stride0=true' if support_stride0 else ''
    pipeline.add_pass('convert-triton-load-store-to-gcu-dma', options)


def add_gcu_combine_ops(pipeline):
    """Combine adjacent GCU operations."""
    pipeline.add_pass('gcu-combine-ops')


def add_gcu_triton_fusion(pipeline):
    """Fuse Triton operations for GCU."""
    pipeline.add_pass('gcu-triton-fusion')


def add_triton_gcu_data_layout_optimize(pipeline):
    """Optimize data layouts for GCU."""
    pipeline.add_pass('triton-gcu-data-layout-optimize')


def add_triton_gcu_pingpong(pipeline, num_stages: int):
    """Add ping-pong buffer optimization pass.

    Args:
        pipeline: MLIR pass pipeline
        num_stages: Number of pipeline stages
    """
    options = f'num_stages={num_stages}'
    pipeline.add_pass('triton-gcu-pingpong', options)


def add_flatten_triton_func(pipeline):
    """Flatten Triton function operations."""
    pipeline.add_pass('flatten-triton-func')


def add_convert_triton_to_gcu(pipeline, vector_length: int = 128):
    """Convert Triton operations to GCU operations.

    Args:
        pipeline: MLIR pass pipeline
        vector_length: Vector length for operations
    """
    options = f'vector-length={vector_length}'
    pipeline.add_pass('convert-triton-to-gcu', options)
