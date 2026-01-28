"""
Compiler integration for pipeline optimization

This module handles the integration of pipeline configuration into
the Triton compilation pipeline.
"""

import os
from typing import Dict, Optional, Any
import triton


class PipelineCompilerHook:
    """
    Compiler hook for injecting pipeline optimization passes.

    This integrates with the Triton compiler's multi-stage pipeline
    to add buffer analysis and pipelining transformation passes.
    """

    @staticmethod
    def inject_pipeline_passes(stages: Dict[str, Any], options: Any, config: Optional[Dict] = None):
        """
        Inject pipelining passes into compilation stages.

        This replaces Triton's built-in pipelining with FlagTree's AdvancedPipeliner
        when @auto_pipeline decorator is used.

        Args:
            stages: Dictionary of compilation stages
            options: Compiler options
            config: Pipeline configuration from @auto_pipeline decorator
        """
        if config is None:
            return

        # Extract pipeline configuration
        global_stages = config.get('global_to_shared_stages', 1)
        register_stages = config.get('shared_to_register_stages', 1)
        async_copy = config.get('enable_async_copy', False)
        swizzle = config.get('enable_swizzle', False)
        min_speedup = config.get('min_speedup', 1.0)
        warp_specialization = config.get('enable_warp_specialization', False)
        multi_buffer_fusion = config.get('enable_multi_buffer_fusion', False)

        # Only inject if pipelining is actually enabled
        if global_stages <= 1 and register_stages <= 1:
            return

        # Get TTGIR stage (after Triton → TritonGPU lowering)
        if 'ttgir' not in stages:
            print("Warning: TTGIR stage not found, cannot apply pipelining")
            return

        original_ttgir_fn = stages['ttgir']

        # Create a replacement TTGIR function that uses AdvancedPipeliner INSTEAD of Triton's built-in
        def ttgir_with_advanced_pipelining(src, metadata):
            from .._C.libtriton import passes, ir, nvidia
            from ..runtime.driver import driver


            # Get backend options from options object
            num_warps = getattr(options, 'num_warps', 4)
            num_ctas = getattr(options, 'num_ctas', 1)

            # Get capability from the active device
            try:
                target = driver.active.get_current_target()
                capability = target.arch
            except:
                capability = 80  # Default to Ampere

            mod = src
            cluster_info = nvidia.ClusterInfo()
            if hasattr(options, 'cluster_dims') and options.cluster_dims is not None:
                cluster_info.clusterDimX = options.cluster_dims[0]
                cluster_info.clusterDimY = options.cluster_dims[1]
                cluster_info.clusterDimZ = options.cluster_dims[2]

            # TTIR -> TTGIR conversion
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", num_warps, 32, num_ctas)

            # Standard TTGIR optimization passes
            passes.ttgpuir.add_coalesce(pm)
            if capability // 10 >= 8:
                passes.ttgpuir.add_f32_dot_tc(pm)
            nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_optimize_thread_locality(pm)
            passes.ttgpuir.add_accelerate_matmul(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
            passes.common.add_cse(pm)

            if capability // 10 >= 8:
                passes.ttgpuir.add_combine_tensor_select_and_if(pm)

                _debug = os.environ.get('FLAGTREE_DEBUG_PIPELINE', '0') == '1'

                # Always use Triton's well-tested built-in pipeline for Global→Shared
                # This generates proper cp.async operations with correct synchronization
                if _debug:
                    print(f"[FlagTree] Using Triton's built-in pipeline: num_stages={global_stages}")
                passes.ttgpuir.add_pipeline(pm, global_stages)

                # Run AdvancedPipeliner AFTER for additional optimizations:
                # - Shared→Register pipelining (register_stages > 1)
                # - Memory swizzle optimization (swizzle)
                # - Multi-buffer fusion (multi_buffer_fusion)
                use_advanced = (register_stages > 1 or swizzle or multi_buffer_fusion)
                if use_advanced:
                    if _debug:
                        print(f"[FlagTree] Running AdvancedPipeliner for additional optimizations:")
                        print(f"           register_stages={register_stages}, swizzle={swizzle}, fusion={multi_buffer_fusion}")
                    # Pass global_stages=1 to skip Global→Shared (already done)
                    passes.ttgpuir.add_advanced_pipeliner(pm, 1, register_stages,
                                                          False, swizzle, min_speedup,
                                                          warp_specialization, multi_buffer_fusion)

            # Enhanced optimization passes - run multiple iterations for better results
            passes.ttgpuir.add_prefetch(pm)
            passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_reduce_data_duplication(pm)
            passes.ttgpuir.add_reorder_instructions(pm)
            passes.common.add_cse(pm)

            # Second optimization iteration - can find more opportunities after first pass
            if use_advanced:
                passes.ttgpuir.add_prefetch(pm)
                passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
                passes.ttgpuir.add_remove_layout_conversions(pm)
                passes.ttgpuir.add_reorder_instructions(pm)
                passes.common.add_cse(pm)

            passes.common.add_symbol_dce(pm)

            if capability // 10 >= 9:
                nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
                nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
            passes.common.add_canonicalizer(pm)

            pm.run(mod)
            metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)

            return mod

        # Replace the TTGIR stage with our custom implementation
        stages['ttgir'] = ttgir_with_advanced_pipelining

    @staticmethod
    def extract_config_from_kernel(kernel_fn) -> Optional[Dict]:
        """
        Extract pipeline configuration from kernel function attributes.

        Args:
            kernel_fn: JITFunction with potential _pipeline_config attribute

        Returns:
            Dictionary of pipeline configuration or None
        """
        if hasattr(kernel_fn, '_pipeline_config'):
            config = kernel_fn._pipeline_config
            if hasattr(config, 'to_dict'):
                return config.to_dict()
            elif isinstance(config, dict):
                return config

        if hasattr(kernel_fn, 'fn') and hasattr(kernel_fn.fn, '_pipeline_config'):
            config = kernel_fn.fn._pipeline_config
            if hasattr(config, 'to_dict'):
                return config.to_dict()
            elif isinstance(config, dict):
                return config

        return None


def enable_pipelining_globally(enabled: bool = True):
    """
    Enable/disable pipelining optimization globally for all kernels.

    Args:
        enabled: Whether to enable pipelining

    Note:
        This is a global setting. Individual kernels can override with @auto_pipeline.
        Disabled by default for safety.
    """
    import os
    os.environ['TRITON_ENABLE_PIPELINING'] = '1' if enabled else '0'


def is_pipelining_enabled() -> bool:
    """Check if pipelining is globally enabled"""
    import os
    return os.environ.get('TRITON_ENABLE_PIPELINING', '0') == '1'


def get_pipeline_stats(kernel_fn) -> Dict[str, Any]:
    """
    Get pipelining statistics for a compiled kernel.

    Args:
        kernel_fn: Compiled JITFunction

    Returns:
        Dictionary with pipelining statistics:
            - enabled: Whether pipelining was applied
            - buffers_pipelined: Number of buffers pipelined
            - stages_used: [global_stages, register_stages]
            - speedup_estimate: Expected speedup

    Example:
        @triton.jit
        @auto_pipeline(PipelineConfig(global_to_shared_stages=3))
        def kernel(...):
            ...

        kernel[grid](...)
        stats = get_pipeline_stats(kernel)
        print(f"Speedup estimate: {stats['speedup_estimate']:.2f}x")
    """
    # This would be populated by the compiler
    # For now, return default structure
    config = PipelineCompilerHook.extract_config_from_kernel(kernel_fn)

    if not config:
        return {
            'enabled': False,
            'buffers_pipelined': 0,
            'stages_used': [1, 1],
            'speedup_estimate': 1.0,
        }

    return {
        'enabled': True,
        'buffers_pipelined': 0,  # Would be filled by compiler
        'stages_used': [
            config.get('global_to_shared_stages', 1),
            config.get('shared_to_register_stages', 1)
        ],
        'speedup_estimate': 1.0,  # Would be computed by analytical model
    }


# Integration with triton.autotune
def extend_autotune_with_pipelining():
    """
    Extend triton.autotune to include pipeline configurations in search space.

    This allows autotuner to explore different pipeline stage counts
    along with traditional parameters like BLOCK_SIZE and num_warps.

    Example:
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=2, pipeline_stages=2),
                triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, pipeline_stages=3),
            ],
            key=['M', 'N', 'K'],
        )
        @triton.jit
        def kernel(...):
            ...

    Note:
        Requires integration with triton.Config and Autotuner classes
    """
    # This would extend triton.Config to support pipeline_stages parameter
    # Implementation would modify triton/runtime/autotuner.py
    pass


__all__ = [
    'PipelineCompilerHook',
    'enable_pipelining_globally',
    'is_pipelining_enabled',
    'get_pipeline_stats',
    'extend_autotune_with_pipelining',
]
