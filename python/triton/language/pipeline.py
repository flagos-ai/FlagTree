"""
Triton Language Pipeline Optimization API with TLX Integration

This module provides high-level Python APIs for enabling advanced
multi-level pipelining optimization in Triton kernels, including
TLX (Triton Low-level Language Extensions) for warp specialization.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from enum import Enum
import triton
import triton.language as tl


class WarpRole(Enum):
    """Warp roles for warp specialization"""
    PRODUCER = "producer"      # Prefetch data from global to shared memory
    CONSUMER = "consumer"      # Compute using data from shared memory
    MIXED = "mixed"           # Both producer and consumer (default Triton behavior)
    DEFAULT = "default"       # Use default scheduling


@dataclass
class WarpSpecConfig:
    """
    Configuration for TLX warp specialization.

    Warp specialization dedicates warps to specific tasks:
    - Producer warps: prefetch data from global to shared memory
    - Consumer warps: perform compute using data in shared memory

    This enables better overlap between memory and compute operations.

    Attributes:
        num_producer_warps: Number of warps dedicated to data prefetching
        num_consumer_warps: Number of warps dedicated to computation
        producer_registers: Register budget for producer warps (default: auto)
        consumer_registers: Register budget for consumer warps (default: auto)
        num_pipeline_stages: Number of pipeline stages for producer/consumer overlap
        enable_pingpong: Enable ping-pong buffering for better overlap

    Example:
        warp_config = WarpSpecConfig(
            num_producer_warps=1,
            num_consumer_warps=3,
            num_pipeline_stages=2
        )
    """
    num_producer_warps: int = 1
    num_consumer_warps: int = 3
    producer_registers: Optional[int] = None
    consumer_registers: Optional[int] = None
    num_pipeline_stages: int = 2
    enable_pingpong: bool = False

    def __post_init__(self):
        if self.num_producer_warps < 1:
            raise ValueError("num_producer_warps must be >= 1")
        if self.num_consumer_warps < 1:
            raise ValueError("num_consumer_warps must be >= 1")
        if self.num_pipeline_stages < 1:
            raise ValueError("num_pipeline_stages must be >= 1")

    @property
    def total_warps(self):
        return self.num_producer_warps + self.num_consumer_warps


@dataclass
class TLXBufferConfig:
    """
    Configuration for TLX local buffer allocation.

    Attributes:
        shape: Shape of the buffer (excluding pipeline dimension)
        dtype: Data type of buffer elements
        num_buffers: Number of pipeline buffers (for double/triple buffering)
        storage: Memory storage kind ('smem' or 'tmem')
        enable_swizzle: Apply swizzling to reduce bank conflicts
        layout: Optional custom layout encoding
    """
    shape: Tuple[int, ...]
    dtype: any  # tl.dtype
    num_buffers: int = 2
    storage: str = "smem"  # "smem" or "tmem"
    enable_swizzle: bool = True
    layout: Optional[any] = None

    def __post_init__(self):
        if self.num_buffers < 1:
            raise ValueError("num_buffers must be >= 1")
        if self.storage not in ["smem", "tmem"]:
            raise ValueError(f"storage must be 'smem' or 'tmem', got {self.storage}")


@dataclass
class PipelineConfig:
    """
    Configuration for multi-level pipelining optimization with TLX support.

    Attributes:
        global_to_shared_stages: Number of stages for global -> shared memory pipeline (default: 1, no pipelining)
        shared_to_register_stages: Number of stages for shared -> register pipeline (default: 1, no pipelining)
        enable_async_copy: Use hardware async copy (cp.async on Ampere+, TMA on Hopper+)
        enable_swizzle: Apply swizzling pattern to reduce shared memory bank conflicts
        min_speedup: Minimum expected speedup threshold (default 1.0 = no threshold)
        enable_warp_specialization: Enable dedicated producer/consumer warps for better overlap
        enable_multi_buffer_fusion: Enable shared sync barriers for K/V buffers in attention
        warp_spec_config: TLX warp specialization configuration
        buffer_configs: List of TLX buffer configurations for manual buffer management
        enable_tma: Enable Tensor Memory Accelerator (Hopper+)
        enable_cluster_barriers: Enable cross-CTA cluster barriers (Hopper+)

    Example:
        config = PipelineConfig(
            global_to_shared_stages=3,
            shared_to_register_stages=2,
            enable_async_copy=True,
            enable_swizzle=True,
            enable_warp_specialization=True,
            warp_spec_config=WarpSpecConfig(
                num_producer_warps=1,
                num_consumer_warps=3
            )
        )
    """
    global_to_shared_stages: int = 1
    shared_to_register_stages: int = 1
    enable_async_copy: bool = True
    enable_swizzle: bool = False
    min_speedup: float = 1.0
    enable_warp_specialization: bool = False
    enable_multi_buffer_fusion: bool = False
    warp_spec_config: Optional[WarpSpecConfig] = None
    buffer_configs: List[TLXBufferConfig] = field(default_factory=list)
    enable_tma: bool = False
    enable_cluster_barriers: bool = False

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.global_to_shared_stages < 1:
            raise ValueError("global_to_shared_stages must be >= 1")
        if self.global_to_shared_stages > 5:
            raise ValueError("global_to_shared_stages should not exceed 5 (register pressure)")
        if self.shared_to_register_stages < 1:
            raise ValueError("shared_to_register_stages must be >= 1")
        if self.shared_to_register_stages > 5:
            raise ValueError("shared_to_register_stages should not exceed 5")
        if self.min_speedup < 1.0:
            raise ValueError("min_speedup must be >= 1.0")

        # Auto-create warp spec config if warp specialization is enabled
        if self.enable_warp_specialization and self.warp_spec_config is None:
            self.warp_spec_config = WarpSpecConfig()

    def is_enabled(self):
        """Check if pipelining is actually enabled"""
        return (self.global_to_shared_stages > 1 or
                self.shared_to_register_stages > 1 or
                self.enable_warp_specialization or
                self.enable_multi_buffer_fusion or
                self.enable_tma)

    def uses_tlx(self):
        """Check if TLX features are used"""
        return (self.enable_warp_specialization or
                self.enable_cluster_barriers or
                len(self.buffer_configs) > 0)

    def to_dict(self):
        """Convert to dictionary for compiler"""
        result = {
            'global_to_shared_stages': self.global_to_shared_stages,
            'shared_to_register_stages': self.shared_to_register_stages,
            'enable_async_copy': self.enable_async_copy,
            'enable_swizzle': self.enable_swizzle,
            'min_speedup': self.min_speedup,
            'enable_warp_specialization': self.enable_warp_specialization,
            'enable_multi_buffer_fusion': self.enable_multi_buffer_fusion,
            'enable_tma': self.enable_tma,
            'enable_cluster_barriers': self.enable_cluster_barriers,
        }
        if self.warp_spec_config:
            result['warp_spec'] = {
                'num_producer_warps': self.warp_spec_config.num_producer_warps,
                'num_consumer_warps': self.warp_spec_config.num_consumer_warps,
                'producer_registers': self.warp_spec_config.producer_registers,
                'consumer_registers': self.warp_spec_config.consumer_registers,
                'num_pipeline_stages': self.warp_spec_config.num_pipeline_stages,
                'enable_pingpong': self.warp_spec_config.enable_pingpong,
            }
        return result


def auto_pipeline(config: Optional[PipelineConfig] = None):
    """
    Decorator to enable automatic pipelining optimization on a Triton kernel.

    The compiler will automatically detect buffers and loops that can benefit
    from pipelining and apply the transformation. When TLX features are enabled,
    warp specialization and advanced memory operations are also applied.

    Args:
        config: Pipeline configuration. If None, uses conservative defaults.

    Returns:
        Decorated kernel function with pipelining enabled

    Example:
        # Basic pipelining
        @auto_pipeline(PipelineConfig(
            global_to_shared_stages=3,
            shared_to_register_stages=2,
            enable_async_copy=True
        ))
        @triton.jit
        def matmul_kernel(...):
            ...

        # With TLX warp specialization
        @auto_pipeline(PipelineConfig(
            global_to_shared_stages=3,
            enable_warp_specialization=True,
            warp_spec_config=WarpSpecConfig(
                num_producer_warps=1,
                num_consumer_warps=3
            )
        ))
        @triton.jit
        def warp_specialized_matmul(...):
            ...

    Note:
        - Place @auto_pipeline BEFORE @triton.jit for correct operation
        - Warp specialization requires Hopper+ GPUs for best performance
    """
    def decorator(func):
        if config is None:
            pipeline_cfg = PipelineConfig()
        else:
            pipeline_cfg = config

        # Handle both raw functions and JITFunction wrappers
        from triton.runtime import JITFunction

        if isinstance(func, JITFunction):
            func.fn._pipeline_config = pipeline_cfg
            func.fn._triton_pipeline_enabled = True
            func._pipeline_config = pipeline_cfg
            func._triton_pipeline_enabled = True
        else:
            func._pipeline_config = pipeline_cfg
            func._triton_pipeline_enabled = True

        return func

    return decorator


def warp_specialized_pipeline(
    num_producer_warps: int = 1,
    num_consumer_warps: int = 3,
    num_stages: int = 3,
    enable_pingpong: bool = False
):
    """
    Decorator for TLX warp-specialized pipelining.

    This is a convenience decorator that combines auto_pipeline with
    TLX warp specialization settings optimized for producer-consumer patterns.

    Args:
        num_producer_warps: Number of warps for data prefetching
        num_consumer_warps: Number of warps for computation
        num_stages: Number of pipeline stages
        enable_pingpong: Enable ping-pong buffering

    Returns:
        Decorated kernel with warp specialization enabled

    Example:
        @warp_specialized_pipeline(
            num_producer_warps=1,
            num_consumer_warps=3,
            num_stages=3
        )
        @triton.jit
        def producer_consumer_kernel(...):
            with tl.async_tasks():
                with tl.async_task(num_warps=1):
                    # Producer: prefetch data
                    ...
                with tl.async_task(num_warps=3):
                    # Consumer: compute
                    ...
    """
    warp_config = WarpSpecConfig(
        num_producer_warps=num_producer_warps,
        num_consumer_warps=num_consumer_warps,
        num_pipeline_stages=num_stages,
        enable_pingpong=enable_pingpong
    )
    config = PipelineConfig(
        global_to_shared_stages=num_stages,
        enable_async_copy=True,
        enable_warp_specialization=True,
        warp_spec_config=warp_config
    )
    return auto_pipeline(config)


def pipeline_buffer(tensor_ptr, num_stages: int, memory_scope: str = "shared"):
    """
    Manually mark a tensor pointer for pipelining optimization.

    This provides fine-grained control over which buffers are pipelined,
    as opposed to auto_pipeline which automatically detects candidates.

    Args:
        tensor_ptr: Pointer to buffer to pipeline
        num_stages: Number of circular buffer stages (2-5 recommended)
        memory_scope: Memory hierarchy level - "global", "shared", or "register"

    Returns:
        Annotated tensor pointer with pipeline metadata

    Example:
        @triton.jit
        def manual_pipeline_kernel(...):
            a_smem = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float16)
            a_smem = pipeline_buffer(a_smem, num_stages=3, memory_scope="shared")
            for k in range(0, K, BLOCK_K):
                ...
    """
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")
    if num_stages > 5:
        print(f"Warning: num_stages={num_stages} may cause high register pressure")

    if memory_scope not in ["global", "shared", "register"]:
        raise ValueError(f"Invalid memory_scope: {memory_scope}")

    if hasattr(tensor_ptr, '_pipeline_metadata'):
        tensor_ptr._pipeline_metadata.update({
            'num_stages': num_stages,
            'memory_scope': memory_scope,
            'manual_pipeline': True,
        })
    else:
        try:
            tensor_ptr._pipeline_metadata = {
                'num_stages': num_stages,
                'memory_scope': memory_scope,
                'manual_pipeline': True,
            }
        except AttributeError:
            pass

    return tensor_ptr


def swizzle_buffer(tensor_ptr, swizzle_pattern: int = 8):
    """
    Apply swizzling pattern to reduce shared memory bank conflicts.

    Args:
        tensor_ptr: Pointer to shared memory buffer
        swizzle_pattern: Swizzle pattern size (default: 8)

    Returns:
        Tensor pointer with swizzle metadata
    """
    if swizzle_pattern not in [4, 8, 16, 32]:
        print(f"Warning: swizzle_pattern={swizzle_pattern} is not a common value. "
              f"Recommended: 8 or 16")

    if hasattr(tensor_ptr, '_swizzle_metadata'):
        tensor_ptr._swizzle_metadata['pattern'] = swizzle_pattern
    else:
        try:
            tensor_ptr._swizzle_metadata = {'pattern': swizzle_pattern}
        except AttributeError:
            pass

    return tensor_ptr


# =====================================================
# TLX Integration: Warp Specialization Helpers
# =====================================================


def get_warp_role(warp_id: int, config: WarpSpecConfig) -> WarpRole:
    """
    Determine the role of a warp based on configuration.

    Args:
        warp_id: The warp ID (0 to total_warps-1)
        config: Warp specialization configuration

    Returns:
        WarpRole indicating if warp is producer or consumer
    """
    if warp_id < config.num_producer_warps:
        return WarpRole.PRODUCER
    else:
        return WarpRole.CONSUMER


def create_producer_consumer_barriers(num_stages: int):
    """
    Create barrier configuration for producer-consumer synchronization.

    This helper creates the barrier setup needed for pipelined producer-consumer
    kernels using TLX barriers.

    Args:
        num_stages: Number of pipeline stages

    Returns:
        Dictionary with barrier configuration info

    Example:
        barriers = create_producer_consumer_barriers(3)
        # Use with TLX barrier operations:
        # full_barriers = tlx.alloc_barriers(barriers['num_full_barriers'])
        # empty_barriers = tlx.alloc_barriers(barriers['num_empty_barriers'])
    """
    return {
        'num_full_barriers': num_stages,
        'num_empty_barriers': num_stages,
        'producer_arrive_count': 1,
        'consumer_arrive_count': 1,
    }


# =====================================================
# Convenience Functions for Common Configurations
# =====================================================


def pipeline_config_gemm(enable_warp_spec: bool = False):
    """
    Returns recommended pipeline configuration for GEMM kernels.

    Args:
        enable_warp_spec: Enable TLX warp specialization for better overlap
    """
    config = PipelineConfig(
        global_to_shared_stages=3,
        shared_to_register_stages=2,
        enable_async_copy=True,
        enable_swizzle=True,
        enable_warp_specialization=enable_warp_spec
    )
    if enable_warp_spec:
        config.warp_spec_config = WarpSpecConfig(
            num_producer_warps=1,
            num_consumer_warps=3,
            num_pipeline_stages=3
        )
    return config


def pipeline_config_gemm_hopper():
    """
    Returns optimized pipeline configuration for GEMM on Hopper GPUs.

    Uses TMA and warp specialization for maximum performance.
    """
    return PipelineConfig(
        global_to_shared_stages=3,
        shared_to_register_stages=2,
        enable_async_copy=True,
        enable_swizzle=True,
        enable_warp_specialization=True,
        enable_tma=True,
        warp_spec_config=WarpSpecConfig(
            num_producer_warps=1,
            num_consumer_warps=3,
            num_pipeline_stages=3,
            enable_pingpong=True
        )
    )


def pipeline_config_conv(enable_warp_spec: bool = False):
    """Returns recommended pipeline configuration for Convolution kernels"""
    return PipelineConfig(
        global_to_shared_stages=3,
        shared_to_register_stages=1,
        enable_async_copy=True,
        enable_swizzle=True,
        enable_warp_specialization=enable_warp_spec
    )


def pipeline_config_softmax():
    """Returns recommended pipeline configuration for Softmax kernels"""
    return PipelineConfig(
        global_to_shared_stages=2,
        shared_to_register_stages=1,
        enable_async_copy=True,
        enable_swizzle=False
    )


def pipeline_config_attention(enable_warp_spec: bool = True, enable_flash: bool = True):
    """
    Returns recommended pipeline configuration for Attention kernels.

    Args:
        enable_warp_spec: Enable warp specialization for producer-consumer overlap
        enable_flash: Enable FlashAttention-style optimizations
    """
    config = PipelineConfig(
        global_to_shared_stages=3,
        shared_to_register_stages=1,
        enable_async_copy=True,
        enable_swizzle=False,
        enable_warp_specialization=enable_warp_spec,
        enable_multi_buffer_fusion=enable_flash
    )
    if enable_warp_spec:
        config.warp_spec_config = WarpSpecConfig(
            num_producer_warps=1,
            num_consumer_warps=3,
            num_pipeline_stages=2,
            enable_pingpong=enable_flash
        )
    return config


def pipeline_config_attention_hopper():
    """
    Returns optimized pipeline configuration for FlashAttention on Hopper.

    Uses TLX warp specialization with ping-pong buffering for optimal
    memory-compute overlap.
    """
    return PipelineConfig(
        global_to_shared_stages=3,
        shared_to_register_stages=2,
        enable_async_copy=True,
        enable_swizzle=True,
        enable_warp_specialization=True,
        enable_multi_buffer_fusion=True,
        enable_tma=True,
        warp_spec_config=WarpSpecConfig(
            num_producer_warps=1,
            num_consumer_warps=3,
            num_pipeline_stages=3,
            enable_pingpong=True
        )
    )


# =====================================================
# Auto-tuning with TLX Configurations
# =====================================================


def autotune_pipeline(configs=None, key=None):
    """
    Create an auto-tuning decorator that explores different pipeline configurations.

    Args:
        configs: List of PipelineConfig objects to try. If None, uses defaults.
        key: List of argument names that determine which config to use.

    Returns:
        A decorator that creates an autotuned kernel

    Example:
        @autotune_pipeline(
            configs=[
                PipelineConfig(global_to_shared_stages=2),
                PipelineConfig(global_to_shared_stages=3, enable_warp_specialization=True),
            ],
            key=['M', 'N', 'K']
        )
        @triton.jit
        def matmul_kernel(...):
            ...
    """
    if configs is None:
        configs = [
            PipelineConfig(global_to_shared_stages=2, enable_async_copy=True),
            PipelineConfig(global_to_shared_stages=3, enable_async_copy=True),
            PipelineConfig(global_to_shared_stages=4, enable_async_copy=True),
            PipelineConfig(global_to_shared_stages=3, enable_warp_specialization=True),
        ]

    def decorator(func):
        triton_configs = []
        for i, cfg in enumerate(configs):
            triton_cfg = triton.Config(
                {},
                num_stages=cfg.global_to_shared_stages,
                num_warps=cfg.warp_spec_config.total_warps if cfg.warp_spec_config else 8,
            )
            triton_configs.append(triton_cfg)

        autotuned_func = triton.autotune(
            configs=triton_configs,
            key=key or []
        )(func)

        return autotuned_func

    return decorator


def create_pipeline_configs(
    stages_range=(2, 5),
    warps_options=(4, 8),
    enable_warp_spec_options=(False, True)
):
    """
    Create a list of pipeline configurations for auto-tuning.

    Args:
        stages_range: Tuple of (min_stages, max_stages)
        warps_options: Tuple of num_warps values to try
        enable_warp_spec_options: Tuple of warp specialization options

    Returns:
        List of triton.Config objects for autotuning
    """
    configs = []
    for stages in range(stages_range[0], stages_range[1] + 1):
        for warps in warps_options:
            for warp_spec in enable_warp_spec_options:
                if warp_spec:
                    # Use 1 producer + (warps-1) consumer warps
                    num_producer = 1
                    num_consumer = max(1, warps - 1)
                    configs.append(triton.Config(
                        {'WARP_SPEC': True},
                        num_stages=stages,
                        num_warps=warps
                    ))
                else:
                    configs.append(triton.Config(
                        {'WARP_SPEC': False},
                        num_stages=stages,
                        num_warps=warps
                    ))
    return configs


def create_tlx_autotune_configs(
    stages_range=(2, 4),
    producer_warps_options=(1,),
    consumer_warps_options=(3, 7),
    enable_pingpong_options=(False, True)
):
    """
    Create TLX-specific autotune configurations for warp-specialized kernels.

    Args:
        stages_range: Tuple of (min_stages, max_stages)
        producer_warps_options: Tuple of producer warp counts to try
        consumer_warps_options: Tuple of consumer warp counts to try
        enable_pingpong_options: Tuple of ping-pong options to try

    Returns:
        List of PipelineConfig objects optimized for TLX features
    """
    configs = []
    for stages in range(stages_range[0], stages_range[1] + 1):
        for num_producers in producer_warps_options:
            for num_consumers in consumer_warps_options:
                for pingpong in enable_pingpong_options:
                    config = PipelineConfig(
                        global_to_shared_stages=stages,
                        enable_async_copy=True,
                        enable_warp_specialization=True,
                        warp_spec_config=WarpSpecConfig(
                            num_producer_warps=num_producers,
                            num_consumer_warps=num_consumers,
                            num_pipeline_stages=stages,
                            enable_pingpong=pingpong
                        )
                    )
                    configs.append(config)
    return configs


# Export public API
__all__ = [
    # Core classes
    'PipelineConfig',
    'WarpSpecConfig',
    'WarpRole',
    'TLXBufferConfig',
    # Decorators
    'auto_pipeline',
    'warp_specialized_pipeline',
    'autotune_pipeline',
    # Buffer operations
    'pipeline_buffer',
    'swizzle_buffer',
    # TLX helpers
    'get_warp_role',
    'create_producer_consumer_barriers',
    # Convenience configs
    'pipeline_config_gemm',
    'pipeline_config_gemm_hopper',
    'pipeline_config_conv',
    'pipeline_config_softmax',
    'pipeline_config_attention',
    'pipeline_config_attention_hopper',
    # Autotune utilities
    'create_pipeline_configs',
]
