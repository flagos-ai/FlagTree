# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
import triton.language.core as tl
from typing import Optional, Sequence
from enum import Enum
from . import types as tle

from triton.language.core import (
    constexpr,
    tensor,
    range,
)


class pipeline(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)


@tl.builtin
def alloc(
    shape: tuple,
    dtype: tl.dtype,
    layout: Optional[tle.shared_layout] = None,
    scope: tle.scope = tle.smem,
    nv_mma_shared_layout=True,
    _semantic=None,
) -> tle.buffered_tensor:
    """
    Allocate local memory buffer

    Args:
        shape: Buffer shape
        dtype: Data type
        layout: Memory layout encoding (optional)
        scope: Storage type (default to shared memory)
        _semantic: Semantic analyzer (internal use)

    Returns:
        Allocated buffer tensor

    Raises:
        ValueError: When parameters are invalid
        RuntimeError: When allocation fails
    """
    # Parameter validation
    if not isinstance(shape, (tuple, list)):
        # Try to handle Triton tuple-like objects
        if hasattr(shape, '__iter__'):
            shape = tuple(shape)
        else:
            raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")

    if not isinstance(dtype, tl.dtype):
        raise ValueError(f"Data type must be tl.dtype, but got {type(dtype)}")

    if not isinstance(scope, tle.scope):
        raise ValueError(f"Storage type must be tle.scope, but got {type(scope)}")

    if layout is not None and not isinstance(layout, tle.shared_layout):
        # Handle constexpr None
        if hasattr(layout, 'value') and layout.value is None:
            layout = None
        else:
            raise ValueError(f"Layout must be tle.shared_layout or None, but got {type(layout)}")

    # Semantic analysis
    try:
        from .semantic import TLESemantic
        if isinstance(_semantic, TLESemantic):
            shape, dtype = _semantic.analyze_alloc_operation(shape, dtype, layout, scope)
    except ImportError:
        # If semantic analysis module is not available, continue with warning
        import warnings
        warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

    # Map scope to storage (backward compatibility)
    storage = scope

    try:
        unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
        full_shape = unwrapped_shape
        dtype = tl._unwrap_if_constexpr(dtype)
        elem_type = dtype.to_ir(_semantic.builder)

        # Parse layout (if constexpr)
        layout = tl._unwrap_if_constexpr(layout)

        if layout is None:
            if storage == tle.smem:
                if not nv_mma_shared_layout:
                    layout = tle.swizzled_shared_layout.make_default(rank=len(shape))
                    layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
                        layout.vectorSize,
                        layout.perPhase,
                        layout.maxPhase,
                        layout.order,
                        layout.numCTAsPerCGA,
                        layout.numCTASplit,
                        layout.numCTAOrder,
                    )
                else:
                    layout = tle.nv_mma_shared_layout.make_default(shape, dtype)
                    layout_handle = _semantic.builder.make_nv_mma_shared_encoding_attr(
                        [int(x) for x in layout.shape],
                        layout.order,
                        layout.elemType.to_ir(_semantic.builder),
                        layout.numCTAsPerCGA,
                        layout.numCTASplit,
                        layout.numCTAOrder,
                        layout.fp4Padded,
                        layout.swizzled,
                    )
            else:
                layout = tle.tensor_memory_layout.make_default(shape)
                layout_handle = _semantic.builder.make_tensor_memory_encoding_attr(
                    layout.blockM,
                    layout.blockN,
                    layout.unpacked,
                    layout.CTASplitM,
                    layout.CTASplitN,
                )
        else:
            # Use provided layout
            layout_handle = layout.to_ir(_semantic.builder)

        if storage == tle.smem:
            tensor_handle = _semantic.builder.create_local_alloc(full_shape, elem_type, layout_handle)
        else:
            raise ValueError(f"Storage type {storage} not yet supported")

        return tle.buffered_tensor(tensor_handle, dtype, unwrapped_shape, storage, layout, _semantic)

    except Exception as e:
        raise RuntimeError(f"Memory allocation failed: {str(e)}") from e


class CopyDirection(Enum):
    """Copy direction enum for data transfer operations"""
    GM_TO_LOCAL = "GMTOLOCAL"  # Global memory to local memory
    LOCAL_TO_GM = "LOCALTOGM"  # Local memory to global memory


@tl.builtin
def copy(
    src,
    dst,
    shape,
    offsets: Sequence[constexpr | tensor] = None,
    _semantic=None,
) -> None:
    """
    High-performance data copy operation supporting TMA (Tensor Memory Accelerator) transfers.

    This function provides efficient data movement between different memory spaces, with support for:
    - TMA descriptor-based transfers (for NVIDIA Hopper architecture and later)
    - Standard global ↔ local memory transfers
    - Bidirectional data movement with automatic direction detection

    Transfer Direction Modes:
    - GM_TO_LOCAL: Global memory → Local memory (shared memory)
    - LOCAL_TO_GM: Local memory → Global memory

    Direction is automatically determined based on operand types:
    - If src is tl.tensor/tensor_descriptor and dst is tle.buffered_tensor: GM_TO_LOCAL
    - If src is tle.buffered_tensor and dst is tl.tensor/tensor_descriptor: LOCAL_TO_GM

    Args:
        src: Source data - can be:
            - tl.tensor (global memory tensor)
            - tl.tensor_descriptor (TMA descriptor for global memory)
            - tle.buffered_tensor (local memory buffer)
        dst: Destination - can be:
            - tle.buffered_tensor (local memory buffer)
            - tl.tensor (global memory tensor)
            - tl.tensor_descriptor (TMA descriptor for global memory)
        shape: Tuple specifying the dimensions of the data to copy
        offsets: Sequence of offsets for multi-dimensional addressing. Used with TMA operations
            to specify the starting coordinates within the tensor. Required for TMA copy.
        _semantic: Internal semantic analyzer for validation and compilation (user-provided)

    Raises:
        ValueError: When parameter types are incompatible or offsets missing for TMA operations
        RuntimeError: When copy operation fails during execution or compilation

    Examples:
        Standard global → local copy:
            local_buf = tle.alloc([256, 256], dtype=tl.float32, scope=tle.smem)
            tle.copy(global_tensor, local_buf, [256, 256])

        TMA copy with offsets:
            tle.copy(tma_desc, local_buf, [64, 64], [x_offset, y_offset])
    """

    def normcopy(
        src: tl.tensor,
        dst: tle.buffered_tensor,
        shape: tuple,
        direction,
        _semantic=None,
    ) -> None:

        # Semantic analysis
        try:
            from .semantic import TLESemantic
            if isinstance(_semantic, TLESemantic):
                _semantic.analyze_copy_operation(src, dst, shape)
        except ImportError:
            # If semantic analysis module is not available, continue with warning
            import warnings
            warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

        mask = None
        other = None
        boundary_check = ()
        padding_option = ""
        cache_modifier = ""
        eviction_policy = ""
        volatile = False

        try:
            if (direction == CopyDirection.GM_TO_LOCAL):
                # src is global tensor
                tt_load = _semantic.load(src, mask, other, boundary_check, padding_option, cache_modifier,
                                         eviction_policy, volatile, None)
                _semantic.builder.create_local_store(dst.handle, tt_load.handle)
            else:
                # src is local buffer - copy from shared memory to global memory
                block_type = tl.block_type(src.type.element_ty, src.type.shape)
                tt_local_load = _semantic.builder.create_local_load(block_type.to_ir(_semantic.builder), src.handle)
                load = tl.tensor(tt_local_load, block_type)
                _semantic.store(dst, load, mask, boundary_check, cache_modifier, eviction_policy)
        except Exception as e:
            raise RuntimeError(f"copy operation failed: {str(e)}") from e

    # this api is use for tma copy
    def tmacopy(
        src: tle.buffered_tensor | tl.tensor_descriptor,
        dst: tle.buffered_tensor | tl.tensor_descriptor,
        direction,
        shape: tuple,
        offsets: Sequence[constexpr | tensor],
        _semantic=None,
    ) -> None:
        # Parameter validation
        valid_types = (tle.buffered_tensor, tl.tensor_descriptor)

        if not isinstance(src, valid_types):
            raise ValueError(
                f"Source parameter must be tle.buffered_tensor or tl.tensor_descriptor, but got {type(src).__name__}")

        if not isinstance(dst, valid_types):
            raise ValueError(
                f"Destination parameter must be tle.buffered_tensor or tl.tensor_descriptor, but got {type(dst).__name__}"
            )

        # Auto-determine copy direction based on operand types
        if isinstance(src, tle.buffered_tensor) and isinstance(dst, tl.tensor_descriptor):
            desc = dst
        elif isinstance(src, tl.tensor_descriptor) and isinstance(dst, tle.buffered_tensor):
            desc = src
        else:
            raise ValueError(
                f"Invalid copy combination: src={type(src).__name__}, dst={type(dst).__name__}. "
                "One operand must be tl.tensor_descriptor (global memory) and the other must be tle.buffered_tensor (local memory)"
            )

        if not isinstance(shape, (tuple, list)):
            # Try to handle Triton tuple-like objects
            if hasattr(shape, '__iter__'):
                shape = tuple(shape)
            else:
                raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")

        if not isinstance(offsets, (tuple, list)):
            # Try to handle Triton tuple-like objects
            if hasattr(offsets, '__iter__'):
                offsets = tuple(offsets)
            else:
                raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")

        # Note: Skip shape assertion at this level since it requires _semantic context
        # assert desc.shape == shape, "Shape mismatch between descriptor and provided shape"
        assert len(offsets) == len(desc.shape), "Offsets and shape must have the same length"
        offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
        _semantic.builder.create_tma_copy(src.handle, dst.handle, offsets)
        return

    # Parameter validation
    valid_types = (tl.tensor, tle.buffered_tensor, tl.tensor_descriptor)

    if not isinstance(src, valid_types):
        raise ValueError(
            f"Source parameter must be tl.tensor or tle.buffered_tensor  tl.tensor_descriptor , but got {type(src).__name__}"
        )

    if not isinstance(dst, valid_types):
        raise ValueError(
            f"Destination parameter must be tl.tensor or tle.buffered_tensor  tl.tensor_descriptor, but got {type(dst).__name__}"
        )

    # Auto-determine copy direction based on operand types
    if isinstance(src, tle.buffered_tensor) and isinstance(dst, tl.tensor):
        direction = CopyDirection.LOCAL_TO_GM
        is_normcopy = True
    elif isinstance(src, tl.tensor) and isinstance(dst, tle.buffered_tensor):
        direction = CopyDirection.GM_TO_LOCAL
        is_normcopy = True
    elif isinstance(src, tle.buffered_tensor) and isinstance(dst, tl.tensor_descriptor):
        direction = CopyDirection.LOCAL_TO_GM
        is_normcopy = False
    elif isinstance(src, tl.tensor_descriptor) and isinstance(dst, tle.buffered_tensor):
        direction = CopyDirection.GM_TO_LOCAL
        is_normcopy = False
    else:
        raise ValueError(
            f"Invalid copy combination: src={type(src).__name__}, dst={type(dst).__name__}. "
            "One operand must be tl.tensor (global memory) and the other must be tle.buffered_tensor (local memory)")

    if not isinstance(shape, (tuple, list)):
        # Try to handle Triton tuple-like objects
        if hasattr(shape, '__iter__'):
            shape = tuple(shape)
        else:
            raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")
    if is_normcopy:
        return normcopy(src, dst, shape, direction, _semantic)
    else:
        return tmacopy(src, dst, direction, shape, offsets, _semantic)


@tl.builtin
def local_load(
    buffer: tle.buffered_tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Load data from local memory buffer

    Args:
        buffer: Local memory buffer tensor
        _semantic: Semantic analyzer (internal use)

    Returns:
        Loaded data tensor

    Raises:
        ValueError: When buffer is not initialized or type mismatch
        RuntimeError: When load operation fails
    """
    # Parameter validation
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"Buffer parameter must be tle.buffered_tensor, but got {type(buffer)}")

    # Semantic analysis
    try:
        from .semantic import TLESemantic
        if isinstance(_semantic, TLESemantic):
            _semantic.analyze_local_load_operation(buffer)
    except ImportError:
        # If semantic analysis module is not available, continue with warning
        import warnings
        warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

    try:
        block_type = tl.block_type(buffer.type.element_ty, buffer.type.shape)
        output = _semantic.builder.create_local_load(block_type.to_ir(_semantic.builder), buffer.handle)
        return tl.tensor(output, block_type)
    except Exception as e:
        raise RuntimeError(f"Local load operation failed: {str(e)}") from e


@tl.builtin
def local_store(
    dst: tle.buffered_tensor,
    src: tl.tensor,
    _semantic=None,
) -> None:
    """
    Store a tensor into a local memory buffer.

    Args:
        dst: Destination buffer in local memory (shared memory or tensor memory)
        src: Source tensor to store
        _semantic: Semantic analyzer for validation (internal use)

    Raises:
        RuntimeError: When tensor memory storage is not yet supported
        ValueError: When parameter types are incompatible
    """
    storage = dst.type.storage
    if storage == tle.tmem:
        raise RuntimeError("Tensor memory local_store not yet supported")

    # Perform the store operation
    _semantic.builder.create_local_store(dst.handle, src.handle)
