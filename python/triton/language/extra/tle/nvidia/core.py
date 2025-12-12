# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
import numpy as np
import triton.language.core as tl
from typing import Optional, Tuple, overload
from enum import Enum
from . import types as tle

from triton.language import semantic as tl_semantic
from triton.language.core import (
    _tensor_member_fn,
    _shape_check_impl,
    _unwrap_if_constexpr,
    builtin,
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



@tl.builtin
def copy(
    src: tl.tensor,
    result: tle.buffered_tensor,
    shape: tuple,
    _semantic=None,
) -> None:

    """
    Copy data between global memory and local memory (bidirectional).

    This function supports both:
    - Global memory → Local memory (shared memory/tensor memory)
    - Local memory → Global memory

    The direction is automatically determined based on operand types:
    - If src is tl.tensor and result is tle.buffered_tensor: GM_TO_LOCAL
    - If src is tle.buffered_tensor and result is tl.tensor: LOCAL_TO_GM

    Args:
        src: Source tensor (global memory) or buffer (local memory)
        result: Destination buffer (local memory) or tensor (global memory)
        shape: Copy shape dimensions
        _semantic: Semantic analyzer for validation (internal use)

    Raises:
        ValueError: When parameter types are incompatible
        RuntimeError: When copy operation fails
    """

    # copy data between global memory and sharemem memory
    class CopyDirection(Enum):
        """Copy direction enum for data transfer operations"""
        GM_TO_LOCAL = "GMTOLOCAL"  # Global memory to local memory
        LOCAL_TO_GM = "LOCALTOGM"  # Local memory to global memory


    # Parameter validation
    valid_types = (tl.tensor, tle.buffered_tensor)

    if not isinstance(src, valid_types):
        raise ValueError(f"Source parameter must be tl.tensor or tle.buffered_tensor, but got {type(src).__name__}")

    if not isinstance(result, valid_types):
        raise ValueError(f"Destination parameter must be tl.tensor or tle.buffered_tensor, but got {type(result).__name__}")

    # Auto-determine copy direction based on operand types
    if isinstance(src, tle.buffered_tensor) and isinstance(result, tl.tensor):
        direction = CopyDirection.LOCAL_TO_GM
    elif isinstance(src, tl.tensor) and isinstance(result, tle.buffered_tensor):
        direction = CopyDirection.GM_TO_LOCAL
    else:
        raise ValueError(
            f"Invalid copy combination: src={type(src).__name__}, result={type(result).__name__}. "
            "One operand must be tl.tensor (global memory) and the other must be tle.buffered_tensor (local memory)"
        )

    if not isinstance(shape, (tuple, list)):
        # Try to handle Triton tuple-like objects
        if hasattr(shape, '__iter__'):
            shape = tuple(shape)
        else:
            raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")

    # Semantic analysis
    try:
        from .semantic import TLESemantic
        if isinstance(_semantic, TLESemantic):
            _semantic.analyze_copy_operation(src, result, shape)
    except ImportError:
        # If semantic analysis module is not available, continue with warning
        import warnings
        warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

    mask=None
    other=None
    boundary_check=()
    padding_option=""
    cache_modifier=""
    eviction_policy=""
    volatile=False

    try:
        if (direction == CopyDirection.GM_TO_LOCAL):
            # src is global tensor
            tt_load = _semantic.load(src, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                                    volatile, None)
            _semantic.builder.create_local_store(result.handle, tt_load.handle)
        else:
            # src is local buffer - copy from shared memory to global memory
            block_type = tl.block_type(src.type.element_ty, src.type.shape)
            tt_local_load = _semantic.builder.create_local_load(block_type.to_ir(_semantic.builder), src.handle)
            load = tl.tensor(tt_local_load, block_type)
            _semantic.store(result, load, mask, boundary_check, cache_modifier, eviction_policy)
    except Exception as e:
        raise RuntimeError(f"copy operation failed: {str(e)}") from e
    return 

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
