# flagtree tle
import triton.language.core as tl
from typing import Callable, Optional, Sequence
from enum import Enum
from . import types as tle

from triton.language.core import (
    constexpr,
    tensor,
    range,
)

# Address space 3 matches the shared-memory space used in TritonGPU lowering.
SHARED_MEMORY_ADDRESS_SPACE = 3


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
def memory_space(input, space, _builder=None, _semantic=None):
    '''
    Assign a memory space to the tensor :code:`input`.

    :param input: the input tensor
    :type input: Tensor
    :param space: the memory space to assign. Can be one of "shared_memory", "tensor_memory", "register" or any other target-specific memory space.
    :type space: str
    '''
    space = tl._unwrap_if_constexpr(space)
    input.handle.set_attr("tt.memory_space", _semantic.builder.get_string_attr(space))
    return input


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
            if direction == CopyDirection.GM_TO_LOCAL:
                tt_load = _semantic.load(src, mask, other, boundary_check, padding_option, cache_modifier,
                                         eviction_policy, volatile, None)
                local_ptrs = local_ptr(dst, _semantic=_semantic)
                _semantic.store(local_ptrs, tt_load, mask, boundary_check, cache_modifier, eviction_policy)
            else:
                local_ptrs = local_ptr(src, _semantic=_semantic)
                load = tl.load(local_ptrs, _semantic=_semantic)
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


def _normalize_shape_dims(raw_shape: Sequence) -> tuple[int, ...]:
    normalized = []
    for dim in raw_shape:
        value = tl._unwrap_if_constexpr(dim)
        if not isinstance(value, int):
            raise ValueError("local_ptr index shapes must be statically sized")
        normalized.append(int(value))
    return tuple(normalized)


@tl.builtin
def local_ptr(
    buffer: tle.buffered_tensor,
    indices: Optional[Callable] = None,
    shape: Optional[Sequence[int]] = None,
    _semantic=None,
    _generator=None,
) -> tl.tensor:
    """
    Materialize shared-memory pointers that cover the given buffered tensor.

    Args:
        buffer: Local memory buffer tensor returned by ``tle.alloc``.
        indices: Optional callable that maps loop indices to a tuple of integer
            indices (length == buffer rank). The callable is invoked with one
            loop variable per dimension in ``shape`` and must return integer
            tensors/scalars that index into ``buffer``. When used inside
            ``@triton.jit`` kernels, ``indices`` should be a Triton JIT
            function (or ``functools.partial`` wrapping one) so it can be
            inlined during code generation.
        shape: Output pointer tensor shape. When ``indices`` is provided, this
            defines the loop bounds used to build the pointer view.
        _semantic: Semantic analyzer (internal use).
        _generator: Triton code generator (internal use).

    Returns:
        Tensor of pointers suitable for ``tl.load``/``tl.store``.
    """
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"Buffer parameter must be tle.buffered_tensor, but got {type(buffer)}")

    indices = tl._unwrap_if_constexpr(indices)

    buffer_shape = tuple(
        int(tl._unwrap_if_constexpr(dim)) if not isinstance(dim, int) else dim for dim in buffer.type.shape)

    view_shape: tuple[int, ...]

    if indices is None:
        if shape is not None:
            raise ValueError("local_ptr shape must be omitted when indices is None")
        view_shape = buffer_shape
        indices_fn = None
    else:
        if not callable(indices):
            raise ValueError("local_ptr indices must be a callable")
        if shape is None:
            raise ValueError("local_ptr shape must be provided when indices is callable")
        if isinstance(shape, tl.tuple):
            shape = tuple(shape.values)
        if not isinstance(shape, (tuple, list)):
            raise ValueError("local_ptr shape must be a tuple or list")
        view_shape = _normalize_shape_dims(shape)
        indices_fn = indices

    try:
        from .semantic import TLESemantic
        if isinstance(_semantic, TLESemantic):
            _semantic.analyze_local_pointer_operation(buffer, None)
    except ImportError:
        import warnings
        warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

    ptr_dtype = tl.pointer_type(buffer.type.element_ty, SHARED_MEMORY_ADDRESS_SPACE)
    block_type = tl.block_type(ptr_dtype, list(view_shape))
    insert_block = _semantic.builder.get_insertion_block()
    if insert_block is None:
        raise RuntimeError("TLE local_ptr called without an insertion block")
    block_ir = block_type.to_ir(_semantic.builder)
    local_ptr_op = _semantic.builder.create_local_pointers(block_ir, buffer.handle)

    region = local_ptr_op.get_region(0)
    builder = _semantic.builder

    try:
        from triton.runtime.jit import JITFunction
    except Exception:
        JITFunction = None

    partial_args = []
    partial_kwargs = {}
    fn = indices_fn
    try:
        import functools
        if isinstance(indices_fn, functools.partial):
            fn = indices_fn.func
            partial_args = list(indices_fn.args)
            partial_kwargs = dict(indices_fn.keywords or {})
    except Exception:
        pass

    with tl._insertion_guard(builder):
        arg_types = [tl.int32] * len(view_shape)
        to_ir = lambda T: T.to_ir(builder)
        block = builder.create_block_with_parent(region, list(map(to_ir, arg_types)))
        args = [tensor(block.arg(i), ty) for i, ty in enumerate(arg_types)]
        if indices_fn is None:
            indices_out = tuple(args)
        else:
            if JITFunction is not None and isinstance(fn, JITFunction):
                if _generator is None:
                    raise RuntimeError("local_ptr callable indices require a code generator context")
                call_args = partial_args + list(args)
                indices_out = _generator.call_JitFunction(fn, call_args, kwargs=partial_kwargs)
            else:
                indices_out = fn(*args)
        if isinstance(indices_out, tensor):
            indices_tuple = (indices_out, )
        elif isinstance(indices_out, tl.tuple):
            indices_tuple = tuple(indices_out.values)
        elif isinstance(indices_out, (tuple, list)):
            indices_tuple = tuple(indices_out)
        else:
            raise ValueError("local_ptr indices must return a tl.tensor or tuple of tl.tensor values")
        if len(indices_tuple) != len(buffer_shape):
            raise ValueError(f"local_ptr indices must return {len(buffer_shape)} values, got {len(indices_tuple)}")
        handles = []
        for idx in indices_tuple:
            idx_tensor = idx if isinstance(idx, tensor) else _semantic.to_tensor(idx)
            if not idx_tensor.dtype.is_int():
                raise ValueError("local_ptr indices must use integer dtypes")
            handles.append(idx_tensor.handle)
        builder.create_local_pointers_ret(*handles)

    result_tensor = tl.tensor(local_ptr_op.get_result(0), block_type)

    return result_tensor
