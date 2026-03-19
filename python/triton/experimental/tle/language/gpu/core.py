# flagtree tle
import builtins
import triton.language.core as tl
from typing import Optional, Sequence
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
                    layout.colStride,
                    layout.CTASplitM,
                    layout.CTASplitN,
                    layout.twoCTAs,
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
                local_ptrs = local_ptr(dst, _make_full_indices(dst, _semantic), _semantic=_semantic)
                _semantic.store(local_ptrs, tt_load, mask, boundary_check, cache_modifier, eviction_policy)
            else:
                local_ptrs = local_ptr(src, _make_full_indices(src, _semantic), _semantic=_semantic)
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


# ============================================================================
# extract_tile helper functions (module-level, do not depend on @tl.builtin context)
# ============================================================================


def _try_unwrap_int(val):
    """
    Try to unwrap val as a Python int.
    Supports: int, tl.constexpr(int), objects with .value attribute.
    For runtime tl.tensor, returns None.
    """
    if isinstance(val, int):
        return val
    try:
        v = tl._unwrap_if_constexpr(val)
        if isinstance(v, int):
            return v
    except Exception:
        pass
    try:
        if hasattr(val, 'value') and isinstance(val.value, int):
            return val.value
    except Exception:
        pass
    return None


def _unwrap_tile_shape(tile_shape):
    """Unwrap tile_shape (any form) as List[int], all elements must be compile-time constants."""
    if hasattr(tile_shape, '__iter__') and not isinstance(tile_shape, str):
        result = []
        for s in tile_shape:
            val = s
            while hasattr(val, 'value'):
                val = val.value
            try:
                val = tl._unwrap_if_constexpr(val)
            except Exception:
                pass
            if not isinstance(val, int):
                raise ValueError(
                    f"All tile_shape dims must be int or tl.constexpr, got {type(val)} (original: {type(s)})")
            result.append(val)
        return result
    else:
        val = tile_shape
        while hasattr(val, 'value'):
            val = val.value
        try:
            val = tl._unwrap_if_constexpr(val)
        except Exception:
            pass
        if not isinstance(val, int):
            raise ValueError(f"tile_shape must be int/constexpr, got {type(val)}")
        return [val]


def _linearize_static_multidim_index(index_list, src_shape, tile_shape_ints):
    """
    Linearize multi-dimensional static index (row-major order).
    index_list:      List[int]  tile coordinate in each dimension
    src_shape:       List[int]  source tensor shape in each dimension
    tile_shape_ints: List[int]  tile size in each dimension
    Returns a linearized scalar int
    """
    rank = len(src_shape)
    if len(index_list) != rank:
        raise ValueError(f"Index rank {len(index_list)} must match source rank {rank}")

    grid = []
    for i in builtins.range(rank):
        if src_shape[i] % tile_shape_ints[i] != 0:
            raise ValueError(f"Source dim {i} ({src_shape[i]}) not divisible by tile dim ({tile_shape_ints[i]})")
        grid.append(src_shape[i] // tile_shape_ints[i])

    for i, v in builtins.enumerate(index_list):
        if v < 0 or v >= grid[i]:
            raise ValueError(f"Index[{i}]={v} out of bounds for grid size {grid[i]}")

    # row major linearization
    linear = 0
    stride = 1
    for i in builtins.reversed(builtins.range(rank)):
        linear += index_list[i] * stride
        stride *= grid[i]
    return linear


# ============================================================
# dynamic multidim index linearization
# ============================================================


def _linearize_dynamic_multidim_index(index_tuple, src_shape, tile_shape_ints, _semantic):
    """
    Convert dynamic multi-dimensional tile index to linear index IR.
    Example:
        src_shape = [16,16]
        tile_shape = [4,4]
        tile grid = [4,4]
        index = [i,j]
        linear = i*4 + j
    """

    if any(not isinstance(s, int) for s in src_shape):
        raise ValueError("Source shape must be static for dynamic multi-dim index")
    # compute tile grid
    grid = []
    for s, t in builtins.zip(src_shape, tile_shape_ints):
        if s % t != 0:
            raise ValueError(f"Source dim {s} not divisible by tile dim {t}")
        grid.append(s // t)
    # compute strides
    strides = [1] * len(grid)
    acc = 1
    for i in builtins.reversed(builtins.range(len(grid))):
        strides[i] = acc
        acc *= grid[i]

    # Validation: index_tuple rank must match grid rank
    if len(index_tuple) != len(grid):
        raise ValueError(f"Dynamic multi-dim index rank {len(index_tuple)} does not match grid rank {len(grid)}")

    linear_ir = None
    for i, v in builtins.enumerate(index_tuple):
        stride = strides[i]
        if isinstance(v, tl.tensor):
            term = v.handle
        else:
            iv = _try_unwrap_int(v)
            if iv is None:
                raise ValueError("Dynamic multidim index must contain tl.tensor or int")
            term = _semantic._convert_to_ir_values([iv], require_i64=False)[0]
        if stride != 1:
            stride_ir = _semantic._convert_to_ir_values([stride], require_i64=False)[0]
            term = _semantic.builder.create_mul(term, stride_ir)
        if linear_ir is None:
            linear_ir = term
        else:
            linear_ir = _semantic.builder.create_add(linear_ir, term)
    return linear_ir


@tl.builtin
def extract_tile(
    x: tl.tensor,
    index,
    tile_shape: tuple,
    _semantic=None,
) -> tl.tensor:
    """
    Extract a tile from a tensor at a given tile index.

    Supported index forms:
        1. Multi-dimensional static: tuple/list of int/constexpr (e.g. [1, 2])
           → Linearized at compile time; uses register shuffle or SMEM path depending on alignment.
        2. Scalar static: int or tl.constexpr
           → Treated as already-linearized tile index (compile time constant).
        3. Scalar dynamic: tl.tensor (scalar, i32/i64)
           → Treated as a runtime linear tile index; always uses SMEM relay path.
        4. Multi-dimensional dynamic: tuple/list containing tl.tensor (e.g. [i, j], i/j are tl.tensor)
           → Automatically linearized at runtime in the frontend; supports mixed int/tl.tensor per axis.

    For multi-dimensional dynamic index, the function will automatically compute the row-major linearized tile index as a dynamic IR expression, so users can pass [i, j, ...] directly.

    Args:
        x:          Source tensor (tl.tensor)
        index:      Tile index (see above)
        tile_shape: Tile shape in each dimension (must be compile-time constants)
        _semantic:  Internal semantic analyzer (for lowering)

    Returns:
        Extracted tile tensor with shape = tile_shape
    Raises:
        ValueError: If index or shape is invalid
        RuntimeError: If IR generation fails
    """
    # --- Parameter check ---
    if not isinstance(x, tl.tensor):
        raise ValueError(f"Source must be tl.tensor, but got {type(x)}")

    # --- Unwrap tile_shape (must all be compile-time constants) ---
    tile_shape_ints = _unwrap_tile_shape(tile_shape)

    src_shape = [tl._unwrap_if_constexpr(dim) for dim in x.type.shape]

    # --- Parse index, three cases ---
    #
    #   Case A: tl.tensor → dynamic index, directly pass IR Value handle
    #   Case B: tuple/list of int/constexpr → multi-dim static, linearize then go to Case C
    #   Case C: scalar int/constexpr → static scalar
    #
    is_dynamic = False
    index_value = None  # For static path: Python int
    index_ir_handle = None  # For dynamic path: MLIR Value handle

    if isinstance(index, tl.tensor):
        # Case A: dynamic index, value known only at runtime
        is_dynamic = True
        index_ir_handle = index.handle
    else:
        # Try to unwrap, determine if multi-dim or scalar
        index_unwrapped = index
        try:
            index_unwrapped = tl._unwrap_if_constexpr(index)
        except Exception:
            pass
        try:
            if hasattr(index_unwrapped, 'value'):
                index_unwrapped = index_unwrapped.value
        except Exception:
            pass
        if isinstance(index_unwrapped, (tuple, list, tl.tuple)):
            # Case B: multi-dim index → unwrap each element, then linearize to scalar
            has_tensor = any(isinstance(v, tl.tensor) for v in index_unwrapped)
            if has_tensor:
                # ====================================
                # dynamic multidim index
                # ====================================
                index_ir_handle = _linearize_dynamic_multidim_index(index_unwrapped, src_shape, tile_shape_ints,
                                                                    _semantic)
                is_dynamic = True
            else:
                # ====================================
                # static multidim index
                # ====================================
                idx_ints = []
                for v in index_unwrapped:
                    iv = _try_unwrap_int(v)
                    if iv is None:
                        raise ValueError("Multi-dim index must contain int/constexpr values.")
                    idx_ints.append(iv)

                if any(not isinstance(s, int) for s in src_shape):
                    raise ValueError("Source shape must be static when using a multi-dim index")
                index_value = _linearize_static_multidim_index(idx_ints, src_shape, tile_shape_ints)
        else:
            # Case C: scalar static index
            scalar_int = _try_unwrap_int(index_unwrapped)
            if scalar_int is not None:
                index_value = scalar_int
            else:
                raise ValueError(f"index must be int, constexpr, tuple/list of int/constexpr, "
                                 f"or a scalar tl.tensor; got {type(index)}")
    # --- Basic dimension check ---
    if len(tile_shape_ints) != len(src_shape):
        raise ValueError(f"tile_shape rank ({len(tile_shape_ints)}) must match "
                         f"source rank ({len(src_shape)})")

    # --- Compile-time check for static index ---
    if not is_dynamic:
        for i, (s, t) in builtins.enumerate(builtins.zip(src_shape, tile_shape_ints)):
            if isinstance(s, int) and s % t != 0:
                raise ValueError(f"Source dim {i} ({s}) not divisible by tile dim ({t})")
        if all(isinstance(s, int) for s in src_shape):
            total_tiles = 1
            for s, t in builtins.zip(src_shape, tile_shape_ints):
                total_tiles *= s // t
            if index_value < 0 or index_value >= total_tiles:
                raise ValueError(f"index {index_value} out of range [0, {total_tiles})")

        # Semantic validation (static path)
        try:
            from .semantic import TLESemantic
            if isinstance(_semantic, TLESemantic):
                _semantic.analyze_extract_tile_operation(x, index_value, tile_shape_ints)
        except ImportError:
            pass

    # --- Generate MLIR IR ---
    try:
        if is_dynamic:
            # Dynamic index: directly use the IR handle from the input tl.tensor
            index_ir = index_ir_handle
        else:
            # Static index: encode compile-time constant as IR constant
            index_ir = _semantic._convert_to_ir_values([index_value], require_i64=False)[0]

        output = _semantic.builder.create_extract_tile(x.handle, index_ir, tile_shape_ints)
        block_type = tl.block_type(x.type.element_ty, tile_shape_ints)
        return tl.tensor(output, block_type)
    except Exception as e:
        raise RuntimeError(f"Failed to create extract_tile operation: {str(e)}") from e


@tl.builtin
def insert_tile(
    x: tl.tensor,
    tile: tl.tensor,
    index,
    _semantic=None,
) -> tl.tensor:
    """
    Insert a tile into source tensor.

    index supports:
      1. Multi-dim static index: list/tuple of int/constexpr (e.g. [i, j])
      2. Scalar static index: int / tl.constexpr
      3. Scalar dynamic index: tl.tensor (runtime value)
    """
    # Basic type checks for source and tile tensors.
    if not isinstance(x, tl.tensor):
        raise ValueError(f"Source must be tl.tensor, but got {type(x)}")
    if not isinstance(tile, tl.tensor):
        raise ValueError(f"Tile must be tl.tensor, but got {type(tile)}")

    # Shapes must be compile-time integers so tile-grid math stays static.
    src_shape = [tl._unwrap_if_constexpr(dim) for dim in x.type.shape]
    tile_shape = [tl._unwrap_if_constexpr(dim) for dim in tile.type.shape]
    if any(not isinstance(dim, int) for dim in src_shape):
        raise ValueError("Source shape must be static for insert_tile")
    if any(not isinstance(dim, int) for dim in tile_shape):
        raise ValueError("Tile shape must be static for insert_tile")
    if len(src_shape) != len(tile_shape):
        raise ValueError(f"Source rank ({len(src_shape)}) must match tile rank ({len(tile_shape)})")
    if x.type.element_ty != tile.type.element_ty:
        raise ValueError(f"Element type mismatch: source={x.type.element_ty}, tile={tile.type.element_ty}")

    # Build per-dimension tile grid: how many tiles exist in each axis.
    grid = []
    for i, (src_dim, tile_dim) in enumerate(zip(src_shape, tile_shape)):
        if tile_dim <= 0:
            raise ValueError(f"Tile dimension {i} must be positive, got {tile_dim}")
        if src_dim % tile_dim != 0:
            raise ValueError(f"Source dimension {i}: {src_dim} must be divisible by tile dimension {tile_dim}")
        grid.append(src_dim // tile_dim)

    # Parse index: dynamic scalar tensor or static scalar/multi-dim.
    is_dynamic = False
    index_value = None
    index_ir_handle = None

    if isinstance(index, tl.tensor):
        is_dynamic = True
        index_ir_handle = index.handle
    else:
        index_unwrapped = index
        try:
            index_unwrapped = tl._unwrap_if_constexpr(index_unwrapped)
        except Exception:
            pass
        try:
            if hasattr(index_unwrapped, "value"):
                index_unwrapped = index_unwrapped.value
        except Exception:
            pass

        index_list = None
        if isinstance(index_unwrapped, (tuple, list, tl.tuple)):
            index_list = list(index_unwrapped)
            if len(index_list) != len(src_shape):
                raise ValueError(f"Index rank {len(index_list)} must match source rank {len(src_shape)}")
            has_tensor = any(isinstance(v, tl.tensor) for v in index_list)
            # ------------------------------------------------
            # dynamic multi-dim index
            # ------------------------------------------------
            if has_tensor:
                index_ir_handle = _linearize_dynamic_multidim_index(index_list, src_shape, tile_shape, _semantic)
                is_dynamic = True
            # ------------------------------------------------
            # static multi-dim index
            # ------------------------------------------------
            else:
                idx = []
                for i, v in enumerate(index_list):
                    iv = _try_unwrap_int(v)
                    if iv is None:
                        raise ValueError("Multi-dim index must contain int/constexpr values")
                    if iv < 0 or iv >= grid[i]:
                        raise ValueError(f"Index[{i}]={iv} out of bounds for tile grid (0~{grid[i]-1})")
                    idx.append(iv)
                index_value = _linearize_static_multidim_index(idx, src_shape, tile_shape)
        else:
            # Path B: scalar static index -> treat as already-linearized tile id.
            scalar_int = _try_unwrap_int(index_unwrapped)
            if scalar_int is None:
                raise ValueError(f"index must be int, constexpr, tuple/list of int/constexpr, "
                                 f"or a scalar tl.tensor; got {type(index)}")
            index_value = scalar_int

    # Static index checks + optional semantic pass.
    if not is_dynamic:
        if index_value < 0:
            raise ValueError("Scalar index must be non-negative")

        total_tiles = 1
        for g in grid:
            total_tiles *= g
        if index_value >= total_tiles:
            raise ValueError(f"Scalar index {index_value} out of bounds for total tiles {total_tiles}")

        try:
            from .semantic import TLESemantic
            if isinstance(_semantic, TLESemantic):
                _semantic.analyze_insert_tile_operation(x, tile, index_value)
        except ImportError:
            pass

    # Lower to IR and construct output tensor with the source tensor type.
    try:
        if is_dynamic:
            index_ir = index_ir_handle
        else:
            index_ir = _semantic._convert_to_ir_values([index_value], require_i64=False)[0]
        output = _semantic.builder.create_insert_tile(
            x.handle,
            tile.handle,
            index_ir,
        )
        return tl.tensor(output, x.type)
    except Exception as e:
        raise RuntimeError(f"Failed to create insert_tile operation: {str(e)}") from e


def _expand_index_to_shape(index: tl.tensor, shape: Sequence[int], axis: int, _semantic) -> tl.tensor:
    idx = index
    for _ in builtins.range(axis):
        idx = tl.expand_dims(idx, 0, _semantic=_semantic)
    for _ in builtins.range(len(shape) - axis - 1):
        idx = tl.expand_dims(idx, len(idx.shape), _semantic=_semantic)
    return tl.broadcast_to(idx, *shape, _semantic=_semantic)


def _make_full_indices(buffer: tle.buffered_tensor, _semantic) -> tuple[tl.tensor, ...]:
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    indices = []
    for axis, dim in enumerate(shape):
        idx = tl.arange(0, dim, _semantic=_semantic)
        idx = _expand_index_to_shape(idx, shape, axis, _semantic)
        indices.append(idx)
    return tuple(indices)


@tl.builtin
def local_ptr(
    buffer: tle.buffered_tensor,
    indices: Optional[Sequence] = None,
    _semantic=None,
    _generator=None,
) -> tl.tensor:
    """
    Materialize shared-memory pointers that cover the given buffered tensor.

    Args:
        buffer: Local memory buffer tensor returned by ``tle.alloc``.
        indices: Tuple of integer index tensors. The tuple length must equal
            the rank of ``buffer`` and every tensor must have the same shape.
            The output pointer tensor will have that same shape.
        _semantic: Semantic analyzer (internal use).
        _generator: Triton code generator (internal use).

    Returns:
        Tensor of pointers suitable for ``tl.load``/``tl.store``.
    """
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"Buffer parameter must be tle.buffered_tensor, but got {type(buffer)}")

    # Preferred metadata source: buffered_tensor.type (survives JIT value
    # reconstruction). Keep value attrs as backward-compatibility fallback.
    remote_shard_id = getattr(buffer.type, "_tle_remote_shard_id", None)
    remote_scope = getattr(buffer.type, "_tle_remote_scope", None)
    if remote_shard_id is None:
        remote_shard_id = getattr(buffer, "_tle_remote_shard_id", None)
        remote_scope = getattr(buffer, "_tle_remote_scope", None)
    remote_buffer_marker = remote_shard_id is not None

    indices = tl._unwrap_if_constexpr(indices)
    if indices is None:
        raise ValueError("local_ptr indices must be provided as a tuple of tensors")
    if isinstance(indices, tl.tuple):
        indices_tuple = tuple(indices.values)
    elif isinstance(indices, (tuple, list)):
        indices_tuple = tuple(indices)
    else:
        raise ValueError("local_ptr indices must be a tuple or list of tensors")

    buffer_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    if len(indices_tuple) != len(buffer_shape):
        raise ValueError(f"local_ptr indices must provide {len(buffer_shape)} tensors, got {len(indices_tuple)}")

    idx_tensors: list[tensor] = []
    view_shape: Optional[tuple[int, ...]] = None
    scalar_index_flags: list[bool] = []
    for idx in indices_tuple:
        idx_tensor = idx if isinstance(idx, tensor) else _semantic.to_tensor(idx)
        if not idx_tensor.dtype.is_int():
            raise ValueError("local_ptr indices must use integer dtypes")
        is_scalar_index = not idx_tensor.type.is_block()
        scalar_index_flags.append(is_scalar_index)
        if is_scalar_index:
            idx_tensors.append(idx_tensor)
            continue
        if view_shape is None:
            view_shape = tuple(idx_tensor.shape)
        elif tuple(idx_tensor.shape) != view_shape:
            raise ValueError("local_ptr indices must have identical shapes")
        idx_tensors.append(idx_tensor)

    if not idx_tensors:
        raise ValueError("local_ptr indices cannot be empty")
    all_scalar_indices = all(scalar_index_flags)
    any_scalar_indices = any(scalar_index_flags)
    if any_scalar_indices and not all_scalar_indices:
        raise ValueError("local_ptr indices must be either all scalar or all tensors with identical shapes")
    if not all_scalar_indices and view_shape is None:
        view_shape = tuple()

    try:
        from .semantic import TLESemantic
        if isinstance(_semantic, TLESemantic):
            _semantic.analyze_local_pointer_operation(buffer, idx_tensors)
    except ImportError:
        import warnings
        warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

    ptr_dtype = tl.pointer_type(buffer.type.element_ty, SHARED_MEMORY_ADDRESS_SPACE)
    insert_block = _semantic.builder.get_insertion_block()
    if insert_block is None:
        raise RuntimeError("TLE local_ptr called without an insertion block")
    if all_scalar_indices:
        result_ty = ptr_dtype
        result_ir = ptr_dtype.to_ir(_semantic.builder)
    else:
        result_ty = tl.block_type(ptr_dtype, list(view_shape))
        result_ir = result_ty.to_ir(_semantic.builder)
    handles = [idx.handle for idx in idx_tensors]
    local_ptr_op = _semantic.builder.create_local_pointers(result_ir, buffer.handle, *handles)

    result_tensor = tl.tensor(local_ptr_op.get_result(0), result_ty)

    if remote_buffer_marker:
        if all_scalar_indices:
            raise ValueError("local_ptr does not yet support scalar indices on remote buffers")
        # Keep remote semantics attached to the source buffered tensor and
        # materialize them only when pointer view is requested.
        from triton.experimental.tle.language import distributed as _tle_distributed
        result_tensor = _tle_distributed._remote_pointer(
            result_tensor,
            remote_shard_id,
            scope=remote_scope,
            _semantic=_semantic,
        )

    return result_tensor
