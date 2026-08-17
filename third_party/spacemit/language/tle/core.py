# spine-triton TLE (Triton Language Extension)
# Migrated from FlagTree tle/core.py for spine-triton backend.
# - cumsum: removed (not needed for spine-triton)
# - load: interface preserved, implementation pending
# - extract_tile / insert_tile: fully migrated
import builtins
import triton.language.core as tl

# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@tl.builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, is_async=False, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    # TODO: Enable async load support for spine-triton backend
    # Original FlagTree implementation:
    #   x = tl.load(pointer, mask=mask, other=other, boundary_check=boundary_check,
    #               padding_option=padding_option, cache_modifier=cache_modifier,
    #               eviction_policy=eviction_policy, volatile=volatile, _semantic=_semantic)
    #   x.handle.set_attr("tt.load.async", _semantic.builder.get_bool_attr(is_async))
    #   return x
    raise NotImplementedError("tle.load is not yet implemented for spine-triton. Use tl.load instead.")


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
           -> Linearized at compile time; uses register shuffle or SMEM path depending on alignment.
        2. Scalar static: int or tl.constexpr
           -> Treated as already-linearized tile index (compile time constant).
        3. Scalar dynamic: tl.tensor (scalar, i32/i64)
           -> Treated as a runtime linear tile index; always uses SMEM relay path.
        4. Multi-dimensional dynamic: tuple/list containing tl.tensor (e.g. [i, j], i/j are tl.tensor)
           -> Automatically linearized at runtime in the frontend; supports mixed int/tl.tensor per axis.

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
    #   Case A: tl.tensor -> dynamic index, directly pass IR Value handle
    #   Case B: tuple/list of int/constexpr -> multi-dim static, linearize then go to Case C
    #   Case C: scalar int/constexpr -> static scalar
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
            # Case B: multi-dim index -> unwrap each element, then linearize to scalar
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
    for i, (src_dim, tile_dim) in builtins.enumerate(builtins.zip(src_shape, tile_shape)):
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
                for i, v in builtins.enumerate(index_list):
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
