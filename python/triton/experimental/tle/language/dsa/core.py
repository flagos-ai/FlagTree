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
    PropagateNan,
)
from triton.language import math as tlmath

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
def memory_space(input, space, _semantic=None):
    """
    Annotate a tensor with a target memory-space tag.

    The attribute ``tt.memory_space`` is propagated through the IR and can be
    consumed by downstream DSA passes (e.g. ``--dsa-memory-to-core``) to make
    allocation / placement decisions.

    Args:
        input: Tensor to annotate.
        space: Memory-space name string, e.g. ``"spm"`` or ``"shared_memory"``.
    """
    space = tl._unwrap_if_constexpr(space)
    builder = _semantic.builder
    if builder is not None and hasattr(input, 'handle') and hasattr(input.handle, 'set_attr'):
        input.handle.set_attr("tt.memory_space", builder.get_string_attr(str(space)))
    return input


@tl.builtin
def alloc(
    shape: tuple,
    dtype: tl.dtype,
    layout: Optional[object] = None,
    scope: tle.scope = None,
    _semantic=None,
) -> tle.buffered_tensor:
    """
    Allocate local memory buffer

    Args:
        shape: Buffer shape
        dtype: Data type
        layout: Memory layout encoding (optional)
        scope: Storage type (default to shared memory)

    Returns:
        Allocated buffer tensor

    Raises:
        ValueError: When parameters are invalid
        RuntimeError: When allocation fails
    """
    from .semantic import DSASemantic

    builder = _semantic.builder
    if builder is None:
        raise ValueError("alloc must be used inside @triton.jit")
    if layout is not None:
        raise ValueError("alloc(): layout parameter is not yet support for DSA backend")

    # --- Validate inputs via semantic layer ---
    unwrapped_shape = DSASemantic.validate_alloc_shape(shape)
    elem_dtype = DSASemantic.validate_alloc_dtype(dtype)
    resolved_scope = DSASemantic.validate_alloc_scope(scope)

    elem_ir_ty = elem_dtype.to_ir(builder)

    if not hasattr(builder, "create_dsa_alloc"):
        raise RuntimeError("builder missing create_dsa_alloc for DSA alloc")

    alloc_value = builder.create_dsa_alloc(list(unwrapped_shape), elem_ir_ty)
    buf_ty = tle.buffered_tensor_type(unwrapped_shape, elem_dtype, resolved_scope)
    return tle.buffered_tensor(alloc_value, buf_ty)


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
    Copy data between global memory (GM) and local scratchpad memory (SPM).

    Supported combinations:

    1. **tl.tensor -> buffered_tensor**  (GM -> SPM):
       Load data from a global tensor pointer into a local buffer.
    2. **buffered_tensor -> tl.tensor**  (SPM -> GM):
       Store data from a local buffer into global memory via a tensor pointer.
    3. **buffered_tensor -> buffered_tensor** (SPM -> SPM):
       Direct local-to-local copy (original path, delegates to backend).

    Args:
        src: Source operand - either a ``tl.tensor`` (pointer) or ``buffered_tensor``.
        dst: Destination operand - either a ``tl.tensor`` (pointer) or ``buffered_tensor``.
        shape: Logical shape of the data to copy (used for GM<->Local).
        offsets: Reserved for API compatibility with TMA copy (unused on DSA).
    """
    del offsets  # DSA copy does not use offsets

    builder = _semantic.builder
    if builder is None:
        raise ValueError("copy must be used inside @triton.jit")

    src_is_buf = isinstance(src, tle.buffered_tensor)
    dst_is_buf = isinstance(dst, tle.buffered_tensor)

    # ---- Case 1: buffered_tensor -> buffered_tensor (SPM <-> SPM) ----
    if src_is_buf and dst_is_buf:
        if not hasattr(builder, "create_dsa_copy"):
            raise RuntimeError("builder missing create_dsa_copy for DSA copy")
        builder.create_dsa_copy(src.handle, dst.handle)
        return None

    # ---- Case 2: tl.tensor (GM ptr) -> buffered_tensor (SPM) ----
    if not src_is_buf and dst_is_buf:
        if not isinstance(src, tl.tensor):
            raise ValueError(f"copy src must be tl.tensor (pointer) or buffered_tensor, got {type(src)}")
        # Validate element type compatibility
        src_ptr_dtype = src.dtype
        if hasattr(src_ptr_dtype, 'element_ty'):
            src_elem_dtype = src_ptr_dtype.element_ty
        else:
            src_elem_dtype = src_ptr_dtype
        dst_elem_dtype = dst.type.element_ty
        if src_elem_dtype != dst_elem_dtype:
            raise ValueError(f"copy element type mismatch: src has {src_elem_dtype}, "
                             f"dst has {dst_elem_dtype}")

        # Create identity indices for the dst buffer and local pointers
        indices = _make_full_indices(dst, _semantic)
        dst_ptrs = local_ptr(dst, indices, _semantic=_semantic)

        # Load from GM pointers and store into SPM buffer via local pointers
        loaded = tl.load(src, _semantic=_semantic)
        tl.store(dst_ptrs, loaded, _semantic=_semantic)
        return None

    # ---- Case 3: buffered_tensor (SPM) -> tl.tensor (GM ptr) ----
    if src_is_buf and not dst_is_buf:
        if not isinstance(dst, tl.tensor):
            raise ValueError(f"copy dst must be tl.tensor (pointer) or buffered_tensor, got {type(dst)}")
        dst_ptr_dtype = dst.dtype
        if hasattr(dst_ptr_dtype, 'element_ty'):
            dst_elem_dtype = dst_ptr_dtype.element_ty
        else:
            dst_elem_dtype = dst_ptr_dtype
        src_elem_dtype = src.type.element_ty
        if src_elem_dtype != dst_elem_dtype:
            raise ValueError(f"copy element type mismatch: src has {src_elem_dtype}, "
                             f"dst has {dst_elem_dtype}")

        # Create identity indices for the src buffer and local pointers
        indices = _make_full_indices(src, _semantic)
        src_ptrs = local_ptr(src, indices, _semantic=_semantic)

        # Load from SPM buffer via local pointers and store to GM through dst
        loaded = tl.load(src_ptrs, _semantic=_semantic)
        tl.store(dst, loaded, _semantic=_semantic)
        return None

    # ---- Unsupported combination ----
    raise ValueError("copy requires at least one operand to be a buffered_tensor. "
                     f"Got src={type(src).__name__}, dst={type(dst).__name__}")


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

    Returns:
        Tensor of pointers suitable for ``tl.load``/``tl.store``.
    """
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"Buffer parameter must be tle.buffered_tensor, but got {type(buffer)}")

    builder = _semantic.builder
    if builder is None:
        raise ValueError("local_ptr must be used inside @triton.jit")

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

    ptr_dtype = tl.pointer_type(buffer.type.element_ty)
    insert_block = builder.get_insertion_block()
    if insert_block is None:
        raise RuntimeError("TLE local_ptr called without an insertion block")
    if all_scalar_indices:
        result_ty = ptr_dtype
        result_ir = ptr_dtype.to_ir(builder)
    else:
        result_ty = tl.block_type(ptr_dtype, list(view_shape))
        result_ir = result_ty.to_ir(builder)
    handles = [idx.handle for idx in idx_tensors]
    if not hasattr(builder, "create_dsa_local_pointers"):
        raise RuntimeError("builder missing create_dsa_local_pointers for DSA local_ptr")
    local_ptr_op = builder.create_dsa_local_pointers(result_ir, buffer.handle, *handles)

    result_tensor = tl.tensor(local_ptr_op.get_result(0), result_ty)

    if remote_buffer_marker:
        if all_scalar_indices:
            raise ValueError("local_ptr does not yet support scalar indices on remote buffers")
        if not hasattr(builder, "create_dsa_remote_pointers"):
            raise RuntimeError("builder missing create_dsa_remote_pointers for remote buffers")
        shard_val = (remote_shard_id.handle
                     if isinstance(remote_shard_id, tl.tensor) else _semantic.to_tensor(remote_shard_id).handle)
        remote_op = builder.create_dsa_remote_pointers(
            result_ir,
            result_tensor.handle,
            shard_val,
            scope=remote_scope,
        )
        result_tensor = tl.tensor(remote_op.get_result(0), result_ty)

    return result_tensor


@tl.builtin
def to_tensor(buffer: tle.buffered_tensor, writable: bool = True, _semantic=None) -> tl.tensor:
    """
    Convert a DSA ``buffered_tensor`` (SPM buffer) into a ``tl.tensor`` view.

    This is a zero-copy view: the returned tensor aliases the SPM buffer, so it
    can participate in standard Triton tensor expressions without any data
    movement. Lowered by ``--tle-to-mk`` into ``bufferization.to_tensor``.

    Args:
        buffer: A ``buffered_tensor`` previously allocated with ``tle.language.dsa.alloc``.
        writable: Mark the resulting tensor as writable (default ``True``).

    Returns:
        ``tl.tensor`` aliasing the SPM buffer contents.
    """
    builder = _semantic.builder
    if builder is None:
        raise ValueError("to_tensor must be used inside @triton.jit")
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"to_tensor requires a buffered_tensor, got {type(buffer).__name__}")
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    result_ty = tl.block_type(buffer.type.element_ty, list(shape))
    result_ir = result_ty.to_ir(builder)
    if not hasattr(builder, "create_dsa_to_tensor"):
        raise RuntimeError("builder missing create_dsa_to_tensor for DSA to_tensor")
    handle = builder.create_dsa_to_tensor(result_ir, buffer.handle, bool(writable))
    return tl.tensor(handle, result_ty)


@tl.builtin
def to_buffer(src: tl.tensor, space=None, _semantic=None) -> tle.buffered_tensor:
    """
    Copy a ``tl.tensor`` into a newly-allocated DSA buffer and return it.

    This is the reverse bridge of :func:`to_tensor`: it materialises the tensor
    value into a fresh SPM buffer. Lowered by ``--tle-to-mk`` into
    ``bufferization.to_buffer`` + ``memref.copy``.

    Args:
        src: A ``tl.tensor`` value to store.
        space: Storage scope for the new buffer. Defaults to ``tle.spm``;
            may be an address-space selector such as
            ``tle.language.dsa.tsingmicro.SPM``.

    Returns:
        A new ``buffered_tensor`` containing a copy of ``src``.
    """
    builder = _semantic.builder
    if builder is None:
        raise ValueError("to_buffer must be used inside @triton.jit")
    if not isinstance(src, tl.tensor):
        raise ValueError(f"to_buffer src must be a tl.tensor, got {type(src).__name__}")
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in src.shape)
    if not shape:
        raise ValueError("to_buffer src must be a non-scalar tensor")
    buf = alloc(shape, src.dtype, scope=space, _semantic=_semantic)
    if not hasattr(builder, "create_dsa_to_buffer"):
        raise RuntimeError("builder missing create_dsa_to_buffer for DSA to_buffer")
    builder.create_dsa_to_buffer(src.handle, buf.handle)
    return buf


def _check_binary_operands(opname, lhs, rhs, out):
    """Validate three-operand elementwise arithmetic operands.

    All three must be ``buffered_tensor`` with identical shape and element
    dtype (tle.md: no implicit broadcast in this API layer).
    """
    for name, val in (("lhs", lhs), ("rhs", rhs), ("out", out)):
        if not isinstance(val, tle.buffered_tensor):
            raise ValueError(f"{opname} {name} must be a buffered_tensor, "
                             f"got {type(val).__name__}")
    lhs_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in lhs.type.shape)
    rhs_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in rhs.type.shape)
    out_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in out.type.shape)
    if lhs_shape != rhs_shape or lhs_shape != out_shape:
        raise ValueError(f"{opname} shape mismatch: lhs={lhs_shape}, "
                         f"rhs={rhs_shape}, out={out_shape}")
    lhs_dtype = lhs.type.element_ty
    rhs_dtype = rhs.type.element_ty
    out_dtype = out.type.element_ty
    if lhs_dtype != rhs_dtype or lhs_dtype != out_dtype:
        raise ValueError(f"{opname} dtype mismatch: lhs={lhs_dtype}, "
                         f"rhs={rhs_dtype}, out={out_dtype}")


def _create_binary_builtin(opname, builder_method, builder):
    """Shared builder for dsa binary arithmetic ops."""
    if builder is None:
        raise ValueError(f"{opname} must be used inside @triton.jit")
    if not hasattr(builder, builder_method):
        raise RuntimeError(f"builder missing {builder_method} for DSA {opname}")
    return builder


@tl.builtin
def add(lhs, rhs, out, _semantic=None):
    """``out = lhs + rhs`` elementwise on SPM buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("add", lhs, rhs, out)
    _create_binary_builtin("add", "create_dsa_add", builder).create_dsa_add(lhs.handle, rhs.handle, out.handle)


@tl.builtin
def sub(lhs, rhs, out, _semantic=None):
    """``out = lhs - rhs`` elementwise on SPM buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("sub", lhs, rhs, out)
    _create_binary_builtin("sub", "create_dsa_sub", builder).create_dsa_sub(lhs.handle, rhs.handle, out.handle)


@tl.builtin
def mul(lhs, rhs, out, _semantic=None):
    """``out = lhs * rhs`` elementwise on SPM buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("mul", lhs, rhs, out)
    _create_binary_builtin("mul", "create_dsa_mul", builder).create_dsa_mul(lhs.handle, rhs.handle, out.handle)


@tl.builtin
def max(lhs, rhs, out, _semantic=None):
    """``out = max(lhs, rhs)`` elementwise on SPM buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("max", lhs, rhs, out)
    _create_binary_builtin("max", "create_dsa_maximum", builder).create_dsa_maximum(lhs.handle, rhs.handle, out.handle)


@tl.builtin
def min(lhs, rhs, out, _semantic=None):
    """``out = min(lhs, rhs)`` elementwise on SPM buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("min", lhs, rhs, out)
    _create_binary_builtin("min", "create_dsa_minimum", builder).create_dsa_minimum(lhs.handle, rhs.handle, out.handle)


@tl.builtin
def div(lhs, rhs, out, _semantic=None):
    """``out = lhs / rhs`` elementwise on SPM buffers (three-operand)."""
    builder = _semantic.builder
    _check_binary_operands("div", lhs, rhs, out)
    _create_binary_builtin("div", "create_dsa_div", builder).create_dsa_div(lhs.handle, rhs.handle, out.handle)


def _tle_pick_sum_dtype(in_dtype, dtype):
    if dtype is not None:
        return dtype
    if in_dtype.is_int_signed():
        return tl.int32 if in_dtype.int_bitwidth < 32 else None
    if in_dtype.is_int_unsigned():
        return tl.uint32 if in_dtype.int_bitwidth < 32 else None
    return None


@tl.builtin
def cumsum(input, axis=0, reverse=False, dtype: tl.constexpr = None, _semantic=None):
    """
    Compute exclusive cumulative sum and total sum along ``axis``.

    Returns ``(exclusive_sum, total_sum)``.  The Tsingmicro lowering in this
    branch supports forward scan along the last block dimension.
    """
    axis = tl._unwrap_if_constexpr(axis)
    reverse = tl._unwrap_if_constexpr(reverse)
    dtype = tl._unwrap_if_constexpr(dtype)

    if reverse:
        raise NotImplementedError("tle.cumsum(reverse=True) is not supported on Tsingmicro yet")

    builder = _semantic.builder
    if not isinstance(input, tl.tensor):
        input = _semantic.to_tensor(input)
    input = tl._promote_bfloat16_to_float32(input, _semantic=_semantic)
    if input.dtype.is_bool():
        raise TypeError("tle.cumsum does not support bool input on Tsingmicro")
    out_dtype: tl.constexpr = _tle_pick_sum_dtype(input.dtype, dtype)
    if out_dtype is not None:
        input = input.to(out_dtype, _semantic=_semantic)

    input_ty = input.type
    if not input_ty.is_block():
        zero = tl.full((), 0, input.dtype, _semantic=_semantic)
        return zero, input

    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in input_ty.shape)
    rank = len(shape)
    normalized_axis = axis + rank if axis < 0 else axis
    if normalized_axis != rank - 1:
        raise NotImplementedError("tle.cumsum currently supports only the last block dimension on Tsingmicro")

    exclusive_ty = input_ty
    total_ty = input_ty.scalar if rank == 1 else tl.block_type(input_ty.scalar, list(shape[:-1]))
    pad = shape[-1] * 2
    cumsum_op = builder.create_dsa_cumsum(
        exclusive_ty.to_ir(builder),
        total_ty.to_ir(builder),
        input.handle,
        int(normalized_axis),
        bool(reverse),
        list(shape),
        int(pad),
    )
    exclusive_sum = tl.tensor(cumsum_op.get_result(0), exclusive_ty)
    total_sum = tl.tensor(cumsum_op.get_result(1), total_ty)
    return exclusive_sum, total_sum


# ---------------------------------------------------------------------------
# dsa.extract_slice / dsa.insert_slice / dsa.extract_tile / dsa.insert_tile
#
# Element-level strided slicing (spec 3.3.2.4) is the primary interface; the
# tile forms are a grid-coordinate convenience wrapper that resolves a tile
# index into per-dim offsets and delegates to the same slice builders.
# Offsets may mix static (int/constexpr) and dynamic (scalar tl.tensor) dims,
# encoded with ShapedType::kDynamic as the sentinel in `static_offsets`.
# ---------------------------------------------------------------------------

_DYNAMIC = -(1 << 63)  # int64 min == ShapedType::kDynamic sentinel


def _try_unwrap_int(val):
    """Return ``val`` as a Python int if int/constexpr-like, else ``None``."""
    if isinstance(val, int):
        return val
    v = tl._unwrap_if_constexpr(val)
    return v if isinstance(v, int) else None


def _static_dims(shape, fn):
    """Unwrap a shape into compile-time ints, erroring on dynamic dims."""
    shape = tl._unwrap_if_constexpr(shape)
    dims = [tl._unwrap_if_constexpr(d) for d in shape]
    if any(not isinstance(d, int) for d in dims):
        raise ValueError(f"{fn}: shape must be compile-time constants, got {shape}")
    return dims


def _split_offsets(offsets, fn):
    """Split per-dim offsets (int/constexpr or scalar tl.tensor) into
    ``(static_offsets, dyn_handles)``; dynamic dims use ``_DYNAMIC`` sentinel."""
    offsets = tl._unwrap_if_constexpr(offsets)
    static = []
    dyn = []
    for v in offsets:
        if isinstance(v, tl.tensor):
            if v.shape != ():
                raise ValueError(f"{fn}: dynamic offsets must be scalar tl.tensor")
            static.append(_DYNAMIC)
            dyn.append(v.handle)
        else:
            iv = _try_unwrap_int(v)
            if iv is None:
                raise ValueError(f"{fn}: offsets must be int/constexpr or scalar tl.tensor")
            static.append(iv)
    return static, dyn


def _resolve_tile_offsets(index, src_shape, tile_shape, _semantic, fn):
    """Convert a tile grid index into per-dim slice offsets.

    Supported ``index`` forms (grid semantics, mirrors the former
    ``tle.extract_tile``/``tle.insert_tile`` behaviour):
      1. scalar ``tl.tensor``  -> runtime linear tile id (div/mod chain)
      2. int / constexpr       -> compile-time linear tile id
      3. tuple/list/tl.tuple   -> per-dim coordinates, mixed int/tl.tensor

    Returns ``(static_offsets, dyn_handles)``.
    """
    rank = len(src_shape)
    grid = [s // t for s, t in zip(src_shape, tile_shape)]

    # --- scalar dynamic linear index ---
    if isinstance(index, tl.tensor):
        if index.shape != ():
            raise ValueError(f"{fn}: dynamic index must be a scalar tl.tensor")
        strides = [1] * rank
        acc = 1
        for i in builtins.range(rank - 1, -1, -1):
            strides[i] = acc
            acc *= grid[i]
        static = []
        dyn = []
        lin = index
        for g, st, t in zip(grid, strides, tile_shape):
            div = _semantic.floordiv(lin, _semantic.to_tensor(st))
            mod = _semantic.mod(div, _semantic.to_tensor(g))
            off = _semantic.mul(mod, _semantic.to_tensor(t), sanitize_overflow=True)
            static.append(_DYNAMIC)
            dyn.append(off.handle)
        return static, dyn

    # Unwrap constexpr wrappers before static/multi-dim dispatch.
    unwrapped = tl._unwrap_if_constexpr(index)

    # --- scalar static linear index ---
    scalar = _try_unwrap_int(unwrapped)
    if scalar is not None:
        total = 1
        for g in grid:
            total *= g
        if scalar < 0 or scalar >= total:
            raise ValueError(f"{fn}: index {scalar} out of range [0, {total})")
        coords = []
        rem = scalar
        for g in reversed(grid):
            coords.append(rem % g)
            rem //= g
        coords.reverse()
        return [c * t for c, t in zip(coords, tile_shape)], []

    # --- multi-dim index ---
    if isinstance(unwrapped, tl.tuple):
        index_list = list(unwrapped.values)
    elif isinstance(unwrapped, (tuple, list)):
        index_list = list(unwrapped)
    else:
        raise ValueError(f"{fn}: index must be int/constexpr, tuple/list of int/constexpr, "
                         f"or a scalar tl.tensor; got {type(index)}")

    if len(index_list) != rank:
        raise ValueError(f"{fn}: index rank {len(index_list)} must match source rank {rank}")

    static = []
    dyn = []
    for k, (v, t) in enumerate(zip(index_list, tile_shape)):
        if isinstance(v, tl.tensor):
            if v.shape != ():
                raise ValueError(f"{fn}: dynamic index elements must be scalar")
            off = _semantic.mul(v, _semantic.to_tensor(t), sanitize_overflow=True)
            static.append(_DYNAMIC)
            dyn.append(off.handle)
        else:
            iv = _try_unwrap_int(v)
            if iv is None:
                raise ValueError(f"{fn}: index must contain tensor/int/constexpr")
            if iv < 0 or iv >= grid[k]:
                raise ValueError(f"{fn}: index[{k}]={iv} out of bounds for tile grid "
                                 f"(0~{grid[k] - 1})")
            static.append(iv * t)
    return static, dyn


def _check_static_slice_bounds(fn, src_shape, static_offsets, sizes, strides):
    """Validate the static (non-sentinel) offsets against src/sizes/strides."""
    for i, (src, off, size, stride) in enumerate(zip(src_shape, static_offsets, sizes, strides)):
        if size <= 0:
            raise ValueError(f"{fn}: size[{i}]={size} must be positive")
        if stride <= 0:
            raise ValueError(f"{fn}: stride[{i}]={stride} must be positive")
        if off == _DYNAMIC:
            continue
        if off < 0:
            raise ValueError(f"{fn}: offset[{i}]={off} must be non-negative")
        end = off + (size - 1) * stride + 1
        if end > src:
            raise ValueError(f"{fn}: slice [{off}:{off}+{size}*{stride}] exceeds source dim "
                             f"{i} ({src})")


@tl._tensor_member_fn
@tl.builtin
def extract_slice(x: tl.tensor, offsets, sizes, strides, _semantic=None) -> tl.tensor:
    """Extract a strided slice from ``x``.

    Args:
        x:       Source ``tl.tensor``.
        offsets: Per-dim offsets; each element is an int/constexpr (static) or
            a scalar ``tl.tensor`` (dynamic).
        sizes:   Per-dim slice sizes (compile-time constants); this is the
            result shape (``tensor.extract_slice`` semantics).
        strides: Per-dim strides (compile-time constants).

    Returns a tensor whose shape is ``sizes``.
    """
    if not isinstance(x, tl.tensor):
        raise ValueError(f"extract_slice: source must be tl.tensor, got {type(x)}")

    builder = _semantic.builder
    if builder is None:
        raise ValueError("extract_slice must be used inside @triton.jit")

    src_shape = _static_dims(x.type.shape, "extract_slice")
    sizes = _static_dims(sizes, "extract_slice")
    strides = _static_dims(strides, "extract_slice")
    offsets = tl._unwrap_if_constexpr(offsets)
    if len(offsets) != len(src_shape) or len(sizes) != len(src_shape) \
            or len(strides) != len(src_shape):
        raise ValueError("extract_slice: offsets/sizes/strides rank must match source rank")

    static_offsets, dyn_offsets = _split_offsets(offsets, "extract_slice")
    _check_static_slice_bounds("extract_slice", src_shape, static_offsets, sizes, strides)

    # tensor.extract_slice semantics: sizes is the result shape.
    result_ty = tl.block_type(x.type.element_ty, sizes)
    result_ir = result_ty.to_ir(builder)

    handle = builder.create_dsa_extract_slice(result_ir, x.handle, static_offsets, dyn_offsets, sizes, strides)
    return tl.tensor(handle, result_ty)


@tl._tensor_member_fn
@tl.builtin
def insert_slice(x: tl.tensor, tile: tl.tensor, offsets, sizes=None, strides=None, _semantic=None) -> tl.tensor:
    """Insert ``tile`` into ``x`` at ``offsets``.

    ``sizes`` defaults to ``tile.shape``; ``strides`` defaults to all ones.
    Returns a new tensor with the same shape/type as ``x``.
    """
    if not isinstance(x, tl.tensor):
        raise ValueError(f"insert_slice: source must be tl.tensor, got {type(x)}")
    if not isinstance(tile, tl.tensor):
        raise ValueError(f"insert_slice: tile must be tl.tensor, got {type(tile)}")

    builder = _semantic.builder
    if builder is None:
        raise ValueError("insert_slice must be used inside @triton.jit")

    src_shape = _static_dims(x.type.shape, "insert_slice")
    tile_shape = _static_dims(tile.type.shape, "insert_slice")
    offsets = tl._unwrap_if_constexpr(offsets)
    if len(offsets) != len(src_shape):
        raise ValueError("insert_slice: offsets rank must match source rank")
    if sizes is None:
        sizes = tile_shape
    else:
        sizes = _static_dims(sizes, "insert_slice")
    if strides is None:
        strides = [1] * len(src_shape)
    else:
        strides = _static_dims(strides, "insert_slice")
    if len(sizes) != len(src_shape) or len(strides) != len(src_shape):
        raise ValueError("insert_slice: sizes/strides rank must match source rank")
    if tuple(sizes) != tuple(tile_shape):
        raise ValueError(f"insert_slice: sizes {sizes} must match tile shape {tile_shape}")
    if x.type.element_ty != tile.type.element_ty:
        raise ValueError(f"insert_slice: element type mismatch source={x.type.element_ty}, "
                         f"tile={tile.type.element_ty}")

    static_offsets, dyn_offsets = _split_offsets(offsets, "insert_slice")
    _check_static_slice_bounds("insert_slice", src_shape, static_offsets, sizes, strides)

    handle = builder.create_dsa_insert_slice(x.type.to_ir(builder), x.handle, tile.handle, static_offsets, dyn_offsets,
                                             sizes, strides)
    return tl.tensor(handle, x.type)


@tl._tensor_member_fn
@tl.builtin
def extract_tile(x: tl.tensor, index, tile_shape, _semantic=None) -> tl.tensor:
    """Extract a tile from ``x`` at grid ``index`` (convenience wrapper).

    ``index`` supports per-dim coordinates, a linear tile id, or a dynamic
    scalar tensor; ``tile_shape`` must be compile-time constants. Equivalent
    to :func:`extract_slice` with unit strides and grid-derived offsets.
    """
    if not isinstance(x, tl.tensor):
        raise ValueError(f"extract_tile: source must be tl.tensor, got {type(x)}")

    builder = _semantic.builder
    if builder is None:
        raise ValueError("extract_tile must be used inside @triton.jit")

    src_shape = _static_dims(x.type.shape, "extract_tile")
    tile_shape = _static_dims(tile_shape, "extract_tile")
    if len(tile_shape) != len(src_shape):
        raise ValueError(f"extract_tile: tile_shape rank ({len(tile_shape)}) must match "
                         f"source rank ({len(src_shape)})")
    for i, (s, t) in enumerate(zip(src_shape, tile_shape)):
        if t <= 0:
            raise ValueError(f"extract_tile: tile dim {i} must be positive, got {t}")
        if s % t != 0:
            raise ValueError(f"extract_tile: source dim {i} ({s}) not divisible by tile dim ({t})")

    static_offsets, dyn_offsets = _resolve_tile_offsets(index, src_shape, tile_shape, _semantic, "extract_tile")

    result_ty = tl.block_type(x.type.element_ty, tile_shape)
    result_ir = result_ty.to_ir(builder)

    handle = builder.create_dsa_extract_slice(result_ir, x.handle, static_offsets, dyn_offsets, tile_shape,
                                              [1] * len(tile_shape))
    return tl.tensor(handle, result_ty)


@tl._tensor_member_fn
@tl.builtin
def insert_tile(x: tl.tensor, tile: tl.tensor, index, _semantic=None) -> tl.tensor:
    """Insert ``tile`` into ``x`` at grid ``index`` (convenience wrapper).

    ``tile_shape`` is derived from ``tile``; ``index`` supports the same forms
    as :func:`extract_tile`. Returns a tensor with the shape/type of ``x``.
    """
    if not isinstance(x, tl.tensor):
        raise ValueError(f"insert_tile: source must be tl.tensor, got {type(x)}")
    if not isinstance(tile, tl.tensor):
        raise ValueError(f"insert_tile: tile must be tl.tensor, got {type(tile)}")

    builder = _semantic.builder
    if builder is None:
        raise ValueError("insert_tile must be used inside @triton.jit")

    src_shape = _static_dims(x.type.shape, "insert_tile")
    tile_shape = _static_dims(tile.type.shape, "insert_tile")
    if len(tile_shape) != len(src_shape):
        raise ValueError(f"insert_tile: source rank ({len(src_shape)}) must match tile rank "
                         f"({len(tile_shape)})")
    for i, (s, t) in enumerate(zip(src_shape, tile_shape)):
        if t <= 0:
            raise ValueError(f"insert_tile: tile dim {i} must be positive, got {t}")
        if s % t != 0:
            raise ValueError(f"insert_tile: source dim {i} ({s}) not divisible by tile dim ({t})")
    if x.type.element_ty != tile.type.element_ty:
        raise ValueError(f"insert_tile: element type mismatch source={x.type.element_ty}, "
                         f"tile={tile.type.element_ty}")

    static_offsets, dyn_offsets = _resolve_tile_offsets(index, src_shape, tile_shape, _semantic, "insert_tile")

    handle = builder.create_dsa_insert_slice(x.type.to_ir(builder), x.handle, tile.handle, static_offsets, dyn_offsets,
                                             tile_shape, [1] * len(tile_shape))
    return tl.tensor(handle, x.type)


# Fmt_INT64 in tsingmicro tx81 Data_Format enum (see instr_def / op_def.h).
_DSA_RANDGEN_FMT_INT64 = 11
_DSA_RANDGEN_NUM_STREAMS = 16
_DSA_RANDGEN_ALIGN_BYTES = 128


@tl.builtin
def randgen(seed0, seed1, n_out: tl.constexpr, _builder=None, _semantic=None, _generator=None):
    """
    Hardware tsingmicro TX81 peri for ``randgen``.

    Args:
        seed0: block tensor ``[16]`` of ``int64`` / ``uint64`` seeds (stream a).
        seed1: block tensor ``[16]`` of ``int64`` / ``uint64`` seeds (stream b).
        n_out: number of ``int64`` random outputs; must be a multiple of 16
            (hardware emits 16 values / 128 bytes per step).

    Returns:
        ``(out, seed0_out, seed1_out)`` where ``out`` has shape ``[n_out]`` and
        the seed outputs are updated length-16 seed vectors.

    Notes:
        Output values are raw xorshift128+ ``uint64`` bit patterns stored as
        ``int64``. Convert to Uniform(0,1) / Normal yourself (see ``rand`` /
        ``randn`` helpers), or feed downstream kernels.
    """
    n_out = int(tl._unwrap_if_constexpr(n_out))
    if n_out <= 0 or (n_out % _DSA_RANDGEN_NUM_STREAMS) != 0:
        raise ValueError(f"tle.dsa.randgen n_out must be a positive multiple of "
                         f"{_DSA_RANDGEN_NUM_STREAMS}, got {n_out}")

    builder = _builder if _builder is not None else _semantic.builder
    if not hasattr(builder, "create_dsa_randgen"):
        raise RuntimeError("builder missing create_dsa_randgen for DSA randgen")

    if not isinstance(seed0, tl.tensor):
        seed0 = tl.to_tensor(seed0, _semantic=_semantic)
    if not isinstance(seed1, tl.tensor):
        seed1 = tl.to_tensor(seed1, _semantic=_semantic)

    if seed0.dtype != tl.int64:
        seed0 = seed0.to(tl.int64, _semantic=_semantic)
    if seed1.dtype != tl.int64:
        seed1 = seed1.to(tl.int64, _semantic=_semantic)

    seed0_ty = seed0.type
    seed1_ty = seed1.type
    if (not seed0_ty.is_block() or not seed1_ty.is_block()
            or tuple(int(tl._unwrap_if_constexpr(d)) for d in seed0_ty.shape) != (_DSA_RANDGEN_NUM_STREAMS, )
            or tuple(int(tl._unwrap_if_constexpr(d)) for d in seed1_ty.shape) != (_DSA_RANDGEN_NUM_STREAMS, )):
        raise ValueError("tle.dsa.randgen seeds must be block tensors of shape [16]")

    out_ty = tl.block_type(tl.int64, [n_out])
    byte_count = n_out * 8
    assert byte_count % _DSA_RANDGEN_ALIGN_BYTES == 0

    rand_op = builder.create_dsa_randgen(
        out_ty.to_ir(builder),
        seed0_ty.to_ir(builder),
        seed1_ty.to_ir(builder),
        seed0.handle,
        seed1.handle,
        int(byte_count),
        int(_DSA_RANDGEN_FMT_INT64),
    )
    out = tl.tensor(rand_op.get_result(0), out_ty)
    seed0_out = tl.tensor(rand_op.get_result(1), seed0_ty)
    seed1_out = tl.tensor(rand_op.get_result(2), seed1_ty)
    return out, seed0_out, seed1_out


def _uint32_bits_to_uniform(bits32, semantic):
    """
    Map random int32 bits to Uniform(0, 1) via IEEE754 mantissa stuffing.

    u = bitcast((bits & 0x7FFFFF) | 0x3F800000, f32) - 1.0

    Avoids the sitofp / where / cmp / sub integer chain that lowers to
    per-element scf.for on TX81 (no int vector ALU). Uses only bitwise
    and/or + bitcast + float sub, which can stay on peri / float paths.

    ``semantic`` is the v3.6 SemanticAnalyzer (methods take ``self``, never a
    raw builder).
    """
    # 0x7FFFFF fits signed i32; 0x3F800000 (== float 1.0 bits) is taken as
    # uint32 by to_tensor so the bit pattern is preserved.
    mant_mask = semantic.to_tensor(0x7FFFFF)
    one_bits = semantic.to_tensor(0x3F800000)
    one_f = semantic.to_tensor(1.0)
    mant = semantic.and_(bits32, mant_mask)
    packed = semantic.or_(mant, one_bits)
    f12 = semantic.bitcast(packed, tl.float32)  # [1, 2)
    return semantic.sub(f12, one_f, True)  # [0, 1)


def _i64_as_i32_view(raw_i64, n_i32: int, builder):
    """
    Zero-copy view of an i64 buffer as i32 (little-endian: lo32, hi32, ...).

    ``raw_i64`` must have shape ``[n_i32 // 2]``. Emits vendor-neutral
    ``dsa.bitcast`` (backends alias the buffer; no elementwise ``trunci``).
    """
    n_i64 = int(tl._unwrap_if_constexpr(raw_i64.shape[0]))
    if n_i32 != n_i64 * 2:
        raise ValueError(f"i64→i32 view expects n_i32 == 2 * n_i64, got {n_i32} vs 2*{n_i64}")
    if not hasattr(builder, "create_dsa_bitcast"):
        raise RuntimeError("builder missing create_dsa_bitcast; rebuild libtriton with TLE DSA")
    dst_ty = tl.block_type(tl.int32, [n_i32])
    handle = builder.create_dsa_bitcast(dst_ty.to_ir(builder), raw_i64.handle)
    return tl.tensor(handle, dst_ty)


@tl.builtin
def rand(seed0, seed1, n_out: tl.constexpr, _builder=None, _semantic=None, _generator=None):
    """
    Uniform(0, 1) floats via hardware ``randgen`` + float scaling.

    ``n_out`` must be a multiple of 32 (``randgen`` emits i64; each i64
    contributes two i32 samples via a zero-copy view).

    Returns ``(u, seed0_out, seed1_out)`` with ``u`` shaped ``[n_out]`` float32.
    """
    n_out = int(tl._unwrap_if_constexpr(n_out))
    if n_out <= 0 or (n_out % 32) != 0:
        raise ValueError(f"tle.dsa.rand n_out must be a positive multiple of 32, got {n_out}")

    builder = _builder if _builder is not None else _semantic.builder
    # Half as many i64 draws: lo/hi 32-bit halves become two Uniform samples.
    raw64, seed0_out, seed1_out = randgen(seed0, seed1, n_out // 2, _semantic=_semantic)
    bits32 = _i64_as_i32_view(raw64, n_out, builder)
    u = _uint32_bits_to_uniform(bits32, _semantic)
    return u, seed0_out, seed1_out


@tl.builtin
def randn(seed0, seed1, n_out: tl.constexpr, _builder=None, _semantic=None, _generator=None):
    """
    Normal(0, 1) floats via hardware ``randgen`` + Box-Muller.

    ``n_out`` must be a multiple of 32 (two Uniform halves of size
    ``n_out // 2``, each backed by ``n_out // 4`` i64 draws viewed as i32).

    Uses a single ``randgen`` of length ``n_out // 2`` and pairs consecutive
    Uniform samples ``(u[2i], u[2i+1])`` for Box-Muller.  Calling ``rand``
    twice is unsafe while the peri does not advance ``seed0_out`` /
    ``seed1_out``.

    Returns ``(n, seed0_out, seed1_out)`` with ``n`` shaped ``[n_out]`` float32
    (concatenation of the two Box-Muller outputs).
    """
    n_out = int(tl._unwrap_if_constexpr(n_out))
    half = n_out // 2
    if n_out <= 0 or (n_out % 32) != 0:
        raise ValueError(f"tle.dsa.randn n_out must be a positive multiple of 32, got {n_out}")

    builder = _builder if _builder is not None else _semantic.builder
    # Reuse ``rand`` (Uniform) then Box-Muller with consecutive pairing.
    # Bare ``semantic.to_tensor(float)`` scalars in BM previously produced a
    # systematic Normal bias on TX81 (mean≈0.15, std≈0.94); force f32 via
    # broadcast from ``u_half`` (``u*0+c``). Layout reshape/permute/cat alone
    # is fine — see ``test_tx81_reshape_permute_cat.py``.
    uv, seed0_out, seed1_out = rand(seed0, seed1, n_out, _builder=builder, _semantic=_semantic)

    pairs = _semantic.reshape(uv, [half, 2], False)
    u_half, v_half = _semantic.split(pairs)

    # Force f32-typed scalars via broadcast from u_half (avoids f64 pitfalls).
    zero = _semantic.mul(u_half, _semantic.to_tensor(0.0), True)
    eps = _semantic.add(zero, _semantic.to_tensor(1.0e-7), True)
    two_pi = _semantic.add(zero, _semantic.to_tensor(6.283185307179586), True)
    neg_two = _semantic.add(zero, _semantic.to_tensor(-2.0), True)

    u1 = _semantic.maximum(u_half, eps, PropagateNan.NONE)
    theta = _semantic.mul(two_pi, v_half, True)
    log_u1 = tlmath.log(u1, _semantic=_semantic)
    r = tlmath.sqrt(_semantic.mul(neg_two, log_u1, True), _semantic=_semantic)
    n0 = _semantic.mul(r, tlmath.cos(theta, _semantic=_semantic), True)
    n1 = _semantic.mul(r, tlmath.sin(theta, _semantic=_semantic), True)
    out = _semantic.reshape(_semantic.join(n0, n1), [n_out], False)
    return out, seed0_out, seed1_out
