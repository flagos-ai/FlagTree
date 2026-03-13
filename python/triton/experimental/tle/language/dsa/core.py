# flagtree tle
import builtins
import triton.language.core as tl
import triton.language.semantic as semantic
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
def memory_space(input, space, _builder=None):
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
    if _builder is not None and hasattr(input, 'handle') and hasattr(input.handle, 'set_attr'):
        input.handle.set_attr("tt.memory_space", _builder.get_string_attr(str(space)))
    return input


@tl.builtin
def alloc(
    shape: tuple,
    dtype: tl.dtype,
    layout: Optional[object] = None,
    scope: tle.scope = None,
    _builder=None,
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
    from .semantic import DSASemantic

    if _builder is None:
        raise ValueError("alloc must be used inside @triton.jit")
    if layout is not None:
        raise ValueError("alloc(): layout parameter is not yet support for DSA backend")

    # --- Validate inputs via semantic layer ---
    unwrapped_shape = DSASemantic.validate_alloc_shape(shape)
    elem_dtype = DSASemantic.validate_alloc_dtype(dtype)
    resolved_scope = DSASemantic.validate_alloc_scope(scope)

    elem_ir_ty = elem_dtype.to_ir(_builder)

    if not hasattr(_builder, "create_dsa_alloc"):
        raise RuntimeError("builder missing create_dsa_alloc for DSA alloc")

    alloc_value = _builder.create_dsa_alloc(list(unwrapped_shape), elem_ir_ty)
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
    _builder=None,
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

    if _builder is None:
        raise ValueError("copy must be used inside @triton.jit")

    src_is_buf = isinstance(src, tle.buffered_tensor)
    dst_is_buf = isinstance(dst, tle.buffered_tensor)

    # ---- Case 1: buffered_tensor -> buffered_tensor (SPM <-> SPM) ----
    if src_is_buf and dst_is_buf:
        if not hasattr(_builder, "create_dsa_copy"):
            raise RuntimeError("builder missing create_dsa_copy for DSA copy")
        _builder.create_dsa_copy(src.handle, dst.handle)
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
        if not hasattr(_builder, "create_dsa_copy"):
            raise RuntimeError("builder missing create_dsa_copy for DSA copy")
        _builder.create_dsa_copy(src.handle, dst.handle)
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
        if not hasattr(_builder, "create_dsa_copy"):
            raise RuntimeError("builder missing create_dsa_copy for DSA copy")
        _builder.create_dsa_copy(src.handle, dst.handle)
        return None

    # ---- Unsupported combination ----
    raise ValueError("copy requires at least one operand to be a buffered_tensor. "
                     f"Got src={type(src).__name__}, dst={type(dst).__name__}")


def _expand_index_to_shape(index: tl.tensor, shape: Sequence[int], axis: int, _builder) -> tl.tensor:
    idx = index
    for _ in builtins.range(axis):
        idx = tl.expand_dims(idx, 0, _builder=_builder)
    for _ in builtins.range(len(shape) - axis - 1):
        idx = tl.expand_dims(idx, len(idx.shape), _builder=_builder)
    return tl.broadcast_to(idx, *shape, _builder=_builder)


def _make_full_indices(buffer: tle.buffered_tensor, _builder) -> tuple[tl.tensor, ...]:
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    indices = []
    for axis, dim in enumerate(shape):
        idx = tl.arange(0, dim, _builder=_builder)
        idx = _expand_index_to_shape(idx, shape, axis, _builder)
        indices.append(idx)
    return tuple(indices)


@tl.builtin
def local_ptr(
    buffer: tle.buffered_tensor,
    indices: Optional[Sequence] = None,
    _builder=None,
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

    if _builder is None:
        raise ValueError("local_ptr must be used inside @triton.jit")

    # Preferred metadata source: buffered_tensor.type (survives JIT value
    # reconstruction). Keep value attrs as backward-compatibility fallback.
    remote_shard_id = getattr(buffer.type, "_tle_remote_shard_id", None)
    _ = getattr(buffer.type, "_tle_remote_scope", None)
    if remote_shard_id is None:
        remote_shard_id = getattr(buffer, "_tle_remote_shard_id", None)
        _ = getattr(buffer, "_tle_remote_scope", None)
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
        idx_tensor = idx if isinstance(idx, tensor) else semantic.to_tensor(idx, _builder)
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
    insert_block = _builder.get_insertion_block()
    if insert_block is None:
        raise RuntimeError("TLE local_ptr called without an insertion block")
    if all_scalar_indices:
        result_ty = ptr_dtype
        result_ir = ptr_dtype.to_ir(_builder)
    else:
        result_ty = tl.block_type(ptr_dtype, list(view_shape))
        result_ir = result_ty.to_ir(_builder)
    handles = [idx.handle for idx in idx_tensors]
    if not hasattr(_builder, "create_dsa_local_pointers"):
        raise RuntimeError("builder missing create_dsa_local_pointers for DSA local_ptr")
    local_ptr_op = _builder.create_dsa_local_pointers(result_ir, buffer.handle, *handles)

    result_tensor = tl.tensor(local_ptr_op.get_result(0), result_ty)

    if remote_buffer_marker:
        if all_scalar_indices:
            raise ValueError("local_ptr does not yet support scalar indices on remote buffers")
        if not hasattr(_builder, "create_dsa_remote_pointers"):
            raise RuntimeError("builder missing create_dsa_remote_pointers for remote buffers")
        shard_val = (remote_shard_id.handle if isinstance(remote_shard_id, tl.tensor) else semantic.to_tensor(
            remote_shard_id, _builder).handle)
        remote_op = _builder.create_dsa_remote_pointers(result_ir, result_tensor.handle, shard_val)
        result_tensor = tl.tensor(remote_op.get_result(0), result_ty)

    return result_tensor
