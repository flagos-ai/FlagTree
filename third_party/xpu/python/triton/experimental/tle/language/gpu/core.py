# Triton 3.6 XPU TLE - GPU core APIs
# Phase 1: alloc, local_ptr, copy (连续性搬运)

import builtins
import triton.language.core as tl
from typing import Optional, Sequence
from enum import Enum
from . import types as tle_types
from .types import (
    scope,
    lmem,
    buffered_tensor,
    buffered_tensor_type,
    shared_layout,
    swizzled_shared_layout,
)

from triton.language.core import (
    constexpr,
    tensor,
    range,
)

# =============================================================================
# pipeline — software pipeline loop iterator
# =============================================================================


class pipeline(range):
    """Software pipeline loop iterator for XPU.

    Equivalent to `tl.range` but with explicit pipeline semantics.
    The compiler restructures the loop into prefetch + compute structure.

    Usage:
        for yoff in tle.gpu.pipeline(0, ynumel, YBLOCK, num_stages=2):
            ...
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)


# =============================================================================
# memory_space — annotate tensor with memory space attribute
# =============================================================================


@tl.builtin
def memory_space(input, space, _builder=None, _semantic=None):
    """Assign a memory space attribute to a tensor."""
    space = tl._unwrap_if_constexpr(space)
    input.handle.set_attr("tt.memory_space", _semantic.builder.get_string_attr(space))
    return input


# =============================================================================
# alloc — allocate local/shared memory buffer
# =============================================================================


@tl.builtin
def alloc(
    shape: tuple,
    dtype: tl.dtype,
    layout=None,
    scope: tle_types.scope = None,
    _semantic=None,
    _generator=None,
) -> buffered_tensor:
    """Allocate a local/shared memory buffer on XPU.

    Args:
        shape: Buffer dimensions, e.g. [XBLOCK, YBLOCK]
        dtype: Element type, e.g. tl.float32
        layout: Memory layout (None = default swizzled layout)
        scope: tle.gpu.lmem (default for XPU)

    Returns:
        buffered_tensor handle for use with copy/local_ptr

    Example:
        a_lmem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem)
    """
    if scope is None:
        scope = lmem

    if not isinstance(shape, (tuple, list)):
        if hasattr(shape, '__iter__'):
            shape = tuple(shape)
        else:
            raise ValueError(f"shape must be tuple/list, got {type(shape)}")

    layout = tl._unwrap_if_constexpr(layout)
    if layout is not None and not isinstance(layout, shared_layout):
        if hasattr(layout, 'value') and layout.value is None:
            layout = None
        else:
            raise ValueError(f"layout must be shared_layout or None, got {type(layout)}")

    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    dtype = tl._unwrap_if_constexpr(dtype)

    if layout is None:
        layout = swizzled_shared_layout.make_default(rank=len(unwrapped_shape))

    elem_type = dtype.to_ir(_semantic.builder)

    # GluonOpBuilder has get_swizzled_shared_layout; standard ir.builder does not.
    # For XPU TLE path using ir.builder, we need to get the GluonOpBuilder from the
    # same context, or use an alternative approach.
    builder = _semantic.builder
    if not hasattr(builder, 'get_swizzled_shared_layout'):
        # Wrap the ir.builder context in a GluonOpBuilder for memdesc construction
        from triton._C.libtriton import gluon_ir
        # Get context from code_generator
        context = _generator.context if _generator is not None else None
        if context is None:
            raise RuntimeError("Cannot access MLIRContext from builder. "
                               "Ensure TLE kernel uses @triton.jit decorator.")
        gluon_builder = gluon_ir.GluonOpBuilder(context)
        # Sync insertion point: set the gluon builder to same block/position
        gluon_builder.set_insertion_point_to_end(builder.get_insertion_block())
        layout_ir = layout.to_ir(gluon_builder)
        memdesc_ty = gluon_builder.get_shared_mem_desc_ty(elem_type, unwrapped_shape, layout_ir, unwrapped_shape)
        alloc_handle = gluon_builder.create_local_alloc(memdesc_ty)
    else:
        layout_ir = layout.to_ir(builder)
        memdesc_ty = builder.get_shared_mem_desc_ty(elem_type, unwrapped_shape, layout_ir, unwrapped_shape)
        alloc_handle = builder.create_local_alloc(memdesc_ty)

    return buffered_tensor(alloc_handle, dtype, unwrapped_shape, scope, layout)


# =============================================================================
# copy — GM <-> LM data transfer (TMA descriptor path)
# =============================================================================


class CopyDirection(Enum):
    GM_TO_LOCAL = "GMTOLOCAL"
    LOCAL_TO_GM = "LOCALTOGM"


@tl.builtin
def copy(
    src,
    dst,
    shape,
    offsets: Sequence = None,
    _semantic=None,
    _generator=None,
) -> None:
    """Copy data between global memory and local memory via TMA descriptor.

    Direction is auto-detected from operand types:
    - src=tensor_descriptor, dst=buffered_tensor -> GM_TO_LOCAL
    - src=buffered_tensor, dst=tensor_descriptor -> LOCAL_TO_GM

    Args:
        src: tensor_descriptor or buffered_tensor
        dst: buffered_tensor or tensor_descriptor
        shape: Block dimensions to copy, e.g. [XBLOCK, YBLOCK]
        offsets: Coordinates in global tensor, e.g. [pid * XBLOCK, yoff]

    Example:
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [pid * XBLOCK, yoff])
        tle.gpu.copy(c_lmem, c_desc, [XBLOCK, YBLOCK], [pid * XBLOCK, yoff])
    """
    # Auto-detect direction
    src_is_desc = isinstance(src, (tl.tensor_descriptor, tl.tensor_descriptor_base))
    dst_is_desc = isinstance(dst, (tl.tensor_descriptor, tl.tensor_descriptor_base))
    src_is_buf = isinstance(src, buffered_tensor)
    dst_is_buf = isinstance(dst, buffered_tensor)

    def _is_ptr_tensor(x):
        # A tensor of GM pointers (e.g. `in_ptr0 + addr`): not a descriptor,
        # not an lmem buffer, but a tl.tensor whose scalar type is a pointer.
        return isinstance(x, tl.tensor) and x.type.scalar.is_ptr()

    # ---- Non-descriptor (permuted gather/scatter) normcopy path ----------
    # When one side is a raw GM pointer tensor and the other an lmem buffer,
    # the access is a data-dependent / permuted gather that a TensorDescriptor
    # cannot express. Lower it to an element-wise gather (g2l) / scatter (l2g)
    # that matches the tle_local_ptr per-core layout.
    def _emit_normcopy(fn_name, a_handle, b_handle):
        builder = _semantic.builder
        if not hasattr(builder, fn_name):
            from triton._C.libtriton import gluon_ir
            context = _generator.context if _generator is not None else None
            if context is None:
                raise RuntimeError("Cannot access MLIRContext from builder. "
                                   "Ensure TLE kernel uses @triton.jit decorator.")
            gluon_builder = gluon_ir.GluonOpBuilder(context)
            gluon_builder.restore_insertion_point(builder.get_insertion_point())
            getattr(gluon_builder, fn_name)(a_handle, b_handle)
        else:
            getattr(builder, fn_name)(a_handle, b_handle)

    if _is_ptr_tensor(src) and dst_is_buf:
        _emit_normcopy("create_xpu_normcopy_global_to_local", src.handle, dst.handle)
        return
    if src_is_buf and _is_ptr_tensor(dst):
        _emit_normcopy("create_xpu_normcopy_local_to_global", src.handle, dst.handle)
        return

    if src_is_desc and dst_is_buf:
        direction = CopyDirection.GM_TO_LOCAL
        desc = src
        buf = dst
    elif src_is_buf and dst_is_desc:
        direction = CopyDirection.LOCAL_TO_GM
        desc = dst
        buf = src
    else:
        raise ValueError(f"Invalid copy operands: src={type(src).__name__}, dst={type(dst).__name__}. "
                         "One must be tensor_descriptor and the other buffered_tensor.")

    if not isinstance(shape, (tuple, list)):
        shape = tuple(shape) if hasattr(shape, '__iter__') else [shape]

    if offsets is None:
        raise ValueError("offsets is required for TMA copy (descriptor path)")

    if not isinstance(offsets, (tuple, list)):
        offsets = tuple(offsets) if hasattr(offsets, '__iter__') else [offsets]

    # Convert offsets to IR values
    ir_offsets = []
    for off in offsets:
        off = tl._unwrap_if_constexpr(off)
        if isinstance(off, tl.tensor):
            ir_offsets.append(off.handle)
        elif isinstance(off, int):
            # Create i32 constant
            i32_ty = _semantic.builder.get_int32_ty()
            ir_offsets.append(_semantic.builder.get_int32(off))
        else:
            ir_offsets.append(off.handle if hasattr(off, 'handle') else off)

    # Collect stride IR handles from the descriptor.
    # tensor_descriptor has .strides (tuple of tl.tensor with i64 handles),
    # tensor_descriptor_base does not — fall back to empty list.
    ir_strides = []
    if hasattr(desc, 'strides') and desc.strides:
        for s in desc.strides:
            if hasattr(s, 'handle'):
                ir_strides.append(s.handle)
            else:
                ir_strides.append(s)

    # Collect shape IR handles from the descriptor (the tensor's REAL shape).
    # These are needed so the lowering can clamp tail-block copies to the real
    # tensor extent per dimension (row-tail and col-tail), instead of writing
    # past the logical shape into physical padding.
    ir_shapes = []
    desc_shape = getattr(desc, 'shape', None)
    if desc_shape:
        for d in desc_shape:
            d = tl._unwrap_if_constexpr(d)
            if isinstance(d, tl.tensor):
                ir_shapes.append(d.handle)
            elif isinstance(d, int):
                ir_shapes.append(_semantic.builder.get_int32(d))
            elif hasattr(d, 'handle'):
                ir_shapes.append(d.handle)

    # Emit TMA copy IR
    builder = _semantic.builder
    if not hasattr(builder, 'create_xpu_copy_global_to_local'):
        from triton._C.libtriton import gluon_ir
        context = _generator.context if _generator is not None else None
        if context is None:
            raise RuntimeError("Cannot access MLIRContext from builder. "
                               "Ensure TLE kernel uses @triton.jit decorator.")
        gluon_builder = gluon_ir.GluonOpBuilder(context)
        # Must insert at current builder position, not block end (yield may follow)
        gluon_builder.restore_insertion_point(builder.get_insertion_point())
        if direction == CopyDirection.GM_TO_LOCAL:
            gluon_builder.create_xpu_copy_global_to_local(desc.handle, ir_offsets, buf.handle, ir_strides, ir_shapes)
        else:
            gluon_builder.create_xpu_copy_local_to_global(desc.handle, ir_offsets, buf.handle, ir_strides, ir_shapes)
    else:
        if direction == CopyDirection.GM_TO_LOCAL:
            builder.create_xpu_copy_global_to_local(desc.handle, ir_offsets, buf.handle, ir_strides, ir_shapes)
        else:
            builder.create_xpu_copy_local_to_global(desc.handle, ir_offsets, buf.handle, ir_strides, ir_shapes)


# =============================================================================
# local_load / local_store — direct memdesc-based load/store (no pointer needed)
# =============================================================================


@tl.builtin
def local_load(buffer: buffered_tensor, _semantic=None) -> tl.tensor:
    """Load all data from a local memory buffer.

    Uses triton_3.6 native create_local_load (memdesc → tensor).

    Example:
        a_val = tle.gpu.local_load(a_lmem)
    """
    if not isinstance(buffer, buffered_tensor):
        raise ValueError(f"buffer must be buffered_tensor, got {type(buffer)}")

    ret_ty = tl.block_type(buffer.dtype, list(buffer.shape))
    load_handle = _semantic.builder.create_local_load(ret_ty.to_ir(_semantic.builder), buffer.handle)
    return tl.tensor(load_handle, ret_ty)


@tl.builtin
def local_store(buffer: buffered_tensor, value: tl.tensor, _semantic=None) -> None:
    """Store tensor data into a local memory buffer.

    Uses triton_3.6 native create_local_store (tensor → memdesc).

    Example:
        tle.gpu.local_store(c_lmem, c_val)
    """
    if not isinstance(buffer, buffered_tensor):
        raise ValueError(f"buffer must be buffered_tensor, got {type(buffer)}")
    _semantic.builder.create_local_store(buffer.handle, value.handle)


# =============================================================================
# local_ptr — materialize pointers from buffered_tensor + indices
# =============================================================================


@tl.builtin
def local_ptr(
    buffer: buffered_tensor,
    indices=None,
    _semantic=None,
    _generator=None,
) -> tl.tensor:
    """Materialize local memory pointers from buffer + index tensors.

    Converts a buffered_tensor + multi-dimensional indices into a pointer tensor
    that can be used with tl.load/tl.store.

    Args:
        buffer: buffered_tensor from tle.gpu.alloc
        indices: Tuple of index tensors (row_ids, col_ids, ...)

    Returns:
        Pointer tensor: tensor<[shape] x ptr<dtype, addr_space>>

    Example:
        row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
        col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
        ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
        val = tl.load(ptrs)

    NOTE: This requires `create_local_pointers` in the C++ builder.
          If not available, use local_load/local_store as alternative.
    """
    if not isinstance(buffer, buffered_tensor):
        raise ValueError(f"buffer must be buffered_tensor, got {type(buffer)}")

    buffer_shape = list(buffer.shape)
    indices = tl._unwrap_if_constexpr(indices)

    if indices is None:
        raise ValueError("indices is required for local_ptr")

    if isinstance(indices, tl.tuple):
        indices_tuple = tuple(indices.values)
    elif isinstance(indices, (tuple, list)):
        indices_tuple = tuple(indices)
    else:
        raise ValueError("indices must be a tuple/list of tensors")

    if len(indices_tuple) != len(buffer_shape):
        raise ValueError(f"indices rank ({len(indices_tuple)}) must match buffer rank ({len(buffer_shape)})")

    # Convert indices to IR tensor handles
    idx_handles = []
    view_shape = None
    for idx in indices_tuple:
        if isinstance(idx, tl.tensor):
            idx_handles.append(idx.handle)
            if idx.type.is_block() and view_shape is None:
                view_shape = list(idx.shape)
        else:
            idx_t = _semantic.to_tensor(idx)
            idx_handles.append(idx_t.handle)
            if idx_t.type.is_block() and view_shape is None:
                view_shape = list(idx_t.shape)

    if view_shape is None:
        view_shape = buffer_shape

    # Determine address space: lmem=0
    # Per P800 design doc: address space 0 = local memory pointer
    addr_space = 0 if buffer.type.storage is lmem else 3

    # Result type: tensor<[view_shape] x ptr<elem_ty, addr_space>>
    ptr_scalar_ty = tl.pointer_type(buffer.dtype, addr_space)
    if view_shape:
        result_ty = tl.block_type(ptr_scalar_ty, view_shape)
    else:
        result_ty = ptr_scalar_ty

    result_ir = result_ty.to_ir(_semantic.builder)

    # Call C++ builder method via GluonOpBuilder (create_local_pointers is on GluonOpBuilder)
    builder = _semantic.builder
    if not hasattr(builder, 'create_local_pointers'):
        from triton._C.libtriton import gluon_ir
        context = _generator.context if _generator is not None else None
        if context is None:
            raise RuntimeError("Cannot access MLIRContext from builder. "
                               "Ensure TLE kernel uses @triton.jit decorator.")
        gluon_builder = gluon_ir.GluonOpBuilder(context)
        # Insert at the current codegen position (NOT block end): when local_ptr is
        # called inside an scf.for body, set_insertion_point_to_end would place the
        # op AFTER the loop's scf.yield terminator ("scf.yield must be last op").
        # restore_insertion_point mirrors what copy() does and is correct both at
        # function top-level and inside loop bodies.
        gluon_builder.restore_insertion_point(builder.get_insertion_point())
        local_ptr_val = gluon_builder.create_local_pointers(result_ir, buffer.handle, *idx_handles)
    else:
        local_ptr_val = builder.create_local_pointers(result_ir, buffer.handle, *idx_handles)

    return tl.tensor(local_ptr_val, result_ty)
