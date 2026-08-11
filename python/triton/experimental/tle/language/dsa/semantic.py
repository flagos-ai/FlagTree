# Copyright 2026- Xcoresigma Technology Co., Ltd

from typing import List, Union
from triton.language import core as tl
from triton.language import semantic as tl_semantic
from triton._C.libtriton import ir
from .types import buffer, buffer_type, address_space


def wrap_tensor(x, scalar_ty, ret_shape):
    if ret_shape:
        res_ty = tl.block_type(scalar_ty, ret_shape)
    else:
        # 0d-tensor -> scalar
        res_ty = scalar_ty
    return tl.tensor(x, res_ty)


def scalar_constant(value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    # assert value.numel.value == 1, "only accepts size-1 tensor"
    if isinstance(value, tl.constexpr):
        value = builder.get_int32(value)
        return tl.tensor(value, dtype)

    if isinstance(value, int):
        value = builder.get_int32(value)
        return tl.tensor(value, dtype)

    if value.dtype.is_int():
        return tl.tensor(value.handle, dtype)


def _binary_op_type_checking(input: tl.tensor, other: tl.tensor, builder: ir.builder):
    semantic = tl_semantic.TritonSemantic(builder)
    return semantic.binary_op_type_checking_impl(input, other, True, True)


def copy(src, dst, shape: List[Union[tl.constexpr, int]], inter_no_alias: bool, builder: ir.builder):
    """
    Generate a unified TileIR copy op.

    The CommonIR POC needs TLE DSA and future GPGPU paths to share one IR
    vocabulary. Therefore the public tle.dsa.copy frontend emits tile.copy
    instead of the older TLE-specific dsa_copy op.
    """
    tile_copy(src, dst, shape, inter_no_alias, builder)


def _tile_buffer_binary_op(input: buffer, other: buffer, result: buffer, op_name: str, builder: ir.builder):
    lhs = tile_to_tensor(input, False, builder)
    rhs = tile_to_tensor(other, False, builder)

    if op_name == "add":
        value = tl_semantic.add(lhs, rhs, True, builder)
    elif op_name == "sub":
        value = tl_semantic.sub(lhs, rhs, True, builder)
    elif op_name == "mul":
        value = tl_semantic.mul(lhs, rhs, True, builder)
    elif op_name == "div":
        value = tl_semantic.truediv(lhs, rhs, builder)
    elif op_name == "max":
        value = tl_semantic.maximum(lhs, rhs, tl.PropagateNan.NONE, builder)
    elif op_name == "min":
        value = tl_semantic.minimum(lhs, rhs, tl.PropagateNan.NONE, builder)
    else:
        raise ValueError(f"unsupported tile buffer binary op: {op_name}")

    tile_store_tensor(value, result, builder)


def add(input: buffer, other: buffer, result: buffer, builder: ir.builder):
    _tile_buffer_binary_op(input, other, result, "add", builder)


def sub(input: buffer, other: buffer, result: buffer, builder: ir.builder):
    _tile_buffer_binary_op(input, other, result, "sub", builder)


def mul(input: buffer, other: buffer, result: buffer, builder: ir.builder):
    _tile_buffer_binary_op(input, other, result, "mul", builder)


def div(input: buffer, other: buffer, result: buffer, builder: ir.builder):
    _tile_buffer_binary_op(input, other, result, "div", builder)


def max(input: buffer, other: buffer, result: buffer, builder: ir.builder):
    _tile_buffer_binary_op(input, other, result, "max", builder)


def min(input: buffer, other: buffer, result: buffer, builder: ir.builder):
    _tile_buffer_binary_op(input, other, result, "min", builder)


def alloc(etype: tl.dtype, shape: List[tl.constexpr], address_space: address_space, builder: ir.builder) -> buffer:
    """Allocate a unified TileIR buffer for the public tle.dsa.alloc API."""
    return tile_alloc(etype, shape, address_space, builder)


def to_buffer(
    tensor: tl.tensor,
    address_space: address_space,
    bind_buffer: buffer,
    builder: ir.builder,
) -> buffer:
    """Convert a ranked tensor to a unified TileIR buffer."""
    return tile_to_buffer(tensor, address_space, bind_buffer, builder)


def to_tensor(memref: buffer, writable: bool, builder: ir.builder, target_shape=None) -> tl.tensor:
    """Convert a unified TileIR buffer back to a ranked tensor."""
    return tile_to_tensor(memref, writable, builder, target_shape=target_shape)


def insert_slice(ful: tl.tensor, sub: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int],
                 builder: ir.builder) -> tl.tensor:
    sizes_int = tl._unwrap_shape(sizes)
    strides_int = tl._unwrap_shape(strides)
    assert (len(ful.shape) == len(offsets))
    assert (len(ful.shape) == len(sizes_int))
    assert (len(ful.shape) == len(strides_int))
    assert (all([s >= 1 for s in sizes_int]))
    assert (all([s >= 0 for s in strides_int]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = builder.create_dsa_insert_slice(ful.handle, sub.handle, new_offsets, sizes_int, strides_int)
    return tl.tensor(out, ret_type)


def extract_slice(ful: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int],
                  builder: ir.builder) -> tl.tensor:
    sizes_int = tl._unwrap_shape(sizes)
    strides_int = tl._unwrap_shape(strides)
    assert (len(ful.shape) == len(offsets))
    assert (len(ful.shape) == len(sizes_int))
    assert (len(ful.shape) == len(strides_int))
    assert (all([s >= 1 for s in sizes_int]))
    assert (all([s >= 0 for s in strides_int]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes_int)
    out = builder.create_dsa_extract_slice(ful.handle, new_offsets, sizes_int, strides_int)
    return tl.tensor(out, ret_type)


def extract_element(src: tl.tensor, indice: List[tl.tensor], builder: ir.builder):
    if len(src.shape) != len(indice):
        raise ValueError("Indice's rank must be equal to src tensor's rank")

    new_indice = [i.handle for i in indice]
    result = builder.create_dsa_extract_scalar(src.handle, new_indice)
    return wrap_tensor(result, src.type.scalar, None)


def subview(src: buffer, offsets: List[tl.tensor], sizes: List[tl.constexpr], strides: List[tl.constexpr],
            builder: ir.builder) -> buffer:
    """Extract a subview using unified TileIR for the public tle.dsa.subview API."""
    return tile_subview(src, offsets, sizes, strides, builder)


# ==============================================================================
# TileIR semantic functions — emit tile.* ops
# ==============================================================================


def tile_alloc(etype: tl.dtype, shape: List[tl.constexpr], address_space: address_space, builder: ir.builder) -> buffer:
    """Allocate a buffer using tile.alloc."""
    shape = tl._unwrap_shape(shape)
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be list/tuple")
    etype = tl._unwrap_if_constexpr(etype)
    address_space = tl._unwrap_if_constexpr(address_space)
    element_ty_ir = etype.to_ir(builder)
    addr_space_attr = (address_space.to_ir(builder) if address_space else builder.dsa_get_null_attr())
    tile_buf_ty = builder.tile_get_buffer_type(shape, element_ty_ir, addr_space_attr)
    handle = builder.create_tile_alloc(tile_buf_ty)
    buffer_ty = buffer_type(element_ty=etype, shape=shape, space=address_space)
    return buffer(handle, buffer_ty)


def tile_copy(src, dst, shape: List[Union[tl.constexpr, int]], inter_no_alias: bool, builder: ir.builder):
    """Copy data using tile.copy."""
    shape = [scalar_constant(x, tl.int32, builder) for x in shape]
    builder.create_tile_copy(src.handle, dst.handle, [s.handle for s in shape], inter_no_alias)


def tile_store_tensor(tensor: tl.tensor, dst: buffer, builder: ir.builder):
    """Store a ranked tensor value back into a TileIR buffer."""
    if not isinstance(dst, buffer):
        raise TypeError("dst must be a buffer")
    builder.create_tile_store_tensor(tensor.handle, dst.handle)


def tile_to_buffer(
    tensor: tl.tensor,
    address_space: address_space,
    bind_buffer: buffer,
    builder: ir.builder,
) -> buffer:
    """Convert a ranked tensor to a TileIR buffer using tile.store_tensor."""
    if not isinstance(tensor.shape, (tl.tuple, tuple, list)) or not tensor.shape:
        raise TypeError("scalar type cannot be converted to buffer")

    shape = tl._unwrap_shape(tensor.shape)
    if bind_buffer is not None:
        if not isinstance(bind_buffer, buffer):
            raise TypeError("bind_buffer must be a buffer or None")
        if bind_buffer.shape != list(shape):
            raise ValueError(f"bind_buffer shape {bind_buffer.shape} does not match tensor shape {list(shape)}")
        dst = bind_buffer
    else:
        address_space = tl._unwrap_if_constexpr(address_space)
        dst = tile_alloc(tensor.dtype, shape, address_space, builder)

    tile_store_tensor(tensor, dst, builder)
    return dst


def tile_subview(src: buffer, offsets: List[tl.tensor], sizes: List[tl.constexpr], strides: List[tl.constexpr],
                 builder: ir.builder) -> buffer:
    """Extract subview using tile.subview."""
    new_offsets = [offset.handle for offset in offsets]
    sizes_int = tl._unwrap_shape(sizes)
    strides_int = tl._unwrap_shape(strides)

    result_handle = builder.create_tile_subview(src.handle, new_offsets, sizes_int, strides_int)

    if src.strides:
        src_memory_strides = src.strides
    else:
        src_memory_strides = []
        stride = 1
        for dim_size in reversed(src.shape):
            if dim_size < 0:
                raise ValueError("Cannot compute strides for buffer with dynamic dimensions")
            src_memory_strides.insert(0, stride)
            stride *= dim_size

    result_memory_strides = []
    for src_stride, subview_stride in zip(src_memory_strides, strides_int):
        result_memory_strides.append(src_stride * subview_stride)

    buffer_ty = buffer_type(element_ty=src.dtype, shape=sizes_int, space=src.space, strides=result_memory_strides)
    return buffer(result_handle, buffer_ty)


def tile_to_tensor(memref: buffer, writable: bool, builder: ir.builder, target_shape=None) -> tl.tensor:
    """Convert buffer to tensor view using tile.to_tensor."""
    if not isinstance(memref, buffer):
        raise TypeError("memref must be buffer")

    shape = memref.shape
    if target_shape:
        shape = tl._unwrap_shape(target_shape)
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be list/tuple")
    tensor_type = tl.block_type(memref.dtype, shape)

    handle = builder.create_tile_to_tensor(memref.handle, writable)
    return tl.tensor(handle, tensor_type)


def set_flag(producer_pipe, consumer_pipe, event_id, builder: ir.builder):
    """Cross-engine set_flag using tile.set_flag."""
    builder.create_tile_set_flag(int(producer_pipe), int(consumer_pipe), int(event_id))


def wait_flag(producer_pipe, consumer_pipe, event_id, builder: ir.builder):
    """Cross-engine wait_flag using tile.wait_flag."""
    builder.create_tile_wait_flag(int(producer_pipe), int(consumer_pipe), int(event_id))


def pipe_barrier(pipe, builder: ir.builder):
    """Intra-engine pipe_barrier using tile.pipe_barrier."""
    builder.create_tile_pipe_barrier(int(pipe))


def tensor_to_tile(src: tl.tensor, space: address_space, builder: ir.builder) -> buffer:
    """Convert a tt.ptr/tensor to a !tile.buf.

    Wraps the tensor handle as a buffer — the actual data copy is done
    by tile_copy.  If space is given, allocates a new buffer first.
    """
    shape = getattr(src, 'shape', None)
    if not shape or not isinstance(shape, (tuple, list)):
        # Block pointer / scalar: wrap as-is without shape
        buffer_ty = buffer_type(element_ty=src.dtype, shape=[], space=space)
        return buffer(src.handle, buffer_ty)

    if space is not None:
        space = tl._unwrap_if_constexpr(space)
        addr_space_attr = space.to_ir(builder)
        element_ty_ir = src.dtype.to_ir(builder)
        shape = list(shape)
        tile_buf_ty = builder.tile_get_buffer_type(shape, element_ty_ir, addr_space_attr)
        buf_handle = builder.create_tile_alloc(tile_buf_ty)
        builder.create_tile_copy(src.handle, buf_handle, [], False)
        buffer_ty = buffer_type(element_ty=src.dtype, shape=shape, space=space)
        return buffer(buf_handle, buffer_ty)
    else:
        shape = list(shape)
        buffer_ty = buffer_type(element_ty=src.dtype, shape=shape, space=None)
        return buffer(src.handle, buffer_ty)


def tile_gm_offset(base, indices: List[tl.tensor], strides: List[tl.tensor], builder: ir.builder) -> tl.tensor:
    """Compute GM offset using tile.gm_offset."""
    idx_handles = [i.handle for i in indices]
    stride_handles = [s.handle for s in strides]
    handle = builder.create_tile_gm_offset(base.handle, idx_handles, stride_handles)
    return tl.tensor(handle, base.type)


def tile_cube_launch(a: buffer, b: buffer, acc: buffer, stage_a: buffer, stage_b: buffer, dst, transpose_a: bool,
                     transpose_b: bool, init: bool, mma: str, builder: ir.builder):
    """Launch Cube work using tile.cube_launch."""
    for name, value in [
        ("a", a),
        ("b", b),
        ("acc", acc),
        ("stage_a", stage_a),
        ("stage_b", stage_b),
    ]:
        if not isinstance(value, buffer):
            raise TypeError(f"{name} must be a buffer")
    builder.create_tile_cube_launch(
        a.handle,
        b.handle,
        acc.handle,
        stage_a.handle,
        stage_b.handle,
        dst.handle,
        bool(transpose_a),
        bool(transpose_b),
        bool(init),
        str(mma) if mma is not None else "",
    )


def tile_cube_wait(builder: ir.builder):
    """Wait for Cube work using tile.cube_wait."""
    builder.create_tile_cube_wait()
