# Copyright 2026- Xcoresigma Technology Co., Ltd

from typing import List, Union
from triton.language import core as tl
from triton.language.semantic import (
    binary_op_type_checking_impl, )
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

    if value.dtype.is_int():
        return tl.tensor(value.handle, dtype)


def copy(src, dst, shape: List[Union[tl.constexpr, int]], inter_no_alias: bool, builder: ir.builder):
    """
    Generate tt.copy(src, dst, shape) and return dst-like tensor.
    Lowering to hivm.load/hivm.store is done in MLIR pass.
    """
    shape = [scalar_constant(x, tl.int32, builder) for x in shape]
    builder.create_dsa_copy(src.handle, dst.handle, [s.handle for s in shape], inter_no_alias)


def add(input: tl.tensor, other: tl.tensor, result: tl.tensor, builder: ir.builder):
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    builder.create_dsa_add(input.handle, other.handle, result.handle)


def sub(input: tl.tensor, other: tl.tensor, result: tl.tensor, builder: ir.builder):
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    builder.create_dsa_sub(input.handle, other.handle, result.handle)


def mul(input: tl.tensor, other: tl.tensor, result: tl.tensor, builder: ir.builder):
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    builder.create_dsa_mul(input.handle, other.handle, result.handle)


def div(input: tl.tensor, other: tl.tensor, result: tl.tensor, builder: ir.builder):
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    builder.create_dsa_div(input.handle, other.handle, result.handle)


def max(input: tl.tensor, other: tl.tensor, result: tl.tensor, builder: ir.builder):
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    builder.create_dsa_max(input.handle, result.handle)


def min(input: tl.tensor, other: tl.tensor, result: tl.tensor, builder: ir.builder):
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    builder.create_dsa_min(input.handle, other.handle, result.handle)


def alloc(etype: tl.dtype, shape: List[tl.constexpr], address_space: address_space, builder: ir.builder) -> buffer:
    shape = tl._unwrap_shape(shape)
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be list/tuple")
    etype = tl._constexpr_to_value(etype)
    address_space = tl._constexpr_to_value(address_space)
    element_ty_ir = etype.to_ir(builder)
    addr_space_attr = (address_space.to_ir(builder) if address_space else builder.dsa_get_null_attr())
    memref_ty = builder.dsa_get_buffer_type(shape, element_ty_ir, addr_space_attr)
    handle = builder.create_dsa_alloc(memref_ty)
    buffer_ty = buffer_type(element_ty=etype, shape=shape, space=address_space)
    return buffer(handle, buffer_ty)


def to_buffer(
    tensor: tl.tensor,
    address_space: address_space,
    bind_buffer: buffer,
    builder: ir.builder,
) -> buffer:
    if not isinstance(tensor.shape, (tuple, list)) or not tensor.shape:
        raise TypeError("scalar type cannot be converted to buffer")
    # if isinstance(bind_buffer, buffer):
    #     builder.create_bind_buffer(tensor.handle, bind_buffer.handle)
    #     return bind_buffer
    if bind_buffer is not None:
        raise ValueError("bind_buffer must be a buffer or None")
    address_space = tl._constexpr_to_value(address_space)
    addr_space_attr = (address_space.to_ir(builder) if address_space else builder.dsa_get_null_attr())
    handle = builder.dsa_to_buffer(tensor.handle, addr_space_attr)
    buffer_ty = buffer_type(element_ty=tensor.dtype, shape=tensor.shape, space=address_space)
    return buffer(handle, buffer_ty)


def to_tensor(memref: buffer, writable: bool, builder: ir.builder, target_shape=None) -> tl.tensor:
    if not isinstance(memref, buffer):
        raise TypeError("memref must be buffer")

    need_convert_layout = False
    shape = memref.shape
    if target_shape:
        need_convert_layout = True
        shape = tl._unwrap_shape(target_shape)
        assert shape != memref.shape, "target shape is the same as source shape"
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be list/tuple")
    tensor_type = tl.block_type(memref.dtype, shape)

    memref_value = memref.handle
    if need_convert_layout:
        buffer_ty = buffer_type(
            element_ty=memref.dtype,
            shape=shape,
            space=memref.space,
        )
        memref_value = builder.create_convert_layout(memref_value, buffer_ty.to_ir(builder))

    return tl.tensor(builder.dsa_to_tensor(memref_value, writable), tensor_type)


def insert_slice(ful: tl.tensor, sub: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int],
                 builder: ir.builder) -> tl.tensor:
    assert (len(ful.shape) == len(offsets))
    assert (len(ful.shape) == len(sizes))
    assert (len(ful.shape) == len(strides))
    assert (all([s >= 1 for s in sizes]))
    assert (all([s >= 0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = builder.create_dsa_insert_slice(ful.handle, sub.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)


def extract_slice(ful: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int],
                  builder: ir.builder) -> tl.tensor:
    assert (len(ful.shape) == len(offsets))
    assert (len(ful.shape) == len(sizes))
    assert (len(ful.shape) == len(strides))
    assert (all([s >= 1 for s in sizes]))
    assert (all([s >= 0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes)
    out = builder.create_dsa_extract_slice(ful.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)


def extract_element(src: tl.tensor, indice: List[tl.tensor], builder: ir.builder):
    if len(src.shape) != len(indice):
        raise ValueError("Indice's rank must be equal to src tensor's rank")

    new_indice = [i.handle for i in indice]
    result = builder.create_dsa_extract_scalar(src.handle, new_indice)
    return wrap_tensor(result, src.type.scalar, None)


def subview(src: buffer, offsets: List[tl.tensor], sizes: List[tl.constexpr], strides: List[tl.constexpr],
            builder: ir.builder) -> buffer:

    new_offsets = [offset.handle for offset in offsets]
    sizes_int = tl._unwrap_shape(sizes)
    strides_int = tl._unwrap_shape(strides)

    result_handle = builder.create_dsa_subview(src.handle, new_offsets, sizes_int, strides_int)

    # calculate the memory layout strides of the source buffer
    if src.strides:
        # use the strides of the source buffer
        src_memory_strides = src.strides
    else:
        # calculate the default row-major strides
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

    # create buffer_type with strides
    buffer_ty = buffer_type(element_ty=src.dtype, shape=sizes_int, space=src.space, strides=result_memory_strides)
    return buffer(result_handle, buffer_ty)
