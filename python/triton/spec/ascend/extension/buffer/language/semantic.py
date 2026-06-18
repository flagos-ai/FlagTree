# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import (TypeVar, List)

from triton._C.libtriton import ir
import triton.language.core as tl

from . import core as bl

T = TypeVar('T')


def alloc(etype: tl.dtype, shape: List[tl.constexpr], address_space: bl.address_space, is_mem_unique,
          builder: ir.builder) -> bl.buffer:
    shape = tl._unwrap_shape(shape)
    if etype == tl.int1:
        raise TypeError("Unsupported alloc int1 type")
    if not isinstance(shape, (tl.tuple, list)):
        raise TypeError("shape must be list/tuple")
    etype = tl._unwrap_if_constexpr(etype)
    address_space = tl._unwrap_if_constexpr(address_space)
    element_ty_ir = etype.to_ir(builder)
    addr_space_attr = (address_space.to_ir(builder) if address_space else builder.get_null_attr())
    memref_ty = builder.get_buffer_ty(shape, element_ty_ir, addr_space_attr)
    handle = builder.alloc(memref_ty)
    if is_mem_unique:
        builder.create_annotation_mark(handle, "mem_unique", builder.get_unit_attr())
    builder.create_annotation_mark(handle, "effects", builder.get_str_array_attr(["write", "read"]))

    buffer_ty = bl.buffer_type(element_ty=etype, shape=shape, space=address_space)
    return bl.buffer(handle, buffer_ty)


def to_buffer(
    tensor: tl.tensor,
    address_space: bl.address_space,
    bind_buffer: bl.buffer,
    builder: ir.builder,
) -> bl.buffer:
    if not isinstance(tensor.shape, (tl.tuple, list)) or not tensor.shape:
        raise TypeError("scalar type cannot be converted to buffer")
    if isinstance(bind_buffer, bl.buffer):
        builder.create_bind_buffer(tensor.handle, bind_buffer.handle)
        return bind_buffer
    if bind_buffer is not None:
        raise ValueError("bind_buffer must be a buffer or None")
    address_space = tl._unwrap_if_constexpr(address_space)
    addr_space_attr = (address_space.to_ir(builder) if address_space else builder.get_null_attr())
    handle = builder.to_buffer(tensor.handle, addr_space_attr)
    buffer_ty = bl.buffer_type(element_ty=tensor.dtype, shape=tensor.shape, space=address_space)
    return bl.buffer(handle, buffer_ty)


def to_tensor(memref: bl.buffer, writable: bool, builder: ir.builder, target_shape=None) -> tl.tensor:
    if not isinstance(memref, bl.buffer):
        raise TypeError("memref must be bl.buffer")

    need_convert_layout = False
    shape = memref.shape
    if target_shape:
        need_convert_layout = True
        shape = tl._unwrap_shape(target_shape)
        assert shape != memref.shape, "target shape is the same as source shape"
    if not isinstance(shape, (tl.tuple, list)):
        raise TypeError("shape must be list/tuple")
    tensor_type = tl.block_type(memref.dtype, shape)

    memref_value = memref.handle
    if need_convert_layout:
        buffer_ty = bl.buffer_type(
            element_ty=memref.dtype,
            shape=shape,
            space=memref.space,
        )
        memref_value = builder.create_convert_layout(memref_value, buffer_ty.to_ir(builder))

    return tl.tensor(builder.to_tensor(memref_value, writable), tensor_type)


def subview(src: bl.buffer, offsets: List[tl.tensor], sizes: List[tl.constexpr], strides: List[tl.constexpr],
            builder: ir.builder) -> bl.buffer:

    new_offsets = [offset.handle for offset in offsets]
    sizes_int = tl._unwrap_shape(sizes)
    strides_int = tl._unwrap_shape(strides)

    result_handle = builder.subview(src.handle, new_offsets, sizes_int, strides_int)

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
    buffer_ty = bl.buffer_type(element_ty=src.dtype, shape=sizes_int, space=src.space, strides=result_memory_strides)
    return bl.buffer(result_handle, buffer_ty)
