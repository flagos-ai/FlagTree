# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.

from typing import List, Optional, Union, Tuple
from triton.language import core as tl
from triton.language.semantic import (
    binary_op_type_checking_impl,
)
from triton._C.libtriton import ir
from .types import buffer, buffer_type, address_space

def scalar_constant(value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    # assert value.numel.value == 1, "only accepts size-1 tensor"
    if isinstance(value, tl.constexpr):
        value = builder.get_int32(value)
        return tl.tensor(value, dtype)

    if value.dtype.is_int():
        return tl.tensor(value.handle, dtype)

### def alloc(shape: List[tl.tensor], dtype: tl.dtype, layout, scope, builder: ir.builder) -> tl.tensor:
###     ret_ty = tl.block_type(dtype, shape)
###     return tl.tensor(builder.create_dsa_alloc(shape, str(layout), str(scope),
###                                           dtype.to_ir(builder)), ret_ty)


def copy(src, dst, shape: List[Union[tl.constexpr, int]], builder: ir.builder):
    """
    Generate tt.copy(src, dst, shape) and return dst-like tensor.
    Lowering to hivm.load/hivm.store is done in MLIR pass.
    """
    shape = [scalar_constant(x, tl.int32, builder) for x in shape]
    builder.create_dsa_copy(src.handle, dst.handle, [s.handle for s in shape])


### def to_tensor(buffer: tl.tensor, builder: ir.builder) -> tl.tensor:
###     if not isinstance(buffer, tl.tensor):
###         raise TypeError("buffer must be tensor of pointers")
### 
###     tensor_ty = buffer.type
###     element_ty = tensor_ty.element_ty
###     if not element_ty.is_ptr:
###         raise TypeError("The basic elements of a buffer must be pointers")
### 
###     return tl.tensor(builder.dsa_to_tensor(buffer.handle), tensor_ty)
### 
### def to_buffer(src: tl.tensor, builder: ir.builder) -> tl.tensor:
###     if not isinstance(src, tl.tensor):
###         raise TypeError("src of to_buffer must be tensor")
### 
###     return tl.tensor(builder.dsa_to_buffer(src.handle), src.type)

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

### def dot(inputA: tl.tensor, inputB: tl.tensor, result: tl.tensor, size: List[int], initC: bool, a_transpose: bool, b_transpose: bool, enable_hf32: bool, builder: ir.builder):
###     assert len(size) == 3, f"Please set the M、N、K value."
### 
###     builder.create_dsa_dot(inputA.handle, inputB.handle, result.handle, size, initC, a_transpose, b_transpose, enable_hf32)

def alloc(etype: tl.dtype, shape: List[tl.constexpr], address_space: address_space,
          builder: ir.builder) -> buffer:
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
    if not (bind_buffer is None):
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