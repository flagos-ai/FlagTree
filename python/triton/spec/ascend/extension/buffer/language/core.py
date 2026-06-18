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

__all__ = [
    "address_space",
    "buffer_type",
    "subview",
    "alloc",
    "buffer",
    "to_buffer",
    "to_tensor",
]

import importlib
from typing import TypeVar, List
from functools import wraps

from triton._C.libtriton import ir
import triton.language.core as tl

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
BUFFER_BUILTIN = "__buffer_builtin__"


def builtin(fn: T) -> T:
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    # also set triton_builtin to true so that CodeGenerator will recognize this function
    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, BUFFER_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:
    """Is this a registered buffer language builtin function?"""
    return getattr(fn, BUFFER_BUILTIN, False)


class address_space:
    """Represents a buffer's address space.

    The :code:`address_space` of a buffer is a target-specific attribute.
    """

    def to_ir(self, builder: ir.builder) -> ir.type:
        raise NotImplementedError("Abstract address_space cannot be converted to ir")


class buffer_type(tl.dtype):

    def __init__(self, element_ty: tl.dtype, shape: List, space: address_space = None, strides: List = None):
        self.element_ty = element_ty
        self.shape = shape if isinstance(shape, list) else list(shape)
        self.space = space
        self.strides = strides if strides is not None else []
        self.name = self._make_name()

    def _make_name(self):
        res = '<buffer ' + 'x'.join(str(s) for s in self.shape) + 'x' + str(self.element_ty)
        if self.strides:
            res += ', strides=[' + ', '.join(str(s) for s in self.strides) + ']'
        if self.space:
            res += ', ' + str(self.space)
        return res + '>'

    def to_ir(self, builder: ir.builder) -> ir.type:
        element_ty_ir = self.element_ty.to_ir(builder)
        addr_space_attr = self.space.to_ir(builder) if self.space else builder.get_null_attr()

        # use the method with strides if strides is not empty
        if self.strides:
            return builder.get_buffer_ty_with_strides(self.shape, element_ty_ir, self.strides, addr_space_attr)
        else:
            return builder.get_buffer_ty(self.shape, element_ty_ir, addr_space_attr)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, buffer_type):
            return False
        return (self.element_ty == other.element_ty and self.shape == other.shape and self.space == other.space
                and self.strides == other.strides)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self):
        return self.element_ty

    def mangle(self) -> str:
        elt = self.element_ty.mangle()
        shape = "_".join(map(str, self.shape))
        return f"B{elt}S{shape}S"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int):
        return buffer(handles[cursor], self), cursor + 1


# -----------------------
# buffer
# -----------------------


class buffer(tl.base_value):
    """Represents a region of memory.

    :code:`buffer` is the fundamental data structure for Triton programs using
    the buffer language extension. Most functions in
    :py:mod:`triton.extension.buffer.language` operate on and return buffers.

    Most of the named member functions here are duplicates of the free functions
    in :code:`triton.language`.  For example, :code:`triton.language.sqrt(x)` is
    equivalent to :code:`x.sqrt()`.

    .. rubric:: Constructors
    ..
       For some reason Sphinx includes __init__ before printing the full table
       of methods.  Not what I want, but I can't figure out how to fix it.  Give
       it its own section so it looks intentional. :)
    """

    def __init__(self, handle, buffer_ty: buffer_type):
        """Not called by user code."""
        super().__init__()
        self.handle = handle
        self.type = buffer_ty
        self.dtype = buffer_ty.element_ty.scalar
        self.shape = buffer_ty.shape
        self.space = buffer_ty.space
        self.strides = buffer_ty.strides

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    def __str__(self) -> str:
        # ex. "<16x32xfloat32, address_space>"
        res = '<' + 'x'.join(str(s) for s in self.shape) + 'x' + str(self.dtype)
        if self.space:
            res += ', ' + str(self.space)
        return res + '>'

    @builtin
    def subview(self, offsets: List[tl.constexpr], sizes: List[tl.constexpr], strides: List[tl.constexpr],
                _semantic=None) -> 'buffer':
        return subview(self, offsets, sizes, strides, _semantic=_semantic)

    @builtin
    def to_tensor(self, writable=True, target_shape=None, _semantic=None):
        """Convert this buffer to a tl.tensor"""
        return to_tensor(self, writable=writable, target_shape=target_shape, _semantic=_semantic)


semantic = importlib.import_module(".semantic", package=__package__)


@builtin
def alloc(etype: tl.dtype, shape: List[tl.constexpr], _address_space: address_space = None, is_mem_unique: bool = False,
          _semantic=None) -> buffer:
    """
    Allocates a region of local memory with the specified shape and type.

    :param etype: the element type of the buffer.
    :type etype: tl.dtype
    :param shape: A list of non-negative integers representing the shape of the buffer.
    :type shape: List[tl.constexpr]
    :param _address_space: (Optional) backend-specific local memory address space
    :type _address_space: bl.address_space
    """
    return semantic.alloc(etype, shape, _address_space, is_mem_unique, _semantic.builder)


@builtin
def to_buffer(tensor: tl.tensor, space: address_space = None, bind_buffer: buffer = None, _semantic=None) -> buffer:
    """
    Convert a tensor to a buffer.

    :param tensor: the tensor to convert.
    :type tensor: tl.tensor
    :param space: the address space for the buffer (optional).
    :type space: address_space
    """
    return semantic.to_buffer(tensor, space, bind_buffer, _semantic.builder)


@builtin
def to_tensor(memref: buffer, writable: bool = True, target_shape=None, _semantic=None) -> tl.tensor:
    """
    Create a tl.tensor from a bl.buffer.

    :param memref: the input bl.buffer object.
    :memref type: bl.buffer
    :param writable: If set true, the resultant tensor is considered "writable" during bufferization.
    :type writable: bool
    """
    return semantic.to_tensor(memref, writable, _semantic.builder, target_shape=target_shape)


def check_subview(src, offsets, sizes, strides):
    """
    Check data of subview methods which the data length and the offset value must be 32-byte aligned.

    The conditions for checking data are as follows:
    1. offset value must be 32-bytes aligned.
    2. all strides must be 1.
    3. the first point's offset in the second row of the last dimension must be 32-bytes aligned.

    For instance, the following example fails to satisfy the specified criteria.
        %subview = memref.subview %arg0[1, 1][4, 4][2, 2]
        : memref<8x8xf32, strided<[8, 1], offset: 0>> to
        memref<4x4xf32, strided<[16, 2], offset: 9>>
    offsets = [8, 8] | sizes = [4, 4] | strides = [2, 2]
    result_offset = 9
    second_row_start_offset = 25
    The scene will be go wrong because the follow conditions are not meet.
        1) result_offset is not 32-bytes aligned.
        2) strides = [2, 2], not all strides are equal to 1.
        3) second_row_start_offset are not 32-bytes aligned.
    """
    bytes_per_block = 32
    bits_per_byte = 8
    base_byte = bytes_per_block // (src.dtype.primitive_bitwidth // bits_per_byte)
    result_strides = []
    result_offset = 0
    second_row_start_offset = 0
    length = len(strides)
    src_strides = [1] * length
    if length == 1:
        if offset[0] % base_byte != 0:
            raise TypeError("all strides should be 1 and the offset value should be 32-bytes aligned.")
        return
    for i in range(length - 2, -1, -1):
        src_strides[i] = src_strides[i + 1] * src.shape[i + 1]
    for i in range(0, length):
        if isinstance(offsets[i], tl.tensor):
            return
        result_strides.append(src_strides[i] * strides[i])
        result_offset = result_offset + offsets[i] * src_strides[i]
    second_row_start_offset = result_offset + src_strides[-2] * strides[-2]
    is_unaligned = False
    if sizes[1] > 1:
        is_unaligned = second_row_start_offset % base_byte != 0
    stride_1 = all(s == 1 for s in strides)
    is_unaligned = result_offset % base_byte != 0 or is_unaligned or not stride_1
    if is_unaligned:
        raise TypeError("all strides should be 1 and the offset value should be 32-bytes aligned.")


@builtin
def subview(src: buffer, offsets: List[tl.constexpr], sizes: List[tl.constexpr], strides: List[tl.constexpr],
            _semantic=None) -> buffer:
    '''
    Creates a subview of the source buffer with the specified offsets, sizes, and strides.

    :param src: The source buffer to create a subview from.
    :type src: buffer
    :param offsets: A list of non-negative integers representing the offsets in each dimension.
    :type offsets: List[tl.constexpr]
    :param sizes: A list of non-negative integers representing the sizes in each dimension.
    :type sizes: List[tl.constexpr]
    :param strides: A list of non-negative integers representing the strides in each dimension.
    :type strides: List[tl.constexpr]
    :return: A new buffer representing the subview of the source buffer.
    :rtype: buffer
    '''
    # Validate that sizes and strides contain only constexpr values
    new_sizes = []
    for i, size in enumerate(sizes):
        if isinstance(size, int):
            # Convert regular integers to constexpr
            new_sizes.append(tl.constexpr(size))
        elif isinstance(size, tl.constexpr):
            new_sizes.append(size)
        else:
            raise TypeError(f"sizes[{i}] must be constexpr, got {type(size).__name__}")

    new_strides = []
    for i, stride in enumerate(strides):
        if isinstance(stride, int):
            # Convert regular integers to constexpr
            new_strides.append(tl.constexpr(stride))
        elif isinstance(stride, tl.constexpr):
            new_strides.append(stride)
        else:
            raise TypeError(f"strides[{i}] must be constexpr, got {type(stride).__name__}")

    check_offsets = []
    new_offsets = []
    for offset in offsets:
        if isinstance(offset, tl.constexpr):
            # Check that constexpr offset values cannot be negative
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(_semantic.to_tensor(offset))
            check_offsets.append(offset)
        elif isinstance(offset, int):
            # Convert regular integers to constexpr and then to tensor
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(_semantic.to_tensor(tl.constexpr(offset)))
            check_offsets.append(tl.constexpr(offset))
        else:
            # Assume it's already a tensor
            new_offsets.append(offset)
            check_offsets.append(offset)

    check_subview(src, check_offsets, new_sizes, new_strides)
    return semantic.subview(src, new_offsets, new_sizes, new_strides, _semantic.builder)
