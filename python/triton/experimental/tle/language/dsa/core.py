# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.

import triton.language.core as tl
from triton.language.core import (
    _constexpr_to_value,
    constexpr
)
from triton._C.libtriton import ir

from typing import List, TypeVar
from functools import wraps

from . import semantic as tle_semantic
from .types import address_space, buffer

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
TLE_BUILTIN = "__tle_builtin__"

def builtin(fn: T) -> T:
    """
    Decorator for builtin functions to mark a function as a tle language builtin function.
    """
    assert callable
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, TLE_BUILTIN, True)

    return wrapper

def is_builtin(fn) -> bool:
    """
    Returns whether a function is a builtin function.
    """
    return getattr(fn, TLE_BUILTIN, False)


class range():
    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.range(10, num_stages=3):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    :param num_stages: pipeline the loop into this many stages (so there are
        :code:`num_stages` iterations of the loop in flight at once).

        Note this is subtly different than passing :code:`num_stages` as a
        kernel argument.  The kernel argument only pipelines loads that feed
        into :code:`dot` operations, while this attribute tries to pipeline most
        (though not all) loads in this loop.
    :param loop_unroll_factor: Tells the Triton IR level loop unroller how many
        times to unroll a for loop that this range is used with. Less than 2 for
        this value implies no unrolling.
    :param disallow_acc_multi_buffer: If true, prevent the accumulator of the dot
        operation in the loop to be multi-buffered, if applicable.
    :param flatten: automatically flatten the loop nest starting at this loop to
        create a single flattened loop. The compiler will try to pipeline the
        flattened loop which can avoid stage stalling.
    :param warp_specialize: Enable automatic warp specialization on the loop.
        The compiler will attempt to partition memory, MMA, and vector
        operations in the loop into separate async partitions. This will
        increase the total number of warps required by the kernel.
    :param disable_licm: Tells the compiler it shouldn't hoist loop invariant
        code outside the loop. This is often useful to avoid creating long liveranges
        within a loop.

        Note that warp specialization is only supported on Blackwell GPUs and
        only works on simple matmul loops. Support for arbitrary loops will be
        expanded over time.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False):
        if step is None:
            self.step = constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages
        self.loop_unroll_factor = loop_unroll_factor
        self.disallow_acc_multi_buffer = disallow_acc_multi_buffer
        self.flatten = flatten
        self.warp_specialize = warp_specialize
        self.disable_licm = disable_licm

    def __iter__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

class pipeline(range):
    """
    Iterator that counts upward forever, with software pipeline semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    """
    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)


@builtin
def from_buffer_to_tensor_pointer(src: buffer, _builder=None) -> tl.tensor:
    buffer_ty = src.type
    ele_type = buffer_ty.element_ty
    shape = buffer_ty.shape
    block_type = tl.block_type(ele_type, shape)
    return tl.tensor(src.handle, block_type)

@builtin
def copy(src, dst, shape, _builder=None):
    assert len(shape) != 0, f"Can't deduce copy extents from args"

    shape = _constexpr_to_value(shape)
    tle_semantic.copy(src, dst, shape, _builder)


@builtin
def add(input, other, result, _builder=None):
    input = from_buffer_to_tensor_pointer(input, _builder=_builder)
    other = from_buffer_to_tensor_pointer(other, _builder=_builder)
    result = from_buffer_to_tensor_pointer(result, _builder=_builder)
    tle_semantic.add(input, other, result, _builder)

@builtin
def sub(input, other, result, _builder=None):
    input = from_buffer_to_tensor_pointer(input, _builder=_builder)
    other = from_buffer_to_tensor_pointer(other, _builder=_builder)
    result = from_buffer_to_tensor_pointer(result, _builder=_builder)
    tle_semantic.sub(input, other, result, _builder)

@builtin
def mul(input, other, result, _builder=None):
    input = from_buffer_to_tensor_pointer(input, _builder=_builder)
    other = from_buffer_to_tensor_pointer(other, _builder=_builder)
    result = from_buffer_to_tensor_pointer(result, _builder=_builder)
    tle_semantic.mul(input, other, result, _builder)

@builtin
def div(input, other, result, _builder=None):
    input = from_buffer_to_tensor_pointer(input, _builder=_builder)
    other = from_buffer_to_tensor_pointer(other, _builder=_builder)
    result = from_buffer_to_tensor_pointer(result, _builder=_builder)
    tle_semantic.div(input, other, result, _builder)

@builtin
def max(input, other, result, _builder=None):
    # elementwise binary vector maximum op
    input = from_buffer_to_tensor_pointer(input, _builder=_builder)
    other = from_buffer_to_tensor_pointer(other, _builder=_builder)
    result = from_buffer_to_tensor_pointer(result, _builder=_builder)
    tle_semantic.max(input, other, result, _builder)

@builtin
def min(input, other, result, _builder=None):
    # elementwise binary vector minimum op
    input = from_buffer_to_tensor_pointer(input, _builder=_builder)
    other = from_buffer_to_tensor_pointer(other, _builder=_builder)
    result = from_buffer_to_tensor_pointer(result, _builder=_builder)
    tle_semantic.min(input, other, result, _builder)

### @builtin
### def dot(inputA, inputB, result, size, initC, a_transpose=False, b_transpose=False, enable_hf32=False, _builder=None):
###     initC = _constexpr_to_value(initC)
###     a_transpose = _constexpr_to_value(a_transpose)
###     b_transpose = _constexpr_to_value(b_transpose)
###     enable_hf32 = _constexpr_to_value(enable_hf32)
###     tle_semantic.dot(inputA, inputB, result, size, initC, a_transpose, b_transpose, enable_hf32, _builder)



@builtin
def alloc(shape: List[tl.constexpr], dtype: tl.dtype, mem_addr_space: address_space = None, _builder=None) -> buffer:
    """
    Allocates a region of local memory with the specified shape and type.

    :param etype: the element type of the buffer.
    :type etype: tl.dtype
    :param shape: A list of non-negative integers representing the shape of the buffer.
    :type shape: List[tl.constexpr]
    :param _address_space: (Optional) backend-specific local memory address space
    :type _address_space: bl.address_space
    """
    return tle_semantic.alloc(dtype, shape, mem_addr_space, _builder)


@builtin
def to_buffer(tensor: tl.tensor, space: address_space = None, bind_buffer: buffer = None, _builder=None) -> buffer:
    """
    Convert a tensor to a buffer.

    :param tensor: the tensor to convert.
    :type tensor: tl.tensor
    :param space: the address space for the buffer (optional).
    :type space: address_space
    """
    return tle_semantic.to_buffer(tensor, space, bind_buffer, _builder)


@builtin
def to_tensor(memref: buffer, writable: bool = True, target_shape=None, _builder=None) -> tl.tensor:
    """
    Create a tl.tensor from a bl.buffer.

    :param memref: the input bl.buffer object.
    :memref type: bl.buffer
    :param writable: If set true, the resultant tensor is considered "writable" during bufferization.
    :type writable: bool
    """
    return tle_semantic.to_tensor(memref, writable, _builder, target_shape=target_shape)

@builtin
def subview(src: buffer, offsets: List[tl.constexpr], sizes: List[tl.constexpr], strides: List[tl.constexpr],
            _builder=None) -> buffer:
    pass