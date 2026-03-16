# Copyright 2026- Xcoresigma Technology Co., Ltd

import triton.language.core as tl
from triton.language import semantic as tl_semantic
from triton.language.core import (_constexpr_to_value, tensor, constexpr)

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


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it indicates that there are no dependencies between loop iterations,
     allowing them to be executed in parallel.
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
def copy(src, dst, shape, inter_no_alias=False, _builder=None):
    """Copy data from `src` to `dst` shaped by `shape`.

    :param inter_no_alias: If True, the copy is annotated as no aliasing between different iterations.
    """
    assert len(shape) != 0, "Can't deduce copy extents from args"

    shape = _constexpr_to_value(shape)
    inter_no_alias = _constexpr_to_value(inter_no_alias)
    tle_semantic.copy(src, dst, shape, inter_no_alias, _builder)


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


@builtin
def alloc(shape: List[tl.constexpr], dtype: tl.dtype, mem_addr_space: address_space, _builder=None) -> buffer:
    """
    Allocates a region of local memory with the specified shape and type.

    :param etype: the element type of the buffer.
    :type etype: tl.dtype
    :param shape: A list of non-negative integers representing the shape of the buffer.
    :type shape: List[tl.constexpr]
    :param _address_space: (Optional) backend-specific local memory address space
    :type _address_space: bl.address_space
    """
    assert (mem_addr_space is not None)
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

    new_offsets = []
    for offset in offsets:
        if isinstance(offset, tl.constexpr):
            # Check that constexpr offset values cannot be negative
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(tl_semantic.to_tensor(offset, _builder))
        elif isinstance(offset, int):
            # Convert regular integers to constexpr and then to tensor
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(tl_semantic.to_tensor(tl.constexpr(offset), _builder))
        else:
            # Assume it's already a tensor
            new_offsets.append(offset)

    return tle_semantic.subview(src, new_offsets, new_sizes, new_strides, _builder)


def hint(**kwargs):
    """Dummy function for AST parsing. Not executed during JIT compilation."""
    raise RuntimeError("tle.hint() cannot be called directly.")


@builtin
def insert_slice(ful: tensor, sub: tensor, offsets: List[tensor], sizes: List[int], strides: List[int],
                 _builder=None) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to receive tensor.
    :type ful: Tensor
    :param sub: The tensor to be inserted.
    :type sub: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    assert (len(ful.shape) == len(sizes))
    assert (len(ful.shape) == len(strides))
    new_offsets = [tl_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o for o in offsets]
    out = tle_semantic.insert_slice(ful, sub, new_offsets, sizes, strides, _builder)
    return out


@builtin
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to split.
    :type ful: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert len(ful.shape) > 0
    new_offsets = [tl_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o for o in offsets]
    sub = tle_semantic.extract_slice(ful, new_offsets, sizes, strides, _builder)
    return sub


@builtin
def extract_element(src, indice, _builder=None, _generator=None):
    """
    get_element op reads a ranked tensor and returns one element as specified by the given indices.
    The result of the op is a value with the same type as the elements of the tensor.
    The arity of indices must match the rank of the accessed value.

    :param src: The tensor to be accessed.
    :type src: Tensor
    :param indice:
    :type indice: tuple of ints
    """
    assert len(src.shape) > 0
    new_indice = [tl_semantic.to_tensor(i, _builder) if isinstance(i, constexpr) else i for i in indice]
    return tle_semantic.extract_element(src, new_indice, _builder)
