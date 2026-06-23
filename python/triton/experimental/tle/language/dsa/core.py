# Copyright 2026- Xcoresigma Technology Co., Ltd

import triton.language.core as tl
from triton.language.core import (_unwrap_if_constexpr, tensor, constexpr)

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
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
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
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)


@builtin
def from_buffer_to_tensor_pointer(src: buffer, _semantic=None) -> tl.tensor:
    buffer_ty = src.type
    ele_type = buffer_ty.element_ty
    shape = buffer_ty.shape
    block_type = tl.block_type(ele_type, shape)
    return tl.tensor(src.handle, block_type)


@builtin
def copy(src, dst, shape, inter_no_alias=False, _semantic=None):
    """Copy data from `src` to `dst` shaped by `shape`.

    :param inter_no_alias: If True, the copy is annotated as no aliasing between different iterations.
    """
    assert len(shape) != 0, "Can't deduce copy extents from args"

    shape = _unwrap_if_constexpr(shape)
    inter_no_alias = _unwrap_if_constexpr(inter_no_alias)
    tle_semantic.copy(src, dst, shape, inter_no_alias, _semantic.builder)


@builtin
def add(input, other, result, _semantic=None):
    input = from_buffer_to_tensor_pointer(input, _semantic=_semantic)
    other = from_buffer_to_tensor_pointer(other, _semantic=_semantic)
    result = from_buffer_to_tensor_pointer(result, _semantic=_semantic)
    tle_semantic.add(input, other, result, _semantic.builder)


@builtin
def sub(input, other, result, _semantic=None):
    input = from_buffer_to_tensor_pointer(input, _semantic=_semantic)
    other = from_buffer_to_tensor_pointer(other, _semantic=_semantic)
    result = from_buffer_to_tensor_pointer(result, _semantic=_semantic)
    tle_semantic.sub(input, other, result, _semantic.builder)


@builtin
def mul(input, other, result, _semantic=None):
    input = from_buffer_to_tensor_pointer(input, _semantic=_semantic)
    other = from_buffer_to_tensor_pointer(other, _semantic=_semantic)
    result = from_buffer_to_tensor_pointer(result, _semantic=_semantic)
    tle_semantic.mul(input, other, result, _semantic.builder)


@builtin
def div(input, other, result, _semantic=None):
    input = from_buffer_to_tensor_pointer(input, _semantic=_semantic)
    other = from_buffer_to_tensor_pointer(other, _semantic=_semantic)
    result = from_buffer_to_tensor_pointer(result, _semantic=_semantic)
    tle_semantic.div(input, other, result, _semantic.builder)


@builtin
def max(input, other, result, _semantic=None):
    # elementwise binary vector maximum op
    input = from_buffer_to_tensor_pointer(input, _semantic=_semantic)
    other = from_buffer_to_tensor_pointer(other, _semantic=_semantic)
    result = from_buffer_to_tensor_pointer(result, _semantic=_semantic)
    tle_semantic.max(input, other, result, _semantic.builder)


@builtin
def min(input, other, result, _semantic=None):
    # elementwise binary vector minimum op
    input = from_buffer_to_tensor_pointer(input, _semantic=_semantic)
    other = from_buffer_to_tensor_pointer(other, _semantic=_semantic)
    result = from_buffer_to_tensor_pointer(result, _semantic=_semantic)
    tle_semantic.min(input, other, result, _semantic.builder)


@builtin
def alloc(shape: List[tl.constexpr], dtype: tl.dtype, mem_addr_space: address_space, _semantic=None) -> buffer:
    """
    Allocates a region of local memory with the specified shape and type.
    """
    assert (mem_addr_space is not None)
    return tle_semantic.alloc(dtype, shape, mem_addr_space, _semantic.builder)


@builtin
def to_buffer(tensor: tl.tensor, space: address_space = None, bind_buffer: buffer = None, _semantic=None) -> buffer:
    """
    Convert a tensor to a buffer.
    """
    return tle_semantic.to_buffer(tensor, space, bind_buffer, _semantic.builder)


@builtin
def to_tensor(memref: buffer, writable: bool = True, target_shape=None, _semantic=None) -> tl.tensor:
    """
    Create a tl.tensor from a bl.buffer.
    """
    return tle_semantic.to_tensor(memref, writable, _semantic.builder, target_shape=target_shape)


@builtin
def subview(src: buffer, offsets: List[tl.constexpr], sizes: List[tl.constexpr], strides: List[tl.constexpr],
            _semantic=None) -> buffer:
    '''
    Creates a subview of the source buffer with the specified offsets, sizes, and strides.
    '''
    new_sizes = []
    for i, size in enumerate(sizes):
        if isinstance(size, int):
            new_sizes.append(tl.constexpr(size))
        elif isinstance(size, tl.constexpr):
            new_sizes.append(size)
        else:
            raise TypeError(f"sizes[{i}] must be constexpr, got {type(size).__name__}")

    new_strides = []
    for i, stride in enumerate(strides):
        if isinstance(stride, int):
            new_strides.append(tl.constexpr(stride))
        elif isinstance(stride, tl.constexpr):
            new_strides.append(stride)
        else:
            raise TypeError(f"strides[{i}] must be constexpr, got {type(stride).__name__}")

    new_offsets = []
    for offset in offsets:
        if isinstance(offset, tl.constexpr):
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(_semantic.to_tensor(offset))
        elif isinstance(offset, int):
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(_semantic.to_tensor(tl.constexpr(offset)))
        else:
            new_offsets.append(offset)

    return tle_semantic.subview(src, new_offsets, new_sizes, new_strides, _semantic.builder)


def hint(**kwargs):
    """Dummy function for AST parsing. Not executed during JIT compilation."""
    raise RuntimeError("tle.hint() cannot be called directly.")


@builtin
def insert_slice(ful: tensor, sub: tensor, offsets: List[tensor], sizes: List[int], strides: List[int],
                 _semantic=None) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.
    """
    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    assert (len(ful.shape) == len(sizes))
    assert (len(ful.shape) == len(strides))
    new_offsets = [_semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in offsets]
    out = tle_semantic.insert_slice(ful, sub, new_offsets, sizes, strides, _semantic.builder)
    return out


@builtin
def extract_slice(ful, offsets, sizes, strides, _semantic=None, _generator=None) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.
    """
    assert len(ful.shape) > 0
    new_offsets = [_semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in offsets]
    sub = tle_semantic.extract_slice(ful, new_offsets, sizes, strides, _semantic.builder)
    return sub


@builtin
def extract_element(src, indice, _semantic=None, _generator=None):
    """
    get_element op reads a ranked tensor and returns one element as specified by the given indices.
    """
    assert len(src.shape) > 0
    new_indice = [_semantic.to_tensor(i) if isinstance(i, constexpr) else i for i in indice]
    return tle_semantic.extract_element(src, new_indice, _semantic.builder)
