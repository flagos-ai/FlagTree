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

# Active ``with tle.dsa.hint(...)`` scopes. The 3.5 code generator stored
# these on its TleCodeGenerator; the current frontend can keep the same
# information here because the with-body is traced while __enter__ is active.
_HINT_STACK = []


class _HintContext:

    def __init__(self, hints):
        self.hints = hints

    def __enter__(self):
        _HINT_STACK.append(self.hints)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _HINT_STACK.pop()
        return False


def _current_hint(name, default=False):
    for hints in reversed(_HINT_STACK):
        if name in hints:
            return hints[name]
    return default


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
def copy(src, dst, shape, inter_no_alias=None, _semantic=None):
    """Copy data from `src` to `dst` shaped by `shape`.

    :param inter_no_alias: If True, the copy is annotated as no aliasing between different iterations.
    """
    assert len(shape) != 0, "Can't deduce copy extents from args"

    shape = _unwrap_if_constexpr(shape)
    if inter_no_alias is None:
        inter_no_alias = _current_hint("inter_no_alias", False)
    inter_no_alias = _unwrap_if_constexpr(inter_no_alias)
    tle_semantic.copy(src, dst, shape, inter_no_alias, _semantic.builder)


@builtin
def add(input, other, result, _semantic=None):
    tle_semantic.add(input, other, result, _semantic.builder)


@builtin
def sub(input, other, result, _semantic=None):
    tle_semantic.sub(input, other, result, _semantic.builder)


@builtin
def mul(input, other, result, _semantic=None):
    tle_semantic.mul(input, other, result, _semantic.builder)


@builtin
def div(input, other, result, _semantic=None):
    tle_semantic.div(input, other, result, _semantic.builder)


@builtin
def max(input, other, result, _semantic=None):
    # elementwise binary vector maximum op
    tle_semantic.max(input, other, result, _semantic.builder)


@builtin
def min(input, other, result, _semantic=None):
    # elementwise binary vector minimum op
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

    :param src: The source buffer to create a subview from.
    :type src: buffer
    :param offsets: Offsets in each dimension. Items may be constexpr/int literals or dynamic tl.tensor values.
    :type offsets: List
    :param sizes: A list of non-negative integers representing the sizes in each dimension.
    :type sizes: List[tl.constexpr]
    :param strides: A list of non-negative integers representing the strides in each dimension.
    :type strides: List[tl.constexpr]
    :return: A new buffer representing the subview of the source buffer.
    :rtype: buffer
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


def hint(_semantic=None, **kwargs):
    """Scope hints for the enclosed TLE operations.

    ``with tle.dsa.hint(inter_no_alias=True):`` annotates copies in the body.
    The context object keeps the hints in a small stack while the Triton
    frontend traces the body, so no custom code-generator patch is needed.
    """
    return _HintContext(kwargs)


setattr(hint, TRITON_BUILTIN, True)
setattr(hint, TLE_BUILTIN, True)


def _unwrap_slice_meta(value):
    """Unwrap a constexpr/list/tuple so slice metadata can be iterated."""
    value = _unwrap_if_constexpr(value)
    if isinstance(value, tl.tuple):
        return list(value)
    return value


@builtin
def insert_slice(ful: tensor, sub: tensor, offsets, sizes=None, strides=None, _semantic=None) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.

    ``sizes`` defaults to ``sub.shape`` and ``strides`` defaults to all ones,
    matching the portable 3.6 ``tle.dsa.insert_slice`` API.
    """
    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    offsets = _unwrap_slice_meta(offsets)
    sizes = _unwrap_slice_meta(sizes)
    strides = _unwrap_slice_meta(strides)
    if sizes is None:
        sizes = [_unwrap_if_constexpr(dim) for dim in sub.shape]
    if strides is None:
        strides = [1] * len(sizes)
    assert (len(ful.shape) == len(sizes))
    assert (len(ful.shape) == len(strides))
    new_offsets = [_semantic.to_tensor(o) for o in offsets]
    out = tle_semantic.insert_slice(ful, sub, new_offsets, sizes, strides, _semantic.builder)
    return out


@builtin
def extract_slice(ful, offsets, sizes, strides=None, _semantic=None, _generator=None) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.

    ``strides`` defaults to all ones.
    """
    assert len(ful.shape) > 0
    offsets = _unwrap_slice_meta(offsets)
    sizes = _unwrap_slice_meta(sizes)
    strides = _unwrap_slice_meta(strides)
    if strides is None:
        strides = [1] * len(sizes)
    new_offsets = [_semantic.to_tensor(o) for o in offsets]
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


# ==============================================================================
# CommonIR builtin functions — 3-task flash attention operations
# ==============================================================================


@builtin
def tile_alloc(shape: List[tl.constexpr], dtype: tl.dtype, mem_addr_space: address_space, _semantic=None,
               _generator=None) -> buffer:
    """Allocate a buffer in a specific memory space (CommonIR path)."""
    assert (mem_addr_space is not None)
    return tle_semantic.tile_alloc(dtype, shape, mem_addr_space, _semantic.builder)


@builtin
def tile_copy(src, dst, shape, inter_no_alias=None, _semantic=None, _generator=None):
    """Copy data using CommonIR tile.copy op."""
    assert len(shape) != 0, "Can't deduce copy extents from args"
    shape = _unwrap_if_constexpr(shape)
    if inter_no_alias is None:
        inter_no_alias = _current_hint("inter_no_alias", False)
    inter_no_alias = _unwrap_if_constexpr(inter_no_alias)
    tle_semantic.tile_copy(src, dst, shape, inter_no_alias, _semantic.builder)


@builtin
def tile_subview(src: buffer, offsets: List, sizes: List[tl.constexpr], strides: List[tl.constexpr], _semantic=None,
                 _generator=None) -> buffer:
    """Extract a subview from a buffer using CommonIR tile.subview op."""
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

    return tle_semantic.tile_subview(src, new_offsets, new_sizes, new_strides, _semantic.builder)


@builtin
def tile_to_tensor(memref: buffer, writable: bool = True, target_shape=None, _semantic=None,
                   _generator=None) -> tl.tensor:
    """Create a tl.tensor from a buffer (CommonIR path)."""
    return tle_semantic.tile_to_tensor(memref, writable, _semantic.builder, target_shape=target_shape)


@builtin
def tile_set_flag(producer_pipe, consumer_pipe, event_id, _semantic=None, _generator=None):
    """Set a cross-engine synchronization flag (CommonIR tile.set_flag)."""
    tle_semantic.set_flag(producer_pipe, consumer_pipe, event_id, _semantic.builder)


@builtin
def tile_wait_flag(producer_pipe, consumer_pipe, event_id, _semantic=None, _generator=None):
    """Wait for a cross-engine synchronization flag (CommonIR tile.wait_flag)."""
    tle_semantic.wait_flag(producer_pipe, consumer_pipe, event_id, _semantic.builder)


@builtin
def tile_pipe_barrier(pipe, _semantic=None, _generator=None):
    """Insert an intra-engine pipeline barrier (CommonIR tile.pipe_barrier)."""
    tle_semantic.pipe_barrier(pipe, _semantic.builder)


@builtin
def tensor_to_tile(src: tl.tensor, space: address_space = None, _semantic=None, _generator=None) -> buffer:
    """Convert a tt.ptr/tensor to a !tile.buf in the given memory space.

    Uses tile.copy to load data from the tensor pointer into a newly
    allocated tile.buf, then returns the buffer.
    If space is not specified, defaults to GM (global memory).
    """
    return tle_semantic.tensor_to_tile(src, space, _semantic.builder)


@builtin
def tile_gm_offset(base, indices, strides, _semantic=None, _generator=None) -> tl.tensor:
    """Compute a GM pointer with multi-dimensional offsets (CommonIR tile.gm_offset)."""
    return tle_semantic.tile_gm_offset(base, indices, strides, _semantic.builder)


@builtin
def tile_cube_launch(a: buffer, b: buffer, acc: buffer, stage_a: buffer, stage_b: buffer, dst,
                     transpose_a: bool = False, transpose_b: bool = False, init: bool = False, mma: str = "",
                     _semantic=None, _generator=None):
    """Launch a Cube matmul using CommonIR tile.cube_launch."""
    transpose_a = _unwrap_if_constexpr(transpose_a)
    transpose_b = _unwrap_if_constexpr(transpose_b)
    init = _unwrap_if_constexpr(init)
    mma = _unwrap_if_constexpr(mma)
    tle_semantic.tile_cube_launch(a, b, acc, stage_a, stage_b, dst, transpose_a, transpose_b, init, mma,
                                  _semantic.builder)


@builtin
def tile_cube_wait(_semantic=None, _generator=None):
    """Wait for CommonIR Cube work using tile.cube_wait."""
    tle_semantic.tile_cube_wait(_semantic.builder)


class Workspace:
    __triton_compile_time_value__ = True

    def __init__(self, base, capacity, shape, dtype, strides=None, stage_stride=None):
        self.base = base
        self.capacity = _unwrap_if_constexpr(capacity)
        if not isinstance(self.capacity, int):
            raise TypeError("workspace capacity must be a compile-time integer")
        if self.capacity < 1:
            raise ValueError("workspace capacity must be positive")
        self.shape = tl._unwrap_shape(shape)
        if not isinstance(self.shape, (tuple, list)):
            raise TypeError("workspace shape must be list/tuple")
        self.shape = list(self.shape)
        self.dtype = _unwrap_if_constexpr(dtype)
        if strides is None:
            stride = 1
            computed_strides = []
            for dim in reversed(self.shape):
                computed_strides.insert(0, stride)
                stride *= dim
            self.strides = computed_strides
        else:
            self.strides = list(tl._unwrap_shape(strides))
        if stage_stride is None:
            stage_stride_value = 1
            for dim in self.shape:
                stage_stride_value *= dim
            self.stage_stride = stage_stride_value
        else:
            self.stage_stride = _unwrap_if_constexpr(stage_stride)

    def slot(self, stage, _semantic=None):
        # A workspace slot is pointer arithmetic over the caller-provided GM ring buffer.
        if isinstance(stage, tensor):
            stage_offset = stage.__mul__(self.stage_stride, _semantic=_semantic)
        else:
            stage_offset = stage * self.stage_stride
        if len(self.shape) == 1:
            offs = tl.arange(0, self.shape[0], _semantic=_semantic)
            elem_offset = offs.__mul__(self.strides[0], _semantic=_semantic)
            if isinstance(stage_offset, tensor):
                offset = stage_offset.__add__(elem_offset, _semantic=_semantic)
            else:
                offset = elem_offset.__add__(stage_offset, _semantic=_semantic)
            if isinstance(self.base, tensor):
                return self.base.__add__(offset, _semantic=_semantic)
            return self.base + offset
        if len(self.shape) == 2:
            offs_m = tl.arange(0, self.shape[0], _semantic=_semantic).__getitem__((slice(None), None),
                                                                                  _semantic=_semantic)
            offs_n = tl.arange(0, self.shape[1], _semantic=_semantic).__getitem__((None, slice(None)),
                                                                                  _semantic=_semantic)
            m_offset = offs_m.__mul__(self.strides[0], _semantic=_semantic)
            n_offset = offs_n.__mul__(self.strides[1], _semantic=_semantic)
            if isinstance(stage_offset, tensor):
                row_offset = stage_offset.__add__(m_offset, _semantic=_semantic)
            else:
                row_offset = m_offset.__add__(stage_offset, _semantic=_semantic)
            if isinstance(self.base, tensor):
                rows = self.base.__add__(row_offset, _semantic=_semantic)
            else:
                rows = self.base + row_offset
            if isinstance(rows, tensor):
                return rows.__add__(n_offset, _semantic=_semantic)
            return rows + n_offset
        raise NotImplementedError("workspace supports rank 1 or rank 2 payloads")


def workspace(base, capacity, shape, dtype, strides=None, stage_stride=None, _semantic=None):
    return Workspace(base, capacity, shape, dtype, strides, stage_stride)


# Descriptor constructor: pure Python with no builder work, so it stays callable from
# host code (tests construct workspaces outside kernels); only the builtin marker is
# needed to satisfy JIT reference checks.
setattr(workspace, TRITON_BUILTIN, True)
setattr(workspace, TLE_BUILTIN, True)


@tl.builtin
def tile_load(src, index=None, dst_space=None, _semantic=None, _generator=None) -> tensor:
    """Load a tile from an encoded tensor_view via tile.load."""
    return tle_semantic.tile_load(src, index, dst_space, _semantic.builder)


@tl.builtin
def tile_store(src, value=None, index=None, dst=None, src_space=None, _semantic=None, _generator=None):
    """Store a tile into an encoded tensor_view via tile.store."""
    tle_semantic.tile_store(src, dst, value, index, _semantic.builder, src_space)


__all__ = [
    "Workspace",
    "alloc",
    "copy",
    "pipeline",
    "parallel",
    "to_tensor",
    "to_buffer",
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "hint",
    "extract_slice",
    "insert_slice",
    "extract_element",
    "subview",
    "tile_alloc",
    "tile_copy",
    "tile_subview",
    "tile_to_tensor",
    "tile_set_flag",
    "tile_wait_flag",
    "tile_pipe_barrier",
    "tile_gm_offset",
    "tile_cube_launch",
    "tile_cube_wait",
    "tile_load",
    "tile_store",
    "tensor_to_tile",
    "from_buffer_to_tensor_pointer",
    "workspace",
]
