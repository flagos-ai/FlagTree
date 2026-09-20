# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# flagtree tle
"""
TLE (Triton Language Extensions) Semantic Analysis Module

This module provides semantic analysis and type checking for TLE operations
"""

from __future__ import annotations
import warnings
from typing import List, Optional, Sequence, Tuple, Union

from triton._C.libtriton import ir
import triton.language.core as tl
from triton._common_ir import ENABLED as COMMON_IR_ENABLED
from . import types as tle
from .mthreads import copy as mthreads_copy
from math import prod


def _expand_index_to_shape(index, shape, axis, _semantic):
    result = index
    for _ in range(axis):
        result = tl.expand_dims(result, 0, _semantic=_semantic)
    for _ in range(len(shape) - axis - 1):
        result = tl.expand_dims(result, len(result.shape), _semantic=_semantic)
    return tl.broadcast_to(result, *shape, _semantic=_semantic)


def _make_full_indices(buffer, _semantic):
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    return tuple(
        _expand_index_to_shape(tl.arange(0, dim, _semantic=_semantic), shape, axis, _semantic)
        for axis, dim in enumerate(shape))


def alloc(shape, dtype, storage, layout, layout_handle, init_value, _semantic):
    builder = _semantic.builder
    shape = [int(tl._unwrap_if_constexpr(dim)) for dim in shape]
    element_type = dtype.to_ir(builder)
    if COMMON_IR_ENABLED:
        memory_space = builder.tile_get_string_attr(tle._storage_to_tile_space(storage))
        tile_ty = builder.tile_get_buffer_type(shape, element_type, memory_space)
        handle = builder.create_tile_alloc(tile_ty, layout_handle)
        if init_value is not None:
            builder.create_tile_store_tensor(init_value.handle, handle)
    elif storage == tle.smem:
        if init_value is None:
            handle = builder.create_local_alloc(shape, element_type, layout_handle)
        else:
            result_type = builder.get_memdesc_type(shape, element_type, layout_handle, "smem")
            handle = builder.create_local_alloc(result_type, init_value.handle)
    else:
        raise ValueError(f"Storage type {storage} not yet supported")

    result = tle.buffered_tensor(handle, dtype, list(shape), storage, layout, _semantic)
    return result


def copy(src, dst, shape, offsets, direction, _semantic, completion_barrier=None):
    descriptor = src if isinstance(src,
                                   tl.tensor_descriptor) else dst if isinstance(dst, tl.tensor_descriptor) else None

    if COMMON_IR_ENABLED:
        if descriptor is None:
            buffer = dst if isinstance(dst, tle.buffered_tensor) else src
            pointers = src if buffer is dst else dst
            requested_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in shape)
            buffer_shape = tuple(buffer.type.shape)
            if requested_shape != buffer_shape or tuple(pointers.shape) != buffer_shape:
                raise ValueError("GPU CommonIR copy requires shape and pointer tensor shape to match the full buffer")
            if not pointers.dtype.is_ptr() or pointers.dtype.element_ty != buffer.dtype:
                raise ValueError("GPU CommonIR copy requires pointers with the buffer element type")
            _semantic.builder.create_tile_copy(src.handle, dst.handle, False)
            return
        indices = _semantic._convert_to_ir_values(offsets, require_i64=False)
        buffer = dst if isinstance(dst, tle.buffered_tensor) else src
        memdesc = get_memdesc(buffer, _semantic)
        src_handle = src.handle if src is descriptor else memdesc
        dst_handle = dst.handle if dst is descriptor else memdesc
        if completion_barrier is None:
            _semantic.builder.create_tma_copy(src_handle, dst_handle, indices)
        else:
            _semantic.builder.create_tma_copy(src_handle, dst_handle, indices, completion_barrier.handle,
                                              completion_barrier.expect_bytes)
        return

    if descriptor is not None:
        if mthreads_copy.enabled():
            mthreads_copy.tmacopy(src, dst, direction, shape, offsets, _semantic)
            return
        indices = _semantic._convert_to_ir_values(offsets, require_i64=False)
        _semantic.builder.create_tma_copy(src.handle, dst.handle, indices)
        return

    if mthreads_copy.enabled():
        mthreads_copy.validate_normal_copy(src, dst, shape, direction)
    if isinstance(_semantic, TLESemantic):
        _semantic.analyze_copy_operation(src, dst, shape)

    mask = None
    boundary_check = ()
    cache_modifier = ""
    eviction_policy = ""
    if direction.name == "GM_TO_LOCAL":
        load_extra_args = () if mthreads_copy.enabled() else (None, )
        value = _semantic.load(src, mask, None, boundary_check, "", cache_modifier, eviction_policy, False,
                               *load_extra_args)
        from .core import local_ptr
        ptrs = local_ptr(dst, _make_full_indices(dst, _semantic), _semantic=_semantic)
        _semantic.store(ptrs, value, mask, boundary_check, cache_modifier, eviction_policy)
    else:
        from .core import local_ptr
        ptrs = local_ptr(src, _make_full_indices(src, _semantic), _semantic=_semantic)
        value = tl.load(ptrs, _semantic=_semantic)
        _semantic.store(dst, value, mask, boundary_check, cache_modifier, eviction_policy)


def subview(src, offsets, shape, strides, layout, _semantic):
    builder = _semantic.builder
    if not COMMON_IR_ENABLED:
        result_type = tle.buffered_tensor_type(src.dtype, shape, src.type.storage, layout, _semantic,
                                               alloc_shape=shape).to_ir(builder)
        return builder.create_memdesc_index(result_type, src.handle, offsets[0].handle)
    return builder.create_tile_subview(
        src.handle,
        [offset.handle for offset in offsets],
        shape,
        strides,
        layout.to_ir(builder),
    )


def to_tensor(buffer, writable, result_type, _semantic):
    if COMMON_IR_ENABLED:
        if tuple(result_type.shape) != tuple(buffer.type.shape):
            raise ValueError("GPU CommonIR buffered_tensor.load requires the full buffer shape")
        result = _semantic.builder.create_tile_to_tensor(buffer.handle, writable)
        return tl.tensor(result, result_type)
    from .core import local_ptr
    ptrs = local_ptr(buffer, _make_full_indices(buffer, _semantic), _semantic=_semantic)
    return tl.load(ptrs, _semantic=_semantic)


def store_tensor(value, dst, _semantic):
    if COMMON_IR_ENABLED:
        if tuple(value.shape) != tuple(dst.type.shape) or value.dtype != dst.dtype:
            raise ValueError("GPU CommonIR buffered_tensor.store requires the full buffer shape and element type")
        _semantic.builder.create_tile_store_tensor(value.handle, dst.handle)
        return
    from .core import local_ptr
    ptrs = local_ptr(dst, _make_full_indices(dst, _semantic), _semantic=_semantic)
    _semantic.store(ptrs, value, None, (), "", "")


def local_ptr(result_ir, buffer, indices, _semantic):
    handles = [index.handle for index in indices]
    if not COMMON_IR_ENABLED:
        return _semantic.builder.create_local_pointers(result_ir, buffer.handle, *handles)
    memdesc = get_memdesc(buffer, _semantic)
    return _semantic.builder.create_local_pointers(result_ir, memdesc, *handles)


def get_memdesc(buffer, _semantic):
    """Bridge a CommonIR buffer to the descriptor type consumed by TLE pipe ops."""
    if not COMMON_IR_ENABLED:
        return buffer.handle
    result_type = buffer.type.to_memdesc_ir(_semantic.builder)
    return _semantic.builder.create_tile_get_memdesc(result_type, buffer.handle)


class TLESemanticError(Exception):
    """TLE operation semantic error exception class"""

    def __init__(self, message: str, operation: str = None):
        self.operation = operation
        self.message = message
        if operation:
            super().__init__(f"TLE semantic error in {operation}: {message}")
        else:
            super().__init__(f"TLE semantic error: {message}")


class TLESemantic:
    """Semantic analyzer for TLE operations"""

    def __init__(self, builder: ir.builder):
        self.builder = builder

    def validate_alloc_shape(self, shape: Sequence[Union[int, any]]) -> List[int]:
        """Validate allocation shape"""
        if not shape:
            raise TLESemanticError("Allocation shape cannot be empty", "alloc")

        unwrapped_shape = []
        for i, dim in enumerate(shape):
            if hasattr(dim, 'value'):  # constexpr-like objects
                dim_val = dim.value
            elif isinstance(dim, int):
                dim_val = dim
            else:
                raise TLESemanticError(f"Shape dimension {i} must be integer or constexpr-like, but got {type(dim)}",
                                       "alloc")

            if dim_val <= 0:
                raise TLESemanticError(f"Shape dimension {i} must be positive, but got {dim_val}", "alloc")

            unwrapped_shape.append(dim_val)

        return unwrapped_shape

    def validate_alloc_dtype(self, dtype: tl.dtype) -> tl.dtype:
        """Validate allocation data type"""
        if not isinstance(dtype, tl.dtype):
            raise TLESemanticError(f"Data type must be tl.dtype, but got {type(dtype)}", "alloc")

        supported_types = [
            tl.float32, tl.float16, tl.bfloat16, tl.int8, tl.int16, tl.int32, tl.int64, tl.uint8, tl.uint16, tl.uint32,
            tl.uint64, tl.int1  # boolean type equivalent in Triton
        ]

        if dtype not in supported_types:
            warnings.warn(f"Data type {dtype} may not be fully supported, recommend using standard data types",
                          UserWarning)

        return dtype

    def validate_copy_compatibility(self, src: tl.tensor, dst: tle.buffered_tensor,
                                    copy_shape: Sequence[Union[int, tl.constexpr]]) -> None:

        if src.type.element_ty != dst.type.element_ty:
            raise TLESemanticError(
                f"Source data type {src.type.element_ty} incompatible with destination data type {dst.type.element_ty}",
                "copy")

        copy_shape_unwrapped = [dim.value if hasattr(dim, 'value') else dim for dim in copy_shape]
        src_shape = list(src.type.shape)
        dst_shape = list(dst.type.shape)

        for i, copy_dim in enumerate(copy_shape_unwrapped):
            if i < len(src_shape) and copy_dim > src_shape[i]:
                raise TLESemanticError(f"Copy dimension {i} ({copy_dim}) exceeds source tensor range ({src_shape[i]})",
                                       "copy")
            if i < len(dst_shape) and copy_dim > dst_shape[i]:
                raise TLESemanticError(
                    f"Copy dimension {i} ({copy_dim}) exceeds destination buffer range ({dst_shape[i]})", "copy")

    def validate_local_pointer_buffer(self, buffer: tle.buffered_tensor) -> None:
        """Validate buffer usage for local pointer materialization"""
        if not isinstance(buffer, tle.buffered_tensor):
            raise TLESemanticError(f"Buffer must be tle.buffered_tensor, but got {type(buffer)}", "local_ptr")

    def validate_extract_tile_params(self, src: tl.tensor, index, tile_shape: Sequence[Union[int, any]]) -> None:
        """

        """
        # Check 1: src type
        if not isinstance(src, tl.tensor):
            raise TLESemanticError(f"Source must be tl.tensor, but got {type(src)}", "extract_tile")

        # Check 2: non-empty / provided parameters
        if index is None:
            raise TLESemanticError("Index cannot be None", "extract_tile")
        if not tile_shape:
            raise TLESemanticError("tile_shape cannot be empty", "extract_tile")

        # Check 3: unpack and validate type
        tile_shape_unwrapped = [s.value if hasattr(s, 'value') else s for s in tile_shape]

        # Check if every dim in tile_shape is int or constexpr-like
        if any(not isinstance(s, int) for s in tile_shape_unwrapped):
            raise TLESemanticError("All tile_shape dims must be int or constexpr", "extract_tile")

        # Check 4: positive values
        if any(s <= 0 for s in tile_shape_unwrapped):
            raise TLESemanticError("All tile_shape dims must be positive", "extract_tile")

        # Check 5: dimension match
        src_shape = list(src.type.shape)

        if len(tile_shape_unwrapped) != len(src_shape):
            raise TLESemanticError(
                f"Tile_shape rank ({len(tile_shape_unwrapped)}) must match "
                f"source rank ({len(src_shape)})", "extract_tile")

        # Index check
        if isinstance(index, (tuple, list)):
            idx = [i.value if hasattr(i, 'value') else i for i in index]
            # Check if index rank matches source tensor rank
            if len(idx) != len(src_shape):
                raise TLESemanticError(f"Index rank ({len(idx)}) must match source rank ({len(src_shape)})",
                                       "extract_tile")
            # Check if every index component is int
            if any(not isinstance(v, int) for v in idx):
                raise TLESemanticError("All index values must be int or constexpr", "extract_tile")
            # Check tile grid out-of-bounds
            if all(isinstance(dim, int) for dim in src_shape):
                grid = [src_dim // tile_shape_dim for src_dim, tile_shape_dim in zip(src_shape, tile_shape_unwrapped)]
                for i, v in enumerate(idx):
                    if v < 0 or v >= grid[i]:
                        raise TLESemanticError(f"Index[{i}]={v} out of bounds for tile grid (0~{grid[i]-1})",
                                               "extract_tile")
        else:
            # If linear index (single value)
            val = index.value if hasattr(index, 'value') else index
            # Check must be int and non-negative
            if not isinstance(val, int):
                raise TLESemanticError("Index must be int or constexpr", "extract_tile")
            if val < 0:
                raise TLESemanticError("Index must be non-negative", "extract_tile")
            # Check linear index out-of-bounds
            if all(isinstance(dim, int) for dim in src_shape):
                grid = [src_dim // tile_shape_dim for src_dim, tile_shape_dim in zip(src_shape, tile_shape_unwrapped)]
                total_tiles = prod(grid)
                if val >= total_tiles:
                    raise TLESemanticError(f"Linear index {val} out of bounds for total tiles {total_tiles}",
                                           "extract_tile")

    def validate_insert_tile_params(self, src: tl.tensor, tile: tl.tensor, index) -> None:
        """

        """
        # src / tile type checks
        if not isinstance(src, tl.tensor):
            raise TLESemanticError(f"Source must be tl.tensor, but got {type(src)}", "insert_tile")
        if not isinstance(tile, tl.tensor):
            raise TLESemanticError(f"Tile must be tl.tensor, but got {type(tile)}", "insert_tile")

        src_shape = list(src.type.shape)
        tile_shape = list(tile.type.shape)

        # rank / element type checks
        if len(src_shape) != len(tile_shape):
            raise TLESemanticError(f"Source rank ({len(src_shape)}) must match tile rank ({len(tile_shape)})",
                                   "insert_tile")
        if src.type.element_ty != tile.type.element_ty:
            raise TLESemanticError(f"Element type mismatch: source={src.type.element_ty}, tile={tile.type.element_ty}",
                                   "insert_tile")

        # tile shape checks (int/constexpr-like, positive)
        tile_shape_unwrapped = [d.value if hasattr(d, 'value') else d for d in tile_shape]
        if any(not isinstance(d, int) for d in tile_shape_unwrapped):
            raise TLESemanticError("All tile dimensions must be int or constexpr", "insert_tile")
        if any(d <= 0 for d in tile_shape_unwrapped):
            raise TLESemanticError("All tile dimensions must be positive", "insert_tile")

        # source shape checks and tile grid construction
        src_shape_unwrapped = [d.value if hasattr(d, 'value') else d for d in src_shape]
        if any(not isinstance(d, int) for d in src_shape_unwrapped):
            raise TLESemanticError("Source shape must be static integers for insert_tile", "insert_tile")

        grid = []
        for i, (src_dim, tile_dim) in enumerate(zip(src_shape_unwrapped, tile_shape_unwrapped)):
            if src_dim % tile_dim != 0:
                raise TLESemanticError(
                    f"Source dimension {i}: {src_dim} must be divisible by tile dimension {tile_dim}", "insert_tile")
            grid.append(src_dim // tile_dim)

        # index checks
        if isinstance(index, (tuple, list)):
            idx = [i.value if hasattr(i, 'value') else i for i in index]
            if len(idx) != len(src_shape_unwrapped):
                raise TLESemanticError(f"Index rank ({len(idx)}) must match source rank ({len(src_shape_unwrapped)})",
                                       "insert_tile")
            if any(not isinstance(v, int) for v in idx):
                raise TLESemanticError("All index values must be int or constexpr", "insert_tile")
            for i, v in enumerate(idx):
                if v < 0 or v >= grid[i]:
                    raise TLESemanticError(f"Index[{i}]={v} out of bounds for tile grid (0~{grid[i]-1})", "insert_tile")
        else:
            val = index.value if hasattr(index, 'value') else index
            if not isinstance(val, int):
                raise TLESemanticError("Index must be int or constexpr", "insert_tile")
            if val < 0:
                raise TLESemanticError("Index must be non-negative", "insert_tile")

            total_tiles = prod(grid)
            if val >= total_tiles:
                raise TLESemanticError(f"Linear index {val} out of bounds for total tiles {total_tiles}", "insert_tile")

    def analyze_extract_tile_operation(self, src: tl.tensor, index, tile_shape: Sequence[Union[int, any]]) -> None:
        """Analyze extract_tile operation semantics"""
        self.validate_extract_tile_params(src, index, tile_shape)

    def analyze_insert_tile_operation(
        self,
        src: tl.tensor,
        tile: tl.tensor,
        index,
    ) -> None:
        """Analyze insert_tile operation semantics """
        self.validate_insert_tile_params(src, tile, index)

    def analyze_alloc_operation(self, shape: Sequence[Union[int, any]], dtype: tl.dtype,
                                layout: Optional[tle.shared_layout], storage: tle.scope) -> Tuple[List[int], tl.dtype]:
        """Analyze alloc operation semantics"""
        validated_shape = self.validate_alloc_shape(shape)
        validated_dtype = self.validate_alloc_dtype(dtype)
        return validated_shape, validated_dtype

    def analyze_copy_operation(self, src: tl.tensor, dst: tle.buffered_tensor,
                               copy_shape: Sequence[Union[int, any]]) -> None:
        """Analyze copy operation semantics"""
        self.validate_copy_compatibility(src, dst, copy_shape)

    def analyze_local_pointer_operation(self, buffer: tle.buffered_tensor,
                                        indices: Optional[Sequence[tl.tensor]] = None) -> None:
        """Analyze local_ptr operation semantics"""
        self.validate_local_pointer_buffer(buffer)
        if not indices:
            return
        first_shape = None
        saw_scalar = False
        saw_block = False
        for idx in indices:
            idx_ty = idx.type
            if idx_ty.is_block():
                saw_block = True
                elem_ty = idx_ty.element_ty
                if not elem_ty.is_int():
                    raise TLESemanticError("Index tensor dtype must be integer", "local_ptr")
                if first_shape is None:
                    first_shape = tuple(idx.shape)
                elif tuple(idx.shape) != first_shape:
                    raise TLESemanticError("Index tensors must have identical shapes", "local_ptr")
            else:
                saw_scalar = True
                if not idx_ty.is_int():
                    raise TLESemanticError("Scalar indices must be integers", "local_ptr")
        if saw_scalar and saw_block:
            raise TLESemanticError("Indices must be all scalar or all tensor", "local_ptr")
