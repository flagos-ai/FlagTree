# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
"""
TLE (Triton Language Extensions) Semantic Analysis Module

This module provides semantic analysis and type checking for TLE operations
"""

from __future__ import annotations
import warnings
from typing import List, Optional, Sequence, Tuple, Union

from triton._C.libtriton import ir
from triton import language as tl
from . import types as tle


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

    def validate_local_load_buffer(self, buffer: tle.buffered_tensor) -> None:
        """Validate local load buffer"""
        if not isinstance(buffer, tle.buffered_tensor):
            raise TLESemanticError(f"Buffer must be tle.buffered_tensor, but got {type(buffer)}", "local_load")

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

    def analyze_local_load_operation(self, buffer: tle.buffered_tensor) -> None:
        """Analyze local_load operation semantics"""
        self.validate_local_load_buffer(buffer)
