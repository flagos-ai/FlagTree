"""
DSA Semantic Validation Layer
=============================

Provides early, human-readable error messages for invalid TLE DSA operations
before they reach the MLIR lowering pipeline.  Mirrors the role of
``flagtree_tle``'s ``TLESemantic`` class but adapted for the DSA backend.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import triton.language.core as tl
from . import types as tle


class DSASemanticError(Exception):
    """Raised when a DSA operation fails semantic validation."""
    pass


# Data types supported by the TsingMicro DSA backend for buffer allocation.
_SUPPORTED_ALLOC_DTYPES = frozenset([
    tl.float32,
    tl.float16,
    tl.bfloat16,
    tl.int8,
    tl.int16,
    tl.int32,
    tl.int64,
    tl.uint8,
    tl.uint16,
    tl.uint32,
    tl.uint64,
])


class DSASemantic:
    """Semantic analyzer for DSA TLE operations.

    Each ``validate_*`` method raises :class:`DSASemanticError` with a
    descriptive message if validation fails, and returns silently on
    success.
    """

    # ------------------------------------------------------------------
    # alloc() validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_alloc_shape(shape: Sequence) -> Tuple[int, ...]:
        """Validate and normalise *shape* for ``alloc()``.

        Returns the unwrapped shape tuple on success.
        """
        if not isinstance(shape, (tuple, list)):
            if hasattr(shape, "__iter__"):
                shape = tuple(shape)
            else:
                raise DSASemanticError(f"alloc: shape must be a tuple or list, got {type(shape).__name__}")

        unwrapped = []
        for i, dim in enumerate(shape):
            dim = tl._unwrap_if_constexpr(dim)
            if not isinstance(dim, int) or dim <= 0:
                raise DSASemanticError(f"alloc: shape[{i}] must be a positive integer, got {dim!r}")
            unwrapped.append(dim)
        return tuple(unwrapped)

    @staticmethod
    def validate_alloc_dtype(dtype: tl.dtype) -> tl.dtype:
        """Validate *dtype* for ``alloc()``."""
        dtype = tl._unwrap_if_constexpr(dtype)
        if not isinstance(dtype, tl.dtype):
            raise DSASemanticError(f"alloc: dtype must be a tl.dtype instance, got {type(dtype).__name__}")
        if dtype not in _SUPPORTED_ALLOC_DTYPES:
            supported = ", ".join(str(d) for d in sorted(_SUPPORTED_ALLOC_DTYPES, key=str))
            raise DSASemanticError(f"alloc: unsupported dtype {dtype}. Supported types: {supported}")
        return dtype

    @staticmethod
    def validate_alloc_scope(scope) -> tle.scope:
        """Validate *scope* for ``alloc()``."""
        if scope is None:
            return tle.spm  # default
        if not isinstance(scope, tle.scope):
            raise DSASemanticError(f"alloc: scope must be a tle.scope instance, got {type(scope).__name__}")
        return scope


def _convert_index_to_ir(elem, builder):
    elem = tl._unwrap_if_constexpr(elem)
    if isinstance(elem, int):
        return builder.get_int32(elem)
    if isinstance(elem, tl.tensor):
        return elem.handle
    raise TypeError(f"Unsupported index element type: {type(elem)}")


def _prepare_tv_indices(view, index, builder):
    tile = list(view.tile)
    rank = len(tile)
    assert rank > 0, "tensor_view has no encoding; use `make_partition_view` first"
    idx = list(index)
    assert len(idx) == rank, f"Expected {rank} entries in `index`, got {len(idx)}"
    return [_convert_index_to_ir(i, builder) for i in idx], tile


def _space_to_ir_attr(space, builder):
    if space is None:
        get_null = getattr(builder, "dsa_get_null_attr", None)
        return get_null() if get_null is not None else None
    return space.to_ir(builder)


def tile_load(src, index, dst_space, builder):
    assert index is not None, "`index` is required"
    index_handles, tile = _prepare_tv_indices(src, index, builder)
    result_ty = tl.block_type(src.dtype, tile)
    space_attr = _space_to_ir_attr(dst_space, builder)
    handle = builder.create_tile_load(src.handle, index_handles, result_ty.to_ir(builder), space_attr)
    return tl.tensor(handle, result_ty)


def tile_store(src, dst, value, index, builder, src_space=None):
    assert value is not None, "`value` is required"
    assert index is not None, "`index` is required"
    index_handles, _ = _prepare_tv_indices(src, index, builder)
    space_attr = _space_to_ir_attr(src_space, builder)
    builder.create_tile_store(value.handle, src.handle, index_handles, space_attr)
