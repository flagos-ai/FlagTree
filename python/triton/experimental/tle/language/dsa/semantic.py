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
