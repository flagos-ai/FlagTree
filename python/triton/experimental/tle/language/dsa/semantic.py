"""
DSA Semantic Validation Layer
=============================

Provides early, human-readable error messages for invalid TLE DSA operations
before they reach the MLIR lowering pipeline.  Mirrors the role of
``flagtree_tle``'s ``TLESemantic`` class but adapted for the TsingMicro /
DSA backend.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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

    # ------------------------------------------------------------------
    # copy() validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_copy_operands(src, dst) -> str:
        """Validate *src*/*dst* types for ``copy()`` and return a direction tag.

        Returns one of ``"SPM_TO_SPM"``, ``"GM_TO_SPM"``, ``"SPM_TO_GM"``.

        Raises :class:`DSASemanticError` if the combination is unsupported.
        """
        src_is_buf = isinstance(src, tle.buffered_tensor)
        dst_is_buf = isinstance(dst, tle.buffered_tensor)

        if src_is_buf and dst_is_buf:
            return "SPM_TO_SPM"
        if (not src_is_buf) and dst_is_buf:
            if not isinstance(src, tl.tensor):
                raise DSASemanticError(f"copy: src must be tl.tensor or buffered_tensor, got {type(src).__name__}")
            return "GM_TO_SPM"
        if src_is_buf and (not dst_is_buf):
            if not isinstance(dst, tl.tensor):
                raise DSASemanticError(f"copy: dst must be tl.tensor or buffered_tensor, got {type(dst).__name__}")
            return "SPM_TO_GM"
        raise DSASemanticError("copy: at least one operand must be a buffered_tensor. "
                               f"Got src={type(src).__name__}, dst={type(dst).__name__}")

    @staticmethod
    def validate_copy_dtype_compat(src_dtype, dst_dtype) -> None:
        """Check that element types of *src* and *dst* are compatible."""
        if src_dtype != dst_dtype:
            raise DSASemanticError(f"copy: element type mismatch – src has {src_dtype}, dst has {dst_dtype}")

    # ------------------------------------------------------------------
    # local_ptr() validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_local_ptr_buffer(buffer) -> None:
        """Validate that *buffer* is a proper ``buffered_tensor``."""
        if not isinstance(buffer, tle.buffered_tensor):
            raise DSASemanticError(f"local_ptr: buffer must be a buffered_tensor, got {type(buffer).__name__}")
        if buffer.type.shape is None:
            raise DSASemanticError("local_ptr: buffer shape is None (deferred shapes not yet supported)")

    @staticmethod
    def validate_local_ptr_indices(
        indices: Sequence,
        buffer_rank: int,
    ) -> None:
        """Validate *indices* for ``local_ptr()``.

        Checks:
        - indices length matches buffer rank
        - all indices are integer-typed
        - indices are either all scalar or all tensor with matching shapes
        """
        if indices is None:
            raise DSASemanticError("local_ptr: indices must be provided as a tuple of tensors")
        if len(indices) != buffer_rank:
            raise DSASemanticError(f"local_ptr: expected {buffer_rank} index tensors, got {len(indices)}")

        view_shape: Optional[tuple] = None
        has_scalar = False
        has_tensor = False

        for i, idx in enumerate(indices):
            if not isinstance(idx, tl.tensor):
                raise DSASemanticError(f"local_ptr: indices[{i}] must be a tl.tensor, "
                                       f"got {type(idx).__name__}")
            if not idx.dtype.is_int():
                raise DSASemanticError(f"local_ptr: indices[{i}] must have integer dtype, "
                                       f"got {idx.dtype}")
            is_scalar = not idx.type.is_block()
            if is_scalar:
                has_scalar = True
            else:
                has_tensor = True
                if view_shape is None:
                    view_shape = tuple(idx.shape)
                elif tuple(idx.shape) != view_shape:
                    raise DSASemanticError(f"local_ptr: index tensor shape mismatch at dim {i}: "
                                           f"expected {view_shape}, got {tuple(idx.shape)}")

        if has_scalar and has_tensor:
            raise DSASemanticError("local_ptr: indices must be either all scalar or all "
                                   "tensor with identical shapes (mixed not allowed)")
