# Triton 3.6 XPU TLE - types
# scope, buffered_tensor, layout types for local/shared memory buffers

from typing import List, Optional, Tuple
from abc import abstractmethod

import triton.language.core as tl
from triton._C.libtriton import ir

# =============================================================================
# Memory scope
# =============================================================================


class scope:
    """Storage scope enum for TLE buffers.

    XPU P800 uses:
    - lmem (local memory) — primary fast storage
    """
    SUPPORTED = ['local_memory', 'share_memory']

    def __init__(self, name: str):
        self.name = name
        assert name in scope.SUPPORTED, f"Unsupported scope: {name}, must be one of {scope.SUPPORTED}"

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, scope):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


lmem = scope('local_memory')
smem = scope('share_memory')

# =============================================================================
# Layout
# =============================================================================


class shared_layout:
    """Base class for shared memory layout encodings."""

    def to_ir(self, builder: ir.builder):
        raise NotImplementedError


class swizzled_shared_layout(shared_layout):
    """Swizzled shared memory layout for bank-conflict-free access."""

    def __init__(self, vectorSize, perPhase, maxPhase, order, numCTAsPerCGA=None, numCTASplit=None, numCTAOrder=None):
        self.vectorSize = vectorSize
        self.perPhase = perPhase
        self.maxPhase = maxPhase
        self.order = order
        self.numCTAsPerCGA = numCTAsPerCGA or [1] * len(order)
        self.numCTASplit = numCTASplit or [1] * len(order)
        self.numCTAOrder = numCTAOrder or list(reversed(range(len(order))))

    @classmethod
    def make_default(cls, rank):
        """Create default layout with no swizzling."""
        return cls(
            vectorSize=1,
            perPhase=1,
            maxPhase=1,
            order=list(reversed(range(rank))),
        )

    def to_ir(self, builder: ir.builder):
        # triton_3.6 API: get_swizzled_shared_layout(vec, per_phase, max_phase, order, cga_layout)
        cga_layout = [self.numCTAsPerCGA, self.numCTASplit, self.numCTAOrder]
        return builder.get_swizzled_shared_layout(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            self.order,
            cga_layout,
        )


# =============================================================================
# buffered_tensor — represents an allocated local/shared memory buffer
# =============================================================================


class buffered_tensor_type(tl.block_type):
    """Type descriptor for a TLE buffered tensor (memdesc-backed)."""

    def __init__(self, element_ty: tl.dtype, shape: List[int], storage: scope, layout: Optional[shared_layout] = None,
                 alloc_shape: List[int] = None):
        super().__init__(element_ty, shape)
        self.storage = storage
        self.layout = layout
        self.alloc_shape = list(shape if alloc_shape is None else alloc_shape)
        # Will be set after alloc to hold the semantic/builder reference
        self._builder_ref = None

    def to_ir(self, builder: ir.builder):
        """Convert to MLIR memdesc type via triton_3.6 API."""
        return builder.get_shared_mem_desc_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.layout.to_ir(builder),
            self.alloc_shape,
        )

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape_str = '_'.join(map(str, self.shape))
        return f'buffered_{elt}S{shape_str}'

    def __eq__(self, other):
        return (type(self) is type(other) and self.shape == other.shape and self.storage == other.storage)

    def __hash__(self):
        return hash((tuple(self.shape), self.storage))

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple['buffered_tensor', int]:
        value = buffered_tensor(handles[cursor], self.scalar, self.shape, self.storage, self.layout,
                                alloc_shape=self.alloc_shape)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List) -> None:
        out.append(self.to_ir(builder))


class buffered_tensor(tl.base_value):
    """A TLE buffered tensor handle — represents allocated local/shared memory.

    Created by `tle.gpu.alloc()`. Used as src/dst for `tle.gpu.copy()` and
    as input to `tle.gpu.local_ptr()`.
    """

    def __init__(self, handle, element_ty: tl.dtype, shape: List[int], storage: scope,
                 layout: Optional[shared_layout] = None, alloc_shape: List[int] = None):
        super().__init__()
        self.handle = handle
        self.shape = shape
        self.dtype = element_ty
        self.type = buffered_tensor_type(element_ty, shape, storage, layout, alloc_shape)

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)
