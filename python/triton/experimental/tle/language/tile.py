# flagtree tle
from typing import List, Tuple

from triton._C.libtriton import ir
import triton.language.core as tl
from triton.language.core import tensor

__all__ = [
    "tensor_view_type",
    "tensor_view",
    "make_partition_view",
    "make_strided_view",
    "make_gather_scatter_view",
    "tile_load",
    "tile_store",
]

# -----------------------
# tensor_view
# -----------------------


class tensor_view_type(tl.base_type):
    """Type of an N-dimensional logical TensorView."""

    def __init__(self, element_ty: tl.dtype, rank: int, view_kind=None, tile=(), traversal_strides=(), sparse_dims=(),
                 padding_value="zero", bounds_check=False):
        self.element_ty = element_ty
        self.rank = rank
        self.view_kind = view_kind
        self.tile = tuple(tile)
        self.traversal_strides = tuple(traversal_strides)
        self.sparse_dims = tuple(sparse_dims)
        self.padding_value = padding_value
        self.bounds_check = bounds_check

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple["tensor_view", int]:
        return tensor_view(handles[cursor], self.element_ty, self.rank, self.view_kind, self.tile,
                           self.traversal_strides, self.sparse_dims, self.padding_value, self.bounds_check), cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(
            builder.get_tensor_view_ty(self.element_ty.to_ir(builder), self.rank, self.view_kind or "", self.tile,
                                       self.traversal_strides, self.sparse_dims, self.padding_value, self.bounds_check))

    def __str__(self) -> str:
        suffix = ("" if self.view_kind is None else f", {self.view_kind}, tile={self.tile}, "
                  f"padding={self.padding_value}, bounds_check={self.bounds_check}")
        return f"tensor_view<{self.element_ty}>[{self.rank}{suffix}]"

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return False
        return (self.element_ty == other.element_ty and self.rank == other.rank and self.view_kind == other.view_kind
                and self.tile == other.tile and self.traversal_strides == other.traversal_strides
                and self.sparse_dims == other.sparse_dims and self.padding_value == other.padding_value
                and self.bounds_check == other.bounds_check)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def mangle(self) -> str:
        encoding = "B" if self.view_kind is None else self.view_kind.upper()
        tile = "x".join(str(value) for value in self.tile)
        traversal = "x".join(str(value) for value in self.traversal_strides)
        sparse = "x".join(str(value) for value in self.sparse_dims)
        padding = self.padding_value.upper().replace("-", "N")
        check = "C" if self.bounds_check else "N"
        return f"TV{self.element_ty.mangle()}R{self.rank}K{encoding}T{tile}V{traversal}S{sparse}P{padding}{check}"


class tensor_view(tl.base_value):
    """A logical tensor accessed by `tle.tile_load` and `tle.tile_store`."""

    def __init__(self, handle, element_ty: tl.dtype, rank: int, view_kind=None, tile=(), traversal_strides=(),
                 sparse_dims=(), padding_value="zero", bounds_check=False):
        """Internal constructor."""
        super().__init__()
        self.handle = handle
        self.type = tensor_view_type(element_ty, rank, view_kind, tile, traversal_strides, sparse_dims, padding_value,
                                     bounds_check)

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty

    @property
    def rank(self):
        return self.type.rank

    @property
    def tile(self):
        return self.type.tile

    def __str__(self) -> str:
        return str(self.type)


@tl.builtin
def make_partition_view(base: tensor, shape, strides, tile, padding_value="zero", _semantic=None) -> "tensor_view":
    """Create a tiled partition view over a scalar base pointer.

    ``padding_value`` must be one of ``"zero"``, ``"nan"``, ``"inf"``, or ``"-inf"``.
    """
    from . import semantic
    return semantic.make_partition_view(base, shape, strides, tile, padding_value, _semantic.builder)


@tl.builtin
def make_strided_view(base: tensor, shape, strides, tile, traversal_strides, padding_value="zero",
                      _semantic=None) -> "tensor_view":
    """Create a tiled pointer view whose origins advance by ``traversal_strides``.

    ``padding_value`` must be one of ``"zero"``, ``"nan"``, ``"inf"``, or ``"-inf"``.
    """
    from . import semantic
    return semantic.make_strided_view(base, shape, strides, tile, traversal_strides, padding_value, _semantic.builder)


@tl.builtin
def make_gather_scatter_view(base: tensor, shape, strides, tile, sparse_dim, padding_value="zero", bounds_check=False,
                             _semantic=None) -> "tensor_view":
    """Create a pointer view with tensor-valued indices in selected dimensions.

    ``padding_value`` must be one of ``"zero"``, ``"nan"``, ``"inf"``, or ``"-inf"``.

    ``bounds_check`` is a constant. When true, a sparse index outside ``[0, shape[d])`` reads as
    ``padding_value`` on load and is skipped on store, so an index of ``-1`` marks a row to skip, and
    ``shape`` must be given for every dimension. When false, sparse indices must lie inside
    ``[0, shape[d])``, an out-of-range index is undefined behavior, and ``shape[d]`` of a sparse
    dimension may be ``None``, which keeps it out of the kernel arguments. The ``shape`` of a dense
    dimension is always required because it clips the last tile.
    """
    from . import semantic
    return semantic.make_gather_scatter_view(base, shape, strides, tile, sparse_dim, padding_value, bounds_check,
                                             _semantic.builder)


@tl.builtin
def tile_load(src, index=None, dst_space=None, _semantic=None, _generator=None) -> tensor:
    """Load a tile from an encoded tensor_view via tile.load.

    `dst_space` names the on-chip memory space the loaded tile lands in
    (uses the backend default when unspecified; UB on Ascend).

    Usage:
        tle.tile_load(view, index=(i, j))
        tle.tile_load(view, index=(i, j), dst_space=ascend.UB)
    """
    from . import semantic
    return semantic.tile_load(src, index, dst_space, _semantic.builder)


@tl.builtin
def tile_store(src, value=None, index=None, src_space=None, _semantic=None, _generator=None):
    """Store a tile into an encoded tensor_view via tile.store.

    `src_space` names the on-chip memory space the value tile lives in before
    the store DMA (uses the backend default when unspecified; UB on Ascend).

    Usage:
        tle.tile_store(view, value=tensor_val, index=(i, j))
        tle.tile_store(view, value=tensor_val, index=(i, j), src_space=ascend.UB)
    """
    from . import semantic
    semantic.tile_store(src, value, index, _semantic.builder, src_space)


# Mark the view accessors as TLE builtins, matching the other tile.* frontends.
setattr(tile_load, "__tle_builtin__", True)
setattr(tile_store, "__tle_builtin__", True)
