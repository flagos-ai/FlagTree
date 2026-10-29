# Copyright 2026- Xcoresigma Technology Co., Ltd

from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language import semantic as tl_semantic

from .tile import tensor_view

# Extent given to a sparse dimension whose `shape` entry is omitted.
_UNBOUNDED_EXTENT = 2**63 - 1


def _convert_extent_to_ir(elem, builder: ir.builder):
    """Convert one `shape`/`strides` entry to an i64 IR Value."""
    elem = tl._unwrap_if_constexpr(elem)
    if isinstance(elem, int):
        assert -2**63 <= elem < 2**63, \
            f"TensorView `shape`/`strides` are 64 bit, got {elem} which is out of range"
        return builder.get_int64(elem)
    if isinstance(elem, tl.tensor):
        return elem.handle
    raise TypeError(f"Unsupported shape/strides element type: {type(elem)}")


def _convert_extents_to_ir(list_like, builder: ir.builder):
    if hasattr(list_like, "__iter__"):
        return [_convert_extent_to_ir(elem, builder) for elem in list_like]
    return [_convert_extent_to_ir(list_like, builder)]


def _tv_constants(values, name, expected_length=None):
    values = tl._unwrap_if_constexpr(values)
    values = list(values) if hasattr(values, "__iter__") else [values]
    values = [tl._unwrap_if_constexpr(value) for value in values]
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} entries in `{name}`, but got {len(values)}")
    if not all(isinstance(value, int) for value in values):
        raise ValueError(f"Expected constant integers in `{name}`")
    return values


def _tv_padding_value(value, element_ty):
    value = tl._unwrap_if_constexpr(value)
    if value not in ("zero", "nan", "inf", "-inf"):
        raise ValueError("Expected `padding_value` to be one of: zero, nan, inf, -inf")
    if value != "zero" and not element_ty.is_floating():
        raise ValueError("Expected a floating-point TensorView for nan or infinity padding")
    return value


def _make_tensor_view(base, shape, strides, builder: ir.builder) -> tensor_view:
    """Build the unencoded base view from a scalar pointer."""
    if not base.type.is_ptr() or base.type.element_ty.is_block():
        raise ValueError("Expected `base` to be a pointer type (but not a block pointer type or others)")
    if base.type.element_ty == tl.int1:
        semantic = tl_semantic.TritonSemantic(builder)
        base = semantic.cast(base, tl.pointer_type(tl.int8, base.type.address_space))
    shape = _convert_extents_to_ir(shape, builder)
    strides = _convert_extents_to_ir(strides, builder)
    assert len(shape) == len(strides), "Expected `shape` and `strides` to have the same length"
    handle = builder.create_tensor_view(base.handle, shape, strides)
    return tensor_view(handle, base.type.element_ty, len(shape))


def make_partition_view(base, shape, strides, tile, padding_value, builder: ir.builder) -> tensor_view:
    view = _make_tensor_view(base, shape, strides, builder)
    tile = _tv_constants(tile, "tile", view.rank)
    padding_value = _tv_padding_value(padding_value, view.dtype)
    if not all(value > 0 for value in tile):
        raise ValueError("Expected positive values in `tile`")
    handle = builder.create_partition_view(view.handle, tile, padding_value)
    return tensor_view(handle, view.dtype, view.rank, "partition", tile, padding_value=padding_value)


def make_strided_view(base, shape, strides, tile, traversal_strides, padding_value, builder: ir.builder) -> tensor_view:
    view = _make_tensor_view(base, shape, strides, builder)
    tile = _tv_constants(tile, "tile", view.rank)
    traversal = _tv_constants(traversal_strides, "traversal_strides", view.rank)
    padding_value = _tv_padding_value(padding_value, view.dtype)
    if not all(value > 0 for value in tile + traversal):
        raise ValueError("Expected positive values in `tile` and `traversal_strides`")
    handle = builder.create_strided_view(view.handle, tile, traversal, padding_value)
    return tensor_view(handle, view.dtype, view.rank, "strided", tile, traversal, padding_value=padding_value)


def _gather_scatter_shape(shape, sparse_dims, bounds_check):
    # Without bounds checking a sparse extent is never compared, so an omitted one becomes a constant that
    # adds no kernel argument.
    shape = list(shape) if hasattr(shape, "__iter__") else [shape]
    resolved = []
    for dim, extent in enumerate(shape):
        if tl._unwrap_if_constexpr(extent) is not None:
            resolved.append(extent)
        elif dim not in sparse_dims:
            raise ValueError(f"Expected `shape[{dim}]` to be given: only sparse dimensions may omit it")
        elif bounds_check:
            raise ValueError(f"Expected `shape[{dim}]` to be given when `bounds_check` is enabled")
        else:
            resolved.append(_UNBOUNDED_EXTENT)
    return resolved


def make_gather_scatter_view(base, shape, strides, tile, sparse_dim, padding_value, bounds_check,
                             builder: ir.builder) -> tensor_view:
    bounds_check = tl._unwrap_if_constexpr(bounds_check)
    if not isinstance(bounds_check, bool):
        raise ValueError("Expected `bounds_check` to be a constant boolean")
    sparse_dims = _tv_constants(sparse_dim, "sparse_dim")
    view = _make_tensor_view(base, _gather_scatter_shape(shape, sparse_dims, bounds_check), strides, builder)
    tile = _tv_constants(tile, "tile", view.rank)
    padding_value = _tv_padding_value(padding_value, view.dtype)
    if not all(value > 0 for value in tile):
        raise ValueError("Expected positive values in `tile`")
    if not sparse_dims or len(set(sparse_dims)) != len(sparse_dims):
        raise ValueError("Expected at least one unique dimension in `sparse_dim`")
    if not all(0 <= value < view.rank for value in sparse_dims):
        raise ValueError(f"Expected `sparse_dim` entries in [0, {view.rank})")
    handle = builder.create_gather_scatter_view(view.handle, tile, sparse_dims, padding_value, bounds_check)
    return tensor_view(handle, view.dtype, view.rank, "gather_scatter", tile, (), sparse_dims, padding_value,
                       bounds_check)


def _convert_index_to_ir(elem, builder: ir.builder):
    """Convert a single index element to an IR Value (i32 scalar)."""
    elem = tl._unwrap_if_constexpr(elem)
    if isinstance(elem, int):
        return builder.get_int32(elem)
    if isinstance(elem, tl.tensor):
        return elem.handle
    raise TypeError(f"Unsupported index element type: {type(elem)}")


def _prepare_tv_indices(view, index, builder: ir.builder):
    """Convert an index tuple to IR handles and read the tile from the view."""
    tile = list(view.tile)
    rank = len(tile)
    assert rank > 0, \
        "tensor_view has no encoding; use `make_partition_view` first"

    idx = list(index)
    assert len(idx) == rank, \
        f"Expected {rank} entries in `index`, got {len(idx)}"
    index_handles = [_convert_index_to_ir(i, builder) for i in idx]
    return index_handles, tile


def _space_to_ir_attr(space, builder: ir.builder):
    """Convert a target memory space (e.g. tle.dsa.ascend.UB) to an IR
    attribute the builder attaches to tile.load / tile.store. `None` maps to a
    null attribute, which the backend lowering interprets (UB on Ascend)."""
    if space is None:
        return builder.dsa_get_null_attr()
    return space.to_ir(builder)


def tile_load(src, index, dst_space, builder: ir.builder) -> tl.tensor:
    """Load a tile from an encoded tensor_view at the given index into
    `dst_space` via tile.load (backend default when unspecified; UB on Ascend)."""
    assert index is not None, "`index` is required"
    index_handles, tile = _prepare_tv_indices(src, index, builder)
    result_ty = tl.block_type(src.dtype, tile)
    space_attr = _space_to_ir_attr(dst_space, builder)
    handle = builder.create_tile_load(src.handle, index_handles, result_ty.to_ir(builder), space_attr)
    return tl.tensor(handle, result_ty)


def tile_store(src, value, index, builder: ir.builder, src_space=None):
    """Store a tile into an encoded tensor_view at the given index from
    `src_space` via tile.store (backend default when unspecified; UB on Ascend)."""
    assert value is not None, "`value` is required"
    assert index is not None, "`index` is required"
    index_handles, _ = _prepare_tv_indices(src, index, builder)
    space_attr = _space_to_ir_attr(src_space, builder)
    builder.create_tile_store(value.handle, src.handle, index_handles, space_attr)
