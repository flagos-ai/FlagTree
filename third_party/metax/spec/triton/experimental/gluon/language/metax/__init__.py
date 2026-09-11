from triton._C.libtriton import ir

from .. import _core as ttgl
from .._core import _unwrap_if_constexpr, builtin
from .._layouts import MACAMmaLayout
from .._semantic import _check

__all__ = [
    "MACAMmaLayout",
    "async_copy_global_to_shared",
    "bsm_perm",
    "slice",
    "slice_update",
    "gvm_arrive",
    "barrier",
    "barrier_shared",
    "sched_bound",
    "iglp",
]


@builtin
def async_copy_global_to_shared(smem, pointer, mask=None, other=None, cache_modifier="", eviction_policy="",
                                volatile=False, intrinsic=True, _semantic=None):
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    volatile = _unwrap_if_constexpr(volatile)
    intrinsic = _unwrap_if_constexpr(intrinsic)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)

    _check(pointer.type.is_block(), lambda: "expected pointer to be a tensor")
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape={smem.shape}, pointer.shape={pointer.shape}"
    )
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, pointer.dtype.element_ty)
        pointer, other = _semantic.broadcast_impl_value(pointer, other)

    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle, other_handle,
                                                        cache_modifier, eviction_policy, volatile, intrinsic)


@builtin
def bsm_perm(value, dtype, _semantic=None):
    dtype = _unwrap_if_constexpr(dtype)
    _check(isinstance(value, ttgl.tensor), lambda: "value must be a tensor")
    _check(isinstance(value.type, ttgl.distributed_type), lambda: "value must have a distributed_type")
    ret_ty = ttgl.distributed_type(dtype, value.shape, value.type.layout)
    handle = _semantic.builder.create_bsm_perm(ret_ty.to_ir(_semantic.builder), value.handle)
    return ttgl.tensor(handle, ret_ty)


@builtin
def slice(source, shape, offsets, _semantic=None):
    shape = _normalize_static_int_list("shape", shape)
    offsets = _normalize_slice_offsets("offsets", offsets)
    if isinstance(source, ttgl.shared_memory_descriptor):
        return _slice_memdesc(source, shape, offsets, _semantic)
    return _slice_tensor(source, shape, offsets, _semantic)


def _slice_tensor(source, shape, offsets, _semantic):
    offsets = _normalize_static_int_list("offsets", offsets)
    _check(isinstance(source, ttgl.tensor), lambda: "source must be a tensor")
    _check(isinstance(source.type, ttgl.distributed_type), lambda: "source must have a distributed_type")
    source_shape = _normalize_static_int_list("source.shape", source.shape)
    _check(len(shape) == len(source_shape),
           lambda: f"shape rank must match source rank, got shape={shape}, source.shape={source_shape}")
    _check(len(offsets) == len(source_shape),
           lambda: f"offset rank must match source rank, got offsets={offsets}, source.shape={source_shape}")
    for i, (size, offset, extent) in enumerate(zip(shape, offsets, source_shape)):
        _check(size > 0, lambda i=i, size=size: f"shape[{i}] must be positive, got {size}")
        _check(offset >= 0, lambda i=i, offset=offset: f"offsets[{i}] must be non-negative, got {offset}")
        _check(offset + size <= extent,
               lambda i=i, offset=offset, size=size, extent=extent:
               f"slice dim {i} out of bounds: offset {offset} + size {size} exceeds extent {extent}")
    ret_ty = ttgl.distributed_type(source.dtype, shape, source.type.layout)
    handle = _semantic.builder.create_extract_slice(ret_ty.to_ir(_semantic.builder), source.handle, offsets)
    return ttgl.tensor(handle, ret_ty)


def _slice_memdesc(source, shape, offsets, _semantic):
    source_shape = _normalize_static_int_list("source.shape", source.shape)
    if len(offsets) == len(source_shape) and len(shape) == len(source_shape) - 1:
        desc = source.index(offsets[0], _semantic=_semantic)
        return _slice_memdesc_same_rank(desc, shape, offsets[1:], _semantic)
    if len(offsets) == len(source_shape) and len(shape) == len(source_shape):
        return _slice_memdesc_same_rank(source, shape, offsets, _semantic)
    _check(
        False, lambda: f"shared slice expects shape rank {len(source_shape)} or {len(source_shape) - 1} "
        f"with {len(source_shape)} offsets, got shape={shape}, offsets={offsets}, source.shape={source_shape}")


def _slice_memdesc_same_rank(source, shape, offsets, _semantic):
    source_shape = _normalize_static_int_list("source.shape", source.shape)
    offsets = _normalize_static_int_list("offsets", offsets)
    _check(len(shape) == len(source_shape),
           lambda: f"shape rank must match source rank, got shape={shape}, source.shape={source_shape}")
    _check(len(offsets) == len(source_shape),
           lambda: f"offset rank must match source rank, got offsets={offsets}, source.shape={source_shape}")
    result = source
    for dim, (size, offset, extent) in enumerate(zip(shape, offsets, source_shape)):
        _check(size > 0, lambda dim=dim, size=size: f"shape[{dim}] must be positive, got {size}")
        _check(offset >= 0, lambda dim=dim, offset=offset: f"offsets[{dim}] must be non-negative, got {offset}")
        _check(offset + size <= extent,
               lambda dim=dim, offset=offset, size=size, extent=extent:
               f"shared slice dim {dim} out of bounds: offset {offset} + size {size} exceeds extent {extent}")
        if size != extent or offset != 0:
            result = result.slice(offset, size, dim, _semantic=_semantic)
    return result


@builtin
def slice_update(base, update, offsets, _semantic=None):
    offsets = _normalize_static_int_list("offsets", offsets)
    _check(isinstance(base, ttgl.tensor), lambda: "base must be a tensor")
    _check(isinstance(update, ttgl.tensor), lambda: "update must be a tensor")
    _check(isinstance(base.type, ttgl.distributed_type), lambda: "base must have a distributed_type")
    _check(isinstance(update.type, ttgl.distributed_type), lambda: "update must have a distributed_type")
    _check(base.dtype == update.dtype, lambda: f"base/update dtype mismatch: {base.dtype} vs {update.dtype}")
    base_shape = _normalize_static_int_list("base.shape", base.shape)
    update_shape = _normalize_static_int_list("update.shape", update.shape)
    _check(len(offsets) == len(base_shape),
           lambda: f"offset rank must match base rank, got offsets={offsets}, base.shape={base_shape}")
    _check(len(update_shape) == len(base_shape),
           lambda: f"update rank must match base rank, got update.shape={update_shape}, base.shape={base_shape}")
    for i, (size, offset, extent) in enumerate(zip(update_shape, offsets, base_shape)):
        _check(size > 0, lambda i=i, size=size: f"update.shape[{i}] must be positive, got {size}")
        _check(offset >= 0, lambda i=i, offset=offset: f"offsets[{i}] must be non-negative, got {offset}")
        _check(offset + size <= extent,
               lambda i=i, offset=offset, size=size, extent=extent:
               f"slice_update dim {i} out of bounds: offset {offset} + size {size} exceeds extent {extent}")
    handle = _semantic.builder.create_insert_slice(base.type.to_ir(_semantic.builder), base.handle, update.handle,
                                                   offsets)
    return ttgl.tensor(handle, base.type)


def _normalize_static_int_list(name, values):
    values = _unwrap_if_constexpr(values)
    if not isinstance(values, (list, tuple)):
        try:
            values = list(values)
        except TypeError:
            _check(False, lambda: f"{name} must be a list or tuple")
    normalized = [_unwrap_if_constexpr(value) for value in values]
    for i, value in enumerate(normalized):
        _check(isinstance(value, int), lambda i=i, value=value: f"{name}[{i}] must be a constant int, got {value}")
    return normalized


def _normalize_slice_offsets(name, values):
    values = _unwrap_if_constexpr(values)
    if not isinstance(values, (list, tuple)):
        try:
            values = list(values)
        except TypeError:
            _check(False, lambda: f"{name} must be a list or tuple")
    return [_unwrap_if_constexpr(value) for value in values]


@builtin
def gvm_arrive(num, _semantic=None):
    num = _unwrap_if_constexpr(num)
    _semantic.builder.create_gvm_arrive(num)


@builtin
def barrier(_semantic=None):
    _semantic.builder.create_maca_barrier()


@builtin
def barrier_shared(_semantic=None):
    _semantic.builder.create_maca_barrier_shared()


@builtin
def sched_bound(_semantic=None):
    _semantic.builder.create_maca_sched_bound()


@builtin
def iglp(config_0=0, config_1=-1, config_2=-1, config_3=-1, config_4=-1, config_5=-1, config_6=-1, config_7=-1,
         _semantic=None):
    config_0 = _unwrap_if_constexpr(config_0)
    config_1 = _unwrap_if_constexpr(config_1)
    config_2 = _unwrap_if_constexpr(config_2)
    config_3 = _unwrap_if_constexpr(config_3)
    config_4 = _unwrap_if_constexpr(config_4)
    config_5 = _unwrap_if_constexpr(config_5)
    config_6 = _unwrap_if_constexpr(config_6)
    config_7 = _unwrap_if_constexpr(config_7)
    _semantic.builder.create_maca_iglp(config_0, config_1, config_2, config_3, config_4, config_5, config_6, config_7)
