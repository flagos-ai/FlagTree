"""Explicit gfx936 32KiB raw DMA windows and distributed subtile reads."""
from triton._C.libtriton import hcu
from triton.experimental.gluon.language import _core as core
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from .packet import PacketToken, _builder, _integer

@builtin
def raw_window_owner(dtype, slots=2, _semantic=None):
    dtype, slots = _unwrap_if_constexpr(dtype), _unwrap_if_constexpr(slots)
    assert dtype in (core.float16, core.bfloat16) and type(slots) is int and 1 <= slots <= 2
    b, loc = _builder(_semantic)
    handle = hcu.create_raw_window_owner(b, loc, dtype.to_ir(_semantic.builder), slots)
    from triton.experimental.gluon.language import SwizzledSharedLayout
    return core.shared_memory_descriptor(handle, dtype, [slots,16384],
        SwizzledSharedLayout(1,1,1,[0]), [slots,16384])

def _constants(value, length):
    value = tuple(_unwrap_if_constexpr(x) for x in _unwrap_if_constexpr(value))
    assert len(value) == length and all(type(x) is int for x in value)
    return list(value)

@builtin
def raw_window_copy(base, dest, rows, row_origin, column_origin, enabled,
                    pitch, storage_shape, basis, _semantic=None):
    b, loc = _builder(_semantic)
    base = _semantic.to_tensor(base)
    pitch = _unwrap_if_constexpr(pitch)
    assert base.type.is_ptr() and type(pitch) is int and pitch > 0
    assert dest.shape == [16384]
    enabled = _semantic.to_tensor(enabled)
    assert not enabled.type.is_block()
    handle = hcu.create_raw_window_copy(b, loc, base.handle, dest.handle,
        _integer(_semantic, rows, core.int32), _integer(_semantic, row_origin, core.int32),
        _integer(_semantic, column_origin, core.int32),
        _semantic.cast(enabled, core.int1).handle, pitch,
        _constants(storage_shape,2), _constants(basis,14))
    return PacketToken(handle)

@builtin
def raw_window_read(source, shape, layout, storage_shape, basis,
                    origin=(0,0), order=(0,1), full_wait=False, global_wait=False, _semantic=None):
    shape = _constants(shape,2);layout = _unwrap_if_constexpr(layout)
    assert source.shape == [16384]
    ty = core.distributed_type(source.dtype, shape, layout)
    b, loc = _builder(_semantic)
    full_wait = _unwrap_if_constexpr(full_wait)
    assert type(full_wait) is bool
    global_wait = _unwrap_if_constexpr(global_wait)
    assert type(global_wait) is bool
    arguments = (b, loc, source.handle, ty.to_ir(_semantic.builder),
        _constants(storage_shape,2), _constants(basis,14),
        _constants(origin,2), _constants(order,2))
    if global_wait:
        handle = hcu.create_raw_window_read_completion(*arguments, full_wait, True)
    else:
        create = hcu.create_raw_window_read_full if full_wait else hcu.create_raw_window_read
        handle = create(*arguments)
    return core.tensor(handle, ty)

@builtin
def raw_window_wait(token, _semantic=None):
    assert isinstance(token, PacketToken)
    b, loc = _builder(_semantic)
    hcu.create_raw_window_wait(b, loc, token.handle)

@builtin
def raw_warp_group_id(group_size, _semantic=None):
    group_size = _unwrap_if_constexpr(group_size)
    assert type(group_size) is int and group_size > 0 and group_size & (group_size-1) == 0
    b, loc = _builder(_semantic)
    return core.tensor(hcu.create_raw_warp_group_id(b, loc, group_size), core.int32)
