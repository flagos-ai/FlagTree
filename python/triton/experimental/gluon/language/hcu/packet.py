"""Python wrappers for the existing HCU whole-packet IR, not a new DMA ABI."""
from dataclasses import dataclass
from triton._C.libtriton import hcu
from triton.language.core import base_type, base_value
from triton.experimental.gluon.language import _core as core
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.experimental.gluon.language._layouts import SharedLayout


@dataclass(frozen=True)
class WholePacketLayout(SharedLayout):
    role: int
    tile_shape: tuple

    def __post_init__(self):
        object.__setattr__(self, 'role', _unwrap_if_constexpr(self.role))
        shape = tuple(_unwrap_if_constexpr(x) for x in _unwrap_if_constexpr(self.tile_shape))
        object.__setattr__(self, 'tile_shape', shape)
        assert self.role in (0, 1) and len(shape) == 2
        assert all(x > 0 and not x & (x - 1) for x in shape)
        assert shape[0] * shape[1] == 4096

    def _to_ir(self, builder):
        return hcu.get_whole_packet_layout(builder.get_op_builder(), self.role, list(self.tile_shape))

    def mangle(self):
        return f'HCU_WholePacket_v256_role{self.role}_{self.tile_shape[0]}x{self.tile_shape[1]}'


@dataclass(frozen=True)
class PacketPrefixLayout(SharedLayout):
    """v257 active physical prefix, retaining the reserved 8KiB role half."""
    role: int
    tile_shape: tuple

    def __post_init__(self):
        object.__setattr__(self, 'role', _unwrap_if_constexpr(self.role))
        shape = tuple(_unwrap_if_constexpr(x) for x in _unwrap_if_constexpr(self.tile_shape))
        object.__setattr__(self, 'tile_shape', shape)
        assert self.role in (0, 1) and len(shape) == 2
        assert all(x > 0 and not x & (x - 1) for x in shape)
        assert 512 <= shape[0] * shape[1] <= 4096

    def _to_ir(self, builder):
        return hcu.get_packet_prefix_layout(builder.get_op_builder(), self.role, list(self.tile_shape))

    def mangle(self):
        return f'HCU_PacketPrefix_v257_role{self.role}_{self.tile_shape[0]}x{self.tile_shape[1]}'


@dataclass(frozen=True)
class PacketStorageLayout(SharedLayout):
    """v258 explicit DMA kind and logical-to-storage axis order.

    transfer_kind: 0 for byte-bounded raw DMA, 1 for structured row DMA.
    source_order lists logical axes in source storage order. The role continues
    to select only the reserved A/B half and matching dot operand index.
    """
    role: int
    tile_shape: tuple
    transfer_kind: int
    source_order: tuple = (0, 1)

    def __post_init__(self):
        for key in ('role', 'transfer_kind'):
            object.__setattr__(self, key, _unwrap_if_constexpr(getattr(self, key)))
        for key in ('tile_shape', 'source_order'):
            value = tuple(_unwrap_if_constexpr(x) for x in _unwrap_if_constexpr(getattr(self, key)))
            object.__setattr__(self, key, value)
        shape = self.tile_shape
        assert type(self.role) is int and self.role in (0, 1)
        assert type(self.transfer_kind) is int and self.transfer_kind in (0, 1)
        assert len(shape) == 2 and all(type(x) is int and x > 0 and not x & (x - 1) for x in shape)
        assert 512 <= shape[0] * shape[1] <= 4096
        assert all(type(x) is int for x in self.source_order)
        assert self.source_order in ((0, 1), (1, 0))

    def _to_ir(self, builder):
        return hcu.get_packet_storage_layout(builder.get_op_builder(), self.role,
            list(self.tile_shape), self.transfer_kind, list(self.source_order))

    def mangle(self):
        return (f'HCU_PacketStorage_v258_role{self.role}_{self.tile_shape[0]}x{self.tile_shape[1]}'
                f'_dma{self.transfer_kind}_order{self.source_order[0]}{self.source_order[1]}')


class PacketTokenType(base_type):
    def to_ir(self, builder):
        return hcu.get_whole_packet_token_type(builder.get_op_builder())

    def _flatten_ir_types(self, builder, out):
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        return PacketToken(handles[cursor]), cursor + 1

    def __eq__(self, other):
        return isinstance(other, PacketTokenType)

    def __hash__(self):
        return hash('HCUWholePacketAsyncTokenV1')

    def mangle(self):
        return 'HCUWholePacketAsyncTokenV1'


class PacketToken(base_value):
    def __init__(self, handle):
        self.handle = handle
        self.type = PacketTokenType()

    def _flatten_ir(self, handles):
        handles.append(self.handle)


def _builder(semantic):
    return semantic.builder.get_op_builder(), semantic.builder.get_loc()


def _integer(semantic, value, dtype):
    value = semantic.to_tensor(_unwrap_if_constexpr(value))
    assert value.type.scalar.is_int()
    assert not value.type.is_block(), 'extent/stride/origin must be scalar'
    return semantic.cast(value, dtype).handle


def _basis(value):
    basis = tuple(_unwrap_if_constexpr(x) for x in _unwrap_if_constexpr(value))
    assert len(basis) == 12
    return list(basis)


def _tokens(tokens):
    tokens = _unwrap_if_constexpr(tokens)
    assert all(isinstance(x, PacketToken) for x in tokens)
    return [x.handle for x in tokens]


@builtin
def whole_packet_owner(dtype, slots, _semantic=None):
    dtype, slots = _unwrap_if_constexpr(dtype), _unwrap_if_constexpr(slots)
    assert dtype in (core.bfloat16, core.float16) and 1 <= slots <= 4
    builder, loc = _builder(_semantic)
    handle = hcu.create_whole_packet_owner(builder, loc, dtype.to_ir(_semantic.builder), slots)
    # One leading slot axis above the rank-one flat byte/frame layout.
    from triton.experimental.gluon.language import SwizzledSharedLayout
    layout = SwizzledSharedLayout(1, 1, 1, [0])
    return core.shared_memory_descriptor(handle, dtype, [slots, 8192], layout, [slots, 8192])


@builtin
def packet_copy(base, dest, shape, strides, indices, basis, _semantic=None):
    builder, loc = _builder(_semantic)
    base = _semantic.to_tensor(base)
    assert base.type.is_ptr() and isinstance(dest.layout, (WholePacketLayout, PacketPrefixLayout, PacketStorageLayout))
    assert len(shape) == len(strides) == 2 and len(indices) == 3
    handle = hcu.create_whole_packet_copy(
        builder, loc, base.handle, dest.handle,
        [_integer(_semantic, x, core.int32) for x in shape],
        [_integer(_semantic, x, core.int64) for x in strides],
        [_integer(_semantic, x, core.int32) for x in indices], _basis(basis))
    return PacketToken(handle)


@builtin
def packet_read(source, layout, basis, _semantic=None):
    # Explicitly tokenless: the validated HCU membar path must retain WAR.
    layout = _unwrap_if_constexpr(layout)
    ty = core.distributed_type(source.dtype, source.shape, layout)
    builder, loc = _builder(_semantic)
    result = hcu.create_whole_packet_read(builder, loc, source.handle,
                                          ty.to_ir(_semantic.builder), _basis(basis), None)
    return core.tensor(result, ty)


@builtin
def packet_commit(tokens, _semantic=None):
    builder, loc = _builder(_semantic)
    return PacketToken(hcu.create_whole_packet_commit(builder, loc, _tokens(tokens)))


@builtin
def packet_wait(tokens, num_outstanding, _semantic=None):
    builder, loc = _builder(_semantic)
    count = _unwrap_if_constexpr(num_outstanding)
    assert isinstance(count, int) and 0 <= count <= 63
    return PacketToken(hcu.create_whole_packet_wait(builder, loc, _tokens(tokens), count))


@builtin
def packet_sched_barrier(_semantic=None):
    builder, loc = _builder(_semantic)
    hcu.create_packet_sched_barrier(builder, loc, 0)


@builtin
def packet_buffer_store(value, base, offsets, mask=None, _semantic=None):
    builder, loc = _builder(_semantic)
    value, base, offsets = map(_semantic.to_tensor, (value, base, offsets))
    assert base.type.is_ptr() and offsets.dtype == core.int32
    assert value.shape == offsets.shape and value.type.layout == offsets.type.layout
    if mask is not None:
        mask = _semantic.cast(_semantic.to_tensor(mask), core.int1)
        assert mask.shape == value.shape and mask.type.layout == value.type.layout
    hcu.create_buffer_store(builder, loc, value.handle, base.handle, offsets.handle,
                             None if mask is None else mask.handle)
