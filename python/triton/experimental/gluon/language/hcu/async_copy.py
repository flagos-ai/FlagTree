"""HCU target copies into ordinary shared descriptors and standard async groups."""
from .. import _core as core
from .._core import builtin, _unwrap_if_constexpr
from .._layouts import SharedLinearLayout


@builtin
def wait_local_memory(_semantic=None):
    """Complete this wave's preceding LDS operations without a rendezvous.

    The target's shared counter can also complete scalar memory operations.
    Global vector prefetches remain outstanding. This neither publishes data
    to other waves nor replaces commit_group/wait_group for async copies;
    cross-wave communication still requires its usual synchronization.
    """
    from triton._C.libtriton import hcu
    builder = _semantic.builder
    hcu.create_local_memory_wait(builder.get_op_builder(), builder.get_loc())


@builtin
def cta_barrier(_semantic=None):
    """Rendezvous all physical CTA waves and complete preceding LDS accesses.

    Unlike a partition-local thread barrier, this also joins the other warp
    partitions. Async producers must still explicitly wait for their copies;
    unrelated global register prefetches are not drained by this LDS barrier.
    """
    from triton._C.libtriton import hcu
    builder = _semantic.builder
    hcu.create_cta_barrier(builder.get_op_builder(), builder.get_loc())


@builtin
def warp_group_id(group_size=1, _semantic=None):
    """Uniform group index of the calling wave within its warp partition."""
    from triton._C.libtriton import hcu
    size = _unwrap_if_constexpr(group_size)
    assert type(size) is int and size > 0 and size & (size - 1) == 0
    builder = _semantic.builder
    return core.tensor(hcu.create_warp_group_id(
        builder.get_op_builder(), builder.get_loc(), size), core.int32)


@builtin
def sched_barrier(mask=0, _semantic=None):
    """Restrict target instruction motion; this is not thread synchronization."""
    from triton._C.libtriton import hcu
    mask = _unwrap_if_constexpr(mask)
    assert isinstance(mask, int) and 0 <= mask <= 0x7fffffff
    builder = _semantic.builder
    hcu.create_sched_barrier(builder.get_op_builder(), builder.get_loc(), mask)


@builtin
def matrix_load_to_shared(dest, base, shape, strides, offsets, enable=True, _semantic=None):
    """Copy a rank-two f16/bf16 matrix tile, zero-filling out-of-bounds values.

    Uses an ordinary SharedLinear destination; the compiler derives the source
    ownership and hardware mode. There is no operand role or source-layout ABI.
    gfx936 currently requires wave64, complete 16B vectors/waves, aligned scalar
    base, constant aligned row pitch and /8 column extents/origins. The
    row pitch selects structured mode when its byte field fits; otherwise the
    raw byte-bounded path requires a local span below 2GiB; larger global
    tensors use a tile-local resource with explicit row/column bounds. The
    shared tile must be a contiguous actual allocation-backed view. Unsupported
    requests fail compilation; callers can use ordinary buffer copies instead.
    False enable still overwrites the entire tile with zeros. Use standard
    commit_group/wait_group and shared-load tokens; publication/WAR still apply.
    """
    assert isinstance(dest, core.shared_memory_descriptor)
    assert len(dest.shape) == 2 and isinstance(dest.layout, SharedLinearLayout)
    shape, strides, offsets = [_unwrap_if_constexpr(v) for v in (shape, strides, offsets)]
    assert len(shape) == len(strides) == len(offsets) == 2
    base = _semantic.to_tensor(base)
    assert base.type.is_ptr() and base.type.scalar.element_ty == dest.dtype
    assert dest.dtype in (core.float16, core.bfloat16)
    def integer(value, dtype):
        value = _semantic.to_tensor(_unwrap_if_constexpr(value))
        assert value.type.is_int() and not value.type.is_block(), "matrix copy operands must be scalar integers"
        return _semantic.cast(value, dtype).handle
    token = _semantic.builder.create_matrix_load_to_shared(dest.handle, base.handle,
        [integer(v, core.int32) for v in shape],
        [integer(v, core.int64) for v in strides],
        [integer(v, core.int32) for v in offsets] + [integer(enable, core.int32)])
    return core.async_token(token)
