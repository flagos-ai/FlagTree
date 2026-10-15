from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import _unwrap_if_constexpr, builtin
from triton.experimental.gluon.language._semantic import _check

from .._layouts import IluvatarBlockedLayout, IluvatarSwizzledSharedLayout

__all__ = ["async_copy_global_to_shared", "commit_group", "wait_group"]


@builtin
def async_copy_global_to_shared(smem, pointer, stride, mask=None, other=None, cache_modifier="", eviction_policy="",
                                volatile=False, _semantic=None):
    """Copy a regular 2D tile from global to shared memory with Iluvatar SME.

    ``stride`` is the distance between adjacent rows in elements. Static
    strides must satisfy the SME 64-byte alignment requirement.

    Source pointers must use an :class:`IluvatarBlockedLayout` with
    ``is_sme=True``. Destination shared memory must use an
    :class:`IluvatarSwizzledSharedLayout` with ``use_tcu=True``.
    """
    _check(pointer.type.is_block(), lambda: "expected pointer to be a tensor")
    _check(
        isinstance(pointer.type.layout, IluvatarBlockedLayout) and pointer.type.layout.is_sme,
        lambda: "expected pointer to use an SME IluvatarBlockedLayout")
    _check(
        isinstance(smem.type.layout, IluvatarSwizzledSharedLayout) and smem.type.layout.use_tcu,
        lambda: "expected destination to use a TCU IluvatarSwizzledSharedLayout")
    _check(len(pointer.shape) == 2, lambda: f"SME requires a rank-2 pointer tensor, got rank {len(pointer.shape)}")
    _check(smem.shape == pointer.shape,
           lambda: f"expected smem shape to match pointer shape but got {smem.shape} and {pointer.shape}")

    element_ty = pointer.dtype.element_ty
    _check(element_ty in (ttgl.int8, ttgl.int16, ttgl.int32, ttgl.float16, ttgl.bfloat16, ttgl.float32),
           lambda: f"unsupported SME element type: {element_ty}")

    # One SME transfer covers 16 rows x 64 bytes, so the contiguous dimension of
    # a tile holds 512 / bitwidth elements. The tile must be an exact multiple of
    # what sme_warps_per_cta covers, otherwise part of it would never be copied.
    layout = pointer.type.layout
    sme_warps_per_cta = layout.sme_warps_per_cta
    _check(
        len(sme_warps_per_cta) == len(pointer.shape),
        lambda: "expected the SME IluvatarBlockedLayout to set sme_warps_per_cta")
    contiguous_dim = layout.order[0]
    tile = [16, 16]
    tile[contiguous_dim] = 512 // element_ty.primitive_bitwidth
    for dim in range(2):
        covered = sme_warps_per_cta[dim] * tile[dim]
        _check(
            pointer.shape[dim] % covered == 0,
            lambda dim=dim, covered=covered: f"SME tile dim {dim} ({pointer.shape[dim]}) must be a multiple of "
            f"sme_warps_per_cta[{dim}] * {tile[dim]} = {covered} for {element_ty}")

    # The engine takes each tile's address out of the pointer tensor, so the
    # distribution has to give every warp exactly one hardware tile: only then
    # does a warp hold the address of the tile it is told to transfer. A warp
    # covers size_per_thread * threads_per_warp elements per dimension.
    for dim in range(2):
        warp_tile = layout.size_per_thread[dim] * layout.threads_per_warp[dim]
        _check(
            warp_tile == tile[dim],
            lambda dim=dim, warp_tile=warp_tile: f"SME needs one hardware tile per warp, so size_per_thread[{dim}] * "
            f"threads_per_warp[{dim}] must be {tile[dim]} for {element_ty}, got {warp_tile}")

    # Warps are numbered along `order`, so the fastest-varying warp index decides
    # which tile a warp is paired with. The two grids have to agree on it, and the
    # blocked grid may only be larger on the other dimension, where the surplus
    # warps sit out the transfer.
    fast, slow = layout.order[0], layout.order[1]
    _check(
        layout.warps_per_cta[fast] == sme_warps_per_cta[fast], lambda: f"SME needs warps_per_cta[{fast}] to equal "
        f"sme_warps_per_cta[{fast}], got {layout.warps_per_cta[fast]} and {sme_warps_per_cta[fast]}")
    _check(
        layout.warps_per_cta[slow] >= sme_warps_per_cta[slow],
        lambda: f"SME needs warps_per_cta[{slow}] to be at least sme_warps_per_cta[{slow}], got "
        f"{layout.warps_per_cta[slow]} and {sme_warps_per_cta[slow]}")

    stride = _unwrap_if_constexpr(stride)
    if isinstance(stride, int):
        _check(stride > 0, lambda: f"SME stride must be positive, got {stride}")
        stride_bytes = stride * element_ty.primitive_bitwidth // 8
        _check(stride_bytes % 64 == 0, lambda: f"SME stride must be 64-byte aligned, got {stride_bytes} bytes")
    stride = _semantic.to_tensor(stride)
    _check(not stride.type.is_block() and stride.dtype.is_int(), lambda: "expected stride to be a scalar integer")

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        if pointer.shape != mask.shape:
            pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
        elif pointer.type.layout != mask.type.layout:
            _check(
                isinstance(mask.type.layout, IluvatarBlockedLayout) and mask.type.layout.sme_mask,
                lambda: "an SME mask with a distinct layout must use IluvatarBlockedLayout(..., sme_mask=True)")
            pointer = _semantic.convert_layout(pointer, mask.type.layout)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, element_ty)
        pointer, other = _semantic.broadcast_impl_value(pointer, other)

    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    mask_handle = mask.handle if mask is not None else ttgl.ir.value()
    other_handle = other.handle if other is not None else ttgl.ir.value()
    _semantic.builder.create_iluvatar_sme_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle,
                                                                     other_handle, stride.handle, cache_modifier,
                                                                     eviction_policy, volatile)


@builtin
def commit_group(_semantic=None):
    """Commit outstanding SME async copies into a waitable group.

    This finalizes a set of ``async_copy_global_to_shared`` operations which can
    be waited upon via ``wait_group``.
    """
    _semantic.builder.create_async_commit_group()


@builtin
def wait_group(num_outstanding=0, _semantic=None):
    """Wait for outstanding SME commit groups.

    Blocks until the number of outstanding ``commit_group``s is less than or
    equal to ``num_outstanding``. Uncommitted async copies are waited upon even
    when ``num_outstanding`` is 0. The compiler converts group counts into
    hardware ``G2S_CNT`` before emitting ``llvm.bi.sl.waitcnt``.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)
