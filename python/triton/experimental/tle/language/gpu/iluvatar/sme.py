# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit warp-owned flat SME transfers and transaction completion.

The destination is a pointer into an existing shared allocation. Each active
warp copies 512 or 1024 bytes without a swizzle. The caller must give each
warp its own destination and wait before consuming or reusing that storage.
Cross-warp consumers additionally require a block barrier after the wait.
"""
import triton.language.core as tl
from triton.language.core import _unwrap_if_constexpr
from .copy import enabled


@tl.builtin
def copy(src, dst, byte_offset=0, active=True, nbytes=1024, _semantic=None):
    """Enqueue one natural-order transfer per active warp.

    Pointer and predicate tensors must be uniform within each warp. Source
    addresses require 64-byte alignment and destinations require 8-byte
    alignment. ``byte_offset`` is an int32 offset from the source pointer.
    Only lane zero issues the transfer. ``active=False`` leaves the shared
    destination unchanged and performs no source access.
    ``nbytes`` selects a compile-time extent of 512 or 1024 bytes.
    """
    if not enabled() or not hasattr(_semantic.builder, "create_iluvatar_sme_copy"):
        raise RuntimeError("Flat SME copy requires the Iluvatar TLE backend")
    if not src.dtype.is_ptr() or src.dtype.address_space != 1:
        raise ValueError("Flat SME source must be a global-memory pointer")
    if not dst.dtype.is_ptr() or dst.dtype.address_space != 3:
        raise ValueError("Flat SME destination must be a shared-memory pointer")
    if src.shape != dst.shape:
        raise ValueError("Flat SME pointer shapes must match")
    extent = _unwrap_if_constexpr(nbytes)
    if type(extent) is not int or extent not in (512, 1024):
        raise ValueError("Flat SME nbytes must be a compile-time 512 or 1024")
    offset = _semantic.to_tensor(byte_offset).to(tl.int32, _semantic=_semantic)
    predicate = _semantic.to_tensor(active).to(tl.int1, _semantic=_semantic)
    # Pointer arithmetic may infer a normal per-lane layout even when its
    # warp index had an explicit layout. Anchor the transport operands at
    # the copy boundary so conversion cannot change who issues each block.
    if src.shape:
        from ..core import set_layout
        from ..types import BlockEncoding
        warps = int(_semantic.builder.options.num_warps)
        rank = len(src.shape)
        lanes = [1] * rank
        owners = [1] * rank
        lanes[-1] = 64
        owners[0] = warps
        layout = BlockEncoding([1] * rank, lanes, owners, list(reversed(range(rank))))
        src = set_layout(src, layout, _semantic=_semantic)
        dst = set_layout(dst, layout, _semantic=_semantic)
        if offset.shape:
            offset = set_layout(offset, layout, _semantic=_semantic)
        if predicate.shape:
            predicate = set_layout(predicate, layout, _semantic=_semantic)
    _semantic.builder.create_iluvatar_sme_copy(src.handle, dst.handle, offset.handle, predicate.handle, extent)


@tl.builtin
def wait(max_pending=0, drain_reads=False, _semantic=None):
    """Complete older transfers in each issuing lane's FIFO.

    ``max_pending`` counts individual transactions, rather than commit
    groups, and is in [0, 63]. At the pipeline tail use zero even when the
    steady state retains newer transfers. ``drain_reads=True`` also drains
    each calling lane's independent shared-read queue before slot reuse.
    Each warp rendezvous after its issuing lane completes the wait. This
    operation does not synchronize other warps.
    """
    if not enabled() or not hasattr(_semantic.builder, "create_iluvatar_sme_wait"):
        raise RuntimeError("SME transaction wait requires the Iluvatar TLE backend")
    pending = _unwrap_if_constexpr(max_pending)
    reads = _unwrap_if_constexpr(drain_reads)
    if type(pending) is not int or not 0 <= pending <= 63:
        raise ValueError("SME max_pending must be a compile-time integer in [0, 63]")
    if type(reads) is not bool:
        raise ValueError("SME drain_reads must be a compile-time bool")
    _semantic.builder.create_iluvatar_sme_wait(pending, reads)
