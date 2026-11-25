# Triton 3.6 XPU TLE - distributed access (`tle.remote`)
#
# Intra-node peer memory access for the jupiter backend, matching the flagtree
# `tle.remote` surface (python/triton/experimental/tle/language/distributed.py).
#
# Usage mirrors flagtree exactly:
#
#     dctx = tle.create_dist_tensor(buf)          # buf on the xccl symmetric heap
#     ...
#     remote_mem = tle.remote(dctx, space="device", dtype=tl.float32,
#                             shard_id=peer, offset=i)
#     val = tl.load(remote_mem)                    # intra-node P2P get
#
# `tle.remote` takes the distributed-tensor handle returned by
# `create_dist_tensor` (flagtree passes a `DistributedRtContext`; here the
# `DistTensorCtx` carries the buffer base plus the per-peer xlink offsets) and
# returns a global pointer to element `offset` inside peer `shard_id`'s copy of
# the symmetric buffer:
#
#     peer_addr = buf_base + (peer_heap_base_p2p[peer] - heap_base) + offset*bytes
#
# On flagtree the peer translation is a device-side op (create_remote_pointers ->
# flagcxGetIntraPointerC). xpu5 has no device-side translation: `tle.remote`
# lowers to plain pointer arithmetic. The per-peer xlink offsets are host-known
# constants carried on the handle, so a compile-time `shard_id` folds into a
# constant address, and a runtime (tensor) `shard_id` selects its offset from
# that constant table via a compile-time-unrolled where-tree -- still pure
# pointer arithmetic, no extern symbol or link step. `peer_heap_base_p2p[peer] -
# heap_base` is read on the host by create_dist_tensor via
# xccl_xshmem_get_xshmem_info (0 for self).
#
# Spaces: intra-node "device" and inter-node "node" both resolve to the same
# host-known P2P offset table, because every rank this
# backend can reach is on one C2C/XLINK fabric. `create_dist_tensor` refuses to
# build a handle when some peer is NOT P2P-reachable (a genuine multi-host run),
# so "node" can never silently produce an address on a host it cannot reach --
# see the node branch of remote(). `device_mesh` scopes still raise
# NotImplementedError.
import os as _os
from dataclasses import asdict, dataclass
from enum import Enum as _Enum
from itertools import product

import triton as _triton
import triton.language as _tll
import triton.language.core as tl

__all__ = ["remote", "signal", "signal_wait", "device_mesh", "MeshConfig",
           "distributed_barrier", "shard_id", "n_pes", "_get_local_rank",
           "_get_world_rank", "BarrierKind", "MemoryOrder", "MemoryScope",
           "GroupKind"]

# On XPU an intra-node peer is reached by plain global-memory P2P translation:
# there is no NVIDIA distributed-shared-memory window to `mapa` into, so there
# is only one intra-node name, "device". Anything else gets the ValueError
# listing the two names that do exist.
_INTRA_NODE_SPACES = ("device", )
# The spaces `remote` recognises.  "node" (inter-node) takes a world rank.
_SPACES = _INTRA_NODE_SPACES + ("node", )

# flagtree's signal/signal_wait team names (distributed.py:136) and the rank
# namespace each one indexes.  "intra_node" peers are local ranks inside my
# node, "inter_node" peers are node indices (the peer with my local rank on that
# node), "world" peers are world ranks.
_SIGNAL_SPACES = {
    "intra": "intra_node",
    "intra_node": "intra_node",
    "device": "intra_node",
    "inter": "inter_node",
    "inter_node": "inter_node",
    "world": "world",
}
# `distributed_barrier` takes the same names as `signal`.
_BARRIER_SPACES = _SIGNAL_SPACES
_COOP_KINDS = ("thread", "warp", "block")


# flagtree's argument enums (its distributed.py:111-134). A kernel copied
# verbatim from a flagtree tutorial passes these members rather than strings --
# `tle.GroupKind.BLOCK`, `order=MemoryOrder.ACQ_REL` -- so they have to exist
# here with the same spellings or the copy fails with an AttributeError.
# `_normalize_coopkind`/`_normalize_choice` already read `.value` off anything
# enum-like, so no call site changes.
#
# Note ACQ_REL's VALUE is "acqrel", with no underscore. That is flagtree's
# spelling and therefore the one a copied kernel passes as a bare string.
class BarrierKind(str, _Enum):
    ARRIVE = "arrive"
    WAIT = "wait"
    SYNC = "sync"


class MemoryOrder(str, _Enum):
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acqrel"


class MemoryScope(str, _Enum):
    SYSTEM = "system"
    DEVICE = "device"
    BLOCK = "block"
    THREAD = "thread"


class GroupKind(str, _Enum):
    THREAD = "thread"
    WARP = "warp"
    BLOCK = "block"
    TILE_SPAN = "tile_span"
    LANES = "lanes"
    GRID = "grid"



def _normalize_space(space):
    space = tl._unwrap_if_constexpr(space)
    if not isinstance(space, str):
        raise TypeError(f"space must be str, got {type(space).__name__}")
    if space not in _SPACES:
        raise ValueError(f"space must be one of {_SPACES!r}, got {space!r}")
    return space


def _normalize_context_id(context_id, prefix=""):
    """flagtree's ``context_id`` contract (distributed.py:223,291,1483)."""
    context_id = tl._unwrap_if_constexpr(context_id)
    if isinstance(context_id, bool):
        raise TypeError(f"{prefix}context_id must be an integer, not bool")
    if not isinstance(context_id, int):
        raise TypeError(f"{prefix}context_id must be a compile-time int, got "
                        f"{type(context_id).__name__}")
    if context_id < 0 or context_id > 0x7FFFFFFF:
        raise ValueError(f"{prefix}context_id must be in int32 range, got {context_id}")
    return context_id


def _normalize_node_context_id(context_id):
    return _normalize_context_id(context_id, "node space ")


def _normalize_node_elem_bytes(dtype):
    """flagtree's node-space ``dtype`` contract (distributed.py:1451)."""
    dtype = tl._unwrap_if_constexpr(dtype)
    if not isinstance(dtype, tl.dtype) or dtype.is_ptr():
        raise TypeError("node space dtype must be a scalar Triton dtype, got "
                        f"{type(dtype).__name__}")
    bits = dtype.primitive_bitwidth
    if bits < 8 or bits % 8:
        raise ValueError(f"node space dtype must be byte-addressable, got {dtype}")
    return bits // 8


def _normalize_offset(offset, _semantic):
    off = offset if isinstance(offset, tl.tensor) else _semantic.to_tensor(offset)
    if not off.dtype.is_int():
        raise TypeError(f"offset must be an integer, got {off.dtype}")
    if off.shape:
        raise ValueError(f"offset must be a scalar, got shape {off.shape}")
    if off.dtype != tl.int64:
        off = tl.cast(off, tl.int64, _semantic=_semantic)
    return off


def _dist_handle(tensor):
    """Unwrap and validate a create_dist_tensor handle."""
    handle = tl._unwrap_if_constexpr(tensor)
    if not (hasattr(handle, "data_ptr") and hasattr(handle, "offsets")):
        raise TypeError(
            "this primitive expects the handle returned by tle.create_dist_tensor(buf), "
            f"got {type(handle).__name__}. xpu5 folds the per-peer xlink offset at "
            "compile time, so pass the handle (not a raw pointer).")
    return handle


def _normalize_coopkind(value, name):
    """flagtree's thread/warp/block cooperation kinds."""
    kind = tl._unwrap_if_constexpr(value)
    kind = getattr(kind, "value", kind)
    kind = str(kind).lower()
    if kind not in _COOP_KINDS:
        raise ValueError(f"{name} must be one of {_COOP_KINDS!r}, got {kind!r}")
    return kind


def _normalize_choice(value, name, choices, default):
    """Validate one of flagtree's small string enums (scope / order)."""
    val = tl._unwrap_if_constexpr(value)
    val = getattr(val, "value", val)
    val = default if val is None else str(val).lower()
    if val not in choices:
        raise ValueError(f"{name} must be one of {choices!r}, got {val!r}")
    return val


def _peer_addr(handle, base, peer, _semantic, table=None):
    """Byte address of ``base`` inside **world rank** ``peer``'s copy of the
    symmetric heap.

    xpu5 has no device-side peer translation (flagtree goes through
    create_remote_pointers -> flagcxGetIntraPointerC); the per-peer xlink
    offsets are host-known constants on the handle, so a compile-time peer folds
    into one constant i64, and a runtime peer selects its offset from that
    constant table with a compile-time-unrolled where-tree.  Pure pointer
    arithmetic -- no device translation op, no extern symbol.

    ``table`` replaces the per-peer constants for a base that is not the heap:
    :func:`_peer_flush_addr` passes the flag-range addresses with ``base=0``.
    """
    offs = [int(x) for x in (handle.offsets if table is None else table)]
    nworld = len(offs)
    peer_val = tl._unwrap_if_constexpr(peer)

    if isinstance(peer_val, int) and not isinstance(peer_val, bool):
        if not (0 <= peer_val < nworld):
            raise ValueError(f"peer rank {peer_val} out of range [0, {nworld})")
        return tl.tensor(_semantic.builder.get_int64(base + offs[peer_val]), tl.int64)
    if not isinstance(peer_val, tl.tensor):
        raise TypeError(f"peer rank must be an int or scalar int tensor, got {type(peer_val).__name__}")
    if peer_val.shape:
        raise ValueError(f"peer rank must be a scalar, got shape {peer_val.shape}")
    if not peer_val.dtype.is_int():
        raise TypeError(f"peer rank must be an integer, got {peer_val.dtype}")

    peer_i = peer_val if peer_val.dtype == tl.int32 else tl.cast(peer_val, tl.int32, _semantic=_semantic)
    off = tl.tensor(_semantic.builder.get_int64(0), tl.int64)
    for p in range(nworld):
        pred = _semantic.equal(peer_i, _semantic.to_tensor(p))
        cand = tl.tensor(_semantic.builder.get_int64(offs[p]), tl.int64)
        off = _semantic.where(pred, cand, off)
    base_t = tl.tensor(_semantic.builder.get_int64(base), tl.int64)
    return _semantic.add(base_t, off, False)


def _world_peer(handle, peer, team, _semantic):
    """Map a peer index in one of flagtree's teams onto a world rank.

    ``intra_node`` peers are local ranks inside my node, ``inter_node`` peers
    are node indices (the rank with my local rank on that node) and ``world``
    peers are already world ranks.  node_rank / local_rank / local_world_size
    are host constants carried on the handle, so this is constant folding for a
    compile-time peer and two integer ops for a runtime one.  With a single
    node both team formulas reduce to the identity, which is why the tutorials
    behave the same whether or not they were written for several nodes.
    """
    if team == "world":
        return peer
    lws = int(handle.local_world_size)
    bias, scale = (int(handle.node_rank) * lws, 1) if team == "intra_node" else (int(handle.local_rank), lws)
    peer_val = tl._unwrap_if_constexpr(peer)
    if isinstance(peer_val, int) and not isinstance(peer_val, bool):
        return peer_val * scale + bias
    peer_t = peer_val if isinstance(peer_val, tl.tensor) else _semantic.to_tensor(peer_val)
    if scale != 1:
        peer_t = _semantic.mul(peer_t, _semantic.to_tensor(scale), False)
    return _semantic.add(peer_t, _semantic.to_tensor(bias), False)


@tl.builtin
def remote(tensor=None, shard_id=None, scope=None, space="device", dtype=None,
           offset=None, coopkind=None, context_id=0, _semantic=None, _generator=None):
    """Return a global pointer into peer ``shard_id``'s symmetric-heap buffer.

    ``space`` is ``"device"`` (intra-node, the default) or ``"node"``
    (inter-node). ``tensor`` is the handle from :func:`create_dist_tensor`;
    ``dtype`` is the
    pointee type of the returned pointer (required); ``offset`` is an element
    offset (``int`` or scalar int tensor) into the peer buffer. ``shard_id`` is
    the peer world rank -- a compile-time ``int`` (folded into a constant
    address) or a scalar int tensor (the per-peer xlink offset is selected from
    the host-known table with a compile-time-unrolled where-tree).

    ``coopkind``/``context_id`` are node-space arguments and are rejected on
    the intra-node ``device`` path, where
    ``shard_id`` is a **local** rank inside this rank's node.
    ``space="node"`` takes a **world** rank and is the same C2C put as the
    intra-node space here -- see the node branch below.
    ``tl.load``/``tl.store`` through the result performs a P2P get/put.
    """
    space = _normalize_space(space)
    if space == "node":
        # flagtree's node-space contract (offset forbidden, dtype required and
        # byte-addressable, context_id a compile-time int32, coopkind one of
        # thread/warp/block), so a kernel written against flagtree gets the same
        # argument errors here.
        if offset is not None:
            raise TypeError('tle.remote(..., space="node") does not accept offset; '
                            'add element offsets to the returned pointer')
        if tl._unwrap_if_constexpr(dtype) is None:
            raise TypeError('tle.remote(..., space="node") requires dtype')
        _normalize_node_elem_bytes(dtype)
        _normalize_node_context_id(context_id)
        if coopkind is not None:
            _normalize_coopkind(coopkind, "coopkind")
        # On this backend the node space resolves through the same host-known
        # P2P offset table as the intra-node ones, because every rank it can
        # address is on one C2C/XLINK fabric: create_dist_tensor refuses to
        # build a handle when a peer is not P2P-reachable, so a genuine
        # multi-host run fails on the HOST instead of putting to a bogus
        # address. Lowering an off-node put properly needs xccl to export its
        # KL5 device-side IBGDA (src/device/kl5/deepep/ibgda_device_mlx5.cuh --
        # xshmemi_ibgda_put_nbi_warp / _amo_nonfetch_add, used by
        # deepep/internode_ll.xpu) as a linkable device ABI; it does not today.
    else:
        # coopkind/context_id are node-only; reject them on the intra-node
        # device path.
        node_only_args = ["coopkind"] if coopkind is not None else []
        unwrapped_context_id = tl._unwrap_if_constexpr(context_id)
        if not isinstance(unwrapped_context_id, int) or unwrapped_context_id != 0:
            node_only_args.append("context_id")
        if node_only_args:
            raise TypeError(
                f'{space} space does not accept node-only argument(s): '
                f'{", ".join(node_only_args)}')
    if tl._unwrap_if_constexpr(scope) is not None:
        raise NotImplementedError(
            "tle.remote(scope=device_mesh) coordinate linearization is not "
            "implemented for the XPU TLE backend; pass a world-rank int shard_id")

    handle = _dist_handle(tensor)

    elem_dtype = tl._unwrap_if_constexpr(dtype)
    if elem_dtype is None:
        raise ValueError("tle.remote requires dtype=<scalar tl.dtype>")
    if not isinstance(elem_dtype, tl.dtype) or elem_dtype.is_ptr():
        raise TypeError(f"tle.remote dtype must be a scalar Triton dtype, got {elem_dtype}")

    # "node" indexes the world; "device" indexes this rank's node (a no-op
    # conversion while the whole world is one node).
    team = "world" if space == "node" else "intra_node"
    addr = _peer_addr(handle, int(handle.data_ptr),
                      _world_peer(handle, shard_id, team, _semantic), _semantic)

    # Fold the element `offset` (may be a runtime scalar) as a byte offset.
    if offset is not None:
        off_e = _normalize_offset(offset, _semantic)
        elem_bytes = elem_dtype.primitive_bitwidth // 8
        addr = _semantic.add(addr, _semantic.mul(off_e, elem_bytes, False), False)

    return tl.cast(addr, tl.pointer_type(elem_dtype, 1), _semantic=_semantic)


# ---------------------------------------------------------------------------
# The local spin (`sign_wait`) every wait is built on.
#
# KL5 has no *cross-chip* communication atomic (xccl's system-scope fence
# degrades to GPU scope), so a rank cannot atomically RMW a peer's memory --
# `signal`'s update is a remote GET + add + remote PUT, which is why a slot
# needs a single writer per epoch. A rank CAN atomically read its own memory,
# and that is all a wait needs: xccl syncs ranks exactly this way, publishing
# into the peer's memory over XLINK (POST_SEND) and spin-reading a local word
# until it changes (WAIT_TO_RECV).
#
# The spin must force a coherent re-read every iteration: a plain (even
# `volatile`) load is loop-invariant to the backend, so LLVM hoists it out of
# the spin and the loop collapses to a single read (verified in the xpu5
# xcnasm). A local read-only atomic (`atomic_add(ptr, 0)`) is a memory side
# effect that cannot be hoisted; on xpu5 it lowers to `global_atomic_or ... glc`
# + `s_dcache_inv`, i.e. a globally-coherent, cache-invalidating re-read --
# exactly the KL5 WAIT_TO_RECV spin.
# ---------------------------------------------------------------------------


@_triton.jit
def _spin_until(flag_ptr, value, max_iters: _tll.constexpr = 1073741824):
    """Spin on a *local* int32 at ``flag_ptr`` until it reaches ``value``.

    This is the raw-pointer spin underneath :func:`signal_wait`, and the only
    place it is used directly is an **intra-device grid barrier** -- every
    program bumps a counter in ordinary global memory with ``tl.atomic_add``
    and then waits here for the whole grid.  flagtree has no primitive for
    that: its ``signal_wait`` needs a distributed handle and addresses that
    handle's signal pad.  For anything cross-rank use :func:`signal_wait`.

    Each read is a local read-only atomic (``atomic_add(flag_ptr, 0)``) rather
    than a plain load: a plain/volatile load is loop-invariant, so the backend
    hoists it and the spin collapses to a single read (confirmed in the xpu5
    xcnasm), whereas the atomic is a memory side effect that stays in the loop
    and lowers to a globally-coherent, cache-invalidating re-read
    (``global_atomic_or ... glc`` + ``s_dcache_inv``) -- the KL5 way (xccl
    KL5 ``WAIT_TO_RECV`` spin).  The atomic is local, so it does not need the
    cross-chip atomic this part lacks.  ``max_iters`` caps the spin so a mis-wired
    run fails fast instead of hanging.  Returns the last value read
    (``>= value`` on success).
    """
    got = _tll.atomic_add(flag_ptr, 0)
    i = 0
    while got < value and i < max_iters:
        got = _tll.atomic_add(flag_ptr, 0)
        i = i + 1
    return got


# ---------------------------------------------------------------------------
# flagtree-compatible signal / signal_wait (flagtree distributed.py:174,258).
#
# flagtree updates a FlagCX *signal slot* -- a counter in a pad that belongs to
# the distributed handle, separate from the data buffer. xpu5 has no FlagCX, so
# `create_dist_tensor` allocates that pad on the xccl symmetric heap next to the
# buffer and the handle carries its base; a slot is one int32.
#
# `op="inc"`/`"add"` is a remote read-modify-write (remote GET, add, remote PUT),
# NOT an atomic: KL5 has no cross-chip communication atomic. That is safe
# exactly when a slot has a single writer per epoch, which is how the signal
# namespace is used (one slot per source rank). It is NOT a general atomic
# counter -- two ranks incrementing the same slot can lose an update.
# The remote GET has a useful side effect: its response has to come back from
# the peer before the PUT is issued, and the C2C path serves requests to one
# destination in order, so the payload stores a producer issued earlier have
# been accepted by the time the slot update lands. That is what lets flagtree's
# single-kernel "puts then signal" shape work here, where a plain `tle.sign`
# store would have no ordering at all.
# ---------------------------------------------------------------------------

_SIGNAL_WAIT_MAX_ITERS = int(_os.environ.get("TLE_SIGNAL_WAIT_MAX_ITERS", str(1 << 22)))


def _peer_flush_addr(handle, peer, _semantic):
    """Address in world rank ``peer``'s xshmem flag range.

    Writing here is how a rank orders a flag after the data it covers.  This part has
    no system-scope fence -- xccl says so in
    ``include/nccl_device/impl/c2c_ordering.h``: "fences degrade to GPU scope;
    cross-C2C data/flag ordering comes from the receiver-side address-window
    hold (write to the syscon/L3 flag range is held until prior GM writes to the
    same c2c port land, then committed)".  The flag range is a different NOC
    slave than HBM, so a write to it makes the destination change, the NOC holds
    it until the prior same-port HBM writes return their bresp, and anything
    issued behind it cannot overtake.

    Two things this is NOT yet aligned with on the xccl side, both blocked
    upstream rather than overlooked:

    * ONE PORT.  The hold covers "prior GM writes to the SAME c2c port", so the
      flush and the data it orders must travel the same port.  That holds here
      because both tables come from one place: ``offsets`` is built from
      ``peer_heap_base_remote_p2p`` and ``peer_flags`` from
      ``peer_flag_remote_p2p`` (xccl's ``src/xccl.cpp:722-739``), one entry per
      peer, same mapping family.  xccl's device side instead indexes
      ``peer_heap_base_remote_p2p_port`` / ``peer_flag_remote_p2p_port`` with
      the same ``idx % num_ports`` for both
      (``src/device/kl5/deepep/ibgda_device_mlx5.cuh:279,300``), spreading one
      peer over several ports.  Those per-port tables live only in the
      device-side state and no ``xccl_xshmem_*`` host entry point hands them
      out, so this cannot follow until one does -- and when it does, BOTH
      tables have to grow the port index together, or the flush stops covering
      the data.
    * NO CACHE HINT.  xccl reads and writes flags with ``llca:15 glc`` /
      ``llca:0 glc``; the stores in :func:`signal` carry neither, because
      nothing on this backend lowers a Triton cache modifier to those bits --
      they appear only in hand-written asm inside the xtdk headers.  Giving
      them one is a backend change, not a change here.

    Returns ``None`` for a handle that carries no flag range, and the caller
    then publishes without a release.  That is not a silent downgrade: a handle
    built by hand -- what the ``*_e2e.py`` tests do, a plain device buffer and
    a pad, no communicator and no xshmem -- has no peer to race with, because
    the only "peer" is self and its xlink offset is 0, so the store never
    leaves the chip.  Release semantics come with
    :func:`tle.create_dist_tensor`, which is where the flag range comes from.
    """
    flags = getattr(handle, "peer_flags", None)
    if not flags:
        return None
    return _peer_addr(handle, 0, peer, _semantic, table=flags)


def _signal_pad_base(handle, name):
    base = getattr(handle, "signal_ptr", None)
    if not base:
        raise RuntimeError(
            f"{name} needs the signal pad that tle.create_dist_tensor() allocates, "
            "and this handle carries none -- rebuild it with the current "
            "tle.create_dist_tensor().")
    return int(base)


def _slot_byte_offset(handle, slot_id, _semantic):
    """Byte offset of int32 signal slot ``slot_id`` inside the pad."""
    nslots = int(getattr(handle, "signal_slots", 0))
    slot = tl._unwrap_if_constexpr(slot_id)
    if isinstance(slot, int) and not isinstance(slot, bool):
        if not (0 <= slot < nslots):
            raise ValueError(f"slot_id {slot} out of range [0, {nslots})")
        return tl.tensor(_semantic.builder.get_int64(slot * 4), tl.int64)
    slot_t = slot if isinstance(slot, tl.tensor) else _semantic.to_tensor(slot)
    if not slot_t.dtype.is_int() or slot_t.dtype.is_bool():
        raise TypeError(f"slot_id must be an integer, got {slot_t.dtype}")
    if slot_t.shape:
        raise ValueError(f"slot_id must be a scalar, got shape {slot_t.shape}")
    if slot_t.dtype != tl.int64:
        slot_t = tl.cast(slot_t, tl.int64, _semantic=_semantic)
    return _semantic.mul(slot_t, 4, False)


@tl.builtin
def signal(device_dptr, peer, slot_id, value=None, op="inc",
           space="intra_node", group_kind="block", context_id=0, scope="system",
           _semantic=None, _generator=None):
    """Update synchronization slot ``slot_id`` owned by remote ``peer``.

    Signature matches flagtree's ``tle.signal``
    ``(device_dptr, peer, slot_id, value, op, space, group_kind, context_id,
    scope)``. ``op="inc"`` adds one, ``op="add"`` adds ``value``, ``op="set"``
    publishes ``value`` itself (NVSHMEM's ``SIGNAL_SET`` against its
    ``SIGNAL_ADD``). ``space``
    selects the team ``peer`` is a rank of: ``intra_node``/``intra``/``device``
    (local rank in this node), ``inter_node``/``inter`` (node index) or
    ``world`` (world rank). It only sends the signal -- no data, no completion.

    xpu5 differences: the update is a remote read-modify-write rather than an
    atomic (KL5 has no cross-chip atomic), so a slot must have one writer
    per epoch; ``group_kind``/``context_id``/``scope`` are validated for
    parity but every lane of the group performs the same scalar update.

    ``op="set"`` is the cheap form, and the single-writer rule above is what
    makes it available: a writer that owns the slot knows the value the waiter
    is looking for, so the remote GET the other two ops have to do first
    disappears and a publish costs two one-way stores instead of a round trip
    plus two. That read is not free on this fabric -- see the ring all-reduce
    in ``test/tle/distributed/10-inter-node-all-reduce.py``, where 16 CTAs
    publish twice each per epoch.
    """
    op_name = str(tl._unwrap_if_constexpr(op)).lower()
    if op_name not in ("inc", "add", "set"):
        raise ValueError(f"op must be 'inc', 'add' or 'set', got {op_name!r}")
    space_name = str(tl._unwrap_if_constexpr(space)).lower()
    if space_name not in _SIGNAL_SPACES:
        raise ValueError(f"space must be one of {sorted(_SIGNAL_SPACES)!r}, got {space_name!r}")
    _normalize_coopkind(group_kind, "group_kind")
    _normalize_context_id(context_id)
    _normalize_choice(scope, "scope", ("system", "device"), "system")

    val = tl._unwrap_if_constexpr(value)
    if op_name == "inc":
        if val is not None:
            raise TypeError('tle.signal(op="inc") does not take a value')
    elif val is None:
        raise TypeError(f'tle.signal(op="{op_name}") requires value')
    elif isinstance(val, tl.tensor):
        if val.shape:
            raise ValueError(f"value must be a scalar, got shape {val.shape}")
        if not val.dtype.is_int() or val.dtype.is_bool():
            raise TypeError(f"value must be an integer, got {val.dtype}")
    elif not isinstance(val, int) or isinstance(val, bool):
        raise TypeError(f"value must be an integer, got {type(val).__name__}")

    handle = _dist_handle(device_dptr)
    if op_name == "inc":
        delta = _semantic.to_tensor(1)
    else:
        delta = val if isinstance(val, tl.tensor) else _semantic.to_tensor(val)
    if delta.dtype != tl.int32:
        delta = tl.cast(delta, tl.int32, _semantic=_semantic)

    world_peer = _world_peer(handle, peer, _SIGNAL_SPACES[space_name], _semantic)
    addr = _peer_addr(handle, _signal_pad_base(handle, "tle.signal"), world_peer, _semantic)
    addr = _semantic.add(addr, _slot_byte_offset(handle, slot_id, _semantic), False)
    ptr = tl.cast(addr, tl.pointer_type(tl.int32, 1), _semantic=_semantic)

    if op_name == "set":
        # Absolute publish: the caller owns the slot and brought the value, so
        # there is nothing to read back and the whole op is the two stores
        # below.
        nxt = delta
    else:
        # Remote GET -> add -> remote PUT.  See the section comment above for
        # why this is not an atomic and when that is safe.
        cur = _tll.load(ptr, _semantic=_semantic)
        nxt = _semantic.add(cur, delta, False)
    # Release -- the only kind this chip has.  Poke the peer's flag range first:
    # that write targets a different NOC slave than its HBM, so the NOC holds it
    # (and everything issued behind it, same c2c port, same awid) until our
    # prior writes to that peer's HBM have returned their bresp.  Then publish.
    # Same shape as xccl's st_release_sys_global(flush_addr, buffer, val)
    # (include/nccl_device/impl/c2c_ordering.h, called at
    # src/device/kl5/deepep/internode_ll.xpu:893), and xshmem sets
    # need_c2c_flush unconditionally on this platform.
    #
    # Without it the flag can overtake the data it covers: the nightly bucket on
    # 2026-11-11 lost the inter-node term of 08-inter-node-reduce-scatter at
    # NNODES=2, on the smallest payload in the case list -- got=[1,3,5,7]
    # (the local pair alone) against want=[6,10,14,18].
    #
    # Self is poked too rather than special-cased: the value is immaterial, it
    # is one store, and skipping it would need a runtime branch on the peer.
    flush = _peer_flush_addr(handle, world_peer, _semantic)
    if flush is not None:
        flush_ptr = tl.cast(flush, tl.pointer_type(tl.int32, 1), _semantic=_semantic)
        _tll.store(flush_ptr, nxt, _semantic=_semantic)
    _tll.store(ptr, nxt, _semantic=_semantic)
    return None


@tl.builtin
def signal_wait(device_dptr, slot_id, wait_kind, target=None,
                group_kind="block", context_id=0, order="acquire",
                max_iters=_SIGNAL_WAIT_MAX_ITERS, _semantic=None, _generator=None):
    """Wait until this rank's signal slot ``slot_id`` reaches ``target``.

    Signature matches flagtree's ``tle.signal_wait``
    ``(device_dptr, slot_id, wait_kind, target, group_kind, context_id,
    order)``; ``max_iters`` is an xpu5-only optional cap so a mis-wired run
    fails fast instead of spinning forever (flagtree needs none because FlagCX
    owns the wait). Returns the last value read.

    The spin is :func:`sign_wait`'s local read-only atomic: a plain -- even
    ``volatile`` -- load is loop-invariant to this backend, so it gets hoisted
    out of the loop and the spin collapses to one read.
    """
    kind = str(tl._unwrap_if_constexpr(wait_kind)).lower()
    if kind not in ("signal", "counter", "shadow"):
        raise ValueError(f"wait kind must be signal, counter, or shadow, got {kind!r}")
    if kind == "shadow":
        raise NotImplementedError(
            'tle.signal_wait(wait_kind="shadow") reads the target from FlagCX\'s '
            "locally maintained shadow buffer, which this backend has no equivalent "
            'of; pass an explicit target with wait_kind="signal".')
    tgt = tl._unwrap_if_constexpr(target)
    if tgt is None:
        raise TypeError(f'tle.signal_wait(wait_kind="{kind}") requires target')
    _normalize_coopkind(group_kind, "group kind")
    _normalize_context_id(context_id)
    _normalize_choice(order, "order", ("relaxed", "acquire"), "acquire")
    if _generator is None:
        raise RuntimeError("tle.signal_wait must be called from inside a Triton kernel")

    handle = _dist_handle(device_dptr)
    base = tl.tensor(_semantic.builder.get_int64(_signal_pad_base(handle, "tle.signal_wait")), tl.int64)
    addr = _semantic.add(base, _slot_byte_offset(handle, slot_id, _semantic), False)
    ptr = tl.cast(addr, tl.pointer_type(tl.int32, 1), _semantic=_semantic)

    cap = tl._unwrap_if_constexpr(max_iters)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ValueError(f"max_iters must be a positive compile-time int, got {cap!r}")
    return _generator.call_JitFunction(_spin_until, [ptr, tgt], {"max_iters": tl.constexpr(cap)})


# ---------------------------------------------------------------------------
# device_mesh + distributed_barrier.
#
# flagtree dispatches distributed_barrier on the mesh's axes (its
# distributed.py:1186-1191): a block-only mesh means a cooperative GRID
# barrier, cluster axes mean a cluster/submesh barrier, and an explicit
# `space=` means a FlagCX cross-rank barrier. Two of the three are implemented
# here -- the grid one below and the cross-rank one further down -- both in
# Triton source rather than as dialect ops, because this fork vendors no TLE C++
# dialect and there is no create_distributed_barrier to call. The submesh one
# has no primitive to lower to at all (no sub-grid synchronization on KL5).
#
# GRID barrier. The counter it needs lives on the mesh, which is a HOST object
# (as in flagtree), so the call form stays flagtree's
# `distributed_barrier(mesh)` with no buffer at the call site -- the pointer
# folds in at compile time the way create_dist_tensor's does for remote/signal.
#
# It is a TICKET barrier, which is what lets the counter be allocated once and
# never reset: an arriving program takes a ticket with atomic_add, derives its
# barrier generation from it (`ticket // nprog`) and waits for the counter to
# reach the end of that generation's window. Consecutive barriers, and
# consecutive launches, just walk up the window sequence. A per-launch memset
# would otherwise be the one thing that needed host cooperation.
#
# ttg.global_scratch_alloc is NOT usable for this: TritonGPUOps.td:443 says the
# buffer is "private to the current program", and the lowering offsets it by the
# linearized program id (Utility.cpp:1162-1176), so every program would get its
# own counter.
# ---------------------------------------------------------------------------

_BARRIER_MAX_ITERS = int(_os.environ.get("TLE_BARRIER_MAX_ITERS", str(1 << 22)))
_BARRIER_CHANNELS = 64


def _prod(values):
    result = 1
    for value in values:
        result *= value
    return result


def _as_positive_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{what} must be a positive int, got {value!r}")
    return value


@dataclass
class MeshConfig:
    """A hierarchical topology, level by level (flagtree's ``MeshConfig``).

    The levels, coarse to fine (flagtree distributed.py:321-348):

    * ``node``      inter-host layout
    * ``device``    devices per host
    * ``block``     the finest, per-program level

    Each is either a size or a list of ``(dim_name, size)``; levels left
    ``None`` are dropped by :meth:`to_dict`, which is what ``device_mesh``
    consumes. Only the names matter to dispatch: ``distributed_barrier`` picks
    its barrier from the dimension names, so a ``block``-only config is the grid
    barrier.
    """

    node: object = None
    device: object = None
    block: object = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __repr__(self):
        fields = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"MeshConfig({fields})"


class device_mesh:
    """Logical view of a physical topology, and the owner of a barrier channel.

    ``topology`` is flagtree's form (its ``distributed.py:351``): a mapping from
    a level name to either a size or a list of ``(dim_name, size)``, e.g.
    ``{"block": [("block_x", 8)]}`` or ``{"node": 2, "device": 2}``,
    or a :class:`MeshConfig` naming the same levels. Which barrier
    :func:`distributed_barrier` emits is decided from the dim names, exactly as
    flagtree decides it.

    ``flatten`` / ``reshape`` / ``mesh[...]`` re-view the same participants, as
    in flagtree: the derived mesh keeps the *physical ids* it selected and the
    *launch* shape of the mesh it came from, so a submesh re-labels coordinates
    without changing what was launched.

    xpu5-only: the mesh also owns the counters ``distributed_barrier`` needs
    (``index`` picks one of ``channels``), allocated lazily on first use so a
    mesh can be built on a host with no device. They are never reset -- see the
    ticket-barrier note above. A derived mesh gets counters of its OWN, because
    its participants are not the ones the parent's counters are counting.
    """

    def __init__(self, topology=None, *, channels=_BARRIER_CHANNELS, _shape=None,
                 _dim_names=None, _physical_ids=None, _launch_shape=None,
                 _launch_dim_names=None):
        self._channels = _as_positive_int(channels, "channels")
        self._counters = None
        if topology is None and _shape is not None:
            # Internal constructor, used by reshape/__getitem__ (flagtree
            # distributed.py:366-377).  A derived mesh has no topology dict: it
            # is described by the view it selected.
            if _dim_names is None or _physical_ids is None:
                raise ValueError("internal mesh constructor requires shape/names/physical ids")
            self._shape = tuple(_shape)
            self._dim_names = tuple(_dim_names)
            self._physical_ids = tuple(_physical_ids)
            self._launch_shape = tuple(_shape if _launch_shape is None else _launch_shape)
            self._launch_dim_names = tuple(
                _dim_names if _launch_dim_names is None else _launch_dim_names)
            self._allocate()
            return
        if isinstance(topology, MeshConfig):
            topology = topology.to_dict()
        if topology is None or not isinstance(topology, dict) or not topology:
            raise TypeError("topology must be a non-empty dict or MeshConfig, e.g. "
                            '{"block": [("block_x", 8)]}')
        shape, dim_names = [], []
        for level_name, level_desc in topology.items():
            if not isinstance(level_name, str) or not level_name:
                raise ValueError(f"invalid topology level name: {level_name!r}")
            level_shape, level_names = self._parse_level(level_name, level_desc)
            shape.extend(level_shape)
            dim_names.extend(level_names)
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"dimension names must be unique, got {dim_names}")
        self._shape = tuple(shape)
        self._dim_names = tuple(dim_names)
        self._physical_ids = tuple(range(_prod(shape)))
        self._launch_shape = self._shape
        self._launch_dim_names = self._dim_names
        # Eager when possible; a CPU-only host (the contract tests) keeps None
        # and only fails if a kernel actually asks for the pointer.
        self._allocate()

    @staticmethod
    def _parse_level(level_name, level_desc):
        if isinstance(level_desc, int) and not isinstance(level_desc, bool):
            return [_as_positive_int(level_desc, level_name)], [level_name]
        if not isinstance(level_desc, (tuple, list)) or not level_desc:
            raise TypeError(f"topology[{level_name!r}] must be a positive int or a "
                            "non-empty list of (name, size)")
        shape, names = [], []
        for item in level_desc:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f"topology[{level_name!r}] entries must be (name, size), got {item!r}")
            dim_name, dim_size = item
            if not isinstance(dim_name, str) or not dim_name:
                raise ValueError(f"invalid dimension name in {level_name!r}: {dim_name!r}")
            shape.append(_as_positive_int(dim_size, f"{level_name}.{dim_name}"))
            names.append(dim_name)
        return shape, names

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dim_names(self):
        return self._dim_names

    @property
    def physical_ids(self):
        """The participants this view selected, in row-major order."""
        return self._physical_ids

    @property
    def size(self):
        return len(self._physical_ids)

    # flagtree distinguishes the barrier mesh from the launch mesh; this backend
    # has no mesh-driven launch, so a root mesh's two are the same thing. A
    # derived mesh keeps the launch shape of the mesh it came from.
    @property
    def launch_shape(self):
        return self._launch_shape

    @property
    def launch_dim_names(self):
        return self._launch_dim_names

    def __deepcopy__(self, memo):
        # An immutable topology descriptor, and the owner of a device buffer:
        # sharing the instance keeps the JIT's global-change check (which
        # deepcopies captured values) from cloning the counters into a second
        # allocation that no other rank would be waiting on.
        return self

    def has_axis(self, token):
        """flagtree's `_mesh_has_axis` (its distributed.py:733): substring match
        against the dimension names, not the level names."""
        return any(token in name for name in self._dim_names)

    def flatten(self):
        return self.reshape(self.size)

    def reshape(self, *shape):
        """Re-view the same participants under a new shape (flagtree 463-481)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(shape)
        if not new_shape:
            raise ValueError("new shape cannot be empty")
        new_shape = tuple(_as_positive_int(v, "shape dimension") for v in new_shape)
        if _prod(new_shape) != self.size:
            raise ValueError(f"cannot reshape mesh of size {self.size} into shape {new_shape}")
        if len(new_shape) == self.ndim:
            new_dim_names = self._dim_names
        elif len(new_shape) == 1:
            new_dim_names = ("flat", )
        else:
            new_dim_names = tuple(f"dim{i}" for i in range(len(new_shape)))
        return device_mesh(None, channels=self._channels, _shape=new_shape,
                           _dim_names=new_dim_names, _physical_ids=self._physical_ids,
                           _launch_shape=self._launch_shape,
                           _launch_dim_names=self._launch_dim_names)

    def _normalize_key(self, key):
        if not isinstance(key, tuple):
            key = (key, )
        if any(item is Ellipsis for item in key):
            if key.count(Ellipsis) > 1:
                raise IndexError("an index can only have a single ellipsis")
            ellipsis_pos = key.index(Ellipsis)
            missing = self.ndim - (len(key) - 1)
            if missing < 0:
                raise IndexError("too many indices for device_mesh")
            key = key[:ellipsis_pos] + (slice(None), ) * missing + key[ellipsis_pos + 1:]
        if len(key) > self.ndim:
            raise IndexError("too many indices for device_mesh")
        return key + (slice(None), ) * (self.ndim - len(key))

    def _linear_index(self, coords):
        index = 0
        for coord, dim_size in zip(coords, self._shape):
            index = index * dim_size + coord
        return index

    def __getitem__(self, key):
        """Select a submesh (flagtree 506-541). An int drops its dimension, a
        slice keeps it; the result carries the physical ids it picked."""
        key = self._normalize_key(key)
        selected_per_dim, keep_dim = [], []
        for dim_size, dim_key in zip(self._shape, key):
            if isinstance(dim_key, bool):
                raise TypeError("device_mesh indices must be int/slice/ellipsis, got bool")
            if isinstance(dim_key, int):
                idx = dim_key + dim_size if dim_key < 0 else dim_key
                if idx < 0 or idx >= dim_size:
                    raise IndexError(f"index {dim_key} out of range for dim size {dim_size}")
                selected_per_dim.append([idx])
                keep_dim.append(False)
            elif isinstance(dim_key, slice):
                indices = list(range(*dim_key.indices(dim_size)))
                if not indices:
                    raise ValueError("empty sub-mesh is not supported")
                selected_per_dim.append(indices)
                keep_dim.append(True)
            else:
                raise TypeError("device_mesh indices must be int/slice/ellipsis, got "
                                f"{type(dim_key).__name__}")
        new_shape = tuple(len(ids) for ids, keep in zip(selected_per_dim, keep_dim) if keep)
        new_dim_names = tuple(n for n, keep in zip(self._dim_names, keep_dim) if keep)
        new_physical_ids = tuple(self._physical_ids[self._linear_index(coords)]
                                 for coords in product(*selected_per_dim))
        return device_mesh(None, channels=self._channels, _shape=new_shape,
                           _dim_names=new_dim_names, _physical_ids=new_physical_ids,
                           _launch_shape=self._launch_shape,
                           _launch_dim_names=self._launch_dim_names)

    def _allocate(self):
        """Allocate the counters, zeroed. Called from __init__ when a device is
        reachable so it happens on the HOST, at a deterministic point: doing it
        lazily from inside the builtin would allocate in the middle of a JIT
        compile, at a different moment on every rank, and inside whatever
        allocator context the caller happens to be in."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            self._counters = torch.zeros(self._channels, dtype=torch.int32, device="cuda")
            return True
        except Exception:
            return False

    def counters_ptr(self):
        """Base of this mesh's barrier counters."""
        if self._counters is None and not self._allocate():
            raise RuntimeError(
                "tle.device_mesh could not allocate its barrier counters: no reachable "
                "CUDA device. Build the mesh on the host that will run the kernel.")
        return int(self._counters.data_ptr())

    def _key(self):
        return (self._shape, self._dim_names, self._channels, id(self))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, device_mesh) and self._key() == other._key()

    def __repr__(self):
        return f"device_mesh(shape={self._shape}, dim_names={self._dim_names})"


@_triton.jit
def _ticket_barrier(ctr_ptr, max_iters: _tll.constexpr):
    """Grid-wide barrier on one monotonically increasing counter.

    Take a ticket, derive this barrier's generation from it, wait for the
    counter to reach the end of that generation's window. Needs no reset, so
    the counter can be allocated once and reused across barriers and launches.

    Hard prerequisite, same as any grid barrier: the grid must be CO-RESIDENT
    (grid <= physical cluster count). A program that was never scheduled never
    takes its ticket, and the resident ones would spin forever -- ``max_iters``
    turns that into a fast failure instead of a hang.

    Returns 1 when the whole grid arrived and 0 when the spin gave up, because
    giving up is SILENT otherwise: the kernel runs on, and whatever it was
    ordering reads data the barrier was supposed to have fenced. The flag is
    the caller's only chance to tell that apart from a cross-rank handshake
    that failed for its own reasons (``tle_mega_moe.py``'s dispatch publishes
    it as a distinct sentinel for exactly that reason). The raw counter is of
    no use to a caller -- the target it would be compared against is derived
    from a ticket only this function sees.
    """
    nprog = _tll.num_programs(0) * _tll.num_programs(1) * _tll.num_programs(2)
    ticket = _tll.atomic_add(ctr_ptr, 1)
    target = (ticket // nprog + 1) * nprog
    got = _tll.atomic_add(ctr_ptr, 0)
    i = 0
    while got < target and i < max_iters:
        got = _tll.atomic_add(ctr_ptr, 0)
        i = i + 1
    return (got >= target).to(_tll.int32)


@_triton.jit
def _barrier_epoch(epoch_ptr, BUMP: _tll.constexpr):
    """This rank's barrier generation, from a counter in its own signal pad.

    The arrival slots are never reset -- they cannot be, since resetting one
    would race the peer that is about to write it -- so each barrier has to wait
    for a HIGHER value than the last. ``BUMP`` takes the next generation
    (``barrier_kind="arrive"`` / ``"sync"``); without it the current one is read
    back (``barrier_kind="wait"``), so an arrive in one launch and a wait in the
    next agree on which generation they are talking about.

    The counter is local, so the read-modify-write is a real atomic, and one
    program per rank must call it -- see :func:`distributed_barrier`.
    """
    delta = 1 if BUMP else 0
    return _tll.atomic_add(epoch_ptr, delta) + delta


@_triton.jit
def _fold_arrived(ok, got, target):
    """Fold one wait's result into a running "every member arrived" flag.

    :func:`signal_wait` hands back the last value its spin read, which is
    ``>= target`` exactly when that member's arrival landed and short of it when
    the spin ran out of ``max_iters``. Folding instead of returning the values
    themselves keeps the verdict one word, which is all a caller can act on: a
    barrier is not partly held.
    """
    return ok & (got >= target).to(_tll.int32)


def _barrier_team(handle, space_name):
    """``(this rank's index in the team, team size)`` for a barrier team."""
    team = _BARRIER_SPACES[space_name]
    if team == "world":
        return int(handle.rank), int(handle.world)
    if team == "intra_node":
        return int(handle.local_rank), int(handle.local_world_size)
    lws = max(int(handle.local_world_size), 1)
    return int(handle.node_rank), (int(handle.world) + lws - 1) // lws


# Channel numbering interleaves the teams, so channel `index` of one team is
# never channel `index` of another. See _barrier_window.
_BARRIER_TEAMS = ("world", "intra_node", "inter_node")


def _barrier_window(handle, space_name, index):
    """Where in the signal pad one barrier channel lives.

    A channel is ``world + 1`` slots taken from the TOP of the pad downwards:
    one arrival slot per team member -- the team is at most the world -- plus
    one generation counter, at ``lo + team_size``. The bottom is where the
    tutorials put their per-source slots (slot == source rank), so a barrier
    cannot silently share a slot with a hand-written handshake on the same
    handle.

    Channels are numbered ``index * len(_BARRIER_TEAMS) + team``, which is what
    keeps two DIFFERENT teams off each other's slots. Sizing the channel by the
    team instead -- which is what this did -- makes teams of unequal size
    overlap, because both anchor at the top of the same pad: with two nodes of
    two, a ``world`` channel is 5 slots and a ``device`` channel is 3, so the
    device barrier's two arrivals land inside the world barrier's four and the
    two share the generation counter. The world barrier then waits for a count
    that two of its slots can never reach and falls out at ``max_iters``
    instead -- ``_spin_until`` returns the last value rather than raising, so
    it does not fail, it just stops synchronizing. With a single node every
    team has the same size and the windows coincide exactly, which is why this
    is only reachable under ``NNODES>1``.
    """
    my, size = _barrier_team(handle, space_name)
    nslots = int(getattr(handle, "signal_slots", 0))
    world = int(handle.world)
    width = world + 1
    nteams = len(_BARRIER_TEAMS)
    channel = index * nteams + _BARRIER_TEAMS.index(_BARRIER_SPACES[space_name])
    lo = nslots - (channel + 1) * width
    # Stop at the per-source region rather than at 0: slot == source rank is
    # what 05/06/07 write by hand on the same pad, so a channel that reached
    # below `world` would corrupt a handshake instead of failing. Channel 0 of
    # all three teams always fits (3*(world+1) + world <= 64 for every world
    # this fabric reaches, 13 ranks), so only index >= 1 can run into this.
    if lo < world:
        raise ValueError(
            f'tle.distributed_barrier(space="{space_name}", index={index}) needs '
            f"{width} of the handle's {nslots} signal slots per channel "
            f"(one arrival per rank + 1 generation counter) and runs off the pad "
            f"at slot {lo}, inside the bottom {world} slots the per-source "
            f"handshakes own; "
            f"the {nteams} teams interleave, so this is channel {channel} and only "
            f"{max(0, (nslots - world) // width)} fit -- use index < "
            f"{max(0, ((nslots - world) // width - 1) // nteams + 1)}.")
    return my, size, lo


def _cross_rank_barrier(handle, space_name, index, group_kind, context_id, bump,
                        do_wait, cap, _semantic, _generator):
    """The ``signal`` / ``signal_wait`` handshake, as a primitive.

    Every rank bumps slot ``my`` of every team member's pad and then waits for
    all of them to reach its own generation. That is the shape every cross-rank
    handshake on this backend already has (``distributed/05-*.py``, and
    ``test_tle_distributed_d2d.py``'s barrier kernel); what it adds is the
    generation counter, which is what makes it REUSABLE without a host memset.

    Each arrival slot has exactly one writer -- rank ``p`` owns slot ``lo + p``
    on every pad -- which is the condition under which ``signal``'s remote
    read-modify-write is safe (KL5 has no cross-chip atomic).

    Self is included rather than skipped: the xlink offset for self is 0, so it
    is a local update, and it keeps a world of 1 and the arrive/wait split from
    needing their own cases.

    Returns 1 when every member's arrival reached this generation and 0 when one
    of the spins ran out of ``max_iters``. That flag is the only way a caller can
    tell the two apart: ``_spin_until`` hands back the last value it read rather
    than raising, so a barrier that gave up looks exactly like one that worked
    while having ordered nothing. An arrive-only call waits for nobody and so
    reports 1.
    """
    my, size, lo = _barrier_window(handle, space_name, index)
    epoch_addr = tl.tensor(
        _semantic.builder.get_int64(_signal_pad_base(handle, "tle.distributed_barrier")),
        tl.int64)
    epoch_addr = _semantic.add(epoch_addr, _slot_byte_offset(handle, lo + size, _semantic),
                               False)
    epoch_ptr = tl.cast(epoch_addr, tl.pointer_type(tl.int32, 1), _semantic=_semantic)
    epoch = _generator.call_JitFunction(_barrier_epoch, [epoch_ptr],
                                        {"BUMP": tl.constexpr(bool(bump))})

    # Unrolled on the host: the team size is a constant on the handle, exactly
    # like the per-peer xlink offsets _peer_addr folds.  `signal` takes the team
    team = _BARRIER_SPACES[space_name]
    if bump:
        for peer in range(size):
            signal(handle, peer, slot_id=lo + my, op="inc", space=team,
                   group_kind=group_kind, context_id=context_id,
                   _semantic=_semantic, _generator=_generator)
    arrived = tl.tensor(_semantic.builder.get_int32(1), tl.int32)
    if do_wait:
        for peer in range(size):
            got = signal_wait(handle, lo + peer, "signal", target=epoch,
                              group_kind=group_kind, context_id=context_id,
                              max_iters=cap, _semantic=_semantic, _generator=_generator)
            arrived = _generator.call_JitFunction(_fold_arrived, [arrived, got, epoch], {})
    return arrived


@tl.builtin
def distributed_barrier(mesh=None, device_dptr=None, space=None, group_kind="block",
                        barrier_kind="sync", order="acqrel", index=0, context_id=0,
                        memory_scope="system", max_iters=_BARRIER_MAX_ITERS,
                        _semantic=None, _generator=None):
    """Synchronize the participants a ``mesh`` (or ``space``) describes.

    Signature matches flagtree's ``tle.distributed_barrier``
    (its ``distributed.py:1178``); ``max_iters`` is an xpu5-only optional spin
    cap. Dispatch follows flagtree's order:

    * a mesh with cluster axes -> rejected. There is no cluster level on this
      machine to synchronize, and the check is what keeps such a mesh from
      silently taking the grid-barrier branch below over the wrong
      participants.
    * an explicit ``space=`` -> a **cross-rank** barrier over that team, built
      out of :func:`signal` / :func:`signal_wait` rather than FlagCX. Needs
      ``device_dptr``, the :func:`create_dist_tensor` handle, because the
      arrival slots live in its signal pad. ``space`` takes the team names
      :func:`signal` does (``world``, ``device``/``intra_node``,
      ``inter_node``).
    * a **block-only** mesh -> grid barrier. Every program in the grid arrives
      and waits for all the others, with no host barrier and no buffer at the
      call site: the counter lives on the mesh.

    The two are orthogonal and compose: the cross-rank one synchronizes RANKS
    and must be called by exactly ONE program per rank (one arrival per rank per
    generation -- a second program would consume the next generation's slot
    value and every rank would wait for a count that never arrives). To order a
    whole grid across ranks, put a grid barrier on either side of it, or launch
    the rank barrier on a ``(1,)`` grid of its own, which is what
    ``distributed/05-*.py`` does with its separate signal and wait kernels.

    ``index`` selects a channel. For the grid barrier it indexes the mesh's
    counters; for the cross-rank one it indexes a window in the handle's signal
    pad (``world + 1`` slots each, taken from the top, with the teams
    interleaved so a ``device`` channel can never land inside a ``world`` one),
    and must agree across the participants either way. ``barrier_kind`` splits
    the cross-rank barrier into its halves -- ``arrive`` signals, ``wait``
    spins, ``sync`` does both -- and is validated-but-unused for the grid one,
    which is whole-program. ``group_kind``/``order``/``context_id``/``memory_scope`` are
    validated for parity and otherwise unused: the spin is already a
    globally-coherent re-read.

    Returns, xpu5-only (flagtree's returns nothing): 1 when every participant
    this call waited for arrived and 0 when a spin hit ``max_iters``, from BOTH
    branches -- the grid one over the programs of the grid, the cross-rank one
    over the members of the team (and 1 from ``barrier_kind="arrive"``, which
    waits for nobody).

    The flag exists because the give-up is otherwise invisible: every spin here
    returns the last value it read instead of raising, so a barrier that ordered
    nothing is indistinguishable from one that held, and the kernel runs on into
    the data it was meant to fence. The cross-rank branch used to return None on
    the grounds that its waits are :func:`signal_wait` calls and a caller could
    check one peer for itself; that left the composite verdict -- the only one a
    barrier has -- to be reassembled by hand, and nobody did. the nightly bucket
    #1009 lost the whole inter-node term of ``08-inter-node-reduce-scatter``
    to four blown windows (``_barrier_kernel`` at 32.6s against ~1ms for the
    ones that rendezvoused) and reported it as a VALUE MISMATCH, the second
    build to pay for the same silence -- and a value mismatch is what sent two
    builds looking for a bug in the kernels. What the give-ups were really
    about is written up on ``08``'s ``MAX_ITERS``: a simulator defect, fixed
    outside this tree. This verdict is what made that reachable at all.

    Using the call as a statement is still fine, so neither form changes a
    kernel that does not ask.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None and not isinstance(mesh, device_mesh):
        raise TypeError(f"mesh must be device_mesh or None, got {type(mesh).__name__}")
    kind = _normalize_coopkind(group_kind, "group_kind")
    barrier_kind = _normalize_choice(barrier_kind, "barrier_kind",
                                     ("sync", "arrive", "wait"), "sync")
    # flagtree spells the acquire-release order "acqrel" (its MemoryOrder.ACQ_REL
    # value, distributed.py:117); "acq_rel" was this backend's own spelling.
    # Accept both -- a kernel copied verbatim from a flagtree tutorial passes
    # the former, and the tests here already pass the latter.
    _normalize_choice(order, "order",
                      ("relaxed", "acquire", "release", "acqrel", "acq_rel"), "acqrel")
    _normalize_choice(memory_scope, "memory_scope", ("system", "device", "block", "thread"), "system")
    cid = _normalize_context_id(context_id)

    slot = tl._unwrap_if_constexpr(index)
    if isinstance(slot, bool) or not isinstance(slot, int):
        raise TypeError(f"index must be a compile-time int, got {type(slot).__name__}")
    if slot < 0:
        raise ValueError(f"index must be non-negative, got {slot}")
    cap = tl._unwrap_if_constexpr(max_iters)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ValueError(f"max_iters must be a positive compile-time int, got {cap!r}")

    if mesh is not None and mesh.has_axis("cluster"):
        raise ValueError(
            "tle.distributed_barrier got a mesh with a cluster axis, and there is no "
            "cluster level on this machine to synchronize. The axis is rejected rather "
            "than ignored: a dim name carrying both tokens (e.g. 'block_cluster') would "
            "otherwise take the grid-barrier branch over the wrong participants. "
            "Pass a block-only mesh for a grid barrier.")

    space_arg = tl._unwrap_if_constexpr(space)
    if space_arg is not None:
        space_name = str(getattr(space_arg, "value", space_arg)).lower()
        if space_name not in _BARRIER_SPACES:
            raise ValueError("tle.distributed_barrier space must be one of "
                             f"{sorted(_BARRIER_SPACES)!r}, got {space_name!r}")
        if device_dptr is None:
            raise TypeError(
                'tle.distributed_barrier(space=...) needs device_dptr, the handle '
                "tle.create_dist_tensor() returns: the arrival slots live in its "
                "signal pad. flagtree reads them off a FlagCX communicator instead.")
        if _generator is None:
            raise RuntimeError("tle.distributed_barrier must be called from inside a Triton kernel")
        return _cross_rank_barrier(_dist_handle(device_dptr), space_name, slot, kind, cid,
                                   bump=barrier_kind in ("sync", "arrive"),
                                   do_wait=barrier_kind in ("sync", "wait"),
                                   cap=cap, _semantic=_semantic, _generator=_generator)

    if mesh is None or not mesh.has_axis("block"):
        raise NotImplementedError(
            "tle.distributed_barrier with no space= implements the GRID barrier only, "
            "which flagtree selects with a block-only mesh (e.g. "
            'tle.device_mesh({"block": [("block_x", <grid>)]})); got '
            f"{mesh!r}.")
    if _generator is None:
        raise RuntimeError("tle.distributed_barrier must be called from inside a Triton kernel")
    if slot >= mesh._channels:
        raise ValueError(f"index {slot} out of range [0, {mesh._channels})")

    addr = tl.tensor(_semantic.builder.get_int64(mesh.counters_ptr() + slot * 4), tl.int64)
    ptr = tl.cast(addr, tl.pointer_type(tl.int32, 1), _semantic=_semantic)
    return _generator.call_JitFunction(_ticket_barrier, [ptr],
                                       {"max_iters": tl.constexpr(cap)})


# ---------------------------------------------------------------------------
# Rank queries: n_pes / _get_local_rank / _get_world_rank / shard_id
# (flagtree distributed.py:78-104 and 945).
#
# flagtree reads all three of the first ones off the FlagCX communicator at
# RUNTIME, through dialect ops (`tle.get_device_id`, `tle.get_world_rank`,
# `tle.get_num_pes`). This fork vendors no TLE C++ dialect and needs none: the
# handle `create_dist_tensor` returns already carries the topology as host
# constants (`communication.py` DistTensorCtx: rank / world / local_rank /
# local_world_size / node_rank) and is passed as a `tl.constexpr`, so every
# query folds to an integer constant -- the same thing `_peer_addr` does with
# the per-peer xlink offsets.
#
# The price is one flagtree already pays here: a kernel compiled against a
# handle is specialized for that handle, which is why `DistTensorCtx.__hash__`
# includes its rank. Two ranks therefore compile two different kernels, as they
# already do for `remote`/`signal` (whose addresses are folded too).
# ---------------------------------------------------------------------------

# flagtree's ret_dtype contract: the query returns a scalar integer tensor.
_RANK_QUERY_FIELDS = {
    "tle._get_local_rank": "local_rank",
    "tle._get_world_rank": "rank",
    "tle.n_pes": "local_world_size",
}


def _rank_constant(device_dptr, name, ret_dtype, _semantic):
    """Fold one host-known topology field of the handle into a constant."""
    field = _RANK_QUERY_FIELDS[name]
    handle = _dist_handle(device_dptr)
    if getattr(handle, field, None) is None:
        raise RuntimeError(
            f"{name} needs the topology tle.create_dist_tensor() records on the "
            f"handle, and this one carries no {field!r} -- rebuild it with the "
            "current tle.create_dist_tensor().")
    dtype = tl._unwrap_if_constexpr(ret_dtype)
    if not isinstance(dtype, tl.dtype) or not dtype.is_int() or dtype.is_bool():
        raise TypeError(f"{name} ret_dtype must be an integer Triton dtype, got {dtype}")
    value = tl.tensor(_semantic.builder.get_int32(int(getattr(handle, field))), tl.int32)
    return value if dtype == tl.int32 else tl.cast(value, dtype, _semantic=_semantic)


@tl.builtin
def _get_local_rank(device_dptr, _semantic=None, ret_dtype=tl.int32):
    """This rank's index inside its own node (flagtree's `tle._get_local_rank`).

    flagtree's "current device id": the intra-node rank, i.e. the namespace
    `remote(space="device")` and `signal(space="intra_node")` index.
    """
    return _rank_constant(device_dptr, "tle._get_local_rank", ret_dtype, _semantic)


@tl.builtin
def _get_world_rank(device_dptr, _semantic=None, ret_dtype=tl.int32):
    """This rank's global index (flagtree's `tle._get_world_rank`)."""
    return _rank_constant(device_dptr, "tle._get_world_rank", ret_dtype, _semantic)


@tl.builtin
def n_pes(dev_mem_ptr, _semantic=None, ret_dtype=tl.int32):
    """How many devices this rank's node has (flagtree's `tle.n_pes`).

    flagtree documents it as "the number of devices on the current node", which
    is `LOCAL_WORLD_SIZE` -- NOT the world size. With one node (the default for
    the tests here) the two coincide; under the launcher's `NNODES>1` they do
    not, and that is the case `shard_id(axis="node")` below divides by.
    """
    return _rank_constant(dev_mem_ptr, "tle.n_pes", ret_dtype, _semantic)


def _resolve_launch_axis(mesh, axis):
    """flagtree's `_resolve_launch_axis` (its distributed.py:929)."""
    if isinstance(axis, bool):
        raise TypeError("axis must be int or str, got bool")
    if isinstance(axis, int):
        ndim = len(mesh.launch_shape)
        axis_idx = axis + ndim if axis < 0 else axis
        if axis_idx < 0 or axis_idx >= ndim:
            raise IndexError(f"axis index {axis} out of range for launch ndim {ndim}")
        return axis_idx
    if isinstance(axis, str):
        if axis not in mesh.launch_dim_names:
            raise ValueError(f"unknown mesh axis {axis!r}; available launch axes: "
                             f"{mesh.launch_dim_names}")
        return mesh.launch_dim_names.index(axis)
    raise TypeError(f"axis must be int or str, got {type(axis).__name__}")


@tl.builtin
def shard_id(mesh, axis, device_dptr=None, _semantic=None):
    """This program's coordinate on one axis of ``mesh`` (flagtree's `shard_id`).

    Signature and dispatch match flagtree's ``shard_id(mesh, axis, device_dptr)``
    (its distributed.py:945): ``axis="device"`` returns the
    intra-node rank, ``axis="node"`` the inter-node one (both need
    ``device_dptr``, the :func:`create_dist_tensor` handle), and any other axis
    -- a name or an index into the launch shape, negative allowed -- is decoded
    out of ``tl.program_id(0)`` with the mesh's strides.

    xpu5 differences, both of them already true of the rest of this module:

    * the rank axes fold to constants off the handle rather than reading a
      communicator (see the section comment above);
    * flagtree calls ``_apply_mesh_cluster_launch`` here, which rewrites the
      launch options from the mesh. There is no mesh-driven launch on this
      backend (see ``device_mesh.launch_shape``), so the mesh axes are decoded
      from the grid the caller actually launched; a mesh whose ``launch_shape``
      does not match that grid returns coordinates for the mesh, not the grid.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    axis = tl._unwrap_if_constexpr(axis)

    if axis in ("device", "node") and device_dptr is None:
        raise ValueError(f"device_dptr is required for axis {axis!r}")
    if axis == "device":
        return _get_local_rank(device_dptr, _semantic=_semantic, ret_dtype=tl.int32)
    if axis == "node":
        # Same formula as flagtree's, so there is one definition of "node rank"
        # and not a second one on the handle: world_rank // devices_per_node.
        world_rank = _get_world_rank(device_dptr, _semantic=_semantic, ret_dtype=tl.int32)
        local_world_size = n_pes(device_dptr, _semantic=_semantic, ret_dtype=tl.int32)
        return _semantic.floordiv(world_rank, local_world_size)

    if not isinstance(mesh, device_mesh):
        raise TypeError(f"mesh must be device_mesh, got {type(mesh).__name__}")
    axis_idx = _resolve_launch_axis(mesh, axis)
    launch_shape = tuple(int(v) for v in mesh.launch_shape)
    launch_size = _prod(launch_shape)
    if launch_size <= 0:
        raise ValueError(f"invalid launch mesh shape: {launch_shape}")

    linear = tl.program_id(0, _semantic=_semantic)
    if launch_size > 1:
        linear = _semantic.mod(linear, launch_size)
    stride = _prod(launch_shape[axis_idx + 1:]) if axis_idx + 1 < len(launch_shape) else 1
    coord = linear
    if stride > 1:
        coord = _semantic.floordiv(coord, stride)
    dim = launch_shape[axis_idx]
    if dim > 1:
        coord = _semantic.mod(coord, dim)
    return coord




