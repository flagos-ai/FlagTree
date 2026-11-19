# Copyright 2026- Xcoresigma Technology Co., Ltd

from __future__ import annotations

import triton.language.core as tl
import triton.language as dl
import triton
from triton.language.core import builtin, tensor
from typing import List

from enum import IntEnum


class _SignalOp(IntEnum):
    SET = 1
    ADD = 2


class _CommScope(IntEnum):
    GPU = 1
    INTRA_NODE = 2
    INTER_NODE = 3


# Signaling scope (FlagCX-style API) -> aclshmem comm scope:
#   "block" -> gpu         (intra-device)
#   "rank"  -> intra_node  (intra-node, cross-device)
#   "node"  -> inter_node  (inter-node)
_SIGNAL_SCOPE_TO_COMM_SCOPE = {
    "block": "gpu",
    "rank": "intra_node",
    "node": "inter_node",
}

_SHMEM_DTYPE_MAP = {
    tl.float16: "half",
    tl.float32: "float",
    tl.bfloat16: "bfloat16",
    tl.int8: "int8_t",
    tl.int16: "int16_t",
    tl.int32: "int32_t",
    tl.int64: "int64_t",
    tl.uint8: "uint8_t",
    tl.uint16: "uint16_t",
    tl.uint32: "uint32_t",
    tl.uint64: "uint64_t",
}


def _shmem_dtype(dtype):
    if hasattr(dtype, 'element_ty'):
        return _SHMEM_DTYPE_MAP.get(dtype.element_ty, str(dtype.element_ty))
    return _SHMEM_DTYPE_MAP.get(dtype, str(dtype))


def _scalar_dtype(t):
    d = t.dtype if hasattr(t, 'dtype') else t
    return d.element_ty if hasattr(d, 'element_ty') else d


def _dispatch(
    func,
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            dtype = arg.dtype
            if hasattr(dtype, 'element_ty'):
                dtype = dtype.element_ty
            arg_types.append(dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_types = arg_type_symbol_dict[arg_types][1]
        if not isinstance(ret_types, (List, tuple)):
            ret_types = [ret_types]

        if symbol == "":
            raise ValueError("Symbol can not be empty")
        call = func(
            lib_name,
            lib_path,
            symbol,
            arg_list,
            [ret_type.to_ir(_semantic.builder) for ret_type in ret_types],
            is_pure,
        )

        if len(ret_types) == 0:
            return tensor(call, tl.void)
        if len(ret_types) == 1:
            return tensor(call.get_result(0), ret_types[0])
        return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))


def _extern_call(
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    """Dispatch an external function call. Supports ptr/block arguments."""
    dispatch_args = args.copy()
    for i in range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])

    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    func = _semantic.builder.create_extern_call
    return _dispatch(
        func,
        lib_name,
        lib_path,
        dispatch_args,
        arg_type_symbol_dict,
        is_pure,
        _semantic,
    )


class MeshConfig:
    """Lightweight mesh configuration for Ascend.

    Only ``device`` level is implemented; ``node``, ``block_cluster``,
    ``block`` accept values but trigger assert at usage time.
    """

    def __init__(self, device=None, node=None, block_cluster=None, block=None):
        self.device = device
        self.node = node
        self.block_cluster = block_cluster
        self.block = block

    def __repr__(self):
        return (f"MeshConfig(device={self.device}, node={self.node}, "
                f"block_cluster={self.block_cluster}, block={self.block})")


class device_mesh:
    """Lightweight logical view of physical device topology."""

    def __init__(self, topology: MeshConfig):
        if not isinstance(topology, MeshConfig):
            raise TypeError(f"topology must be MeshConfig, got {type(topology).__name__}")
        self._mesh_config = topology
        # Only device level is implemented
        assert topology.node is None, "TODO: node level not yet implemented"
        assert topology.block_cluster is None, "TODO: block_cluster level not yet implemented"
        assert topology.block is None, "TODO: block level not yet implemented"
        assert topology.device is not None and topology.device > 0, "device count must be > 0"

    @property
    def device_count(self) -> int:
        return self._mesh_config.device

    @property
    def shape(self) -> tuple:
        return (self.device_count, )

    def __repr__(self):
        return f"DeviceMesh(device={self.device_count})"


@builtin
def remote(
    tensor,
    shard_id=None,
    scope=None,
    space: str = None,
    dtype: tl.dtype = None,
    offset: int = None,
    _semantic=None,
):
    """Distributed pointer shuffle: remote pointer access by peer rank.

    Signature aligned with distributed.py::remote().
    Currently only ``device`` level is supported.
    """
    scope = tl._unwrap_if_constexpr(scope)
    if scope is not None:
        assert isinstance(scope, device_mesh), f"scope must be device_mesh, got {type(scope).__name__}"

    space = tl._unwrap_if_constexpr(space)
    if space is not None and not isinstance(space, str):
        raise TypeError(f"space must be str or None, got {type(space).__name__}")
    assert space is None or space == "device", \
        f"TODO: space='{space}' not yet implemented (only 'device' is supported)"

    dtype = tl._unwrap_if_constexpr(dtype)
    if dtype is not None:
        assert dtype in _SHMEM_DTYPE_MAP, \
            f"dtype {dtype} is not in the supported SHMEM type map: {list(_SHMEM_DTYPE_MAP.keys())}"

    offset = tl._unwrap_if_constexpr(offset)
    if offset is not None and not isinstance(offset, (int, tl.tensor)):
        raise TypeError(f"offset must be int, tl.tensor, or None, got {type(offset).__name__}")

    assert not tensor.type.is_block() and tensor.type.is_ptr(), "only support scalar pointer"
    rank = _semantic._convert_elem_to_ir_value(shard_id, require_i64=False)
    remote_ptr = tl.tensor(_semantic.builder.create_symm_at(tensor.handle, rank), tensor.type)

    # offset is accepted for interface compatibility but not yet applied at
    # the HIVM level — the TT addptr is lost during TT→HIVM bufferization.
    # Callers should use `remote_ptr + offset` at the call site instead.
    assert offset is None, "offset not applied at the HIVM level"
    # TODO: wire offset into symm_at / HIVM custom op

    return remote_ptr


@builtin
def shard_id(
    mesh=None,
    axis=-1,
    device_dptr=None,
    _semantic=None,
):
    """Return current shard coordinate.

    Signature aligned with distributed.py::shard_id().
    Currently only ``device`` axis is supported.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None:
        assert isinstance(mesh, device_mesh), f"mesh must be device_mesh, got {type(mesh).__name__}"

    axis = tl._unwrap_if_constexpr(axis)
    if isinstance(axis, str):
        assert axis == "device", f"TODO: axis='{axis}' not yet implemented (only 'device' is supported)"
        axis = 0
        if device_dptr is not None:
            assert isinstance(device_dptr, tl.tensor) and device_dptr.dtype.is_ptr(), \
                "device_dptr must be a pointer tensor"
    elif not isinstance(axis, int):
        raise TypeError(f"axis must be int or str, got {type(axis).__name__}")
    else:
        assert axis == -1 or axis == 0, f"TODO: axis={axis} not yet implemented (only 'device'/0/-1)"

    axis = _semantic._convert_elem_to_ir_value(axis, require_i64=False)
    return tl.tensor(_semantic.builder.create_get_rank(axis), tl.int32)


@builtin
def distributed_barrier(
    mesh=None,
    device_dptr=None,
    space: str = None,
    group_kind=None,
    barrier_kind=None,
    order=None,
    index: int = None,
    _semantic=None,
):
    """
    Distributed barrier across all ranks.
    Currently only ``device`` level is supported.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None:
        assert isinstance(mesh, device_mesh), f"mesh must be device_mesh, got {type(mesh).__name__}"

    space = tl._unwrap_if_constexpr(space)
    if space is not None and not isinstance(space, str):
        raise TypeError(f"space must be str or None, got {type(space).__name__}")
    assert space is None or space == "device", \
        f"TODO: space='{space}' not yet implemented (only 'device' is supported)"

    index = tl._unwrap_if_constexpr(index)
    if index is not None and not isinstance(index, int):
        raise TypeError(f"index must be int or None, got {type(index).__name__}")

    if device_dptr is not None:
        assert isinstance(device_dptr, tl.tensor) and device_dptr.dtype.is_ptr(), \
            "device_dptr must be a pointer tensor"

    return _extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_barrier_all", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def swizzle2d_Nz(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
):
    """Ascend Nz-format 2D swizzle for communication tiles."""
    data_row_loop_num = dl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = dl.cdiv(data_col_shape, tile_col_shape)
    data_loop_num = data_row_loop_num * data_col_loop_num
    rank_stride = rank_size // comm_npu_split
    swizzle_offset = comm_npu_split
    rank_loop_num = dl.cdiv(rank_size, swizzle_offset)
    rank_tile_idx = iter_id // (swizzle_offset * data_loop_num)
    data_rank_tile_idx = iter_id % (swizzle_offset * data_loop_num)
    rank_tile_size = swizzle_offset
    if rank_tile_idx == rank_loop_num - 1:
        rank_tile_size = rank_size - swizzle_offset * rank_tile_idx
    data_tile_idx = data_rank_tile_idx // rank_tile_size
    rank_idx = rank_tile_idx * swizzle_offset + data_rank_tile_idx % rank_tile_size
    rank_idx = (rank_idx * rank_stride) % rank_size + (rank_idx * rank_stride) // rank_size
    rank_idx = (rank_idx + data_tile_idx) % rank_size
    data_row_idx = data_tile_idx // data_col_loop_num
    data_col_idx = data_tile_idx % data_col_loop_num
    comm_row_size = dl.minimum(data_row_shape - data_row_idx * tile_row_shape, tile_row_shape)
    comm_col_size = dl.minimum(data_col_shape - data_col_idx * tile_col_shape, tile_col_shape)
    return data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size


@triton.jit
def gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
):
    """Ascend Nz-format 2D swizzle for GEMM compute tiles."""
    data_row_loop_num = dl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = dl.cdiv(data_col_shape, tile_col_shape)
    col_loop_num = dl.cdiv(data_col_loop_num, swizzle_offset)
    n_tile_idx = iter_id // (swizzle_offset * data_row_loop_num)
    m_n_tile_idx = iter_id % (swizzle_offset * data_row_loop_num)
    n_tile_size = swizzle_offset
    if n_tile_idx == col_loop_num - 1:
        n_tile_size = data_col_loop_num - swizzle_offset * n_tile_idx
    data_row_idx = m_n_tile_idx // n_tile_size
    data_col_idx = n_tile_idx * swizzle_offset + m_n_tile_idx % n_tile_size
    if n_tile_idx % 2 == 1:
        data_row_idx = data_row_loop_num - data_row_idx - 1
    return data_row_idx, data_col_idx


@builtin
def n_pes(dev_mem_ptr, _semantic=None, ret_dtype=tl.int32):
    """The total number of PEs (ranks) in the program.

    TODO: multi-node is not modeled yet — the GPU-side ``n_pes`` returns the
    per-node device count; aligning that semantics needs node-local team
    support (aclshmem_team_n_pes).
    """
    return _extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_n_pes", tl.int32),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@builtin
def get_local_rank(device_dptr, _semantic=None, axis=-1, ret_dtype=tl.int32):
    """Get the current device id (rank) of the current process.

    TODO: multi-node is not modeled yet — returning the intra-node rank needs
    node-local team support (aclshmem_team_my_pe).
    """
    axis = _semantic._convert_elem_to_ir_value(axis, require_i64=False)
    return tl.tensor(_semantic.builder.create_get_rank(axis), tl.int32)


@builtin
def get_global_rank(device_dptr, _semantic=None, axis=-1, ret_dtype=tl.int32):
    """Get the current world rank (global PE id)."""
    axis = _semantic._convert_elem_to_ir_value(axis, require_i64=False)
    return tl.tensor(_semantic.builder.create_get_rank(axis), tl.int32)


def _str_to_dist_signal_op(sig_op):
    """str -> _SignalOp attr value ("set"/"add")."""
    cache = None
    if sig_op:
        if sig_op == "set":
            cache = _SignalOp.SET
        elif sig_op == "add":
            cache = _SignalOp.ADD
        else:
            raise ValueError(f"Signal Op {sig_op} not supported")
    return cache


def _str_to_dist_comm_scopre(comm_scope):
    """str -> _CommScope attr value ("gpu"/"intra_node"/"inter_node")."""
    scope = None
    if comm_scope:
        if comm_scope == "gpu":
            scope = _CommScope.GPU
        elif comm_scope == "intra_node":
            scope = _CommScope.INTRA_NODE
        elif comm_scope == "inter_node":
            scope = _CommScope.INTER_NODE
        else:
            raise ValueError(f"Comm Scope {comm_scope} not supported")
    return scope


@builtin
def signal(device_dptr, peer, slot_id=None, value: int | None = None, op: str = "inc", space: str = "rank",
           group_kind: str = "block", context_id: int = 0, scope: str = "system", _semantic=None):
    """
    One-sided remote signal update: write ``value`` into the peer's
    symmetric signal slot. Signal-only primitive (no payload, no blocking
    on the peer); pair with ``signal_wait`` on the peer to form the
    producer/consumer handshake.
    Args:
        device_dptr: pointer to the symmetric signal buffer of THIS rank
            (element dtype int32 / int64 / uint64).
        peer: id of the target rank (0..rank_size-1).
        slot_id: offsets — accepted for FlagCX API compatibility, ignored on
            Ascend (the signal is written at ``device_dptr``).
        value: 0/1 signal value; required for op="add"/"set".
        op: "inc" (set 1) / "add" / "set" — all lowered as SET via the
            reliable aclshmem_<ty>_p path.
        space: "intra_node" / "inter_node" (default "inter_node"; "world" is
            not modeled on Ascend).
        group_kind / context_id / scope: FlagCX concepts; TODO — not modeled
            on Ascend (block-level only; ordering/visibility via shmem_fence).

    FlagCX: default OP=inc signal+1.
    """
    assert not device_dptr.type.is_block() and device_dptr.type.is_ptr(), \
        "device_dptr must be a scalar pointer to the symmetric signal array"
    assert group_kind in (None, "block"), \
        (f"TODO: group_kind='{group_kind}' not supported on Ascend "
         f"(the aclshmem signal path is block-level)")

    # slot_id is accepted for FlagCX API compatibility but ignored on Ascend:
    # the notify path writes the signal at device_dptr (single flag slot).

    if op == "inc":
        sig_op, signal_val = "set", 1
    elif op in ("add", "set"):
        assert value is not None, f"op='{op}' requires a value"
        sig_op, signal_val = "set", value
    else:
        raise ValueError(f"signal op '{op}' not supported (use 'inc', 'add' or 'set')")

    assert space in (None, "block", "rank", "node"), \
        f"space '{space}' not supported (use 'block', 'rank' or 'node')"

    comm_scope = _SIGNAL_SCOPE_TO_COMM_SCOPE[tl._unwrap_if_constexpr(space) or "rank"]

    # Not Support context_id / scope with Ascend
    sig_ptr = device_dptr
    peer_ir = _semantic._convert_elem_to_ir_value(peer, require_i64=False)
    signal_ir = _semantic._convert_elem_to_ir_value(signal_val, require_i64=True)
    sig_op_attr = _str_to_dist_signal_op(sig_op)
    comm_attr = _str_to_dist_comm_scopre(comm_scope)
    return tl.tensor(
        _semantic.builder.create_notify(sig_ptr.handle, signal_ir, peer_ir, sig_op_attr, comm_attr),
        tl.void,
    )


@builtin
def signal_wait(device_dptr, slot_id=None, wait_kind="signal", target=1, group_kind="block", context_id=0,
                order="acquire", scope="rank", _semantic=None):
    """
    Args:
        device_dptr: pointer to the symm signal buf.
        slot_id: slot element offset (default None = slot at device_dptr).
        wait_kind: only the threshold wait ("signal") is supported on Ascend
            (the GPU-side "counter"/"shadow" kinds are not modeled).
        target: the threshold value the wait compares against.
        group_kind / context_id: FlagCX S-Path concepts — accepted for API
            compatibility, not modeled on Ascend (block-level, single context).
        order: memory order, "acquire" / "release".
        scope: signaling scope — "block" (intra-device), "rank" (intra-node)
            or "node" (inter-node); mapped to the aclshmem comm scope.

    Returns an int32 token that must be passed to ``consume_token`` to build
    a data dependency on the memory guarded by this signal.
    """
    wait_kind = tl._unwrap_if_constexpr(wait_kind)
    if wait_kind != "signal":
        raise ValueError(f"wait_kind '{wait_kind}' not supported on Ascend "
                         f"(only the threshold 'signal' wait)")

    assert group_kind in (None, "block"), \
        (f"TODO: group_kind='{group_kind}' not supported on Ascend "
         f"(the aclshmem signal path is block-level)")

    assert order in ("acquire", "release"), \
        f"order '{order}' not supported (use 'acquire' or 'release')"

    assert scope in (None, "block", "rank", "node"), \
        f"scope '{scope}' not supported (use 'block', 'rank' or 'node')"

    if not device_dptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported device_dptr type {device_dptr.type.__repr__()} in `signal_wait`")
    slot_id = tl._unwrap_if_constexpr(slot_id)

    if slot_id is None:
        sig_ptr = device_dptr  # wait the slot at device_dptr
    else:
        sig_ptr = _semantic.add(device_dptr, slot_id, True)
    num_barriers = _semantic.to_tensor(1)
    elem_ty = device_dptr.dtype.element_ty
    require_i64 = False
    if elem_ty.is_int64() or elem_ty.is_uint64():
        require_i64 = True
    target_ir = _semantic._convert_elem_to_ir_value(target, require_i64=require_i64)
    # NOTE: `scope` is FlagCX API metadata only on Ascend — the aclshmem
    # threshold wait is a plain equal-spin on local GM (the peer's write
    # arrives via topology-routed engines regardless of scope), so the
    # MEM_SYNC_SCOPE attr is fixed to the default GPU bucket.
    mem_scope_ir = _semantic._str_to_scope("gpu")
    order = _semantic._str_to_sem(order)
    return tl.tensor(
        _semantic.builder.create_signal_wait(
            sig_ptr.handle,
            num_barriers.handle,
            target_ir,
            mem_scope_ir,
            order,
            tl.int32.to_ir(_semantic.builder),
        ),
        tl.int32,
    )


@builtin
def consume_token(value, token, _semantic=None):
    assert token.type.scalar.is_int(), "token must be of int type"
    handle = _semantic.builder.create_consume_token(value.handle, token.handle)
    if isinstance(value, tl.tensor_descriptor):
        return tl.tensor_descriptor(handle, value.shape, value.strides, value.block_type)
    else:
        return tl.tensor(handle, value.type)


@builtin
def shmem_put_nbi(src, dst, nelems, pe, _semantic=None):
    """Copy contiguous data from local GM to remote GM (non-blocking)."""
    dtype = src.dtype
    symbol = f"aclshmem_{_shmem_dtype(dtype)}_putmem_nbi"
    s = _semantic.to_tensor
    return _extern_call(
        "libshmem_device",
        "",
        [dst, src, nelems, pe],
        {(_scalar_dtype(s(dst)), _scalar_dtype(s(src)), tl.int32, tl.int32): (symbol, ())},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def shmem_get_nbi(dst, src, nelems, pe, _semantic=None):
    """Copy contiguous data from remote GM to local GM (non-blocking).

    Args: dst (tensor): local destination buffer
          src (tensor): remote source buffer (on peer PE)
          nelems (int): number of elements
          pe (int): source PE rank
    """
    dtype = dst.dtype
    symbol = f"aclshmem_{_shmem_dtype(dtype)}_getmem_nbi"
    s = _semantic.to_tensor
    return _extern_call(
        "libshmem_device",
        "",
        [dst, src, nelems, pe],
        {(_scalar_dtype(s(dst)), _scalar_dtype(s(src)), tl.int32, tl.int32): (symbol, ())},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def shmem_fence(_semantic=None):
    """Fence: order all prior remote memory operations of this thread
    (e.g. stores into peer symmetric memory) before all subsequent ones
    (e.g. ``notify`` signals). Call it between the data stores and the
    notify loop, mirroring ``libshmem_device.fence()``."""
    return _extern_call(
        "libshmem_device",
        "",
        [],
        {(): ("aclshmem_fence", ())},
        is_pure=False,
        _semantic=_semantic,
    )
