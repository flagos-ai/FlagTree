import argparse
import ctypes
import os
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
from triton.language.extra.libdevice import ffs

from triton.experimental.tle.raw.nvshmem.utils import (
    copy_on_stream,
    init_nvshmem_by_torch_pg,
    init_torch_distributed,
    load_common_host,
    load_host,
    print_perf,
    set_signal_cuda_ptr,
    tensor_from_pointer,
)


def _device_dialect(function_name):
    return dialect(
        name="cuda",
        library="nvshmem",
        compiler="clang",
        file=Path(__file__).parent / "ag-gemm-device.cu",
        extern_func_name=function_name,
    )


@_device_dialect("ag_mark_local_ready")
def mark_local_ready(*args, **kwargs):
    ...


@_device_dialect("ag_wait_ready")
def wait_ready(*args, **kwargs):
    ...


@_device_dialect("ag_putmem_signal_block")
def putmem_signal_block(*args, **kwargs):
    ...


@triton.jit(do_not_specialize=["rank"])
def set_local_ready(ready, rank):
    tle_raw.call(mark_local_ready, [ready, rank])


@triton.jit(do_not_specialize=["rank", "local_world_size", "world_size"])
def nvshmem_device_producer_p2p_put_block(
    ag_buffer,
    signal_buffer,
    elements_per_rank,
    element_size,
    signal_target,
    rank,
    local_world_size,
    world_size,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size
    bytes_per_rank = tl.cast(elements_per_rank, tl.uint64) * tl.cast(element_size, tl.uint64)
    signal_value = tl.cast(signal_target, tl.uint64)

    for i in range(pid, num_nodes - 1, num_pid):
        peer = local_rank + ((node_rank + i + 1) % num_nodes) * local_world_size
        elements_per_rank_u64 = tl.cast(elements_per_rank, tl.uint64)
        segment = tl.cast(rank, tl.uint64) * elements_per_rank_u64
        tle_raw.call(
            putmem_signal_block,
            [
                ag_buffer + segment,
                ag_buffer + segment,
                bytes_per_rank,
                signal_buffer + rank,
                signal_value,
                peer,
            ],
        )


def cp_engine_producer_all_gather_intra_node(
    host,
    local_tensor,
    ag_buffer,
    signal_buffer,
    rank,
    local_world_size,
    ag_intranode_stream,
    signal_target=1,
):
    """Pull each local peer's segment into this rank's workspace."""
    m_per_rank, k = local_tensor.shape
    elements_per_rank = m_per_rank * k
    nbytes = elements_per_rank * local_tensor.element_size()

    for i in range(1, local_world_size):
        src_rank = (rank + i) % local_world_size
        peer_workspace = host.ag_gemm_peer_workspace_ptr(
            ctypes.c_void_p(ag_buffer.data_ptr()),
            src_rank,
        )
        assert peer_workspace, f"failed to get peer workspace pointer for PE {src_rank}"

        segment = src_rank * elements_per_rank
        src_ptr = peer_workspace + segment * local_tensor.element_size()
        dst_ptr = ag_buffer.data_ptr() + segment * local_tensor.element_size()
        copy_on_stream(dst_ptr, src_ptr, nbytes, ag_intranode_stream)
        set_signal_cuda_ptr(
            signal_buffer.data_ptr() + src_rank * ctypes.sizeof(ctypes.c_uint64),
            signal_target,
            ag_intranode_stream,
        )


def cp_engine_producer_all_gather_inter_node(
    host,
    local_tensor,
    ag_buffer,
    signal_buffer,
    rank,
    local_world_size,
    world_size,
    ag_intranode_stream,
    ag_internode_stream,
    signal_target=1,
):
    """AllGather via inter-node push and intra-node fanout."""
    assert world_size % local_world_size == 0
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size
    num_nodes = world_size // local_world_size
    m_per_rank, k = local_tensor.shape
    elements_per_rank = m_per_rank * k
    nbytes = elements_per_rank * local_tensor.element_size()
    element_size = local_tensor.element_size()

    with torch.cuda.stream(ag_internode_stream):
        nvshmem_device_producer_p2p_put_block[(num_nodes - 1, )](
            ag_buffer,
            signal_buffer,
            elements_per_rank,
            element_size,
            signal_target,
            rank,
            local_world_size,
            world_size,
            num_warps=32,
        )

    # Pull segments from local peers.
    for i in range(1, local_world_size):
        src_local_rank = (local_rank + i) % local_world_size
        src_rank = node_rank * local_world_size + src_local_rank
        peer_workspace = host.ag_gemm_peer_workspace_ptr(
            ctypes.c_void_p(ag_buffer.data_ptr()),
            src_rank,
        )
        assert peer_workspace, f"failed to get peer workspace pointer for PE {src_rank}"
        segment = src_rank * elements_per_rank
        src_ptr = peer_workspace + segment * element_size
        dst_ptr = ag_buffer.data_ptr() + segment * element_size
        copy_on_stream(dst_ptr, src_ptr, nbytes, ag_intranode_stream)
        set_signal_cuda_ptr(
            signal_buffer.data_ptr() + src_rank * ctypes.sizeof(ctypes.c_uint64),
            signal_target,
            ag_intranode_stream,
        )

    # Fan each inter-node receive out to local peers.
    for i in range(1, num_nodes):
        recv_node = (node_rank + num_nodes - i) % num_nodes
        recv_rank = recv_node * local_world_size + local_rank
        host.ag_gemm_signal_wait_until_on_stream(
            ctypes.c_void_p(signal_buffer.data_ptr() + recv_rank * ctypes.sizeof(ctypes.c_uint64)),
            signal_target,
            ag_intranode_stream.cuda_stream,
        )
        segment = recv_rank * elements_per_rank
        src_ptr = ag_buffer.data_ptr() + segment * element_size
        for j in range(1, local_world_size):
            dst_local_rank = (local_rank + local_world_size - j) % local_world_size
            dst_rank = node_rank * local_world_size + dst_local_rank
            peer_workspace = host.ag_gemm_peer_workspace_ptr(
                ctypes.c_void_p(ag_buffer.data_ptr()),
                dst_rank,
            )
            peer_ready = host.ag_gemm_peer_ready_ptr(
                ctypes.c_void_p(signal_buffer.data_ptr()),
                dst_rank,
            )
            assert peer_workspace, f"failed to get peer workspace pointer for PE {dst_rank}"
            assert peer_ready, f"failed to get peer ready pointer for PE {dst_rank}"
            dst_ptr = peer_workspace + segment * element_size
            copy_on_stream(dst_ptr, src_ptr, nbytes, ag_intranode_stream)
            set_signal_cuda_ptr(
                peer_ready + recv_rank * ctypes.sizeof(ctypes.c_uint64),
                signal_target,
                ag_intranode_stream,
            )


@triton.jit
def _lane_id():
    return tl.inline_asm_elementwise("mov.u32 $0, %laneid;", constraints="=r", args=[], dtype=tl.int32, is_pure=True,
                                     pack=1)


@triton.jit
def _shfl_sync(value, lane):
    return tl.inline_asm_elementwise("shfl.sync.idx.b32 $0, $1, $2, 31, $3;", constraints="=r,r,r,r",
                                     args=[value, lane, 0xFFFFFFFF], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _shfl_up_sync(value, delta):
    return tl.inline_asm_elementwise("shfl.sync.up.b32 $0, $1, $2, 0, $3;", constraints="=r,r,r,r",
                                     args=[value, delta, 0xFFFFFFFF], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _shfl_down_sync(value, delta):
    return tl.inline_asm_elementwise("shfl.sync.down.b32 $0, $1, $2, 31, $3;", constraints="=r,r,r,r",
                                     args=[value, delta, 0xFFFFFFFF], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ballot_sync(predicate):
    return tl.inline_asm_elementwise("{.reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2;}",
                                     constraints="=r,r,r", args=[predicate,
                                                                 0xFFFFFFFF], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _warp_prefix_sum(value, lane_id, length):
    offset = 1
    while offset < min(length * 2, 32):
        previous = _shfl_up_sync(value, offset)
        if lane_id >= offset:
            value += previous
        offset *= 2
    return value


@triton.jit(do_not_specialize=["rank"])
def _threadblock_swizzle_allgather_gemm(tiled_m, M, rank, WORLD_SIZE: tl.constexpr, NNODES: tl.constexpr,
                                        BLOCK_SIZE_M: tl.constexpr):
    """Prioritize local-node then local-rank rows, including unaligned node boundaries."""
    LOCAL_WORLD_SIZE: tl.constexpr = WORLD_SIZE // NNODES
    node_id = rank // LOCAL_WORLD_SIZE
    m_per_rank = M // WORLD_SIZE
    m_per_node = M // NNODES
    lane_id = _lane_id()

    if lane_id < NNODES:
        node = (lane_id + node_id) % NNODES
        node_m_begin = m_per_node * node
        node_m_end = m_per_node * (node + 1)
        tiled_node_begin = node_m_begin // BLOCK_SIZE_M
        previous_tiled_node_end = (node_m_begin - 1) // BLOCK_SIZE_M
        tiled_node_end = (node_m_end - 1) // BLOCK_SIZE_M
        next_tiled_node_begin = node_m_end // BLOCK_SIZE_M

        if lane_id == 0 and node_m_begin != 0:
            if previous_tiled_node_end == tiled_node_begin:
                tiled_node_begin += 1
        if lane_id == 0 and node_m_end != M:
            if next_tiled_node_begin == tiled_node_end:
                tiled_node_end -= 1
        if lane_id != NNODES - 1 and node_m_end != M:
            if next_tiled_node_begin == tiled_node_end:
                tiled_node_end -= 1
        swizzled_node_tiles = tiled_node_end - tiled_node_begin + 1
    else:
        swizzled_node_tiles = 0

    swizzled_node_offsets = _warp_prefix_sum(swizzled_node_tiles, lane_id, NNODES) - swizzled_node_tiles
    node_tiles_left = _shfl_down_sync(swizzled_node_tiles, NNODES - node_id)
    node_tiles_right = _shfl_up_sync(swizzled_node_tiles, node_id)
    node_tiles = 0
    if lane_id < node_id:
        node_tiles = node_tiles_left
    elif lane_id < NNODES:
        node_tiles = node_tiles_right

    node_tile_offsets = _warp_prefix_sum(node_tiles, lane_id, NNODES) - node_tiles
    mask = _ballot_sync(tiled_m < swizzled_node_offsets)
    swizzled_node = ffs(mask) - 2

    mapped_node = (swizzled_node + node_id) % NNODES
    mapped_node_offset = _shfl_sync(swizzled_node_offsets, swizzled_node)
    mapped_node_tiles = _shfl_sync(swizzled_node_tiles, swizzled_node)
    tiled_m_in_node = tiled_m - mapped_node_offset

    local_rank = rank % LOCAL_WORLD_SIZE
    local_m_begin = m_per_node * mapped_node + m_per_rank * local_rank
    local_tile_begin = tl.cdiv(local_m_begin, BLOCK_SIZE_M)
    destination_node_offset = _shfl_sync(node_tile_offsets, mapped_node)
    local_rank_offset = max(0, local_tile_begin - destination_node_offset)
    mapped_tiled_m_in_node = (tiled_m_in_node + local_rank_offset) % mapped_node_tiles
    return destination_node_offset + mapped_tiled_m_in_node


@triton.jit
def swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def consumer_gemm_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    ready,
    M,
    N,
    K,
    RANK: tl.constexpr,
    NUM_RANKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    nnodes: tl.constexpr = NUM_RANKS // LOCAL_WORLD_SIZE
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    if TRANS_B:
        # Logical [K, N], backed by contiguous [N, K].
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[N, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // NUM_RANKS
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            pid_m, pid_n = swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

            if nnodes == 1:
                pid_m = (pid_m + (RANK % NUM_RANKS) * pid_ms_per_rank) % num_pid_m
            else:
                pid_m = _threadblock_swizzle_allgather_gemm(
                    pid_m,
                    M,
                    RANK,
                    NUM_RANKS,
                    nnodes,
                    BLOCK_SIZE_M,
                )

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            source_rank_begin = offs_am // M_per_rank
            source_rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            for source_rank in range(source_rank_begin, source_rank_end + 1):
                tle_raw.call(wait_ready, [ready, source_rank])

        offs_k = ki * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        if TRANS_B:
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        else:
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def local_copy(a_local, workspace, ready, rank):
    ready.zero_()
    m_per_rank = a_local.shape[0]
    workspace[rank * m_per_rank:(rank + 1) * m_per_rank].copy_(a_local)
    set_local_ready[(1, )](ready, rank, num_warps=1)


def gemm_persistent(
    a,
    b,
    rank,
    num_ranks,
    ready,
    local_world_size,
    trans_b=True,
    block_m=128,
    block_n=256,
    block_k=64,
    group_m=8,
    num_warps=8,
    num_stages=3,
    epilogue_subtile=True,
):
    assert a.shape[1] == b.shape[0], "incompatible GEMM dimensions"
    assert a.dtype == b.dtype, "incompatible GEMM dtypes"
    assert num_ranks % local_world_size == 0, "num_ranks must be divisible by local_world_size"

    num_nodes = num_ranks // local_world_size
    assert num_nodes <= 32, "threadblock swizzle supports at most one warp (32 nodes)"

    m, k = a.shape
    n_per_rank = b.shape[1]
    if trans_b:
        assert b.stride() == (1, k), "transposed B must be backed by contiguous [N, K] storage"
    else:
        assert b.is_contiguous(), "non-transposed B must be contiguous"

    c = torch.empty((m, n_per_rank), dtype=a.dtype, device=a.device)
    total_tiles = triton.cdiv(m, block_m) * triton.cdiv(n_per_rank, block_n)
    num_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (min(total_tiles, num_gemm_sms), )

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    consumer_gemm_persistent[grid](
        a,
        b,
        c,
        ready,
        m,
        n_per_rank,
        k,
        rank,
        num_ranks,
        block_m,
        block_n,
        block_k,
        group_m,
        local_world_size,
        NUM_SMS=num_gemm_sms,
        EPILOGUE_SUBTILE=epilogue_subtile,
        TRANS_B=trans_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c


def ag_gemm_op(
    a,
    b,
    c,
    rank,
    num_ranks,
    workspace,
    ready,
    ag_intranode_stream,
    ag_internode_stream,
    host,
    local_world_size,
    trans_b=True,
    block_m=128,
    block_n=256,
    block_k=64,
    group_m=8,
    num_warps=8,
    num_stages=3,
    epilogue_subtile=True,
):
    assert a.shape[1] == b.shape[0], "incompatible GEMM dimensions"
    assert a.dtype == b.dtype == c.dtype, "incompatible GEMM dtypes"
    assert num_ranks % local_world_size == 0, "num_ranks must be divisible by local_world_size"

    num_nodes = num_ranks // local_world_size
    assert num_nodes <= 32, "threadblock swizzle supports at most one warp (32 nodes)"

    is_multinode = num_nodes > 1
    if is_multinode and ag_internode_stream is None:
        raise ValueError("ag_internode_stream is required for multi-node AllGather")

    # Reserve one SM per remote node for NVSHMEM producers.
    num_ag_sms = num_nodes - 1

    current_stream = torch.cuda.current_stream(b.device)
    host.ag_gemm_barrier_all_on_stream(current_stream.cuda_stream)
    if is_multinode:
        ag_internode_stream.wait_stream(current_stream)
    ag_intranode_stream.wait_stream(current_stream)

    if not is_multinode:
        cp_engine_producer_all_gather_intra_node(
            host=host,
            local_tensor=a,
            ag_buffer=workspace,
            signal_buffer=ready,
            rank=rank,
            local_world_size=local_world_size,
            ag_intranode_stream=ag_intranode_stream,
        )
    else:
        cp_engine_producer_all_gather_inter_node(
            host=host,
            local_tensor=a,
            ag_buffer=workspace,
            signal_buffer=ready,
            rank=rank,
            local_world_size=local_world_size,
            world_size=num_ranks,
            ag_intranode_stream=ag_intranode_stream,
            ag_internode_stream=ag_internode_stream,
        )

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank = b.shape[1]
    if trans_b:
        assert b.stride() == (1, K), "transposed B must be backed by contiguous [N, K] storage"
    else:
        assert b.is_contiguous(), "non-transposed B must be contiguous"

    total_tiles = triton.cdiv(M, block_m) * triton.cdiv(N_per_rank, block_n)
    num_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count - num_ag_sms
    assert num_gemm_sms > 0, "not enough SMs for AG producer and GEMM consumer"
    grid = (min(total_tiles, num_gemm_sms), )

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    consumer_gemm_persistent[grid](
        workspace,
        b,
        c,
        ready,
        M,
        N_per_rank,
        K,
        rank,
        num_ranks,
        block_m,
        block_n,
        block_k,
        group_m,
        local_world_size,
        NUM_SMS=num_gemm_sms,
        EPILOGUE_SUBTILE=epilogue_subtile,
        TRANS_B=trans_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    current_stream.wait_stream(ag_intranode_stream)
    if is_multinode:
        current_stream.wait_stream(ag_internode_stream)

    return c


def torch_ag_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
):
    M_per_rank, K = A.shape
    A_full = torch.empty([M_per_rank * tp_group.size(), K], dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_full, A, group=tp_group)
    return torch.matmul(A_full, B)


def configure_host_library(host):
    host.ag_gemm_workspace_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    host.ag_gemm_workspace_create.restype = ctypes.c_int
    host.ag_gemm_workspace_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    host.ag_gemm_workspace_destroy.restype = ctypes.c_int

    # Preserve the full 64-bit device address.
    host.ag_gemm_peer_workspace_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.ag_gemm_peer_workspace_ptr.restype = ctypes.c_void_p
    host.ag_gemm_peer_ready_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.ag_gemm_peer_ready_ptr.restype = ctypes.c_void_p
    host.ag_gemm_barrier_all_on_stream.argtypes = [ctypes.c_void_p]
    host.ag_gemm_barrier_all_on_stream.restype = None
    host.ag_gemm_signal_wait_until_on_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    host.ag_gemm_signal_wait_until_on_stream.restype = None


def create_workspace(host, world_size, m_per_rank, k, dtype, device):
    workspace_ptr = ctypes.c_void_p()
    ready_ptr = ctypes.c_void_p()
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    local_pe = ctypes.c_int()
    local_npes = ctypes.c_int()
    result = host.ag_gemm_workspace_create(
        m_per_rank * k,
        dtype.itemsize,
        ctypes.byref(workspace_ptr),
        ctypes.byref(ready_ptr),
        ctypes.byref(mype),
        ctypes.byref(npes),
        ctypes.byref(local_pe),
        ctypes.byref(local_npes),
    )
    assert result == 0, f"workspace allocation failed: {result}"
    assert npes.value == world_size
    workspace = tensor_from_pointer(
        workspace_ptr,
        (world_size * m_per_rank, k),
        dtype,
        device,
    )
    ready = tensor_from_pointer(
        ready_ptr,
        (world_size, ),
        torch.uint64,
        device,
    )
    return workspace_ptr, ready_ptr, workspace, ready, mype, local_pe, local_npes


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup_iters):
        func()
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iters):
        output = func()
    stop_event.record()
    torch.cuda.synchronize()

    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


def parse_args():
    parser = argparse.ArgumentParser(description="NVSHMEM allgather GEMM")
    parser.add_argument("--M", type=int, default=8192, help="global M across all ranks")
    parser.add_argument("--N", type=int, default=8192, help="global N across all ranks")
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--case", choices=("check", "perf"), default="check")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    assert world_size >= 2, "AG-GEMM requires at least two GPUs"
    assert world_size % local_world_size == 0, "WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE"
    assert args.M % world_size == 0, f"M={args.M} must be divisible by WORLD_SIZE={world_size}"
    assert args.N % world_size == 0, f"N={args.N} must be divisible by WORLD_SIZE={world_size}"

    m_per_rank = args.M // world_size
    n_per_rank = args.N // world_size

    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("FlagTree TLE AG-GEMM requires sm90 or newer")

    group = init_torch_distributed()
    host = load_host(Path(__file__).with_name("ag-gemm-host.cu"))
    common = load_common_host()
    configure_host_library(host)
    init_nvshmem_by_torch_pg(common, group)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    a_local = torch.randn((m_per_rank, args.K), dtype=dtype, device=device)
    b = torch.randn((n_per_rank, args.K), dtype=dtype, device=device).T
    c = torch.empty((args.M, n_per_rank), dtype=dtype, device=device)
    workspace_ptr, ready_ptr, workspace, ready, mype, local_pe, local_npes = create_workspace(
        host,
        world_size,
        m_per_rank,
        args.K,
        dtype,
        device,
    )
    assert mype.value == rank
    assert local_pe.value == local_rank
    assert local_npes.value == local_world_size

    ag_intranode_stream = torch.cuda.Stream(device=device, priority=-1)
    ag_internode_stream = torch.cuda.Stream(device=device, priority=-1) if world_size > local_world_size else None

    def triton_func():
        local_copy(a_local, workspace, ready, rank)
        return ag_gemm_op(
            a_local,
            b,
            c,
            rank,
            world_size,
            workspace,
            ready,
            ag_intranode_stream,
            ag_internode_stream,
            host,
            local_world_size,
        )

    def torch_func():
        return torch_ag_gemm(a_local, b, group)

    def run_correctness():
        if rank == 0:
            print("[check] start correctness validation", flush=True)
        result = triton_func()
        golden = torch_func()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, golden, atol=1e-3, rtol=1e-3)
        if rank == 0:
            print("[check] Pass!", flush=True)

    def run_benchmark():
        if rank == 0:
            print(f"[bench] start benchmark: warmup={args.warmup}, iters={args.iters}", flush=True)
        _, duration_ms = perf_func(triton_func, iters=args.iters, warmup_iters=args.warmup)
        print_perf(
            "triton ag-gemm",
            duration_ms,
            group,
            rank,
            world_size,
        )

    try:
        if args.case == "check":
            run_correctness()
        else:
            run_benchmark()
    finally:
        result = host.ag_gemm_workspace_destroy(workspace_ptr, ready_ptr)
        assert result == 0
        result = common.nvshmem_finalize_from_torch_distributed()
        assert result == 0
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
