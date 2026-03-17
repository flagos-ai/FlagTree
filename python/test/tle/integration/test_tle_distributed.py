# flagtree tle
"""
TLE Distributed (M3) Integration Tests

These tests validate:
- `tle.distributed_barrier` can be used in JIT kernels.
- `tle.remote` can annotate shared-memory buffered tensors and participate in load/store.
- `tle.remote` can build cluster-cooperative GEMM-style data reuse patterns.

Current lowering targets NVIDIA Hopper cluster instructions, so tests run only
on CUDA devices with compute capability >= 9.0.
"""

import pytest
import re
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

BLOCK_CLUSTER_MESH = tle.device_mesh({"block_cluster": [("cluster_x", 2)]})
BLOCK_CLUSTER_MESH_8 = tle.device_mesh({"block_cluster": [("cluster_x", 8)]})
BLOCK_CLUSTER_MESH_2X2 = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
BLOCK_GRID_MESH_8 = tle.device_mesh({"block": [("block_x", 8)]})
BLOCK_CLUSTER_SUBMESH_ROW0 = BLOCK_CLUSTER_MESH_2X2[0, :]
BLOCK_CLUSTER_SUBMESH_ROW1 = BLOCK_CLUSTER_MESH_2X2[1, :]
BLOCK_CLUSTER_SUBMESH_COL0 = BLOCK_CLUSTER_MESH_2X2[:, 0]
BLOCK_CLUSTER_SUBMESH_COL1 = BLOCK_CLUSTER_MESH_2X2[:, 1]


def _require_hopper_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA GPU")
    major, _minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("Requires NVIDIA Hopper (sm90+) for cluster instructions")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_hopper_cuda()


@triton.jit
def _distributed_barrier_copy_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tle.distributed_barrier()
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _remote_roundtrip_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    remote_smem = tle.remote(smem, 0)
    local_ptr = tle.gpu.local_ptr(remote_smem, (tl.arange(0, BLOCK), ))
    tl.store(local_ptr, vals, mask=mask)

    out_vals = tl.load(local_ptr, mask=mask, other=0.0)
    tl.store(out_ptr + offs, out_vals, mask=mask)


@triton.jit
def _remote_peer_smem_kernel(out_ptr, shard_id_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    vals = tl.cast(offs + pid * BLOCK, tl.float32)

    smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (offs, ))
    tl.store(local_ptr, vals)
    tle.distributed_barrier(mesh)

    shard_id = tl.load(shard_id_ptr + pid)
    remote_smem = tle.remote(smem, shard_id, scope=mesh)
    peer_ptr = tle.gpu.local_ptr(remote_smem, (offs, ))
    peer_vals = tl.load(peer_ptr)
    tl.store(out_ptr + pid * BLOCK + offs, peer_vals)


@triton.jit
def _remote_mixed_local_remote_same_buffer_kernel(out_ptr, shard_id_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    vals = tl.cast(offs + pid * BLOCK, tl.float32)
    rank = tle.shard_id(mesh, "cluster_x")

    smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    shard_id = tl.load(shard_id_ptr + pid)
    remote_smem = tle.remote(smem, shard_id, scope=mesh)

    local_ptr = tle.gpu.local_ptr(smem, (offs, ))
    tl.store(local_ptr, vals)
    tle.distributed_barrier(mesh)

    if rank == 0:
        mixed = tl.load(local_ptr)
    else:
        remote_ptr = tle.gpu.local_ptr(remote_smem, (offs, ))
        mixed = tl.load(remote_ptr)
    tl.store(out_ptr + pid * BLOCK + offs, mixed)


@triton.jit
def _remote_peer_smem_2d_kernel(
    out_ptr,
    shard_id_ptr,
    mesh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    cols = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    vals = tl.cast(rows * BLOCK_K + cols + pid * BLOCK_M * BLOCK_K, tl.float32)

    smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (rows, cols))
    tl.store(local_ptr, vals)
    tle.distributed_barrier(mesh)

    shard_id = tl.load(shard_id_ptr + pid)
    remote_buffer = tle.remote(smem, shard_id, scope=mesh)
    remote_ptr = tle.gpu.local_ptr(remote_buffer, (rows, cols))
    peer_vals = tl.load(remote_ptr)
    # Ensure peer CTA finishes remote reads before either side reuses smem.
    tle.distributed_barrier(mesh)

    out_ptrs = out_ptr + pid * BLOCK_M * BLOCK_K + rows * BLOCK_K + cols
    tl.store(out_ptrs, peer_vals)


@triton.jit
def _remote_const_shard_load_kernel(out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    vals = tl.cast(offs + pid * BLOCK, tl.float16)

    smem = tle.gpu.alloc([BLOCK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (offs, ))
    tl.store(local_ptr, vals)
    tle.distributed_barrier(mesh)

    # Compile-time shard id path should lower to remote cluster load.
    remote_buffer = tle.remote(smem, 0, scope=mesh)
    remote_ptr = tle.gpu.local_ptr(remote_buffer, (offs, ))
    peer_vals = tl.load(remote_ptr)
    tl.store(out_ptr + pid * BLOCK + offs, peer_vals)


@triton.jit
def _remote_const_shard_vectorized_load_kernel(
    out_ptr,
    mesh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    cols = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    vals = tl.cast(rows * BLOCK_K + cols + pid * BLOCK_M * BLOCK_K, tl.float16)

    smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (rows, cols))
    tl.store(local_ptr, vals)
    tle.distributed_barrier(mesh)

    remote_buffer = tle.remote(smem, 0, scope=mesh)
    remote_ptr = tle.gpu.local_ptr(remote_buffer, (rows, cols))
    peer_vals = tl.load(remote_ptr)
    tle.distributed_barrier(mesh)

    out_ptrs = out_ptr + pid * BLOCK_M * BLOCK_K + rows * BLOCK_K + cols
    tl.store(out_ptrs, peer_vals)


@triton.jit
def _remote_pointer_input_disallowed_kernel(out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc([BLOCK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (offs, ))
    remote_ptr = tle.remote(local_ptr, 0, scope=mesh)
    vals = tl.load(remote_ptr)
    tl.store(out_ptr + offs, vals)


@triton.jit
def _remote_buffer_const_shard_vectorized_load_kernel(
    out_ptr,
    mesh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    cols = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    vals = tl.cast(rows * BLOCK_K + cols + pid * BLOCK_M * BLOCK_K, tl.float16)

    smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (rows, cols))
    tl.store(local_ptr, vals)
    tle.distributed_barrier(mesh)

    remote_buffer = tle.remote(smem, 0, scope=mesh)
    remote_ptr = tle.gpu.local_ptr(remote_buffer, (rows, cols))
    peer_vals = tl.load(remote_ptr)
    tle.distributed_barrier(mesh)

    out_ptrs = out_ptr + pid * BLOCK_M * BLOCK_K + rows * BLOCK_K + cols
    tl.store(out_ptrs, peer_vals)


@triton.jit
def _remote_buffer_const_shard_vectorized_load_rank3_kernel(
    out_ptr,
    mesh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SLOT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    cols = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    vals = tl.cast(rows * BLOCK_K + cols + pid * BLOCK_M * BLOCK_K, tl.float16)
    slots = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32) + SLOT

    smem = tle.gpu.alloc([2, BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (slots, rows, cols))
    tl.store(local_ptr, vals)
    tle.distributed_barrier(mesh)

    remote_buffer = tle.remote(smem, 0, scope=mesh)
    remote_ptr = tle.gpu.local_ptr(remote_buffer, (slots, rows, cols))
    peer_vals = tl.load(remote_ptr)
    tle.distributed_barrier(mesh)

    out_ptrs = out_ptr + pid * BLOCK_M * BLOCK_K + rows * BLOCK_K + cols
    tl.store(out_ptrs, peer_vals)


@triton.jit
def _submesh_barrier_lowering_kernel(out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    vals = tl.full((BLOCK, ), pid, tl.int32)
    tle.distributed_barrier(mesh)
    tl.store(out_ptr + pid * BLOCK + offs, vals)


@triton.jit
def _remote_rank0_dsmem_atomic_add_kernel(out_ptr, mesh: tl.constexpr):
    rank = tle.shard_id(mesh, "cluster_x")
    idx = tl.arange(0, 1)
    smem = tle.gpu.alloc([1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    counter_ptr = tle.gpu.local_ptr(smem, (idx, ))

    if rank == 0:
        tl.store(counter_ptr, 0)
    tle.distributed_barrier(mesh)

    remote_rank0 = tle.remote(smem, 0, scope=mesh)
    remote_ptr = tle.gpu.local_ptr(remote_rank0, (idx, ))
    tl.atomic_add(remote_ptr, 1, sem="relaxed", scope="cta")
    tle.distributed_barrier(mesh)

    if rank == 0:
        counter = tl.load(counter_ptr)
        tl.store(out_ptr + idx, counter)


@triton.jit
def _remote_scan_shared_scratch_kernel(out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    rank = tle.shard_id(mesh, "cluster_x")
    offs = tl.arange(0, BLOCK)

    smem = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    local_ptr = tle.gpu.local_ptr(smem, (offs, ))
    if rank == 0:
        tl.store(local_ptr, offs)
    tle.distributed_barrier(mesh)

    if rank == 0:
        vals = tl.load(local_ptr)
        prefix = tl.cumsum(vals, axis=0)
        tl.store(local_ptr, prefix)
    tle.distributed_barrier(mesh)

    remote_rank0 = tle.remote(smem, 0, scope=mesh)
    remote_ptr = tle.gpu.local_ptr(remote_rank0, (offs, ))
    remote_vals = tl.load(remote_ptr)
    tl.store(out_ptr + rank * BLOCK + offs, remote_vals)


@triton.jit
def _distributed_barrier_multiblock_counter_kernel(counter_ptr, out_ptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    cluster_id = pid // 2
    counter_lane_ptr = counter_ptr + cluster_id

    tl.atomic_add(counter_lane_ptr, 1)
    tle.distributed_barrier(mesh)

    seen = tl.load(counter_lane_ptr)
    tl.store(out_ptr + pid, seen)


@triton.jit
def _distributed_barrier_grid_counter_kernel(counter_ptr, out_ptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    tl.atomic_add(counter_ptr, 1)
    tle.distributed_barrier(mesh)
    seen = tl.load(counter_ptr)
    tl.store(out_ptr + pid, seen)


@triton.jit
def _submesh_row_group_barrier_kernel(
    counter_row0_ptr,
    counter_row1_ptr,
    out_row0_ptr,
    out_row1_ptr,
    row0_mesh: tl.constexpr,
    row1_mesh: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid < 2:
        tl.atomic_add(counter_row0_ptr, 1)
        tle.distributed_barrier(row0_mesh)
        seen_row0 = tl.load(counter_row0_ptr)
        tl.store(out_row0_ptr + pid, seen_row0)
    else:
        tl.store(out_row0_ptr + pid, -1)

    if pid >= 2:
        tl.atomic_add(counter_row1_ptr, 3)
        tle.distributed_barrier(row1_mesh)
        seen_row1 = tl.load(counter_row1_ptr)
        tl.store(out_row1_ptr + pid, seen_row1)
    else:
        tl.store(out_row1_ptr + pid, -1)


@triton.jit
def _submesh_col_group_barrier_kernel(
    counter_col0_ptr,
    counter_col1_ptr,
    out_col0_ptr,
    out_col1_ptr,
    col0_mesh: tl.constexpr,
    col1_mesh: tl.constexpr,
):
    pid = tl.program_id(0)
    is_col0 = (pid & 1) == 0

    if is_col0:
        tl.atomic_add(counter_col0_ptr, 1)
        tle.distributed_barrier(col0_mesh)
        seen_col0 = tl.load(counter_col0_ptr)
        tl.store(out_col0_ptr + pid, seen_col0)
    else:
        tl.store(out_col0_ptr + pid, -1)

    if not is_col0:
        tl.atomic_add(counter_col1_ptr, 5)
        tle.distributed_barrier(col1_mesh)
        seen_col1 = tl.load(counter_col1_ptr)
        tl.store(out_col1_ptr + pid, seen_col1)
    else:
        tl.store(out_col1_ptr + pid, -1)


@triton.jit
def _remote_cluster_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    shard_id_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cc,
    stride_cm,
    stride_cn,
    mesh: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    cluster_id = pid // CLUSTER_SIZE
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_idx_m = tl.broadcast_to(offs_m[:, None], (BM, BK))
    a_idx_k = tl.broadcast_to(offs_k[None, :], (BM, BK))
    a_flat_idx = a_idx_m * BK + a_idx_k
    a_buf = tle.gpu.alloc([BM * BK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    a_local_ptr = tle.gpu.local_ptr(a_buf, (a_flat_idx, ))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    b_col = cluster_rank * BN + offs_n
    remote_shard_id = tl.load(shard_id_ptr + pid)
    a_buf_remote = tle.remote(a_buf, remote_shard_id, scope=mesh)

    for k0 in range(0, K, BK):
        if cluster_rank == 0:
            a_tile_gptr = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
            a_tile = tl.load(a_tile_gptr)
            tl.store(a_local_ptr, a_tile)

        tle.distributed_barrier(mesh)

        if cluster_rank == 0:
            for kk in range(0, BK):
                a_col_idx = offs_m * BK + kk
                a_col_ptr = tle.gpu.local_ptr(a_buf, (a_col_idx, ))
                a_vals = tl.cast(tl.load(a_col_ptr), tl.float32)
                b_vals = tl.cast(tl.load(b_ptr + (k0 + kk) * stride_bk + b_col * stride_bn), tl.float32)
                acc += tl.expand_dims(a_vals, 1) * tl.expand_dims(b_vals, 0)
        else:
            for kk in range(0, BK):
                a_col_idx = offs_m * BK + kk
                a_col_ptr = tle.gpu.local_ptr(a_buf_remote, (a_col_idx, ))
                a_vals = tl.cast(tl.load(a_col_ptr), tl.float32)
                b_vals = tl.cast(tl.load(b_ptr + (k0 + kk) * stride_bk + b_col * stride_bn), tl.float32)
                acc += tl.expand_dims(a_vals, 1) * tl.expand_dims(b_vals, 0)

        tle.distributed_barrier(mesh)

    c_col = cluster_rank * BN + offs_n
    c_cluster_base = c_ptr + cluster_id * stride_cc
    c_ptrs = c_cluster_base + offs_m[:, None] * stride_cm + c_col[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def _shard_id_axis_kernel(out_ptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    sid = tle.shard_id(mesh, "cluster_x")
    tl.store(out_ptr + pid, sid)


@triton.jit
def _remote_dot_dotk32_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    shard_id_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    mesh: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DOT_K: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    A_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    cluster_id = pid // CLUSTER_SIZE
    num_pid_n = tl.cdiv(N, BN)
    num_pid_n_group = tl.cdiv(num_pid_n, CLUSTER_SIZE)
    pid_m = cluster_id // num_pid_n_group
    pid_ng = cluster_id % num_pid_n_group
    pid_n = pid_ng * CLUSTER_SIZE + cluster_rank

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_rows_full = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
    a_cols_full = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
    a_rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, DOT_K))
    a_buf = tle.gpu.alloc([A_SLOTS, BM, BK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    shard_id = tl.load(shard_id_ptr + pid)
    a_buf_remote = tle.remote(a_buf, shard_id, scope=mesh)
    for k0 in range(0, K, BK):
        iter_idx = k0 // BK
        slot = iter_idx % A_SLOTS
        slot_full = tl.zeros((BM, BK), dtype=tl.int32) + slot
        if cluster_rank == 0:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
            a_tile = tl.load(a_ptrs)
            a_local_ptr_tile = tle.gpu.local_ptr(a_buf, (slot_full, a_rows_full, a_cols_full))
            tl.store(a_local_ptr_tile, a_tile)

        tle.distributed_barrier(mesh)

        for ks in range(0, BK, DOT_K):
            k_local = ks + tl.arange(0, DOT_K)
            a_cols = tl.broadcast_to(k_local[None, :], (BM, DOT_K))
            slot_dot = tl.zeros((BM, DOT_K), dtype=tl.int32) + slot
            a_ptr_local = tle.gpu.local_ptr(a_buf, (slot_dot, a_rows, a_cols))
            if cluster_rank == 0:
                a = tl.load(a_ptr_local)
            else:
                a_ptr_remote = tle.gpu.local_ptr(a_buf_remote, (slot_dot, a_rows, a_cols))
                a = tl.load(a_ptr_remote)

            b_ptrs = b_ptr + (k0 + k_local)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)

        tle.distributed_barrier(mesh)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _remote_dot_masked_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    shard_id_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    mesh: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DOT_K: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
    A_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    cluster_id = pid // CLUSTER_SIZE
    num_pid_n = tl.cdiv(N, BN)
    num_pid_n_group = tl.cdiv(num_pid_n, CLUSTER_SIZE)
    pid_m = cluster_id // num_pid_n_group
    pid_ng = cluster_id % num_pid_n_group
    pid_n = pid_ng * CLUSTER_SIZE + cluster_rank

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_rows_full = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
    a_cols_full = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
    a_rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, DOT_K))
    a_buf = tle.gpu.alloc([A_SLOTS, BM, BK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    shard_id = tl.load(shard_id_ptr + pid)
    a_buf_remote = tle.remote(a_buf, shard_id, scope=mesh)
    for k0 in range(0, K, BK):
        iter_idx = k0 // BK
        slot = iter_idx % A_SLOTS
        slot_full = tl.zeros((BM, BK), dtype=tl.int32) + slot
        if cluster_rank == 0:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
            if USE_MASK:
                a_mask_tile = (offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K)
                a_tile = tl.load(a_ptrs, mask=a_mask_tile, other=0.0)
            else:
                a_tile = tl.load(a_ptrs)
            a_local_ptr_tile = tle.gpu.local_ptr(a_buf, (slot_full, a_rows_full, a_cols_full))
            if USE_MASK:
                tl.store(a_local_ptr_tile, a_tile, mask=a_mask_tile)
            else:
                tl.store(a_local_ptr_tile, a_tile)

        tle.distributed_barrier(mesh)

        for ks in range(0, BK, DOT_K):
            k_local = ks + tl.arange(0, DOT_K)
            a_cols = tl.broadcast_to(k_local[None, :], (BM, DOT_K))
            slot_dot = tl.zeros((BM, DOT_K), dtype=tl.int32) + slot
            a_ptr_local = tle.gpu.local_ptr(a_buf, (slot_dot, a_rows, a_cols))
            if USE_MASK:
                a_mask = (offs_m[:, None] < M) & ((k0 + k_local)[None, :] < K)
                if cluster_rank == 0:
                    a = tl.load(a_ptr_local, mask=a_mask, other=0.0)
                else:
                    a_ptr_remote = tle.gpu.local_ptr(a_buf_remote, (slot_dot, a_rows, a_cols))
                    a = tl.load(a_ptr_remote, mask=a_mask, other=0.0)
                b_ptrs = b_ptr + (k0 + k_local)[:, None] * stride_bk + offs_n[None, :] * stride_bn
                b_mask = ((k0 + k_local)[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            else:
                if cluster_rank == 0:
                    a = tl.load(a_ptr_local)
                else:
                    a_ptr_remote = tle.gpu.local_ptr(a_buf_remote, (slot_dot, a_rows, a_cols))
                    a = tl.load(a_ptr_remote)
                b_ptrs = b_ptr + (k0 + k_local)[:, None] * stride_bk + offs_n[None, :] * stride_bn
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)

        tle.distributed_barrier(mesh)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if USE_MASK:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _remote_dot_prefetch_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    shard_id_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    mesh: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DOT_K: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    A_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    cluster_id = pid // CLUSTER_SIZE
    num_pid_n = tl.cdiv(N, BN)
    num_pid_n_group = tl.cdiv(num_pid_n, CLUSTER_SIZE)
    pid_m = cluster_id // num_pid_n_group
    pid_ng = cluster_id % num_pid_n_group
    pid_n = pid_ng * CLUSTER_SIZE + cluster_rank

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_rows_full = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
    a_cols_full = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
    a_rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, DOT_K))
    a_buf = tle.gpu.alloc([A_SLOTS, BM, BK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    shard_id = tl.load(shard_id_ptr + pid)
    a_buf_remote = tle.remote(a_buf, shard_id, scope=mesh)

    slot0 = 0
    slot0_full = tl.zeros((BM, BK), dtype=tl.int32) + slot0
    if cluster_rank == 0:
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs)
        a_local_ptr_tile = tle.gpu.local_ptr(a_buf, (slot0_full, a_rows_full, a_cols_full))
        tl.store(a_local_ptr_tile, a_tile)
    tle.distributed_barrier(mesh)

    for k0 in range(0, K, BK):
        iter_idx = k0 // BK
        slot = iter_idx % A_SLOTS
        for ks in range(0, BK, DOT_K):
            k_local = ks + tl.arange(0, DOT_K)
            a_cols = tl.broadcast_to(k_local[None, :], (BM, DOT_K))
            slot_dot = tl.zeros((BM, DOT_K), dtype=tl.int32) + slot
            a_ptr_remote = tle.gpu.local_ptr(a_buf_remote, (slot_dot, a_rows, a_cols))
            a = tl.load(a_ptr_remote)
            b_ptrs = b_ptr + (k0 + k_local)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)

        # Prevent same-slot overwrite/read race when A_SLOTS == 1.
        if A_SLOTS == 1:
            tle.distributed_barrier(mesh)

        next_k0 = k0 + BK
        has_next = next_k0 < K
        next_slot = (iter_idx + 1) % A_SLOTS
        next_slot_full = tl.zeros((BM, BK), dtype=tl.int32) + next_slot
        if has_next and cluster_rank == 0:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (next_k0 + offs_k)[None, :] * stride_ak
            a_tile = tl.load(a_ptrs)
            a_local_ptr_tile = tle.gpu.local_ptr(a_buf, (next_slot_full, a_rows_full, a_cols_full))
            tl.store(a_local_ptr_tile, a_tile)

        tle.distributed_barrier(mesh)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


class TestTLEDistributed:

    def test_distributed_barrier_copy(self):
        block = 128
        numel = block
        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        _distributed_barrier_copy_kernel[(1, )](x, out, numel, BLOCK=block)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_remote_roundtrip(self):
        block = 128
        numel = block
        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        _remote_roundtrip_kernel[(1, )](x, out, numel, BLOCK=block)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_remote_read_peer_smem_same_cluster(self):
        block = 64
        grid = 2
        cluster_size = 2
        num_programs = grid * cluster_size
        out = torch.empty((num_programs * block, ), device="cuda", dtype=torch.float32)
        shard_id = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)

        compiled = _remote_peer_smem_kernel.warmup(
            out,
            shard_id_ptr=shard_id,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        assert ("tle.remote_pointers" in compiled.asm["ttgir"]) or ("tle.remote_shard_id_carrier"
                                                                    in compiled.asm["ttgir"])
        assert "\"ttg.num-ctas\" = 1" in compiled.asm["ttgir"]
        assert "mapa.shared::cluster" in compiled.asm["ptx"]

        _remote_peer_smem_kernel[(grid, )](out, shard_id_ptr=shard_id, mesh=BLOCK_CLUSTER_MESH, BLOCK=block, num_ctas=1,
                                           num_warps=4)
        expected_chunks = []
        for pid in range(num_programs):
            peer_pid = pid ^ 1
            expected_chunks.append(
                torch.arange(peer_pid * block, (peer_pid + 1) * block, device="cuda", dtype=torch.float32))
        expected = torch.cat(expected_chunks, dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_remote_mixed_local_remote_same_buffer(self):
        block = 64
        grid = 2
        cluster_size = 2
        num_programs = grid * cluster_size
        out = torch.empty((num_programs * block, ), device="cuda", dtype=torch.float32)
        shard_id = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)

        compiled = _remote_mixed_local_remote_same_buffer_kernel.warmup(
            out,
            shard_id_ptr=shard_id,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        ttgir = compiled.asm["ttgir"]
        ptx = compiled.asm["ptx"]
        assert "\"tle.local_pointers\"" in ttgir
        assert "\"tle.remote_pointers\"" in ttgir
        assert "mapa.shared::cluster" in ptx

        _remote_mixed_local_remote_same_buffer_kernel[(grid, )](out, shard_id_ptr=shard_id, mesh=BLOCK_CLUSTER_MESH,
                                                                BLOCK=block, num_ctas=1, num_warps=4)
        torch.cuda.synchronize()

        expected_chunks = []
        for pid in range(num_programs):
            if pid % cluster_size == 0:
                expected_chunks.append(torch.arange(pid * block, (pid + 1) * block, device="cuda", dtype=torch.float32))
            else:
                peer_pid = pid ^ 1
                expected_chunks.append(
                    torch.arange(peer_pid * block, (peer_pid + 1) * block, device="cuda", dtype=torch.float32))
        expected = torch.cat(expected_chunks, dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_remote_read_peer_smem_2d_runtime_shard_id_same_cluster(self):
        block_m = 64
        block_k = 16
        grid = 1
        cluster_size = 2
        num_programs = grid * cluster_size
        tile_elems = block_m * block_k

        out = torch.empty((num_programs * tile_elems, ), device="cuda", dtype=torch.float32)
        shard_id = torch.tensor([1, 0], device="cuda", dtype=torch.int32)

        compiled = _remote_peer_smem_2d_kernel.warmup(
            out,
            shard_id_ptr=shard_id,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        assert "tle.remote_pointers" in compiled.asm["ttgir"]
        assert re.search(r"ld\.shared::cluster(\.v(2|4))?\.b32", compiled.asm["ptx"]) is not None

        _remote_peer_smem_2d_kernel[(grid, )](
            out,
            shard_id_ptr=shard_id,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected_chunks = []
        for pid in range(num_programs):
            peer_pid = pid ^ 1
            expected_chunks.append(
                torch.arange(peer_pid * tile_elems, (peer_pid + 1) * tile_elems, device="cuda", dtype=torch.float32))
        expected = torch.cat(expected_chunks, dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_remote_const_shard_load_lowering_same_cluster(self):
        block = 128
        grid = 2
        cluster_size = 2
        num_programs = grid * cluster_size
        out = torch.empty((num_programs * block, ), device="cuda", dtype=torch.float16)

        compiled = _remote_const_shard_load_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        ptx = compiled.asm["ptx"]
        ttgir = compiled.asm["ttgir"]
        assert "mapa.shared::cluster" in ptx
        assert "\"tle.remote_pointers\"" in ttgir
        assert "%c0_i32" in ttgir

        _remote_const_shard_load_kernel[(grid, )](
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK=block,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected_chunks = []
        for pid in range(num_programs):
            peer_pid = (pid // cluster_size) * cluster_size
            expected_chunks.append(
                torch.arange(peer_pid * block, (peer_pid + 1) * block, device="cuda", dtype=torch.float16))
        expected = torch.cat(expected_chunks, dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_remote_const_shard_load_high_block_encoding_no_regression(self):
        block = 256
        out = torch.empty((2 * block, ), device="cuda", dtype=torch.float16)

        compiled = _remote_const_shard_load_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK=block,
            grid=(1, ),
            num_ctas=1,
            num_warps=4,
        )
        ttgir = compiled.asm["ttgir"]
        ptx = compiled.asm["ptx"]

        # local/remote pointer tensors should keep the load-friendly encoding.
        assert "tensor<256x!tt.ptr<f16, 3>, #blocked>" in ttgir
        assert re.search(
            r"\"tle\.remote_pointers\"\(%[^,]+,\s*%[^)]+\)\s*:\s*"
            r"\(tensor<256x!tt\.ptr<f16,\s*3>,\s*#blocked>,\s*i32\)\s*->\s*"
            r"tensor<256x!tt\.ptr<f16,\s*3>,\s*#blocked>",
            ttgir,
        ) is not None
        # Avoid re-introducing a degraded blocked1 -> blocked convert path.
        assert "ttg.convert_layout %remote_ptr : tensor<256x!tt.ptr<f16, 3>, #blocked1> -> tensor<256x!tt.ptr<f16, 3>, #blocked>" not in ttgir
        assert "ld.shared::cluster.b16" in ptx

    def test_remote_const_shard_vectorized_load_lowering_same_cluster(self):
        block_m = 32
        block_k = 8
        grid = 1
        cluster_size = 2
        num_programs = grid * cluster_size
        tile_elems = block_m * block_k
        out = torch.empty((num_programs * tile_elems, ), device="cuda", dtype=torch.float16)

        compiled = _remote_const_shard_vectorized_load_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        ptx = compiled.asm["ptx"]
        assert re.search(r"ld\.shared::cluster(\.v(2|4))?\.b16", ptx) is not None

        _remote_const_shard_vectorized_load_kernel[(grid, )](
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected_chunk = torch.arange(0, tile_elems, device="cuda", dtype=torch.float16)
        expected = torch.cat([expected_chunk, expected_chunk], dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_remote_pointer_input_disallowed(self):
        out = torch.empty((32, ), device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match="only accepts tle.buffered_tensor"):
            _remote_pointer_input_disallowed_kernel.warmup(
                out,
                mesh=BLOCK_CLUSTER_MESH,
                BLOCK=32,
                grid=(1, ),
                num_ctas=1,
                num_warps=4,
            )

    def test_remote_buffer_const_shard_vectorized_load_lowering_same_cluster(self):
        block_m = 32
        block_k = 8
        grid = 1
        cluster_size = 2
        num_programs = grid * cluster_size
        tile_elems = block_m * block_k
        out = torch.empty((num_programs * tile_elems, ), device="cuda", dtype=torch.float16)

        compiled = _remote_buffer_const_shard_vectorized_load_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        ttgir = compiled.asm["ttgir"]
        ptx = compiled.asm["ptx"]
        assert "\"tle.remote_pointers\"" in ttgir
        assert re.search(r"ld\.shared::cluster(\.v(2|4))?\.b16", ptx) is not None

        _remote_buffer_const_shard_vectorized_load_kernel[(grid, )](
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected_chunk = torch.arange(0, tile_elems, device="cuda", dtype=torch.float16)
        expected = torch.cat([expected_chunk, expected_chunk], dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_remote_buffer_const_shard_vectorized_load_rank3_lowering_same_cluster(self):
        block_m = 32
        block_k = 8
        grid = 1
        cluster_size = 2
        num_programs = grid * cluster_size
        tile_elems = block_m * block_k
        out = torch.empty((num_programs * tile_elems, ), device="cuda", dtype=torch.float16)

        compiled = _remote_buffer_const_shard_vectorized_load_rank3_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            SLOT=1,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        ttgir = compiled.asm["ttgir"]
        ptx = compiled.asm["ptx"]
        assert "\"tle.remote_pointers\"" in ttgir
        assert re.search(r"ld\.shared::cluster(\.v(2|4))?\.b16", ptx) is not None

        _remote_buffer_const_shard_vectorized_load_rank3_kernel[(grid, )](
            out,
            mesh=BLOCK_CLUSTER_MESH,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            SLOT=1,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected_chunk = torch.arange(0, tile_elems, device="cuda", dtype=torch.float16)
        expected = torch.cat([expected_chunk, expected_chunk], dim=0)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("submesh", [BLOCK_CLUSTER_SUBMESH_ROW0, BLOCK_CLUSTER_SUBMESH_COL0])
    def test_distributed_barrier_submesh_lowering(self, submesh):
        block = 32
        out = torch.empty((4 * block, ), device="cuda", dtype=torch.int32)

        compiled = _submesh_barrier_lowering_kernel.warmup(
            out,
            mesh=submesh,
            BLOCK=block,
            grid=(1, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 2, 1)
        ptx = compiled.asm["ptx"]
        assert "atom.shared::cluster.add.u32" in ptx

        _submesh_barrier_lowering_kernel[(1, )](out, mesh=submesh, BLOCK=block, num_ctas=1, num_warps=4)
        torch.cuda.synchronize()

    def test_remote_rank0_dsmem_atomic_add_lowering_cluster8(self):
        out = torch.empty((1, ), device="cuda", dtype=torch.int32)
        compiled = _remote_rank0_dsmem_atomic_add_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH_8,
            grid=(1, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (8, 1, 1)
        ptx = compiled.asm["ptx"]
        assert "atom.shared::cluster.cta.relaxed.add.u32" in ptx
        assert "atom.shared.shared::cluster" not in ptx

    def test_remote_rank0_dsmem_atomic_add_runtime_cluster8_stable(self):
        out = torch.empty((1, ), device="cuda", dtype=torch.int32)
        for _ in range(512):
            _remote_rank0_dsmem_atomic_add_kernel[(1, )](
                out,
                mesh=BLOCK_CLUSTER_MESH_8,
                num_ctas=1,
                num_warps=4,
            )
            torch.cuda.synchronize()
            assert int(out.item()) == 8

    def test_remote_scan_shared_scratch_compile_regression_cluster8(self):
        block = 64
        out = torch.empty((8 * block, ), device="cuda", dtype=torch.int32)
        compiled = _remote_scan_shared_scratch_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH_8,
            BLOCK=block,
            grid=(1, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (8, 1, 1)
        assert "\"tt.scan\"" in compiled.asm["ttgir"]

    def test_distributed_barrier_multiblock_counter(self):
        grid = 2
        cluster_size = 2
        num_programs = grid * cluster_size

        counter = torch.zeros((grid, ), device="cuda", dtype=torch.int32)
        out = torch.empty((num_programs, ), device="cuda", dtype=torch.int32)

        compiled = _distributed_barrier_multiblock_counter_kernel.warmup(
            counter,
            out,
            mesh=BLOCK_CLUSTER_MESH,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)

        _distributed_barrier_multiblock_counter_kernel[(grid, )](
            counter,
            out,
            mesh=BLOCK_CLUSTER_MESH,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected_counter = torch.full_like(counter, cluster_size)
        expected_out = torch.full_like(out, cluster_size)
        torch.testing.assert_close(counter, expected_counter, atol=0, rtol=0)
        torch.testing.assert_close(out, expected_out, atol=0, rtol=0)

    def test_distributed_barrier_grid_counter(self, with_allocator):
        grid = 8
        counter = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        out = torch.empty((grid, ), device="cuda", dtype=torch.int32)

        compiled = _distributed_barrier_grid_counter_kernel.warmup(
            counter,
            out,
            mesh=BLOCK_GRID_MESH_8,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (1, 1, 1)
        assert compiled.metadata.launch_cooperative_grid is True
        assert 'group_kind = "grid"' in compiled.asm["ttgir"]
        ptx = compiled.asm["ptx"]
        # Match CUDA cooperative_groups/details/sync.h semantics:
        # atom.add.release.gpu + ld.acquire.gpu polling.
        assert "atom.add.release.gpu.u32" in ptx
        assert "ld.acquire.gpu.u32" in ptx
        assert "atom.global.or.b32" not in ptx

        _distributed_barrier_grid_counter_kernel[(grid, )](
            counter,
            out,
            mesh=BLOCK_GRID_MESH_8,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected = torch.full_like(out, grid)
        torch.testing.assert_close(counter, torch.tensor([grid], device="cuda", dtype=torch.int32), atol=0, rtol=0)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_distributed_barrier_grid_counter_repeated_launch_stable(self, with_allocator):
        # Minimal repro for previous occasional hangs:
        # pure grid barrier kernel, repeated cooperative launches.
        grid = 3
        mesh = tle.device_mesh({"block": [("block_x", grid)]})
        counter = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        out = torch.empty((grid, ), device="cuda", dtype=torch.int32)

        compiled = _distributed_barrier_grid_counter_kernel.warmup(
            counter,
            out,
            mesh=mesh,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (1, 1, 1)
        assert compiled.metadata.launch_cooperative_grid is True
        assert 'group_kind = "grid"' in compiled.asm["ttgir"]
        ptx = compiled.asm["ptx"]
        assert "atom.add.release.gpu.u32" in ptx
        assert "ld.acquire.gpu.u32" in ptx

        for _ in range(256):
            counter.zero_()
            _distributed_barrier_grid_counter_kernel[(grid, )](
                counter,
                out,
                mesh=mesh,
                num_ctas=1,
                num_warps=4,
            )
            torch.cuda.synchronize()
            assert int(counter.item()) == grid
        assert bool(torch.all(out == grid))

    def test_distributed_barrier_grid_counter_runtime_zero_init_scratch(self, with_allocator):
        grid = 3
        mesh = tle.device_mesh({"block": [("block_x", grid)]})
        counter = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        out = torch.empty((grid, ), device="cuda", dtype=torch.int32)

        from triton._internal_testing import default_alloc_fn

        def dirty_alloc(size: int, align: int, stream):
            # Return intentionally dirty scratch to validate launcher-side zero-init.
            return torch.full((size, ), 0x7F, device="cuda", dtype=torch.int8)

        triton.set_allocator(dirty_alloc)
        try:
            _distributed_barrier_grid_counter_kernel.warmup(
                counter,
                out,
                mesh=mesh,
                grid=(grid, ),
                num_ctas=1,
                num_warps=4,
            )
            for _ in range(64):
                counter.zero_()
                _distributed_barrier_grid_counter_kernel[(grid, )](
                    counter,
                    out,
                    mesh=mesh,
                    num_ctas=1,
                    num_warps=4,
                )
                torch.cuda.synchronize()
                assert int(counter.item()) == grid
            assert bool(torch.all(out == grid))
        finally:
            triton.set_allocator(default_alloc_fn)

    def test_distributed_barrier_row_group_independence(self):
        grid = 2
        counter_row0 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        counter_row1 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        out_row0 = torch.empty((4, ), device="cuda", dtype=torch.int32)
        out_row1 = torch.empty((4, ), device="cuda", dtype=torch.int32)

        compiled = _submesh_row_group_barrier_kernel.warmup(
            counter_row0,
            counter_row1,
            out_row0,
            out_row1,
            row0_mesh=BLOCK_CLUSTER_SUBMESH_ROW0,
            row1_mesh=BLOCK_CLUSTER_SUBMESH_ROW1,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 2, 1)
        assert compiled.asm["ptx"].count("atom.shared::cluster.add.u32") >= 2

        _submesh_row_group_barrier_kernel[(grid, )](
            counter_row0,
            counter_row1,
            out_row0,
            out_row1,
            row0_mesh=BLOCK_CLUSTER_SUBMESH_ROW0,
            row1_mesh=BLOCK_CLUSTER_SUBMESH_ROW1,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        row0_count = int(counter_row0.cpu().item())
        row1_count = int(counter_row1.cpu().item())
        assert row0_count > 0
        assert row1_count == 3 * row0_count
        torch.testing.assert_close(out_row0,
                                   torch.tensor([row0_count, row0_count, -1, -1], device="cuda", dtype=torch.int32))
        torch.testing.assert_close(out_row1,
                                   torch.tensor([-1, -1, row1_count, row1_count], device="cuda", dtype=torch.int32))

    def test_distributed_barrier_col_group_independence(self):
        grid = 2
        counter_col0 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        counter_col1 = torch.zeros((1, ), device="cuda", dtype=torch.int32)
        out_col0 = torch.empty((4, ), device="cuda", dtype=torch.int32)
        out_col1 = torch.empty((4, ), device="cuda", dtype=torch.int32)

        compiled = _submesh_col_group_barrier_kernel.warmup(
            counter_col0,
            counter_col1,
            out_col0,
            out_col1,
            col0_mesh=BLOCK_CLUSTER_SUBMESH_COL0,
            col1_mesh=BLOCK_CLUSTER_SUBMESH_COL1,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 2, 1)
        assert compiled.asm["ptx"].count("atom.shared::cluster.add.u32") >= 2

        _submesh_col_group_barrier_kernel[(grid, )](
            counter_col0,
            counter_col1,
            out_col0,
            out_col1,
            col0_mesh=BLOCK_CLUSTER_SUBMESH_COL0,
            col1_mesh=BLOCK_CLUSTER_SUBMESH_COL1,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        col0_count = int(counter_col0.cpu().item())
        col1_count = int(counter_col1.cpu().item())
        assert col0_count > 0
        assert col1_count == 5 * col0_count
        torch.testing.assert_close(out_col0,
                                   torch.tensor([col0_count, -1, col0_count, -1], device="cuda", dtype=torch.int32))
        torch.testing.assert_close(out_col1,
                                   torch.tensor([-1, col1_count, -1, col1_count], device="cuda", dtype=torch.int32))

    def test_remote_cluster_gemm_reuse_a_tile(self):
        clusters = 2
        cluster_size = 2
        bm = 16
        bn = 16
        bk = 16
        k = 32
        m = bm
        n = 2 * bn
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        c = torch.zeros((clusters, m, n), device="cuda", dtype=torch.float32)
        shard_id = torch.zeros((clusters * cluster_size, ), device="cuda", dtype=torch.int32)

        compiled = _remote_cluster_gemm_kernel.warmup(
            a,
            b,
            c,
            shard_id,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            c.stride(2),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            K=k,
            CLUSTER_SIZE=cluster_size,
            grid=(clusters, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        assert "\"ttg.num-ctas\" = 1" in compiled.asm["ttgir"]
        assert ("mapa.shared::cluster" in compiled.asm["ptx"]) or ("tle.remote_pointers" in compiled.asm["ttgir"])

        _remote_cluster_gemm_kernel[(clusters, )](
            a,
            b,
            c,
            shard_id,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            c.stride(2),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            K=k,
            CLUSTER_SIZE=cluster_size,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected = torch.matmul(a.float(), b.float())
        torch.testing.assert_close(c[0], expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(c[1], expected, atol=1e-2, rtol=1e-2)

    def test_shard_id_cluster_axis(self):
        grid = 3
        cluster_size = 2
        num_programs = grid * cluster_size
        out = torch.empty((num_programs, ), device="cuda", dtype=torch.int32)

        compiled = _shard_id_axis_kernel.warmup(
            out,
            mesh=BLOCK_CLUSTER_MESH,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)

        _shard_id_axis_kernel[(grid, )](
            out,
            mesh=BLOCK_CLUSTER_MESH,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected = torch.tensor([(i % cluster_size) for i in range(num_programs)], device="cuda", dtype=torch.int32)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "bk,dot_k,k,m,n,num_warps,num_stages,a_slots",
        [
            (32, 16, 64, 64, 128, 4, 2, 2),
            (32, 32, 64, 64, 128, 4, 2, 2),
            (32, 16, 128, 128, 256, 8, 3, 4),
            (64, 16, 64, 64, 128, 4, 2, 2),
            (64, 32, 64, 64, 128, 4, 2, 2),
            (64, 16, 128, 128, 256, 8, 3, 4),
            (64, 16, 256, 128, 512, 8, 3, 2),
            (128, 32, 256, 128, 512, 8, 2, 2),
        ],
    )
    def test_remote_dot_multi_config(self, bk, dot_k, k, m, n, num_warps, num_stages, a_slots):
        bm = 64
        bn = 64
        cluster_size = 2
        num_pid_n = triton.cdiv(n, bn)
        num_pid_n_group = triton.cdiv(num_pid_n, cluster_size)
        grid = (triton.cdiv(m, bm) * num_pid_n_group, )
        assert bk % dot_k == 0
        num_programs = grid[0] * cluster_size

        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        c = torch.empty((m, n), device="cuda", dtype=torch.float16)
        shard_id = torch.empty((num_programs, ), device="cuda", dtype=torch.int32)
        shard_id[0::2] = 1
        shard_id[1::2] = 0
        assert shard_id.numel() == num_programs

        compiled = _remote_dot_dotk32_kernel.warmup(
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            A_SLOTS=a_slots,
            grid=grid,
            num_ctas=1,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        assert "\"ttg.num-ctas\" = 1" in compiled.asm["ttgir"]

        _remote_dot_dotk32_kernel[grid](
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            A_SLOTS=a_slots,
            num_ctas=1,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        torch.cuda.synchronize()

        ref = torch.matmul(a, b)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-1)

    def test_remote_dot_no_index_repacking_local_staging(self):
        bm = 64
        bn = 64
        bk = 32
        dot_k = 16
        m = 128
        n = 256
        k = 128
        cluster_size = 2
        num_pid_n = triton.cdiv(n, bn)
        num_pid_n_group = triton.cdiv(num_pid_n, cluster_size)
        grid = (triton.cdiv(m, bm) * num_pid_n_group, )
        num_programs = grid[0] * cluster_size

        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        c = torch.empty((m, n), device="cuda", dtype=torch.float16)
        shard_id = torch.empty((num_programs, ), device="cuda", dtype=torch.int32)
        shard_id[0::2] = 1
        shard_id[1::2] = 0

        compiled = _remote_dot_dotk32_kernel.warmup(
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            A_SLOTS=2,
            grid=grid,
            num_ctas=1,
            num_warps=4,
            num_stages=2,
        )
        ttgir = compiled.asm["ttgir"]
        # Guard against regressions that repack index tensors through
        # local_alloc/local_load in the remote A path.
        assert "ttg.local_alloc %" not in ttgir
        # Guard against regressions that insert pointer convert_layout before load
        # on the remote/local shared pointer path.
        assert re.search(r"ttg\\.convert_layout .*-> tensor<.*!tt\\.ptr", ttgir) is None

    def test_remote_dot_mapa_base_pointer_mapped_once(self):
        bm = 64
        bn = 64
        bk = 32
        dot_k = 16
        m = 64
        n = 128
        k = 64
        cluster_size = 2
        num_pid_n = triton.cdiv(n, bn)
        num_pid_n_group = triton.cdiv(num_pid_n, cluster_size)
        grid = (triton.cdiv(m, bm) * num_pid_n_group, )
        num_programs = grid[0] * cluster_size

        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        c = torch.empty((m, n), device="cuda", dtype=torch.float16)
        shard_id = torch.empty((num_programs, ), device="cuda", dtype=torch.int32)
        shard_id[0::2] = 1
        shard_id[1::2] = 0

        compiled = _remote_dot_dotk32_kernel.warmup(
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            A_SLOTS=2,
            grid=grid,
            num_ctas=1,
            num_warps=4,
            num_stages=2,
        )
        ptx = compiled.asm["ptx"]
        # Regression guard: remote path should map cluster shared base pointer
        # with bounded mapa usage, not emit lane-wise mapa ops. Different
        # targets/ptxas versions may unroll this into a small fixed number.
        assert ptx.count("mapa.shared::cluster") <= 16

    @pytest.mark.parametrize("num_stages", [2, 3])
    def test_remote_dot_prefetch_bk64_single_slot_correctness(self, num_stages):
        bm = 64
        bn = 64
        bk = 64
        dot_k = 16
        m = 128
        n = 256
        k = 256
        cluster_size = 2
        a_slots = 1
        num_pid_n = triton.cdiv(n, bn)
        num_pid_n_group = triton.cdiv(num_pid_n, cluster_size)
        grid = (triton.cdiv(m, bm) * num_pid_n_group, )
        num_programs = grid[0] * cluster_size

        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        c = torch.empty((m, n), device="cuda", dtype=torch.float16)
        shard_id = torch.zeros((num_programs, ), device="cuda", dtype=torch.int32)

        compiled = _remote_dot_prefetch_kernel.warmup(
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            A_SLOTS=a_slots,
            grid=grid,
            num_ctas=1,
            num_warps=8,
            num_stages=num_stages,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        assert "\"ttg.num-ctas\" = 1" in compiled.asm["ttgir"]

        _remote_dot_prefetch_kernel[grid](
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            A_SLOTS=a_slots,
            num_ctas=1,
            num_warps=8,
            num_stages=num_stages,
        )
        torch.cuda.synchronize()

        ref = torch.matmul(a, b)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize(
        "m,n,k,bk,num_warps,num_stages,a_slots",
        [
            (70, 130, 96, 32, 4, 2, 1),
            (70, 130, 128, 64, 4, 2, 1),
            (96, 190, 130, 32, 8, 3, 2),
            (96, 190, 130, 64, 8, 3, 2),
        ],
    )
    def test_remote_dot_masked_multi_config(self, m, n, k, bk, num_warps, num_stages, a_slots):
        bm = 64
        bn = 64
        dot_k = 32
        cluster_size = 2
        assert bk % dot_k == 0

        num_pid_n = triton.cdiv(n, bn)
        num_pid_n_group = triton.cdiv(num_pid_n, cluster_size)
        grid = (triton.cdiv(m, bm) * num_pid_n_group, )
        num_programs = grid[0] * cluster_size

        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        c = torch.empty((m, n), device="cuda", dtype=torch.float16)
        shard_id = torch.empty((num_programs, ), device="cuda", dtype=torch.int32)
        shard_id[0::2] = 1
        shard_id[1::2] = 0
        use_mask = True

        compiled = _remote_dot_masked_kernel.warmup(
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            USE_MASK=use_mask,
            A_SLOTS=a_slots,
            grid=grid,
            num_ctas=1,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        assert compiled.metadata.cluster_dims == (2, 1, 1)
        assert "\"ttg.num-ctas\" = 1" in compiled.asm["ttgir"]

        _remote_dot_masked_kernel[grid](
            a,
            b,
            c,
            shard_id,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            mesh=BLOCK_CLUSTER_MESH,
            BM=bm,
            BN=bn,
            BK=bk,
            DOT_K=dot_k,
            CLUSTER_SIZE=cluster_size,
            USE_MASK=use_mask,
            A_SLOTS=a_slots,
            num_ctas=1,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        torch.cuda.synchronize()

        ref = torch.matmul(a, b)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-1)
