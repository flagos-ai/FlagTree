import argparse
import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

from triton.experimental.tle.raw.nvshmem.utils import (
    load_common_host,
    load_host,
    init_torch_distributed,
    init_nvshmem_by_torch_pg,
    tensor_from_pointer,
)


def _device_dialect(function_name):
    return dialect(
        name="cuda",
        compiler="clang",
        file=Path(__file__).parent / "gemm-ar-device.cu",
        extern_func_name=function_name,
    )


@_device_dialect("ar_mark_tile_ready")
def mark_tile_ready(*args, **kwargs):
    ...


@_device_dialect("ar_wait_tile_ready")
def wait_tile_ready(*args, **kwargs):
    ...


@_device_dialect("ar_multimem_ar_vector_multicast_store")
def multimem_ar_vector_multicast_store(*args, **kwargs):
    ...


@_device_dialect("ar_multimem_store_barrier")
def multimem_store_barrier(*args, **kwargs):
    ...


def configure_host_library(host):
    host.gemm_ar_workspace_create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    host.gemm_ar_workspace_create.restype = ctypes.c_int
    host.gemm_ar_workspace_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    host.gemm_ar_workspace_destroy.restype = ctypes.c_int
    host.gemm_ar_multimem_output_create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    host.gemm_ar_multimem_output_create.restype = ctypes.c_int
    host.gemm_ar_multimem_output_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    host.gemm_ar_multimem_output_destroy.restype = ctypes.c_int
    host.gemm_ar_peer_workspace_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.gemm_ar_peer_workspace_ptr.restype = ctypes.c_void_p
    host.gemm_ar_mc_ptr.argtypes = [ctypes.c_void_p]
    host.gemm_ar_mc_ptr.restype = ctypes.c_void_p


def create_workspace(host, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, world_size, dtype, device):
    workspace_ptr = ctypes.c_void_p()
    ready_ptr = ctypes.c_void_p()
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n

    result = host.gemm_ar_workspace_create(M * N, ctypes.byref(workspace_ptr), ctypes.byref(ready_ptr), num_tiles)
    assert result == 0, f"workspace allocation failed: {result}"

    workspace = tensor_from_pointer(workspace_ptr, (M, N), dtype, device)
    ready = tensor_from_pointer(ready_ptr, (world_size, num_tiles), torch.uint64, device)

    return workspace_ptr, ready_ptr, workspace, ready, num_tiles


def create_multimem_output(host, M, N, world_size, num_comm_sms, dtype, device):
    output_ptr = ctypes.c_void_p()
    barrier_ptr = ctypes.c_void_p()
    result = host.gemm_ar_multimem_output_create(
        M * N,
        ctypes.byref(output_ptr),
        ctypes.byref(barrier_ptr),
        num_comm_sms,
    )
    assert result == 0, f"multimem output allocation failed: {result}"

    output = tensor_from_pointer(output_ptr, (M, N), dtype, device)
    barrier = tensor_from_pointer(
        barrier_ptr,
        (world_size, num_comm_sms),
        torch.uint64,
        device,
    )
    return output_ptr, barrier_ptr, output, barrier


def create_peer_pointer_table(host, symmetric_ptr, world_size, device):
    peer_addresses = []
    for peer in range(world_size):
        peer_ptr = host.gemm_ar_peer_workspace_ptr(symmetric_ptr, peer)
        assert peer_ptr, f"peer {peer} symmetric pointer is unavailable"
        peer_addresses.append(peer_ptr)
    return torch.tensor(peer_addresses, dtype=torch.uint64, device=device)


@triton.jit
def thread_idx_x():
    return tl.inline_asm_elementwise(
        "mov.u32 $0, %tid.x;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["epoch"])
def consumer_all_reduce_kernel(
    mc_ptr,
    mc_out_ptr,
    peer_workspace_ptrs,
    peer_ready_ptrs,
    multimem_ready,
    peer_multimem_ready_ptrs,
    ar_out,
    epoch,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    USE_MULTIMEM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tile_cols = tl.cdiv(N, BLOCK_SIZE_N)
    num_tile_rows = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles = num_tile_rows * num_tile_cols
    thread_idx = thread_idx_x()
    BLOCK_DIM: tl.constexpr = 32 * 32
    VEC_SIZE: tl.constexpr = 128 // 16
    VEC_PER_ROW: tl.constexpr = BLOCK_SIZE_N // VEC_SIZE
    tl.static_assert(BLOCK_SIZE_N % VEC_SIZE == 0)

    world_size_i32 = tl.full((), WORLD_SIZE, tl.int32)
    rank_i32 = tl.full((), RANK, tl.int32)
    epoch_u64 = tl.cast(epoch, tl.uint64)
    if USE_MULTIMEM:
        for tile_id in range(pid + RANK * NUM_COMM_SMS, num_tiles, NUM_COMM_SMS * WORLD_SIZE):
            tle_raw.call(wait_tile_ready, [peer_ready_ptrs, world_size_i32, tile_id, num_tiles, epoch_u64])
            tile_row = tile_id // num_tile_cols
            tile_col = tile_id % num_tile_cols
            tile_rows = min(M - tile_row * BLOCK_SIZE_M, BLOCK_SIZE_M)
            vecs_per_tile = tile_rows * VEC_PER_ROW
            for idx in range(thread_idx, vecs_per_tile, BLOCK_DIM):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                offset = ((tile_row * BLOCK_SIZE_M + row_id) * N + tile_col * BLOCK_SIZE_N + col_id * VEC_SIZE)
                tle_raw.call(multimem_ar_vector_multicast_store, [mc_ptr, mc_out_ptr, offset])
        tle_raw.call(multimem_store_barrier,
                     [multimem_ready, peer_multimem_ready_ptrs, rank_i32, world_size_i32, epoch_u64])
    else:
        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            tle_raw.call(wait_tile_ready, [peer_ready_ptrs, world_size_i32, tile_id, num_tiles, epoch_u64])
            tile_row = tile_id // num_tile_cols
            tile_col = tile_id % num_tile_cols
            col_offsets = tile_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < N
            tile_rows = min(M - tile_row * BLOCK_SIZE_M, BLOCK_SIZE_M)
            out_dtype = ar_out.dtype.element_ty
            for row in range(tile_rows):
                row_offset = (tile_row * BLOCK_SIZE_M + row) * N
                accumulator = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)

                for i in tl.static_range(0, WORLD_SIZE):
                    peer = (RANK + WORLD_SIZE - i) % WORLD_SIZE
                    peer_ptr = tl.load(peer_workspace_ptrs + peer)
                    peer_ptr = peer_ptr.to(tl.pointer_type(out_dtype))
                    value = tl.load(peer_ptr + row_offset + col_offsets, mask=col_mask, other=0.0)
                    accumulator += value.to(tl.float32)

                tl.store(
                    ar_out + row_offset + col_offsets,
                    accumulator.to(out_dtype),
                    mask=col_mask,
                )


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(do_not_specialize=["epoch"])
def gemm_ar_producer(
    a_ptr,
    b_ptr,
    c_ptr,
    ready,
    epoch,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    RANK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_GEMM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )

    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )

    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    rank_i32 = tl.full((), RANK, tl.int32)
    epoch_u64 = tl.cast(epoch, tl.uint64)
    dtype = c_ptr.dtype.element_ty

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS, flatten=False):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in tl.range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a, b, accumulator)

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

        ready_tile_id = pid_m * num_pid_n + pid_n
        tle_raw.call(mark_tile_ready, [ready, ready_tile_id, rank_i32, num_tiles, epoch_u64])


def parse_args():
    parser = argparse.ArgumentParser(description="NVSHMEM GEMM allreduce")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K-per-rank", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--case", choices=("check", "perf"), default="check")
    parser.add_argument("--epilogue-subtile", action="store_true")
    parser.add_argument("--use-multimem", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    M = args.M
    N = args.N
    K_per_rank = args.K_per_rank
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    num_comm_sms = 12

    assert world_size >= 2, "GEMM all-reduce requires at least two GPUs"
    assert world_size == local_world_size, ("This example is designed for single-node testing: "
                                            "WORLD_SIZE must equal LOCAL_WORLD_SIZE")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    group = init_torch_distributed()
    host_source = Path(__file__).with_name("gemm-ar-host.cu")
    host = load_host(host_source)
    common = load_common_host()
    configure_host_library(host)
    init_nvshmem_by_torch_pg(common, group)

    dtype = torch.float16
    torch.manual_seed(1234 + rank)
    a = torch.randn((M, K_per_rank), dtype=dtype, device=device) * 0.1
    b = torch.randn((K_per_rank, N), dtype=dtype, device=device) * 0.1

    workspace_ptr, ready_ptr, workspace, ready, num_tiles = create_workspace(host, M, N, BLOCK_M, BLOCK_N, world_size,
                                                                             dtype, device)
    peer_ready_ptrs = create_peer_pointer_table(host, ready_ptr, world_size, device)
    if args.use_multimem:
        peer_workspace_ptrs = ready
        output_ptr, multimem_ready_ptr, ar_out, multimem_ready = create_multimem_output(
            host,
            M,
            N,
            world_size,
            num_comm_sms,
            dtype,
            device,
        )
        mc_out_ptr_val = host.gemm_ar_mc_ptr(output_ptr)
        assert mc_out_ptr_val, "output multicast pointer is unavailable"
        mc_ar_out = tensor_from_pointer(ctypes.c_void_p(mc_out_ptr_val), (M, N), dtype, device)
        peer_multimem_ready_ptrs = create_peer_pointer_table(host, multimem_ready_ptr, world_size, device)
    else:
        peer_workspace_ptrs = create_peer_pointer_table(host, workspace_ptr, world_size, device)
        ar_out = torch.empty_like(workspace)
        mc_ar_out = ar_out
        multimem_ready = ready
        peer_multimem_ready_ptrs = ready

    compute_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)

    if args.use_multimem:
        mc_ptr_val = host.gemm_ar_mc_ptr(workspace_ptr)
        assert mc_ptr_val, "multicast pointer is unavailable"
        mc_workspace = tensor_from_pointer(ctypes.c_void_p(mc_ptr_val), (M, N), dtype, device)
    else:
        mc_workspace = workspace

    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert num_comm_sms < device_sms, "NUM_COMM_SMS must be smaller than the device SM count"
    num_gemm_sms = min(num_tiles, device_sms - num_comm_sms)

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    def run_once(epoch):
        current_stream = torch.cuda.current_stream(device)

        compute_stream.wait_stream(current_stream)
        comm_stream.wait_stream(current_stream)
        with torch.cuda.stream(compute_stream):
            producer_args = (
                a,
                b,
                workspace,
                ready,
                epoch,
                M,
                N,
                K_per_rank,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                workspace.stride(0),
                workspace.stride(1),
            )
            producer_meta = {
                "RANK": rank,
                "GROUP_SIZE_M": 8,
                "NUM_GEMM_SMS": num_gemm_sms,
                "BLOCK_SIZE_M": BLOCK_M,
                "BLOCK_SIZE_N": BLOCK_N,
                "BLOCK_SIZE_K": BLOCK_K,
            }
            gemm_ar_producer[(num_gemm_sms, )](
                *producer_args,
                **producer_meta,
                EPILOGUE_SUBTILE=args.epilogue_subtile,
                num_warps=8,
                num_stages=4 if args.epilogue_subtile else 3,
            )
        with torch.cuda.stream(comm_stream):
            consumer_all_reduce_kernel[(num_comm_sms, )](
                mc_workspace,
                mc_ar_out,
                peer_workspace_ptrs,
                peer_ready_ptrs,
                multimem_ready,
                peer_multimem_ready_ptrs,
                ar_out,
                epoch,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
                NUM_COMM_SMS=num_comm_sms,
                RANK=rank,
                WORLD_SIZE=world_size,
                USE_MULTIMEM=args.use_multimem,
                num_warps=32,
            )
        current_stream.wait_stream(compute_stream)
        current_stream.wait_stream(comm_stream)

    def finish_iteration():
        # Epoch avoids resetting ready, but every rank must finish reading workspace before it is reused.
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)

    epoch = 1
    if args.case == "check":
        run_once(epoch)
        finish_iteration()

        expected = torch.matmul(a, b)
        torch.distributed.all_reduce(expected, group=group)
        torch.testing.assert_close(ar_out, expected, atol=1e-3, rtol=1e-3)
        if rank == 0:
            print("[check] Pass!", flush=True)
    else:
        torch.distributed.barrier(group=group)
        torch.cuda.synchronize()
        for _ in range(args.warmup):
            run_once(epoch)
            if not args.use_multimem:
                finish_iteration()
            epoch += 1
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run_once(epoch)
            if not args.use_multimem:
                finish_iteration()
            epoch += 1
        stop.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(stop) / args.iters

        max_elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
        sum_elapsed = max_elapsed.clone()
        torch.distributed.all_reduce(max_elapsed, op=torch.distributed.ReduceOp.MAX, group=group)
        torch.distributed.all_reduce(sum_elapsed, op=torch.distributed.ReduceOp.SUM, group=group)
        if rank == 0:
            avg_elapsed = sum_elapsed.item() / world_size
            print(f"GEMM all-reduce latency: max_ms={max_elapsed.item():.4f}, "
                  f"avg_ms={avg_elapsed:.4f}")

    if args.use_multimem:
        result = host.gemm_ar_multimem_output_destroy(output_ptr, multimem_ready_ptr)
        assert result == 0, f"multimem output destroy failed: {result}"
    result = host.gemm_ar_workspace_destroy(workspace_ptr, ready_ptr)
    assert result == 0, f"workspace destroy failed: {result}"
    result = common.nvshmem_finalize_from_torch_distributed()
    assert result == 0, f"NVSHMEM finalize failed: {result}"
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
