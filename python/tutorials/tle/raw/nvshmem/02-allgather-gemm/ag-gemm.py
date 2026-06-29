import argparse
import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

from common.utils import (
    load_library,
    install_cumodule_hook,
    init_torch_distributed,
    init_nvshmem_by_torch_pg,
    tensor_from_pointer,
    prepare_clang_bitcode,
)


def _device_dialect(function_name):
    return dialect(
        name="cuda",
        compiler="clang",
        target="bc",
        file=Path(__file__).parent / "ag-gemm-device.cu",
        extern_file=Path(__file__).parent / "ag-gemm-device-extern-call.py",
        extern_func_name=function_name,
    )


@_device_dialect("ag_publish_local_chunk")
def publish_chunk(*args, **kwargs):
    ...


@_device_dialect("ag_mark_local_ready")
def mark_local_ready(*args, **kwargs):
    ...


@_device_dialect("ag_wait_ready")
def wait_ready(*args, **kwargs):
    ...


@triton.jit
def set_local_ready(ready, rank, num_chunks):
    tle_raw.call(mark_local_ready, [ready, rank, num_chunks])


@triton.jit
def allgather_producer(
    workspace,
    ready,
    elements_per_rank,
    elements_per_chunk,
    num_chunks,
    rank,
    world_size,
):
    tle_raw.call(
        publish_chunk,
        [
            workspace,
            ready,
            elements_per_rank,
            elements_per_chunk,
            num_chunks,
            rank,
            world_size,
        ],
    )


@triton.jit
def ag_gemm_consumer(
    a_ptr,
    b_ptr,
    c_ptr,
    ready,
    M,
    N,
    K,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CHUNK_M: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    READY_VALUE: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    m_per_rank = M // WORLD_SIZE
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m_offset = tl.cdiv(m_per_rank * RANK, BLOCK_M)
    pid_m = (pid_m + pid_m_offset) % num_pid_m

    tile_m = pid_m * BLOCK_M
    source_rank = tile_m // m_per_rank
    source_rank_row = tile_m - source_rank * m_per_rank
    chunk_id = source_rank_row // CHUNK_M
    signal_index = source_rank * NUM_CHUNKS + chunk_id
    tle_raw.call(wait_ready, [ready, signal_index])

    offs_m = tile_m + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k * BLOCK_K + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * K + k_offsets[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + k_offsets[:, None]
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


def ag_gemm_op(
    a,
    b,
    c,
    rank,
    world_size,
    workspace,
    ready,
    comm_stream,
    compute_stream,
    extern_libs,
    chunk_m,
    local_world_size,
    block_m=128,
    block_n=256,
    block_k=64,
    group_m=8,
    stages=3,
    ready_value=1,
):
    assert a.shape[1] == b.shape[1], "incompatible GEMM dimensions"
    assert a.dtype == b.dtype == c.dtype, "incompatible GEMM dtypes"

    m, k = workspace.shape
    m_per_rank = m // world_size
    n_per_rank = b.shape[0]
    num_chunks = m_per_rank // chunk_m
    elements_per_chunk = chunk_m * k
    total_tiles = triton.cdiv(m, block_m) * triton.cdiv(n_per_rank, block_n)
    grid = (total_tiles, )

    local_ready = torch.cuda.Event()
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_done = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_done = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(b.device)

    with torch.cuda.stream(compute_stream):
        compute_stream.wait_stream(current_stream)
        local_ready.record(compute_stream)

    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(local_ready)
        comm_start.record(comm_stream)
        allgather_producer[((world_size - 1) * num_chunks, )](
            workspace,
            ready,
            m_per_rank * k,
            elements_per_chunk,
            num_chunks,
            rank,
            world_size,
            num_warps=32,
            extern_libs=extern_libs,
        )
        comm_done.record(comm_stream)

    with torch.cuda.stream(compute_stream):
        compute_start.record(compute_stream)
        ag_gemm_consumer[grid](
            workspace,
            b,
            c,
            ready,
            m,
            n_per_rank,
            k,
            rank,
            world_size,
            block_m,
            block_n,
            block_k,
            group_m,
            chunk_m,
            num_chunks,
            ready_value,
            local_world_size,
            num_warps=8,
            num_stages=stages,
            extern_libs=extern_libs,
        )
        compute_done.record(compute_stream)

    current_stream.wait_event(comm_done)
    current_stream.wait_event(compute_done)
    return c, {
        "comm": (comm_start, comm_done),
        "compute": (compute_start, compute_done),
    }


def triton_prepare(
    a_local,
    workspace,
    ready,
    rank,
    extern_libs,
    chunk_m,
):
    m_per_rank = a_local.shape[0]
    num_chunks = m_per_rank // chunk_m

    ready.zero_()
    workspace[rank * m_per_rank:(rank + 1) * m_per_rank].copy_(a_local)
    set_local_ready[(num_chunks, )](
        ready,
        rank,
        num_chunks,
        num_warps=1,
        extern_libs=extern_libs,
    )


def torch_ag_gemm(group, a_local, b, gathered):
    torch.distributed.all_gather_into_tensor(gathered, a_local, group=group)
    return torch.matmul(gathered, b.T)


def create_workspace(host, world_size, m_per_rank, k, num_chunks, device):
    workspace_ptr = ctypes.c_void_p()
    ready_ptr = ctypes.c_void_p()
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    local_pe = ctypes.c_int()
    local_npes = ctypes.c_int()
    result = host.ag_gemm_workspace_create(
        m_per_rank * k,
        num_chunks,
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
        torch.float16,
        device,
    )
    ready = tensor_from_pointer(
        ready_ptr,
        (world_size, num_chunks),
        torch.uint64,
        device,
    )
    return workspace_ptr, ready_ptr, workspace, ready, mype, local_pe, local_npes


def perf(fn, group, warmup, iters, prepare_fn=None):
    assert warmup >= 0 and iters > 0

    def unpack_result(result):
        if isinstance(result, tuple) and len(result) == 2:
            maybe_events = result[1]
            if isinstance(maybe_events, dict):
                return result[0], maybe_events
        return result, {}

    def prepare():
        torch.distributed.barrier(group=group)
        if prepare_fn is not None:
            prepare_fn()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)

    output = None
    for _ in range(warmup):
        prepare()
        output, _ = unpack_result(fn())
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies_ms = []
    comm_latencies_ms = []
    compute_latencies_ms = []
    for _ in range(iters):
        prepare()
        start.record()
        output, profile_events = unpack_result(fn())
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))
        if "comm" in profile_events:
            comm_start, comm_end = profile_events["comm"]
            comm_latencies_ms.append(comm_start.elapsed_time(comm_end))
        if "compute" in profile_events:
            compute_start, compute_end = profile_events["compute"]
            compute_latencies_ms.append(compute_start.elapsed_time(compute_end))

    profile = {"total": sum(latencies_ms) / len(latencies_ms)}
    if comm_latencies_ms:
        profile["comm"] = sum(comm_latencies_ms) / len(comm_latencies_ms)
    if compute_latencies_ms:
        profile["compute"] = (sum(compute_latencies_ms) / len(compute_latencies_ms))
    return output, profile


def print_perf(
    name: str,
    value: float,
    group,
    rank: int,
    world_size: int,
    unit: str = "ms",
):
    for index in range(world_size):
        torch.distributed.barrier(group=group)
        if rank == index:
            print(f"{name} #{rank}: {value:.4f} {unit}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunked NVSHMEM all-gather GEMM with Triton-distributed-style benchmarking")
    parser.add_argument("--m-per-rank", type=int, default=1024)
    parser.add_argument(
        "--chunk-m",
        type=int,
        default=1024,
        help="rows per independently transferred and signaled A chunk",
    )
    parser.add_argument(
        "--n-per-rank",
        type=int,
        default=4096,
        help="local output width (the local B shard has shape N_per_rank x K)",
    )
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)

    parser.add_argument(
        "--mode",
        choices=("check", "perf"),
        default="check",
        help=("check: run correctness only; "
              "perf: run benchmark only"),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    group = init_torch_distributed()
    rank = group.rank()
    world_size = group.size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    assert world_size >= 2, "AG-GEMM overlap requires at least two GPUs"
    assert world_size == local_world_size, ("this example follows Triton-distributed's single-node AG-GEMM path: "
                                            "WORLD_SIZE must equal LOCAL_WORLD_SIZE")

    a_local = torch.randn((args.m_per_rank, args.k), device=device, dtype=torch.float16)
    b = torch.randn((args.n_per_rank, args.k), device=device, dtype=torch.float16)

    common_path = Path(__file__).parents[1] / "common" / "common-host.so"
    host_path = Path(__file__).with_name("ag-gemm-host.so")
    common = load_library(common_path)
    host = load_library(host_path)
    init_nvshmem_by_torch_pg(common, group)
    install_cumodule_hook(common)

    bitcode_path = Path(__file__).with_name("ag-gemm-device.bc")
    extern_libs = prepare_clang_bitcode(
        common, local_rank, bitcode_path, publish_chunk,
        public_api_names=["ag_mark_local_ready", "ag_publish_local_chunk", "ag_wait_ready"])

    num_chunks = args.m_per_rank // args.chunk_m
    (
        workspace_ptr,
        ready_ptr,
        workspace,
        ready,
        mype,
        local_pe,
        local_npes,
    ) = create_workspace(
        host,
        world_size,
        args.m_per_rank,
        args.k,
        num_chunks,
        device,
    )
    assert mype.value == rank
    assert local_pe.value == local_rank

    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    c = torch.empty(
        (world_size * args.m_per_rank, args.n_per_rank),
        dtype=a_local.dtype,
        device=device,
    )
    gathered = torch.empty(
        (world_size * args.m_per_rank, args.k),
        dtype=a_local.dtype,
        device=device,
    )

    try:

        def prepare_triton_mode():
            triton_prepare(
                a_local,
                workspace,
                ready,
                rank,
                extern_libs,
                args.chunk_m,
            )

        def triton_func():
            return ag_gemm_op(
                a_local,
                b,
                c,
                rank,
                world_size,
                workspace,
                ready,
                comm_stream,
                compute_stream,
                extern_libs,
                args.chunk_m,
                local_world_size,
            )

        def torch_func():
            return torch_ag_gemm(group, a_local, b, gathered)

        def run_correctness():
            if rank == 0:
                print("[check] start correctness validation", flush=True)
            prepare_triton_mode()
            output, _ = triton_func()
            torch.cuda.synchronize(device)
            golden = torch_func()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output, golden, atol=2e-2, rtol=2e-2)
            torch.distributed.barrier(group=group)
            if rank == 0:
                print("[check] Pass!", flush=True)

        def run_benchmark():
            if rank == 0:
                print(
                    f"[bench] start benchmark: warmup={args.warmup}, "
                    f"iters={args.iters}",
                    flush=True,
                )
            _, triton_profile = perf(
                triton_func,
                group,
                args.warmup,
                args.iters,
                prepare_fn=prepare_triton_mode,
            )
            torch.cuda.synchronize(device)

            _, torch_profile = perf(
                torch_func,
                group,
                args.warmup,
                args.iters,
            )
            torch.cuda.synchronize(device)
            torch.distributed.barrier(group=group)

            print_perf(
                "dist-triton ag-gemm",
                triton_profile["total"],
                group,
                rank,
                world_size,
            )
            print_perf(
                "torch ag-gemm",
                torch_profile["total"],
                group,
                rank,
                world_size,
            )
            print_perf(
                "speedup",
                torch_profile["total"] / triton_profile["total"],
                group,
                rank,
                world_size,
                unit="x",
            )

            if rank == 0:
                print(f"configuration: GPUs={world_size}, "
                      f"A_local=({args.m_per_rank}, {args.k}), "
                      f"B_local=({args.n_per_rank}, {args.k}), "
                      f"chunk_m={args.chunk_m}, chunks/rank={num_chunks}")
                if "comm" in triton_profile and "compute" in triton_profile:
                    print(
                        f"dist-triton detail: comm={triton_profile['comm']:.4f} ms, "
                        f"compute={triton_profile['compute']:.4f} ms",
                        flush=True,
                    )

        if args.mode == "check":
            run_correctness()
        if args.mode == "perf":
            run_benchmark()

    finally:
        torch.cuda.synchronize(device)
        result = host.ag_gemm_workspace_destroy(workspace_ptr, ready_ptr)
        assert result == 0
        torch.distributed.barrier(group=group)
        result = common.nvshmem_finalize_from_torch_distributed()
        assert result == 0
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
