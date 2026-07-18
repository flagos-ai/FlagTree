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

from common.utils import (
    print_perf_mean,
    load_library,
    install_cumodule_hook,
    init_torch_distributed,
    init_nvshmem_by_torch_pg,
    tensor_from_pointer,
    prepare_clang_bitcode,
    _set_signal_cuda_ptr,
    cudart,
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


@_device_dialect("ag_publish_local_rank")
def publish_rank(*args, **kwargs):
    ...


@_device_dialect("ag_mark_local_ready")
def mark_local_ready(*args, **kwargs):
    ...


@_device_dialect("ag_wait_ready")
def wait_ready(*args, **kwargs):
    ...


@triton.jit
def set_local_ready(ready, rank):
    tle_raw.call(mark_local_ready, [ready, rank])


@triton.jit
def allgather_producer(
    workspace,
    ready,
    elements_per_rank,
    rank,
    world_size,
):
    tle_raw.call(
        publish_rank,
        [
            workspace,
            ready,
            elements_per_rank,
            rank,
            world_size,
        ],
    )


def cp_engine_producer_all_gather_put(host, local_tensor, ag_buffer, signal_buffer, M_per_rank, N, signal_target, rank,
                                      local_world_size, world_size, intranode_ag_stream):
    local_rank = rank % local_world_size
    nbytes = M_per_rank * N * local_tensor.element_size()
    for i in range(1, local_world_size):
        segment = rank * M_per_rank * N
        local_dst_rank = (local_rank + local_world_size - i) % local_world_size
        peer_workspace = host.ag_gemm_peer_workspace_ptr(
            ctypes.c_void_p(ag_buffer.data_ptr()),
            local_dst_rank,
        )
        peer_ready = host.ag_gemm_peer_ready_ptr(
            ctypes.c_void_p(signal_buffer.data_ptr()),
            local_dst_rank,
        )
        assert peer_workspace, f"failed to get peer workspace pointer for PE {local_dst_rank}"
        assert peer_ready, f"failed to get peer ready pointer for PE {local_dst_rank}"

        src_ptr = ag_buffer.data_ptr() + segment * local_tensor.element_size()
        dst_ptr = peer_workspace + segment * local_tensor.element_size()
        (err, ) = cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            intranode_ag_stream.cuda_stream,
        )
        _set_signal_cuda_ptr(
            peer_ready + rank * ctypes.sizeof(ctypes.c_uint64),
            signal_target,
            intranode_ag_stream,
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
    NUM_RANKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = RANK // LOCAL_WORLD_SIZE
    nnodes = NUM_RANKS // LOCAL_WORLD_SIZE

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
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

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # swizzle m
            if nnodes == 1:
                alpha = 0
                beta = 0
                pid_m = (pid_m + ((((RANK ^ alpha) + beta) % NUM_RANKS) * pid_ms_per_rank)) % num_pid_m
            else:
                m_rank = pid_m // pid_ms_per_rank
                pid_m_intra_rank = pid_m - m_rank * pid_ms_per_rank
                m_node_id = m_rank // LOCAL_WORLD_SIZE
                m_local_rank = m_rank % LOCAL_WORLD_SIZE
                swizzle_m_node_id = (m_node_id + node_id) % nnodes
                swizzle_m_local_rank = (m_local_rank + RANK) % LOCAL_WORLD_SIZE
                swizzle_m_rank = swizzle_m_node_id * LOCAL_WORLD_SIZE + swizzle_m_local_rank

                pid_m = swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            source_rank = offs_am // M_per_rank
            tle_raw.call(wait_ready, [ready, source_rank])

        offs_k = ki * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

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


def ag_gemm_op(
    a,
    b,
    c,
    rank,
    num_ranks,
    workspace,
    ready,
    comm_stream,
    compute_stream,
    host,
    extern_libs,
    local_world_size,
    block_m=128,
    block_n=256,
    block_k=64,
    group_m=8,
    num_warps=8,
    num_stages=3,
    profile=False,
    profile_events=None,
):
    assert a.shape[1] == b.shape[1], "incompatible GEMM dimensions"
    assert a.dtype == b.dtype == c.dtype, "incompatible GEMM dtypes"

    m, k = workspace.shape
    m_per_rank = m // num_ranks
    n_per_rank = b.shape[0]
    total_tiles = triton.cdiv(m, block_m) * triton.cdiv(n_per_rank, block_n)

    num_ag_sms = 0  # The copy engine does not occupy SM resources.
    num_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count - num_ag_sms

    grid = (min(total_tiles, num_gemm_sms), )

    if profile:
        assert profile_events is not None, "profile_events must be pre-created when profile=True"
    else:
        profile_events = {}

    current_stream = torch.cuda.current_stream(b.device)
    comm_stream.wait_stream(current_stream)
    if profile:
        profile_events["comm"][0].record(comm_stream)
    cp_engine_producer_all_gather_put(host=host, local_tensor=a, ag_buffer=workspace, signal_buffer=ready,
                                      M_per_rank=m_per_rank, N=k, signal_target=1, rank=rank,
                                      local_world_size=local_world_size, world_size=num_ranks,
                                      intranode_ag_stream=comm_stream)
    if profile:
        profile_events["comm"][1].record(comm_stream)

    compiled = None

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    with torch.cuda.stream(compute_stream):
        compute_stream.wait_stream(current_stream)

        if profile:
            profile_events["compute"][0].record(compute_stream)
        compiled = ag_gemm_consumer[grid](
            workspace,
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
            EPILOGUE_SUBTILE=False,
            num_warps=num_warps,
            num_stages=num_stages,
            extern_libs=extern_libs,
        )

        if profile:
            profile_events["compute"][1].record(compute_stream)

    current_stream.wait_stream(comm_stream)
    current_stream.wait_stream(compute_stream)

    if profile:
        return compiled, profile_events

    return compiled


def create_profile_events():
    return {
        "comm": (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        ),
        "compute": (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        ),
    }


def triton_prepare(
    a_local,
    workspace,
    ready,
    rank,
    extern_libs,
):
    ready.zero_()
    m_per_rank = a_local.shape[0]
    workspace[rank * m_per_rank:(rank + 1) * m_per_rank].copy_(a_local)
    set_local_ready[(1, )](
        ready,
        rank,
        num_warps=1,
        extern_libs=extern_libs,
    )


def torch_ag_gemm(group, a_local, b, gathered):
    torch.distributed.all_gather_into_tensor(gathered, a_local, group=group)
    return torch.matmul(gathered, b.T)


def configure_host_library(host):
    host.ag_gemm_workspace_create.argtypes = [
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

    # The return value is a 64-bit address.
    # Without declaring it as c_void_p, it will be truncated to 32 bits.
    host.ag_gemm_peer_workspace_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.ag_gemm_peer_workspace_ptr.restype = ctypes.c_void_p
    host.ag_gemm_peer_ready_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.ag_gemm_peer_ready_ptr.restype = ctypes.c_void_p


def create_workspace(host, world_size, m_per_rank, k, device):
    workspace_ptr = ctypes.c_void_p()
    ready_ptr = ctypes.c_void_p()
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    local_pe = ctypes.c_int()
    local_npes = ctypes.c_int()
    result = host.ag_gemm_workspace_create(
        m_per_rank * k,
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
        (world_size, ),
        torch.uint64,
        device,
    )
    return workspace_ptr, ready_ptr, workspace, ready, mype, local_pe, local_npes


def perf(fn, group, warmup, iters, prepare_fn=None):
    assert warmup >= 0 and iters > 0

    def unpack_result(result):
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[1]
        return {}

    def prepare():
        torch.distributed.barrier(group=group)
        if prepare_fn is not None:
            prepare_fn()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)

    for _ in range(warmup):
        prepare()
        fn()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    comm_latency = []
    compute_latency = []
    profile_event_list = []

    start.record()
    for _ in range(iters):
        prepare()
        profile_events = unpack_result(fn())
        profile_event_list.append(profile_events)
    end.record()
    torch.cuda.synchronize()

    for profile_events in profile_event_list:
        if "comm" in profile_events:
            comm_latency.append(profile_events["comm"][0].elapsed_time(profile_events["comm"][1]))

        if "compute" in profile_events:
            compute_latency.append(profile_events["compute"][0].elapsed_time(profile_events["compute"][1]))

    profile = {"total": start.elapsed_time(end) / iters}
    if comm_latency:
        profile["comm"] = sum(comm_latency) / len(comm_latency)
    if compute_latency:
        profile["compute"] = (sum(compute_latency) / len(compute_latency))
    return profile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunked NVSHMEM all-gather GEMM with Triton-distributed-style benchmarking")
    parser.add_argument("--m-per-rank", type=int, default=1024)
    parser.add_argument("--n-per-rank", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable separate comm/compute timing for dist-triton ag-gemm",
    )

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
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    assert world_size >= 2, "AG-GEMM requires at least two GPUs"
    assert world_size == local_world_size, ("This example is designed for single-node testing: "
                                            "WORLD_SIZE must equal LOCAL_WORLD_SIZE")

    if torch.cuda.get_device_capability()[0] < 9:
        print("Skip the test because the device is not sm90 or higher")
        import sys
        sys.exit()

    group = init_torch_distributed()
    common_path = Path(__file__).parents[1] / "common" / "common-host.so"
    host_path = Path(__file__).with_name("ag-gemm-host.so")
    common = load_library(common_path)
    host = load_library(host_path)
    configure_host_library(host)
    init_nvshmem_by_torch_pg(common, group)
    install_cumodule_hook(common)

    bitcode_path = Path(__file__).with_name("ag-gemm-device.bc")
    source_path = Path(__file__).with_name("ag-gemm-device.cu")
    extern_libs = prepare_clang_bitcode(
        common, local_rank, bitcode_path, source_path, publish_rank,
        public_api_names=["ag_mark_local_ready", "ag_publish_local_rank", "ag_wait_ready"])

    dtype = torch.float16
    a_local = torch.randn((args.m_per_rank, args.k), dtype=dtype, device=device)
    b = torch.randn((args.n_per_rank, args.k), dtype=dtype, device=device)
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
        device,
    )
    assert mype.value == rank
    assert local_pe.value == local_rank

    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=-1)

    def prepare_triton_mode():
        triton_prepare(
            a_local,
            workspace,
            ready,
            rank,
            extern_libs,
        )

    def triton_func():
        triton_profile_events = create_profile_events() if args.profile else None
        return ag_gemm_op(a_local, b, c, rank, world_size, workspace, ready, comm_stream, compute_stream, host,
                          extern_libs, local_world_size, profile=args.profile, profile_events=triton_profile_events)

    def torch_func():
        return torch_ag_gemm(group, a_local, b, gathered)

    def run_correctness():
        if rank == 0:
            print("[check] start correctness validation", flush=True)

        prepare_triton_mode()
        triton_func()
        result = host.ag_gemm_workspace_destroy(workspace_ptr, ready_ptr)
        assert result == 0
        result = common.nvshmem_finalize_from_torch_distributed()
        assert result == 0
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)

        golden = torch_func()
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)

        torch.distributed.destroy_process_group()
        torch.testing.assert_close(c, golden, atol=1e-3, rtol=1e-3)
        if rank == 0:
            print("[check] Pass!", flush=True)
            print(f"configuration: GPUs={world_size}, "
                  f"A_local=({args.m_per_rank}, {args.k}), "
                  f"B_local=({args.n_per_rank}, {args.k}), ")

    # For performance testing
    # it's best to test them separately
    # For example, when testing Triton, comment out the Torch part.
    def run_benchmark():
        if rank == 0:
            print(
                f"[bench] start benchmark: warmup={args.warmup}, "
                f"iters={args.iters}",
                flush=True,
            )

        triton_profile = perf(
            triton_func,
            group,
            args.warmup,
            args.iters,
            prepare_fn=prepare_triton_mode,
        )
        result = host.ag_gemm_workspace_destroy(workspace_ptr, ready_ptr)
        assert result == 0
        result = common.nvshmem_finalize_from_torch_distributed()
        assert result == 0
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)

        torch_profile = perf(
            torch_func,
            group,
            args.warmup,
            args.iters,
        )
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)

        print_perf_mean(
            "triton ag-gemm",
            triton_profile["total"],
            group,
            rank,
            world_size,
        )
        print_perf_mean(
            "torch ag-gemm",
            torch_profile["total"],
            group,
            rank,
            world_size,
        )
        print_perf_mean(
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
                  f"B_local=({args.n_per_rank}, {args.k}).")
            if "comm" in triton_profile and "compute" in triton_profile:
                print(
                    f"dist-triton detail: \n"
                    f"comm={triton_profile['comm']:.3f} ms,\n"
                    f"compute={triton_profile['compute']:.3f} ms",
                    flush=True,
                )

        torch.distributed.destroy_process_group()

    if args.mode == "check":
        run_correctness()
    if args.mode == "perf":
        run_benchmark()


if __name__ == "__main__":
    main()
