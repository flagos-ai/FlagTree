# source /opt/dtk/env.sh
# export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
# unset DUSHMEM_BOOTSTRAP
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29502 gemm-ar.py
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29502 gemm-ar.py perf
"""K-split GEMM overlapped with a tile allreduce.

tl.dot on gfx936 tops out near 135 TFLOPS for this shape; hipBLAS is about 300.
The producer is therefore hipBLAS, cut into row chunks. After each chunk a raw
release marks those tiles. A Triton kernel on a second stream waits on every
rank's flag and sums the tile, so finished rows reduce while later rows are
still in the GEMM.

The consumer kernel loads every peer in one launch, so those dushmem_ptr
results are passed in as a device array. That array is not a SHMEM API.
Intra-node only, no proxy fallback. _peer_table asserts both of these:

- gemm_ar_intra_node: dushmem_team_n_pes(DUSHMEMX_TEAM_NODE) == dushmem_n_pes()
- dushmem_ptr(peer) is non-null for every PE
"""

import ctypes
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.hcu.utils import (
    compile_host_library,
    init_dushmem_by_torch_pg,
    init_torch_distributed,
    load_common_host,
    tensor_from_pointer,
)

HERE = Path(__file__).parent
BLOCK_M = 128
BLOCK_N = 128
M = 1024
N = 1024
K_PER_RANK = 512
# M is a multiple of the tile. Chunk size and consumer CUs follow _schedule.
PERF_SHAPES = (
    (4096, 4096, 2048),
    (8192, 8192, 2048),
    (8192, 8192, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 4096),
    (16384, 16384, 8192),
)
WARMUP = 5
ITERS = 10


def _schedule(m, k_per_rank):
    """Row chunk and consumer CUs. Measured on gfx936, fp16, 2 ranks.

    64 consumer CUs slow every shape. M=8192 overlaps better in 1024-row
    chunks; M=4096 and M=16384 stay at 2048. A long K is GEMM-bound, so the
    consumer drops to 16 CUs and leaves the rest to hipBLAS.
    """
    if m < 8192:
        return 2048, 32
    if m < 16384:
        return 1024, 32 if k_per_rank <= 2048 else 16
    return 2048, 16 if k_per_rank >= 8192 else 32


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=HERE / "gemm-ar-device.hip", extern_func_name="ar_wait_tile_ready", deferred=True)
def wait_tile_ready(*args, **kwargs):
    ...


def _load_host():
    host = compile_host_library(HERE / "gemm-ar-host.hip", Path("/tmp/dushmem-gemm-ar-host.so"))
    host.gemm_ar_workspace_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    host.gemm_ar_workspace_create.restype = ctypes.c_int
    host.gemm_ar_peer_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.gemm_ar_peer_ptr.restype = ctypes.c_void_p
    host.gemm_ar_intra_node.argtypes = []
    host.gemm_ar_intra_node.restype = ctypes.c_int
    host.gemm_ar_workspace_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    host.gemm_ar_workspace_destroy.restype = ctypes.c_int
    host.gemm_ar_mark_tiles.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    host.gemm_ar_mark_tiles.restype = ctypes.c_int
    return host


def _peer_table(host, base_ptr, world_size, device):
    """Local mappings of every PE's symmetric heap. Intra-node P2P only."""
    assert host.gemm_ar_intra_node() == 1, ("tile allreduce is intra-node only: "
                                            "DUSHMEMX_TEAM_NODE must include every PE")
    addresses = []
    for peer in range(world_size):
        peer_ptr = host.gemm_ar_peer_ptr(base_ptr, peer)
        assert peer_ptr, f"dushmem_ptr(peer={peer}) is null; that PE is not P2P-mapped"
        addresses.append(peer_ptr)
    return torch.tensor(addresses, dtype=torch.uint64, device=device)


@triton.jit(do_not_specialize=["epoch"])
def consumer_all_reduce(
    peer_workspace_ptrs,
    peer_ready_ptrs,
    ar_out,
    epoch,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_tiles_i32 = num_tiles.to(tl.int32)
    world_i32 = tl.full((), WORLD_SIZE, tl.int32)
    epoch_u64 = tl.cast(epoch, tl.uint64)
    out_dtype = ar_out.dtype.element_ty

    for tile_id in tl.range(pid, num_tiles, NUM_COMM_SMS, flatten=False):
        tle_raw.call(
            wait_tile_ready,
            [peer_ready_ptrs, world_i32, tile_id.to(tl.int32), num_tiles_i32, epoch_u64],
            output_indices=[],
        )
        pid_m = tile_id // num_pid_n
        pid_n = tile_id - pid_m * num_pid_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for peer in tl.static_range(WORLD_SIZE):
            peer_ptr = tl.load(peer_workspace_ptrs + peer).to(tl.pointer_type(out_dtype))
            accumulator += tl.load(peer_ptr + offs_m[:, None] * N + offs_n[None, :]).to(tl.float32)
        tl.store(ar_out + offs_m[:, None] * N + offs_n[None, :], accumulator.to(out_dtype))


def _launch_overlap(host, epoch, a, b, workspace, ready, peer_workspace, peer_ready, ar_out, compute_stream,
                    comm_stream, num_comm_sms, world_size, rank, chunk_m):
    current = torch.cuda.current_stream(a.device)
    compute_stream.wait_stream(current)
    comm_stream.wait_stream(current)
    m, n = a.shape[0], b.shape[1]
    num_pid_n = n // BLOCK_N
    num_tiles = (m // BLOCK_M) * num_pid_n
    stream_ptr = ctypes.c_void_p(compute_stream.cuda_stream)
    with torch.cuda.stream(compute_stream):
        for m0 in range(0, m, chunk_m):
            m1 = min(m0 + chunk_m, m)
            torch.matmul(a[m0:m1], b, out=workspace[m0:m1])
            tile_begin = (m0 // BLOCK_M) * num_pid_n
            tile_count = ((m1 - m0) // BLOCK_M) * num_pid_n
            status = host.gemm_ar_mark_tiles(
                ctypes.c_void_p(ready.data_ptr()),
                rank,
                num_tiles,
                tile_begin,
                tile_count,
                epoch,
                stream_ptr,
            )
            if status != 0:
                raise RuntimeError(f"mark tiles failed: {status}")
    with torch.cuda.stream(comm_stream):
        consumer_all_reduce[(num_comm_sms, )](
            peer_workspace,
            peer_ready,
            ar_out,
            epoch,
            m,
            n,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            NUM_COMM_SMS=num_comm_sms,
            WORLD_SIZE=world_size,
            num_warps=4,
        )
    current.wait_stream(compute_stream)
    current.wait_stream(comm_stream)


def _torch_serial(a, b, partial, nccl_group):
    """GEMM, then allreduce, both on the current stream."""
    torch.matmul(a, b, out=partial)
    torch.distributed.all_reduce(partial, group=nccl_group)


def _elapsed_ms(fn, iters):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def _max_ms(ms, device, nccl_group):
    local = torch.tensor([ms], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX, group=nccl_group)
    return local.item()


def _overlap_ms(host, epoch_box, a, b, workspace, ready, peer_workspace, peer_ready, ar_out, compute_stream,
                comm_stream, num_comm, world_size, rank, chunk_m, device, nccl_group):

    def overlap_once():
        _launch_overlap(
            host,
            epoch_box[0],
            a,
            b,
            workspace,
            ready,
            peer_workspace,
            peer_ready,
            ar_out,
            compute_stream,
            comm_stream,
            num_comm,
            world_size,
            rank,
            chunk_m,
        )
        epoch_box[0] += 1

    return _max_ms(_elapsed_ms(overlap_once, ITERS), device, nccl_group)


def _run_shape(host, nccl_group, device, rank, world_size, m, n, k_per_rank, perf):
    if m % BLOCK_M or n % BLOCK_N:
        raise SystemExit("M and N must be divisible by the tile")
    dtype = torch.float16
    num_tiles = triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N)
    workspace_ptr = ctypes.c_void_p()
    ready_ptr = ctypes.c_void_p()
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    status = host.gemm_ar_workspace_create(
        m * n,
        dtype.itemsize,
        ctypes.byref(workspace_ptr),
        ctypes.byref(ready_ptr),
        num_tiles,
        ctypes.byref(mype),
        ctypes.byref(npes),
    )
    if status != 0:
        raise SystemExit(f"workspace create failed: {status}")
    if mype.value != rank or npes.value != world_size:
        raise SystemExit(f"rank mismatch torch={rank} dushmem={mype.value}")

    workspace = tensor_from_pointer(workspace_ptr, (m, n), dtype, device)
    ready = tensor_from_pointer(ready_ptr, (world_size, num_tiles), torch.uint64, device)
    peer_workspace = _peer_table(host, workspace_ptr, world_size, device)
    peer_ready = _peer_table(host, ready_ptr, world_size, device)

    torch.manual_seed(1000 + rank)
    a = torch.randn((m, k_per_rank), dtype=dtype, device=device) * 0.1
    b = torch.randn((k_per_rank, n), dtype=dtype, device=device) * 0.1
    ar_out = torch.empty((m, n), dtype=dtype, device=device)
    torch_out = torch.empty((m, n), dtype=dtype, device=device)

    chunk_m, num_comm = _schedule(m, k_per_rank)
    num_comm_sms = min(num_comm, num_tiles)
    compute_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    epoch = 1
    _launch_overlap(
        host,
        epoch,
        a,
        b,
        workspace,
        ready,
        peer_workspace,
        peer_ready,
        ar_out,
        compute_stream,
        comm_stream,
        num_comm_sms,
        world_size,
        rank,
        chunk_m,
    )
    torch.cuda.synchronize()

    _torch_serial(a, b, torch_out, nccl_group)
    torch.cuda.synchronize()
    err = (ar_out - torch_out).abs().max().item()
    ok = torch.allclose(ar_out, torch_out, rtol=2e-2, atol=2e-2)
    print(f"gemm-ar pe={rank} M={m} N={n} K={k_per_rank} allclose={bool(ok)} max_abs_err={err}", flush=True)
    if not ok:
        raise SystemExit(1)

    if perf:
        epoch_box = [epoch + 1]
        # Bind the tensors as defaults. A later `del` makes a bare closure look undefined to ruff.
        torch_ms = _max_ms(
            _elapsed_ms(
                lambda a=a, b=b, torch_out=torch_out: _torch_serial(a, b, torch_out, nccl_group),
                ITERS,
            ),
            device,
            nccl_group,
        )
        k_total = k_per_rank * world_size
        flops = 2.0 * m * n * k_total
        overlap_ms = _overlap_ms(
            host,
            epoch_box,
            a,
            b,
            workspace,
            ready,
            peer_workspace,
            peer_ready,
            ar_out,
            compute_stream,
            comm_stream,
            num_comm_sms,
            world_size,
            rank,
            chunk_m,
            device,
            nccl_group,
        )
        if rank == 0:
            print(
                f"shape M={m} N={n} K_per_rank={k_per_rank} chunk={chunk_m} comm={num_comm_sms} "
                f"overlap_ms={overlap_ms:.3f} overlap_tflops={flops / (overlap_ms * 1e-3) / 1e12:.2f} "
                f"torch_ms={torch_ms:.3f} torch_tflops={flops / (torch_ms * 1e-3) / 1e12:.2f} "
                f"speedup={torch_ms / overlap_ms:.3f}",
                flush=True,
            )

    del a, b, workspace, ready, ar_out, torch_out, peer_workspace, peer_ready
    destroy = host.gemm_ar_workspace_destroy(workspace_ptr, ready_ptr)
    if destroy != 0:
        raise SystemExit(f"workspace destroy failed: {destroy}")


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    perf = mode == "perf"
    if world_size < 2:
        raise SystemExit("gemm-ar needs at least 2 ranks")

    group = init_torch_distributed()
    init_dushmem_by_torch_pg(load_common_host(), group)
    nccl_group = torch.distributed.new_group(backend="nccl")
    host = _load_host()
    device = torch.device("cuda", local_rank)
    shapes = PERF_SHAPES if perf else ((M, N, K_PER_RANK), )
    for m, n, k_per_rank in shapes:
        _run_shape(host, nccl_group, device, rank, world_size, m, n, k_per_rank, perf)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
