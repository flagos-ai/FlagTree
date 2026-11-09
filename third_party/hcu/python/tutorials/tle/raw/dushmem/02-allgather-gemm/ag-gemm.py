# source /opt/dtk/env.sh
# export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
# unset DUSHMEM_BOOTSTRAP
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29501 ag-gemm.py
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29501 ag-gemm.py perf
"""Allgather A with DUSHMEM, GEMM with hipBLAS.

tl.dot on gfx936 is far behind hipBLAS, so the multiply is torch.matmul.
Bulk copies use the intra-node P2P copy engine (dushmem_ptr + hipMemcpyAsync),
the same class of path RCCL uses on xGMI. Host dushmemx_putmem_on_stream goes
through the proxy thread and is not that path. Device put's 16-byte copy is
stubbed out (GPUFUSION_UNSUPPORTED_PTX_OPERATION) in the bitcode build.

Intra-node is the NVSHMEM pattern: dushmem_ptr of the peer you are copying,
then hipMemcpyAsync. The pointer is fetched at the copy. There is no peer
table and no put/get/fcollect. Two asserts, no proxy fallback:

- ag_gemm_intra_node: dushmem_team_n_pes(DUSHMEMX_TEAM_NODE) == dushmem_n_pes()
- dushmem_ptr(peer) is non-null (null means no P2P mapping)

Fine-grain ready flags on the large-tile path stay in Triton + deferred raw.

Schedules (2×gfx936, fp16):
- split (M>=8192, 2 ranks): local torch GEMM overlaps the P2P push; deferred
  acquire, then remote GEMM on a torch copy of the arrived rows.
- gather (smaller M, or more ranks): P2P push, dushmemx_signal_wait_until on
  the same stream, one GEMM. A second stream does not pay for itself here.
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
M = 2048
N = 2048
K = 1024
PERF_SHAPES = (
    (4096, 4096, 2048),
    (8192, 8192, 2048),
    (8192, 8192, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 4096),
)
WARMUP = 5
ITERS = 10


def _schedule(m, world_size):
    """Per-shape path. Tuned on 2×gfx936; split needs a large enough local GEMM."""
    if world_size == 2 and m >= 8192:
        return "split"
    return "gather"


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=HERE / "ag-gemm-device.hip", extern_func_name="ag_mark_local_ready", deferred=True)
def mark_local_ready(*args, **kwargs):
    ...


@dialect(name="hcu", file=HERE / "ag-gemm-device.hip", extern_func_name="ag_wait_ready", deferred=True)
def wait_ready(*args, **kwargs):
    ...


def _load_host():
    host = compile_host_library(HERE / "ag-gemm-host.hip", Path("/tmp/dushmem-ag-gemm-host.so"))
    host.ag_gemm_workspace_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    host.ag_gemm_workspace_create.restype = ctypes.c_int
    host.ag_gemm_peer_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.ag_gemm_peer_ptr.restype = ctypes.c_void_p
    host.ag_gemm_intra_node.argtypes = []
    host.ag_gemm_intra_node.restype = ctypes.c_int
    host.ag_gemm_copy_on_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    host.ag_gemm_copy_on_stream.restype = ctypes.c_int
    host.ag_gemm_signal_on_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    host.ag_gemm_signal_on_stream.restype = ctypes.c_int
    host.ag_gemm_signal_wait_on_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    host.ag_gemm_signal_wait_on_stream.restype = ctypes.c_int
    host.ag_gemm_workspace_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    host.ag_gemm_workspace_destroy.restype = ctypes.c_int
    return host


@triton.jit(do_not_specialize=["epoch"])
def set_local_ready(ready, rank, epoch):
    tle_raw.call(mark_local_ready, [ready, rank, tl.cast(epoch, tl.uint64)], output_indices=[])


@triton.jit(do_not_specialize=["epoch"])
def wait_rank_ready(ready, signal_index, epoch):
    tle_raw.call(wait_ready, [ready, signal_index, tl.cast(epoch, tl.uint64)], output_indices=[])


def _publish_local(a_local, workspace, ready, rank, epoch):
    m_per_rank = a_local.shape[0]
    row0 = rank * m_per_rank
    workspace[row0:row0 + m_per_rank].copy_(a_local)
    set_local_ready[(1, )](ready, rank, epoch, num_warps=1)


def _mapped_ptr(host, base, peer):
    """One intra-node mapping. Same role as nvshmem_ptr: null if not P2P."""
    ptr = host.ag_gemm_peer_ptr(base, peer)
    assert ptr, f"dushmem_ptr(peer={peer}) is null; that PE is not P2P-mapped"
    return int(ptr)


def _p2p_push(host, epoch, a_local, workspace_ptr, ready_ptr, comm_stream, world_size, rank, nbytes):
    """Copy-engine push of this rank's rows, then a system-scope release on the peer."""
    stream_comm = ctypes.c_void_p(comm_stream.cuda_stream)
    src = ctypes.c_void_p(a_local.data_ptr())
    with torch.cuda.stream(comm_stream):
        for step in range(1, world_size):
            peer = (rank + step) % world_size
            status = host.ag_gemm_copy_on_stream(
                stream_comm,
                ctypes.c_void_p(_mapped_ptr(host, workspace_ptr, peer) + rank * nbytes),
                src,
                nbytes,
            )
            if status != 0:
                raise RuntimeError(f"p2p copy failed: {status}")
            status = host.ag_gemm_signal_on_stream(
                stream_comm,
                ctypes.c_void_p(_mapped_ptr(host, ready_ptr, peer) + rank * 8),
                epoch,
            )
            if status != 0:
                raise RuntimeError(f"p2p signal failed: {status}")


def _launch_split(host, epoch, a_local, b, workspace_ptr, ready_ptr, workspace, ready, remote_a, c, compute_stream,
                  comm_stream, world_size, rank, m_per_rank, nbytes):
    """Local torch GEMM covers the P2P push; deferred acquire then remote torch GEMM."""
    current = torch.cuda.current_stream(b.device)
    row0 = rank * m_per_rank
    row1 = row0 + m_per_rank

    compute_stream.wait_stream(current)
    comm_stream.wait_stream(current)

    with torch.cuda.stream(compute_stream):
        torch.matmul(a_local, b, out=c[row0:row1])

    _p2p_push(host, epoch, a_local, workspace_ptr, ready_ptr, comm_stream, world_size, rank, nbytes)

    with torch.cuda.stream(compute_stream):
        for step in range(1, world_size):
            src = (rank + step) % world_size
            wait_rank_ready[(1, )](ready, src, epoch, num_warps=1)
            r0 = src * m_per_rank
            r1 = r0 + m_per_rank
            remote_a.copy_(workspace[r0:r1])
            torch.matmul(remote_a, b, out=c[r0:r1])

    current.wait_stream(compute_stream)
    current.wait_stream(comm_stream)


def _launch_gather(host, epoch, a_local, b, workspace_ptr, ready_ptr, workspace, full_a, c, world_size, rank,
                   m_per_rank, nbytes):
    """Small-tile path: P2P push, stream signal-wait, one hipBLAS GEMM.

    signal_wait_until stays on the copy stream. A Triton acquire here is an
    extra launch (~0.03ms) that this tile cannot hide.
    """
    row0 = rank * m_per_rank
    row1 = row0 + m_per_rank
    current = torch.cuda.current_stream(b.device)
    stream = ctypes.c_void_p(current.cuda_stream)
    _p2p_push(host, epoch, a_local, workspace_ptr, ready_ptr, current, world_size, rank, nbytes)
    # DUSHMEM_CMP_GE
    for step in range(1, world_size):
        src = (rank + step) % world_size
        status = host.ag_gemm_signal_wait_on_stream(
            ctypes.c_void_p(int(ready_ptr.value) + src * 8),
            5,
            epoch,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"signal_wait failed: {status}")
    full_a[row0:row1].copy_(a_local)
    for step in range(1, world_size):
        src = (rank + step) % world_size
        r0 = src * m_per_rank
        full_a[r0:r0 + m_per_rank].copy_(workspace[r0:r0 + m_per_rank])
    torch.matmul(full_a, b, out=c)


def _torch_serial(a_local, b, full_a, out, nccl_group):
    torch.distributed.all_gather_into_tensor(full_a, a_local, group=nccl_group)
    torch.matmul(full_a, b, out=out)


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


def _run_shape(host, nccl_group, device, rank, world_size, m, n, k, perf):
    if m % world_size or n % world_size:
        raise SystemExit("M and N must be divisible by the world size")
    m_per_rank = m // world_size
    n_per_rank = n // world_size
    dtype = torch.float16

    workspace_ptr = ctypes.c_void_p()
    ready_ptr = ctypes.c_void_p()
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    status = host.ag_gemm_workspace_create(
        m_per_rank * k,
        dtype.itemsize,
        world_size,
        ctypes.byref(workspace_ptr),
        ctypes.byref(ready_ptr),
        ctypes.byref(mype),
        ctypes.byref(npes),
    )
    if status != 0:
        raise SystemExit(f"workspace create failed: {status}")
    if mype.value != rank or npes.value != world_size:
        raise SystemExit(f"rank mismatch torch={rank} dushmem={mype.value}")

    workspace = tensor_from_pointer(workspace_ptr, (m, k), dtype, device)
    ready = tensor_from_pointer(ready_ptr, (world_size, ), torch.uint64, device)

    torch.manual_seed(2000 + rank)
    a_local = torch.randn((m_per_rank, k), dtype=dtype, device=device) * 0.1
    b = torch.randn((k, n_per_rank), dtype=dtype, device=device) * 0.1
    c = torch.empty((m, n_per_rank), dtype=dtype, device=device)
    remote_a = torch.empty((m_per_rank, k), dtype=dtype, device=device)
    torch_out = torch.empty((m, n_per_rank), dtype=dtype, device=device)
    full_a = torch.empty((m, k), dtype=dtype, device=device)

    compute_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    nbytes = m_per_rank * k * dtype.itemsize
    assert host.ag_gemm_intra_node() == 1, ("copy-engine allgather is intra-node only: "
                                            "DUSHMEMX_TEAM_NODE must include every PE")
    path = _schedule(m, world_size)
    epoch = 1
    _publish_local(a_local, workspace, ready, rank, epoch)
    torch.cuda.synchronize()

    def launch(ep):
        if path == "split":
            _launch_split(
                host,
                ep,
                a_local,
                b,
                workspace_ptr,
                ready_ptr,
                workspace,
                ready,
                remote_a,
                c,
                compute_stream,
                comm_stream,
                world_size,
                rank,
                m_per_rank,
                nbytes,
            )
        else:
            _launch_gather(
                host,
                ep,
                a_local,
                b,
                workspace_ptr,
                ready_ptr,
                workspace,
                full_a,
                c,
                world_size,
                rank,
                m_per_rank,
                nbytes,
            )

    launch(epoch)
    torch.cuda.synchronize()

    _torch_serial(a_local, b, full_a, torch_out, nccl_group)
    torch.cuda.synchronize()
    err = (c - torch_out).abs().max().item()
    ok = torch.allclose(c, torch_out, rtol=2e-2, atol=2e-2)
    print(f"ag-gemm pe={rank} M={m} N={n} K={k} allclose={bool(ok)} max_abs_err={err}", flush=True)
    if not ok:
        raise SystemExit(1)

    if perf:
        epoch_box = [epoch + 1]

        def overlap_once():
            launch(epoch_box[0])
            epoch_box[0] += 1

        overlap_ms = _max_ms(_elapsed_ms(overlap_once, ITERS), device, nccl_group)
        torch_ms = _max_ms(
            _elapsed_ms(lambda: _torch_serial(a_local, b, full_a, torch_out, nccl_group), ITERS),
            device,
            nccl_group,
        )
        flops = 2.0 * m * n_per_rank * k
        if rank == 0:
            print(
                f"shape M={m} N={n} K={k} path={path} "
                f"overlap_ms={overlap_ms:.3f} overlap_tflops={flops / (overlap_ms * 1e-3) / 1e12:.2f} "
                f"torch_ms={torch_ms:.3f} torch_tflops={flops / (torch_ms * 1e-3) / 1e12:.2f} "
                f"speedup={torch_ms / overlap_ms:.3f}",
                flush=True,
            )

    # Drop the views before dushmem_free. `del` makes ruff treat the names as
    # unbound in the nested launch/benchmark closures.
    a_local = b = workspace = ready = c = remote_a = torch_out = full_a = None
    destroy = host.ag_gemm_workspace_destroy(workspace_ptr, ready_ptr)
    if destroy != 0:
        raise SystemExit(f"workspace destroy failed: {destroy}")


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    perf = len(sys.argv) > 1 and sys.argv[1] == "perf"
    if world_size < 2:
        raise SystemExit("ag-gemm needs at least 2 ranks")

    group = init_torch_distributed()
    init_dushmem_by_torch_pg(load_common_host(), group)
    nccl_group = torch.distributed.new_group(backend="nccl")
    host = _load_host()
    device = torch.device("cuda", local_rank)
    shapes = PERF_SHAPES if perf else ((M, N, K), )
    for m, n, k in shapes:
        _run_shape(host, nccl_group, device, rank, world_size, m, n, k, perf)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
