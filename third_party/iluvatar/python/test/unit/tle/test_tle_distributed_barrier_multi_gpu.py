"""Multi-device check that the device-space distributed_barrier really syncs.

The compile-only coverage lives in test_tle_distributed_barrier.py. It can only
prove that the right FlagCX symbol is called with the right arguments; a barrier
that returns immediately would pass all of it. This file runs the barrier on two
devices and checks both that it completes and that it actually orders the ranks:
rank 0 queues GPU work ahead of its barrier, so rank 1's barrier cannot retire
before that work is done.

Launch it with torchrun on two devices (see third_party/iluvatar/test_triton.sh);
under a plain pytest run it skips, so it stays part of the single-device suite
without needing a distributed setup.
"""
import os
import time

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

N = 64
REPEATS = 8
# Enough queued work that scheduling noise cannot explain the wait, while
# staying far below the CI timeout.
DELAY_MATMULS = 40
DELAY_MATMUL_N = 2048
MIN_OBSERVED_WAIT_S = 0.02


@triton.jit
def _barrier_kernel(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh=mesh, device_dptr=device_dptr, space="device")
    tl.store(out_ptr + tl.program_id(0), 1)


@triton.jit
def _arrive_wait_kernel(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh=mesh, device_dptr=device_dptr, space="device", barrier_kind="arrive")
    tle.distributed_barrier(mesh=mesh, device_dptr=device_dptr, space="device", barrier_kind="wait")
    tl.store(out_ptr + tl.program_id(0), 1)


def _setup():
    mem_pool = tle.get_mem_pool()
    with torch.cuda.use_mem_pool(mem_pool):
        x = torch.randn((N, N), dtype=torch.float32, device="cuda")
    return tle.create_dist_tensor(x)


def _run_barrier_check():
    import torch.distributed as dist

    device_dptr = _setup()
    rank = dist.get_rank()
    mesh = tle.device_mesh({"device": dist.get_world_size()})

    # Repeat to catch a barrier that leaves its index in a bad state: the second
    # iteration would then hang or fall through.
    for i in range(REPEATS):
        out = torch.zeros((1, ), dtype=torch.int32, device="cuda")
        _barrier_kernel[(1, )](out_ptr=out, device_dptr=device_dptr, mesh=mesh)
        torch.cuda.synchronize()
        assert out.cpu().tolist() == [1], f"rank {rank}: iteration {i} did not complete"

    out = torch.zeros((1, ), dtype=torch.int32, device="cuda")
    _arrive_wait_kernel[(1, )](out_ptr=out, device_dptr=device_dptr, mesh=mesh)
    torch.cuda.synchronize()
    assert out.cpu().tolist() == [1], f"rank {rank}: arrive/wait pair did not complete"

    _run_ordering_check(device_dptr, mesh, rank)

    tle.cleanup_communicator()


def _run_ordering_check(device_dptr, mesh, rank):
    import torch.distributed as dist

    out = torch.zeros((1, ), dtype=torch.int32, device="cuda")
    # Warm up so kernel compilation is not part of the measurement.
    _barrier_kernel[(1, )](out_ptr=out, device_dptr=device_dptr, mesh=mesh)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        a = torch.randn((DELAY_MATMUL_N, DELAY_MATMUL_N), device="cuda")
        b = torch.randn((DELAY_MATMUL_N, DELAY_MATMUL_N), device="cuda")

    torch.cuda.synchronize()
    dist.barrier()

    started = time.perf_counter()
    if rank == 0:
        # Queued on the same stream, so the barrier kernel below cannot start
        # until this finishes.
        for _ in range(DELAY_MATMULS):
            a = a @ b
    _barrier_kernel[(1, )](out_ptr=out, device_dptr=device_dptr, mesh=mesh)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if rank == 1:
        assert elapsed >= MIN_OBSERVED_WAIT_S, (
            f"rank 1 barrier returned after {elapsed:.4f}s, expected to wait for rank 0; "
            "the device barrier is not synchronizing")
    print(f"[Rank {rank}] ordering check elapsed={elapsed:.4f}s", flush=True)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="needs two devices under torchrun",
)
def test_device_barrier_synchronizes_ranks():
    _run_barrier_check()


if __name__ == "__main__":
    _run_barrier_check()
