# Copyright 2026-     FlagOS Contributors
"""01-ring: @dialect(name=\"cann\", library=\"cannshmem\") peer read via aclshmem_*.

Same distributed semantics as tutorials/tle/dsa/10-tle-d2d-barrier.py:
each rank fills a symmetric buffer with ``arange(N) + rank * 1000``, then the
device kernel reads the *next* PE's buffer into ``out`` and barriers.

Requires a cann/shmem source install (full headers incl. src/host_device):

  export FLAGTREE_CANNSHMEM_HOME=/path/to/shmem/install/shmem
  source /path/to/shmem/install/set_env.sh   # optional

  ASH_MASTER_PORT=18780 torchrun --nproc_per_node=2 --master_port=29780 \\
      python/tutorials/tle/raw/cann/dist/01-ring-cannshmem/ring.py
"""

from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.cann.cannshmem.utils import (
    bench_us,
    cleanup_symmetric_tensor,
    create_symmetric_tensor,
    init_cannshmem_by_torch_pg,
)

DEVICE = "npu"
N = 8


@dialect(
    name="cann",
    library="cannshmem",
    file=Path(__file__).parent / "ring.cpp",
    extern_func_name="cannshmem_ring_read",
    pipeline={"core": "vector", "pipe": "PIPE_MTE2"},
)
def ring_read(*args, **kwargs):
    ...


@triton.jit
def ring_kernel(symm, out, N_ELEMS: tl.constexpr):
    offs = tl.arange(0, N_ELEMS)
    # PIPE_MTE2 custom op writes UB; stage result then store to GM.
    out_t = tl.zeros([N_ELEMS], dtype=tl.float32)
    out_t = tle_raw.call(ring_read, [symm, out_t, N_ELEMS], output_indices=[1])
    tl.store(out + offs, out_t)


def _run_once(symm, out):
    ring_kernel[(1, 1, 1)](
        symm,
        out,
        N_ELEMS=N,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )


def main():
    init_cannshmem_by_torch_pg()
    rank = dist.get_rank()
    world = dist.get_world_size()
    peer = (rank + 1) % world

    local = (torch.arange(N, dtype=torch.float32, device=DEVICE) + rank * 1000).contiguous()
    symm = create_symmetric_tensor(local)
    symm.copy_(local)
    out = torch.full((N, ), -1.0, dtype=torch.float32, device=DEVICE)

    dist.barrier()
    _run_once(symm, out)
    torch.npu.synchronize()
    dist.barrier()

    expected = torch.arange(N, dtype=torch.float32, device=DEVICE) + peer * 1000
    ok = torch.allclose(out, expected)
    print(f"[rank {rank}] read peer={peer} "
          f"out[:4]={out[:4].cpu().tolist()} "
          f"exp[:4]={expected[:4].cpu().tolist()} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        cleanup_symmetric_tensor(symm)
        raise SystemExit(1)

    us = bench_us(lambda: _run_once(symm, out), warmup=3, iters=10)
    if rank == 0:
        print(f"cannshmem ring avg {us:.1f} us (n={N}, world={world})")
        print("01-ring PASS")

    cleanup_symmetric_tensor(symm)


if __name__ == "__main__":
    main()
