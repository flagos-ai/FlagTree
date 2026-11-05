# Copyright 2026- FlagOS Contributors
"""01-allreduce: raw AscendC device HCCL AllReduce (sum).

Each rank fills a small fp16 buffer with ``rank + 1``. Device-side
``Hccl::AllReduce`` (Prepare → Commit → Wait) reduces across ranks.
Expected result on every rank: ``sum(1..world)`` for every element.

Host still needs the AICPU KFC server (``HcclKfcSession``) before the
AICore kernel; that is the same non-GE contract as cann-hccl
``AicpuUnfoldKernelLaunch``.

Run:
  ASCEND_RT_VISIBLE_DEVICES=0,1 \\
  torchrun --nproc_per_node=2 --master_port=29811 \\
      python/tutorials/tle/raw/cann/dist/02-allreduce-hccl/allreduce.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.cann.hccl.utils import (
    HcclKfcSession,
    get_hccl_comm,
    init_torch_distributed,
)

HERE = Path(__file__).parent
DEVICE = "npu"
N = 64


@dialect(
    name="cann",
    library="hccl",
    file=HERE / "allreduce.cpp",
    extern_func_name="hccl_allreduce",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def hccl_allreduce(*args, **kwargs):
    ...


@triton.jit
def allreduce_kernel(buf, ctx_u64, status, N_ELEMS: tl.constexpr):
    dummy = tl.zeros([16], dtype=tl.float16)
    _ = tle_raw.call(
        hccl_allreduce,
        [buf, N_ELEMS, ctx_u64, status, dummy],
        output_indices=[4],
    )


def main() -> None:
    init_torch_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Local value = rank+1; after sum-AllReduce every rank should see 1+...+world.
    buf = torch.full((N, ), float(rank + 1), dtype=torch.float16, device=DEVICE)
    expected = float(sum(range(1, world + 1)))

    session = HcclKfcSession.open(get_hccl_comm(rank), count=N)
    ctx_box = torch.tensor([session.context], dtype=torch.uint64, device=DEVICE)
    status = torch.zeros((1, ), dtype=torch.int32, device=DEVICE)

    dist.barrier()
    # Pass the same buffer into host launch packing (send/recv for ApiParamDef).
    session.ensure_server(scratch=buf)

    allreduce_kernel[(1, 1, 1)](
        buf,
        ctx_box,
        status,
        N_ELEMS=N,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )

    try:
        session.finish()
    except RuntimeError as exc:
        print(f"[rank {rank}/{world}] FAIL server: {exc}", flush=True)
        os._exit(2)

    torch.npu.current_stream().synchronize()
    st = int(status.item())
    if st != 0:
        raise RuntimeError(f"device AllReduce failed, status={st}")

    got = buf.float().cpu()
    max_abs = (got - expected).abs().max().item()
    print(
        f"[rank {rank}/{world}] status={st} expected={expected} "
        f"buf[0]={got[0].item():.4f} max_abs={max_abs:.6f}",
        flush=True,
    )
    if max_abs > 1e-2:
        raise RuntimeError(f"AllReduce result mismatch: expected {expected}, max_abs={max_abs}")

    dist.barrier()
    if rank == 0:
        print("device HCCL AllReduce PASS", flush=True)


if __name__ == "__main__":
    main()
