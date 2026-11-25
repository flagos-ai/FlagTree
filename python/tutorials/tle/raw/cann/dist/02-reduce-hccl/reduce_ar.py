# Copyright 2026- FlagOS Contributors
"""02-reduce-hccl: device AscendC AllReduce with refactored host KFC lifecycle.

Host (hcomm-aligned)::

  session = HcclKfcSession.open(comm, count=n)   # CreateOpResCtx once
  for each op:
      session.configure(...); session.start()    # AICPU TASK_DELIVER
      kernel[...]                                # Init→Commit→Wait→Finalize
      session.finish()                           # sync attach after Finalize

Checks:
  - correctness vs torch.distributed.all_reduce
  - multi-iter server period (reuse same OpRes; start/finish each op)

  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 reduce_ar.py
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
    MC2_BUFFER_TYPE_WINDOW_IN,
    HcclKfcSession,
    get_hccl_comm,
)

DEVICE = "npu"
_CPP = Path(__file__).parent / "reduce_ar.cpp"

# Default smoke shapes (fp16, divisible by tile_cnt).
CASES = [
    # (n_elems, tile_cnt, tag)
    (1024, 1, "1k_t1"),
    (8192, 1, "8k_t1"),
    (8192, 2, "8k_t2"),
    (2048 * 4096, 1, "mm_c_t1"),  # same C as w4 large MM output strip
    (2048 * 4096, 2, "mm_c_t2"),
]

_SESSION: HcclKfcSession | None = None
_CTX = None
_STATUS = None


@dialect(
    name="cann",
    library="hccl",
    file=_CPP,
    extern_func_name="hccl_reduce_ar",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def hccl_reduce_ar(*args, **kwargs):
    ...


@triton.jit
def reduce_ar_kernel(buf, ctx, status, N_ELEMS, TILE_CNT, USE_WIN, DO_COPY):
    dummy = tl.zeros([16], dtype=tl.float16)
    _ = tle_raw.call(
        hccl_reduce_ar,
        [buf, N_ELEMS, TILE_CNT, USE_WIN, DO_COPY, ctx, status, dummy],
        output_indices=[7],
    )


def prepare(rank: int, max_count: int) -> None:
    global _SESSION, _CTX, _STATUS
    if _SESSION is None:
        _SESSION = HcclKfcSession.open(get_hccl_comm(rank), count=max(max_count, 1))
        _CTX = torch.tensor([_SESSION.context], dtype=torch.uint64, device=DEVICE)
        _STATUS = torch.zeros((1, ), dtype=torch.int32, device=DEVICE)
    else:
        _SESSION.activate()


def allreduce_inplace(buf: torch.Tensor, *, tile_cnt: int = 1, do_copy: int = 1) -> None:
    """One full op: configure → start → AICore → finish (server period)."""
    assert _SESSION is not None and _CTX is not None and _STATUS is not None
    assert buf.dtype == torch.float16 and buf.is_contiguous()
    n = int(buf.numel())
    assert n % tile_cnt == 0

    _STATUS.zero_()
    use_win = MC2_BUFFER_TYPE_WINDOW_IN
    _SESSION.configure(
        int(buf.data_ptr()),
        int(buf.data_ptr()),
        total_cnt=n,
        turn_num=tile_cnt,
        use_buffer_type=use_win,
    )
    _SESSION.start()
    try:
        reduce_ar_kernel[(1, )](
            buf,
            _CTX,
            _STATUS,
            N_ELEMS=n,
            TILE_CNT=tile_cnt,
            USE_WIN=use_win,
            DO_COPY=do_copy,
            enable_legacy_insert_load_store_for_mix_cv=True,
        )
        torch.npu.synchronize()
        st = int(_STATUS.item())
        if st != 0:
            raise RuntimeError(f"hccl_reduce_ar status={st}")
    finally:
        # Always finish so next op can start() cleanly (period fix).
        _SESSION.finish()


def _ref_allreduce(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def _check(tag: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 1e-2) -> bool:
    # fp16 accumulate across ranks — allow mild atol.
    ok = torch.allclose(got.float(), ref.float(), rtol=1e-2, atol=atol)
    if not ok:
        diff = (got.float() - ref.float()).abs()
        print(
            f"FAIL {tag} max_abs={float(diff.max()):.4g} "
            f"mean_abs={float(diff.mean()):.4g}",
            flush=True,
        )
    return ok


def main() -> None:
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    max_n = max(n for n, _, _ in CASES)
    prepare(rank, max_n)

    period_iters = int(os.environ.get("REDUCE_PERIOD_ITERS", "8"))
    passed = 0
    failed = 0

    for n_elems, tile_cnt, tag in CASES:
        # --- single-shot correctness ---
        torch.manual_seed(1000 + rank)
        x = torch.randn(n_elems, dtype=torch.float16, device=DEVICE)
        ref = _ref_allreduce(x)
        buf = x.clone()
        allreduce_inplace(buf, tile_cnt=tile_cnt, do_copy=1)
        dist.barrier()
        ok = _check(f"{tag}/once", buf, ref)
        if rank == 0:
            print(f"{'PASS' if ok else 'FAIL'} {tag}/once n={n_elems} tile={tile_cnt}", flush=True)
        passed += int(ok)
        failed += int(not ok)

        # --- server period: reuse OpRes, start/finish each iter ---
        period_ok = True
        for i in range(period_iters):
            torch.manual_seed(2000 + rank + i * 17)
            x = torch.randn(n_elems, dtype=torch.float16, device=DEVICE)
            ref = _ref_allreduce(x)
            buf = x.clone()
            try:
                allreduce_inplace(buf, tile_cnt=tile_cnt, do_copy=1)
            except Exception as e:
                period_ok = False
                if rank == 0:
                    print(f"FAIL {tag}/period iter={i}: {e}", flush=True)
                break
            if not _check(f"{tag}/period[{i}]", buf, ref):
                period_ok = False
                break
            dist.barrier()
        if rank == 0:
            print(
                f"{'PASS' if period_ok else 'FAIL'} {tag}/period "
                f"iters={period_iters} (reuse session OpRes)",
                flush=True,
            )
        passed += int(period_ok)
        failed += int(not period_ok)

    dist.barrier()
    if rank == 0:
        print(f"summary world={world} PASS={passed} FAIL={failed}", flush=True)
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
