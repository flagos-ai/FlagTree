# Copyright 2026- FlagOS Contributors
"""MatmulAllReduce: Triton MM + WINDOW_IN Commit-per-strip (ops InnerProcess).

ops reference: matmul_all_reduce_910_general.h / matmul_all_reduce_base.h
  Init: cGM = GetWindowsInAddr; AllReduce(Prepare)
  for t in tileCnt:
      mmOp.Process()             # → Triton tl.dot into WINDOW strip
      PostProcEachTurn(Commit)   # → Mc2SyncAll + Commit (no copy)
  Wait + Finalize → user out

Host: read windowsIn once, wrap as fp16 tensor, pass as normal Triton ptr
(not tl.load(cgm).to(pointer_type) — that path fails Ascend Linalg).
"""

from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

import torch
import torch_npu
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

# Defaults; override via FUSED_M / FUSED_N / FUSED_K / FUSED_WORLD for sweeps.
M_LARGE = int(os.environ.get("FUSED_M", "2048"))
N_LARGE = int(os.environ.get("FUSED_N", "4096"))
K_LARGE = int(os.environ.get("FUSED_K", "1024"))
WORLD = int(os.environ.get("FUSED_WORLD", "4"))

# Defaults: tile=2 is a good latency/overlap point. tile=1 is also correct after
# AIV SyncAll + post-JIT prime launch (see _prime_gemm_ar).
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
TILE_CNT = 2
USED_CORE_NUM = 20

# Device GM storage for Hccl client bytes across phase 0/1/2 (vector forbids globals).
_HCCL_SKEL_I64 = 512

BACKEND = "raw_try"
IMPLEMENTED = True

_CPP = Path(__file__).parent / "reduce_hccl.cpp"
_SYNC_CPP = Path(__file__).parent / "mc2_sync.cpp"
_SESSION = None
_CTX = None
_HBOX = None
_STATUS = None
_SKEL_BUF = None
_WIN_PTR = None
_WIN_TEN = None
_WIN_NELEMS = 0
_OP_STARTED = False
_KERNEL_WARMED = False
# First real device launch after JIT can produce wrong WINDOW/AR for tile=1
# (iter0 FAIL, iter1+ PASS). Discard one prime launch per compiled binary.
_KERNEL_PRIMED = False

LAST_KERNEL_US = 0.0
_KERNEL_US_SUM = 0.0
_KERNEL_US_CNT = 0

_OFF_RANK_ID = 16
_OFF_WIN_SIZE = 24
_OFF_WINDOWS_IN0 = 32
ACL_MEMCPY_DEVICE_TO_HOST = 2


def reset_kernel_timer() -> None:
    global LAST_KERNEL_US, _KERNEL_US_SUM, _KERNEL_US_CNT
    LAST_KERNEL_US = 0.0
    _KERNEL_US_SUM = 0.0
    _KERNEL_US_CNT = 0


def mean_kernel_us() -> float:
    if _KERNEL_US_CNT == 0:
        return 0.0
    return _KERNEL_US_SUM / _KERNEL_US_CNT


def _aic_core_num() -> int:
    return int(torch.npu.get_device_properties(0).cube_core_num)


def _parallel_params(m: int) -> tuple[int, int, int]:
    tile_cnt = int(os.environ.get("FUSED_TILE_CNT", TILE_CNT))
    tile_cnt = max(tile_cnt, 1)
    if m % tile_cnt != 0:
        tile_cnt = 1
    strip_m = m // tile_cnt
    grid = int(os.environ.get("FUSED_USED_CORE_NUM", USED_CORE_NUM))
    grid = max(grid, 1)
    grid = min(grid, _aic_core_num())
    return strip_m, tile_cnt, grid


def _acl():
    for p in (
        "/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
        "/usr/local/Ascend/cann/lib64/libascendcl.so",
    ):
        if Path(p).is_file():
            lib = ctypes.CDLL(p)
            lib.aclrtMemcpy.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                ctypes.c_size_t, ctypes.c_int,
            ]
            lib.aclrtMemcpy.restype = ctypes.c_int
            return lib
    raise RuntimeError("libascendcl.so not found")


def _read_windows_in(context: int) -> tuple[int, int]:
    acl = _acl()
    rank_id = ctypes.c_uint32()
    win_size = ctypes.c_uint64()
    win_ptr = ctypes.c_uint64()
    for dst, size, off in (
        (ctypes.byref(rank_id), 4, _OFF_RANK_ID),
        (ctypes.byref(win_size), 8, _OFF_WIN_SIZE),
    ):
        rc = acl.aclrtMemcpy(
            dst, size, ctypes.c_void_p(context + off), size, ACL_MEMCPY_DEVICE_TO_HOST,
        )
        if rc != 0:
            raise RuntimeError(f"read OpRes field off={off} rc={rc}")
    rc = acl.aclrtMemcpy(
        ctypes.byref(win_ptr), 8,
        ctypes.c_void_p(context + _OFF_WINDOWS_IN0 + 8 * int(rank_id.value)), 8,
        ACL_MEMCPY_DEVICE_TO_HOST,
    )
    if rc != 0:
        raise RuntimeError(f"read windowsIn rc={rc}")
    return int(win_ptr.value), int(win_size.value)


def _window_view(n_elems: int) -> torch.Tensor:
    """Wrap HCCL windowsIn as an fp16 tensor so Triton gets a real !tt.ptr<f16>."""
    global _WIN_TEN, _WIN_NELEMS
    assert _WIN_PTR is not None and _WIN_PTR != 0
    n_elems = int(n_elems)
    if _WIN_TEN is not None and _WIN_NELEMS == n_elems and _WIN_TEN.data_ptr() == _WIN_PTR:
        return _WIN_TEN
    nbytes = n_elems * 2
    dev = torch.device("npu", torch.npu.current_device())
    storage = torch_npu._C._construct_storage_from_data_pointer(_WIN_PTR, dev, nbytes)
    ten = torch.empty(0, dtype=torch.float16, device=dev)
    ten.set_(storage, storage_offset=0, size=(n_elems,), stride=(1,))
    if int(ten.data_ptr()) != int(_WIN_PTR):
        raise RuntimeError(
            f"window view data_ptr mismatch: ten=0x{ten.data_ptr():x} win=0x{_WIN_PTR:x}"
        )
    _WIN_TEN = ten
    _WIN_NELEMS = n_elems
    return ten


@dialect(
    name="cann",
    library="hccl",
    file=_SYNC_CPP,
    extern_func_name="mc2_sync_all_cube",
    pipeline={"core": "cube", "pipe": "PIPE_FIX"},
)
def mc2_sync_all_cube(*args, **kwargs):
    ...


@dialect(
    name="cann",
    library="hccl",
    file=_SYNC_CPP,
    extern_func_name="mc2_sync_all_vector",
    pipeline={"core": "vector", "pipe": "PIPE_MTE3"},
)
def mc2_sync_all_vector(*args, **kwargs):
    ...


@dialect(
    name="cann",
    library="hccl",
    file=_CPP,
    extern_func_name="hccl_reduce_skel",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def hccl_reduce_skel(*args, **kwargs):
    ...


def prepare(rank: int, world: int, max_m: int, max_n: int) -> None:
    global _SESSION, _CTX, _WIN_PTR, _WIN_TEN, _WIN_NELEMS, WORLD
    del world
    need_elems = max(int(max_m) * int(max_n), 1)
    if _SESSION is None:
        _SESSION = HcclKfcSession.open(get_hccl_comm(rank), count=need_elems)
        _CTX = torch.tensor([_SESSION.context], dtype=torch.uint64, device=DEVICE)
        _WIN_PTR, win_size = _read_windows_in(_SESSION.context)
        need = need_elems * 2
        if _WIN_PTR == 0 or win_size < need:
            raise RuntimeError(
                f"WINDOW_IN too small: win=0x{_WIN_PTR:x} winSize={win_size} need={need}"
            )
        _WIN_TEN = None
        _WIN_NELEMS = 0
    else:
        # Session is sized at open; caller must prepare() with the sweep max first.
        _SESSION.activate()
        _WIN_PTR, win_size = _read_windows_in(_SESSION.context)
        need = need_elems * 2
        if _WIN_PTR == 0 or win_size < need:
            raise RuntimeError(
                f"WINDOW_IN too small for shape: winSize={win_size} need={need} "
                f"(re-open session with larger max_m*max_n)"
            )


def begin_op(out_flat, *, m: int, n: int, tile_cnt: int | None = None) -> None:
    """Configure tiling only; AICPU start happens after warmup in run_gemm_ar."""
    global _OP_STARTED
    assert _SESSION is not None
    if tile_cnt is None:
        _, tile_cnt, _ = _parallel_params(m)
    # send/recv tiling fields: user out is AllReduce dst; WINDOW is device-side cGM.
    _SESSION.configure(
        int(out_flat.data_ptr()),
        int(out_flat.data_ptr()),
        total_cnt=m * n,
        turn_num=tile_cnt,
        use_buffer_type=MC2_BUFFER_TYPE_WINDOW_IN,
    )
    _OP_STARTED = False


def end_op() -> None:
    global _OP_STARTED
    assert _SESSION is not None
    if _SESSION.started:
        _SESSION.finish()
    _OP_STARTED = False


def _ensure_bufs() -> None:
    global _HBOX, _STATUS, _SKEL_BUF
    if _HBOX is None:
        _HBOX = torch.full((1,), -1, dtype=torch.int64, device=DEVICE)
        _STATUS = torch.zeros((1,), dtype=torch.int32, device=DEVICE)
        _SKEL_BUF = torch.zeros((_HCCL_SKEL_I64,), dtype=torch.int64, device=DEVICE)


def _warmup_gemm_ar(a, b, out_flat, win, *, strip_m: int, tile_cnt: int, grid: int) -> None:
    """JIT only (must finish before KFC start). Clears prime flag on new binary."""
    global _KERNEL_WARMED, _KERNEL_PRIMED
    if _KERNEL_WARMED:
        return
    assert _CTX is not None
    _ensure_bufs()
    for op in (mc2_sync_all_cube, mc2_sync_all_vector, hccl_reduce_skel):
        _ = op.bitcode
    m, k = a.shape
    n = b.shape[1]
    n_elems = m * n
    use_win = MC2_BUFFER_TYPE_WINDOW_IN
    gemm_ar_kernel.run(
        a, b, out_flat, win, _CTX, _HBOX, _STATUS, _SKEL_BUF,
        m, n, k, strip_m, tile_cnt, n_elems, use_win, grid,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        grid=(grid,),
        warmup=True,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
    _KERNEL_WARMED = True
    _KERNEL_PRIMED = False


def _prime_gemm_ar(a, b, out_flat, win, *, strip_m: int, tile_cnt: int, grid: int) -> None:
    """One discarded real launch after JIT — needed for tile=1 correctness."""
    global _KERNEL_PRIMED, _OP_STARTED
    if _KERNEL_PRIMED:
        return
    assert _SESSION is not None and _CTX is not None
    _ensure_bufs()
    m, k = a.shape
    n = b.shape[1]
    n_elems = m * n
    use_win = MC2_BUFFER_TYPE_WINDOW_IN
    _STATUS.zero_()
    _HBOX.fill_(-1)
    if not _SESSION.started:
        _SESSION.start()
        _OP_STARTED = True
    gemm_ar_kernel[(grid,)](
        a, b, out_flat, win, _CTX, _HBOX, _STATUS, _SKEL_BUF,
        m, n, k, strip_m, tile_cnt, n_elems, use_win, grid,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
    torch.npu.current_stream().synchronize()
    if _SESSION.started:
        _SESSION.finish()
        _OP_STARTED = False
    _KERNEL_PRIMED = True


@triton.jit
def gemm_ar_kernel(
    a_ptr, b_ptr, out_ptr, win_ptr, ctx, hbox, status, skel_buf,
    M, N, K, STRIP_ROWS, TILE_CNT, N_ELEMS, USE_WIN, NUM_PROGRAMS,
    stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # ops InnerProcess: MM → WINDOW strip, SyncAll, Commit; Wait/Finalize last.
    # out_ptr = AllReduce dst; win_ptr = windowsIn (host-wrapped, real ptr arg).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(STRIP_ROWS, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    dummy = tl.zeros([16], dtype=tl.float16)

    dummy = tle_raw.call(
        hccl_reduce_skel,
        [out_ptr, N_ELEMS, TILE_CNT, USE_WIN, 0, 0, ctx, hbox, status, skel_buf, dummy],
        output_indices=[10],
    )

    for t in range(0, TILE_CNT):
        for tile_id in range(pid, num_tiles, NUM_PROGRAMS):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_local_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_am = t * STRIP_ROWS + offs_local_m
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, K, BLOCK_K):
                offs_k = kk + tl.arange(0, BLOCK_K)
                a = tl.load(
                    a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
                    other=0.0,
                )
                acc = tl.dot(a, b, acc)

            c = acc.to(tl.float16)
            # ops: mm writes cGM (= windowsIn) for this strip
            tl.store(
                win_ptr + (t * STRIP_ROWS + offs_local_m)[:, None] * N + offs_bn[None, :],
                c,
                mask=(offs_local_m[:, None] < STRIP_ROWS)
                & (offs_am[:, None] < M)
                & (offs_bn[None, :] < N),
            )

        # PostProcEachTurn: cube flush + AIV SyncAll (all programs) + Commit.
        # Cube CrossCore alone does not wait for other cubes; without AIV SyncAll,
        # block0 Commit races unfinished WINDOW stores (tile=1 / strip0 FAIL).
        dummy = tle_raw.call(mc2_sync_all_cube, [dummy], output_indices=[0])
        dummy = tle_raw.call(mc2_sync_all_vector, [dummy], output_indices=[0])
        dummy = tle_raw.call(
            hccl_reduce_skel,
            [out_ptr, N_ELEMS, TILE_CNT, USE_WIN, 1, t, ctx, hbox, status, skel_buf, dummy],
            output_indices=[10],
        )

    dummy = tle_raw.call(
        hccl_reduce_skel,
        [out_ptr, N_ELEMS, TILE_CNT, USE_WIN, 2, 0, ctx, hbox, status, skel_buf, dummy],
        output_indices=[10],
    )
    _ = BLOCK_K
    _ = dummy


def run_gemm_ar(a, b, symm, out_flat, rank: int, world: int) -> None:
    global LAST_KERNEL_US, _KERNEL_US_SUM, _KERNEL_US_CNT
    global _HBOX, _STATUS, _OP_STARTED, M_LARGE, N_LARGE, K_LARGE, WORLD
    del symm
    WORLD = int(world)
    m, k = a.shape
    n = b.shape[1]
    M_LARGE, N_LARGE, K_LARGE = int(m), int(n), int(k)
    n_elems = m * n
    strip_m, tile_cnt, grid = _parallel_params(m)

    prepare(rank, world, m, n)
    assert _SESSION is not None and _CTX is not None and _WIN_PTR is not None
    begin_op(out_flat, m=m, n=n, tile_cnt=tile_cnt)

    win = _window_view(n_elems)
    # Compile before any KFC start (AICPU timeout if JIT runs after start).
    _warmup_gemm_ar(a, b, out_flat, win, strip_m=strip_m, tile_cnt=tile_cnt, grid=grid)
    # Discard first real launch after JIT (tile=1 otherwise fails vs torch_npu).
    if not _KERNEL_PRIMED:
        _prime_gemm_ar(a, b, out_flat, win, strip_m=strip_m, tile_cnt=tile_cnt, grid=grid)
        begin_op(out_flat, m=m, n=n, tile_cnt=tile_cnt)

    _ensure_bufs()
    _STATUS.zero_()
    _HBOX.fill_(-1)
    use_win = MC2_BUFFER_TYPE_WINDOW_IN

    if not _SESSION.started:
        _SESSION.start()
        _OP_STARTED = True

    t0 = time.perf_counter()
    gemm_ar_kernel[(grid,)](
        a, b, out_flat, win, _CTX, _HBOX, _STATUS, _SKEL_BUF,
        m, n, k, strip_m, tile_cnt, n_elems, use_win, grid,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
    torch.npu.current_stream().synchronize()
    LAST_KERNEL_US = (time.perf_counter() - t0) * 1e6
    _KERNEL_US_SUM += LAST_KERNEL_US
    _KERNEL_US_CNT += 1

    st = int(_STATUS.item())
    if st != 0:
        raise RuntimeError(f"hccl_reduce_skel status={st}")
