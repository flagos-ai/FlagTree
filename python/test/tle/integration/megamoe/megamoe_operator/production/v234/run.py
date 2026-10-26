"""V234 standalone: v220 with the real SM90 W1 weight/SF layout contract.

This immutable snapshot contains the complete Python implementation and host
harness.  It does not import or monkey-patch an earlier experiment version.
Relative to v208, the L1 A+SFA producer transfers its single required SF row
through a matching ``[1, BLOCK_M]`` descriptor and SMEM view.  L2 retains its
two-row SF transfer.

One process per rank (mpirun -np R). Rank r owns experts [r*EPR, (r+1)*EPR) and
holds NTOK local tokens, exactly like sm90_fp8_mega_moe.cuh.

`sym_buffer.map(ptr, rank)` becomes a peer pointer into the NVSHMEM symmetric
heap: the host builds an int64[NPES] table from nvshmem_ptr(buf, pe) and the
kernel does `tl.load(tab + pe).to(tl.pointer_type(dtype))`. All remote traffic is
then ordinary NVLink P2P load/store plus system-scope atomics -- which is exactly
what UserHopper does (it has no device-side nvshmem calls either).

The four sym_buffer.map sites of the CUDA kernel, and their TLE form:
  D3   *sym_buffer.map(dst_ptr, owner)      -> peer_i32(q_tab, owner)      [VECTOR of owners]
  D4   recv_count/_sum on the owner         -> peer_i64(recv/rsum_tab, owner), scope="sys"
  pull remote token / SF / weight           -> peer_fp8/f32(tok/sf/w_tab, src_rank)
  L2   combine buffer on the SOURCE rank    -> peer_bf16(cb_tab, dst_rank)  [VECTOR of dsts]
  comm::nvlink_barrier (barrier.cuh:38-80)  -> SM0 release.sys-adds every peer's signal
                                               slot, spins on its own with acquire.sys,
                                               wrapped in intra-rank grid syncs.

SFA IS NOW TMA TOO. Earlier I wrongly called the SF tile a "strided gather": the SF
buffers are MN-major, `sf[k_group * padded_tokens + token]`, so a block's SF tile is
BLOCK_M *contiguous* floats -- a perfectly valid TMA box `[1, BLOCK_M]`. That is
exactly what CUDA does (`tma::copy<BLOCK_M, 1, 0, float>`), issuing ONE box for L1
(per-128 SF) and TWO adjacent k-groups for L2 (per-64 SF), landing at smem +0 and
+BLOCK_M (sm90_fp8_mega_moe.cuh:786-805). Here that is two pipe fields sfa_lo/sfa_hi.

The WEIGHT SF (SFB) is deliberately NOT staged: CUDA says so itself at the B loader,
"TMA load B (weight SF is now loaded directly by math warps from global)". So the math
role loads sfb straight from global, as it does in CUDA.

Consequence: every pipe field is now a TMA field, so the "a pipe commit may not mix TMA
and tl.store fields" constraint disappears and the two pipes collapse back into ONE.

The D8 FP8 token load/store pair uses the same descriptorless
``cp.async.bulk`` TMA1D mechanism as UserHopper. Activation SF and top-k weight
remain ordinary load/store. Each pull stream owns one 4 KiB token stage and its
raw mbarrier state; the raw helper is limited to route selection and D6-D9.

V22 DIAGNOSTIC: move the activation-SF SMEM loads after async WGMMA issue and
before `wgmma_wait`, matching the overlap window in UserHopper SM90
`sm90_fp8_mega_moe.cuh:976-1007,1054-1093`. Weight-SF prefetch is intentionally
unchanged; that separate idea was already tested by v8.

V23 L2 EPILOGUE: stage the BF16 accumulator and one destination pointer per row
in linear SMEM, then cross a narrow raw-CUDA boundary which remaps the physical
math warpgroup exactly like UserHopper: 16 lanes own one row and each lane issues
one 16-byte load/store. This fixes both reasons v12-v14 stayed scalar: a gathered
peer base is not compiler-contiguous, and the WGMMA accumulator fragment does not
give each lane eight adjacent BF16 values. The raw ABI carries only integer-encoded
SMEM addresses because mixed addrspace(3)/addrspace(1) pointers do not link in the
current TLE raw implementation.

TEST-ONLY: `meta` and `l2_out` are also placed on the symmetric heap so each rank
can read a peer's copy and check the cross-rank scatter EXACTLY.

Run this file directly; set ``MEGAMOE_NP`` for the rank count.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

SUPPORT_DIR = Path(__file__).resolve().parent.parent
os.environ["TLE_MULTI_TMA_WRITERS"] = "1"
os.environ.setdefault("W_CACHE_TAG", "v234-v220-realdata-w1-layout")
NUM_RANKS = int(os.environ.get("MEGAMOE_NP", "2"))
# TLE_PYTHON_OVERRIDE: mpirun workers are spawned with THIS interpreter, not the one
# that started the parent. Without an override, launching with a different venv
# silently still runs the OLD triton in the workers (cost: a full fake verification).
TLE_PYTHON = Path(os.environ.get("TLE_PYTHON_OVERRIDE", sys.executable))

# MUST precede the triton import: ranks sharing one cache dir deadlock on its lock.
rank_env = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0"))
cache_tag = os.environ.get("W_CACHE_TAG", "bench")
cache_root = Path(os.environ.get("W_CACHE_ROOT", "/workspace/megakernel/.cache"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NUM_RANKS)))
os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / f"tle-megamoe-{cache_tag}-rank-{rank_env}"))

sys.path.insert(0, str(SUPPORT_DIR))
from production_runtime import NVSHMEM_HOME, _compile_nvshmem_host_so, _import_env  # noqa: E402

_env = _import_env()
torch, triton, tl, tle = _env["torch"], _env["triton"], _env["tl"], _env["tle"]
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: E402
from triton.language.extra import libdevice  # noqa: E402
from triton.experimental.tle.raw import dialect  # noqa: E402
import triton.experimental.tle.language.raw as tle_raw  # noqa: E402

# Keep FlagTree runnable on this host's validated clang-17 toolchain without
# modifying the shared virtualenv.  The raw helpers used by v31 already compile
# and pass 8-rank correctness with this compiler.
from triton.experimental.tle.raw.cuda import runtime as _tle_raw_cuda_runtime  # noqa: E402

_tle_raw_cuda_runtime._MIN_CLANG_MAJOR = 17
_tle_raw_cuda_runtime._resolve_clang.cache_clear()

RAW_DIR = Path(__file__).resolve().parent

# Versioned experiments may opt into releasing a GEMM pipe slot immediately
# after WGMMA has stopped consuming its SMEM operands.  Keep the historical
# v33-v40 ordering as the default; v150 flips this module constant before the
# JIT dependency graph is built.
EARLY_PIPE_RELEASE = tl.constexpr(False)
# V157 opts into a per-thread mbarrier release after WGMMA completion. Keep
# every historical version on the elected-thread release path by default.
PARTICIPANT_PIPE_RELEASE = False
# Default (non-specialized) loader role width.  Kept at four for all existing
# versions; v159 overrides it to test whether TLE can avoid two idle frontend
# warps and move the CTA shape toward UserHopper's 384 threads.
LOADER_WARPS = 4
# UserHopper's TMA loaders do not join the dispatch/epilogue pre-pull barrier.
# They poll the finalized rsum status words directly and can therefore prepare
# their scheduler before dispatch releases the math warps.  Historical versions
# retain the conservative meta_ready wait; v160 opts into the CUDA-shaped path.
LOADER_RSUM_EARLY = False


@dialect(
    name="cuda",
    file=RAW_DIR / "raw_l2_wide_scatter.cu",
    extern_func_name="TleL2WideScatter",
)
def l2_wide_scatter_edsl(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=RAW_DIR / "raw_l2_tma_scatter.cu",
    extern_func_name="TleL2TmaScatter",
)
def l2_tma_scatter_edsl(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=RAW_DIR / "raw_d8_tma1d_pull.cu",
    extern=RAW_DIR / "raw_d8_tma1d_pull_extern.py",
    extern_func_name="TleD8Tma1d",
)
def d8_tma1d_edsl(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=RAW_DIR / "raw_d8_unified_dispatch.cu",
    extern=RAW_DIR / "raw_d8_unified_dispatch_extern.py",
    extern_func_name="TleD8UnifiedDispatchPull",
)
def d8_unified_dispatch_edsl(*args, **kwargs):
    ...


# ---------------------------------------------------------------------------
# peer-pointer primitives  ( == sym_buffer.map )
# ---------------------------------------------------------------------------
@triton.jit
def peer_i8(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int8))


@triton.jit
def peer_i32(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int32))


@triton.jit
def peer_i64(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int64))


@triton.jit
def peer_f32(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.float32))


@triton.jit
def peer_bf16(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.bfloat16))


@triton.jit
def peer_fp8(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.float8e4nv))


@triton.jit
def fence_sys():
    _ = tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def smem_generic_addr(ptr):
    """Convert TLE's shared offset to the generic address expected by raw CUDA."""
    offset = tl.cast(ptr, tl.uint64)
    return tl.inline_asm_elementwise(
        "cvta.shared.u64 $0, $1;",
        "=l,l",
        [offset],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rank_barrier(counter, NUM_SMS: tl.constexpr):
    """comm::grid_sync across this rank's own CTAs. Partition-safe."""
    tl.debug_barrier()
    tl.atomic_add(counter, 1, sem="release")
    v = tl.atomic_add(counter, 0, sem="acquire")
    while v < NUM_SMS:
        v = tl.atomic_add(counter, 0, sem="acquire")
    tl.debug_barrier()


@triton.jit
def dispatch_sync(USE_D8_TMA1D: tl.constexpr):
    """Dispatch-only wrapper keeps the role's barrier call graph unambiguous."""
    tl.debug_barrier()


@triton.jit
def dispatch_rank_barrier(counter, NUM_SMS: tl.constexpr, USE_D8_TMA1D: tl.constexpr):
    dispatch_sync(USE_D8_TMA1D)
    tl.atomic_add(counter, 1, sem="release")
    v = tl.atomic_add(counter, 0, sem="acquire")
    while v < NUM_SMS:
        v = tl.atomic_add(counter, 0, sem="acquire")
    dispatch_sync(USE_D8_TMA1D)


@triton.jit
def dispatch_nvlink_barrier(sig_tab, slot, ctr, sm_idx, MY_PE: tl.constexpr, NPES: tl.constexpr, NUM_SMS: tl.constexpr,
                            USE_D8_TMA1D: tl.constexpr, FAST_NVLINK_BARRIER: tl.constexpr):
    # D3 already performed a rank-wide grid sync.  CUDA therefore calls the
    # pre-pull NVLink barrier with sync_prologue=false and has the first NPES
    # threads signal peers concurrently.  Keep the old path as the A/B control.
    dispatch_sync(USE_D8_TMA1D)
    fence_sys()
    if not FAST_NVLINK_BARRIER:
        dispatch_rank_barrier(ctr, NUM_SMS, USE_D8_TMA1D)
    if sm_idx == 0:
        if FAST_NVLINK_BARRIER:
            pe = tl.arange(0, 8)
            tl.atomic_add(peer_i32(sig_tab, pe) + slot, 1, mask=pe < NPES, sem="release", scope="sys")
        else:
            for pe in tl.static_range(0, NPES):
                tl.atomic_add(peer_i32(sig_tab, pe) + slot, 1, sem="release", scope="sys")
        sl = peer_i32(sig_tab, MY_PE)
        v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
        while v < NPES:
            v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
    dispatch_rank_barrier(ctr + 1, NUM_SMS, USE_D8_TMA1D)


@triton.jit
def nvlink_barrier(sig_tab, slot, ctr, sm_idx, MY_PE: tl.constexpr, NPES: tl.constexpr, NUM_SMS: tl.constexpr):
    """comm::nvlink_barrier, barrier.cuh:38-80."""
    fence_sys()
    rank_barrier(ctr, NUM_SMS)
    if sm_idx == 0:
        # v208: match UserHopper's one-thread-per-peer signal ownership.
        pe = tl.arange(0, 8)
        tl.atomic_add(peer_i32(sig_tab, pe) + slot, 1, mask=pe < NPES, sem="release", scope="sys")
        sl = peer_i32(sig_tab, MY_PE)
        v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
        while v < NPES:
            v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
    rank_barrier(ctr + 1, NUM_SMS)


@triton.jit
def wait_flag(ptr):
    v = tl.atomic_add(ptr, 0, sem="acquire")
    while v < 1:
        v = tl.atomic_add(ptr, 0, sem="acquire")


@triton.jit
def pool_layout(rsum, EPR: tl.constexpr, EPR_POW2: tl.constexpr, BLOCK_M: tl.constexpr):
    offs = tl.arange(0, EPR_POW2)
    ok = offs < EPR
    n_vec = (tl.load(rsum + offs, mask=ok, other=0) & 0xffffffff).to(tl.int32)
    blk_vec = tl.where(ok, (n_vec + BLOCK_M - 1) // BLOCK_M, 0)
    off_vec = tl.cumsum(blk_vec, axis=0) - blk_vec
    return n_vec, blk_vec, off_vec


@triton.jit
def split_last(x, D0: tl.constexpr, D1: tl.constexpr):
    """[D0, D1] -> two [D0, D1//2] along the LAST axis."""
    t = tl.reshape(x, (D0, 2, D1 // 2))
    t = tl.permute(t, (0, 2, 1))
    return tl.split(t)


@triton.jit
def split_first(x, D0: tl.constexpr, D1: tl.constexpr):
    """[D0, D1] -> two [D0//2, D1] along the FIRST axis."""
    t = tl.reshape(x, (2, D0 // 2, D1))
    t = tl.permute(t, (1, 2, 0))
    return tl.split(t)


@triton.jit
def pick(vec, idx, EPR_POW2: tl.constexpr):
    """Extract one dynamic scalar without one-hot reduction lowering."""
    index = tl.full((1, ), idx, tl.int32)
    return tl.reshape(tl.gather(vec, index, axis=0), ())


# ---------------------------------------------------------------------------
# symmetric-heap byte helpers (host plumbing only)
# ---------------------------------------------------------------------------
@triton.jit
def sym_zero(tab, PE, NB, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(peer_i8(tab, PE) + offs, tl.zeros((BLOCK, ), dtype=tl.int8), mask=offs < NB)


@triton.jit
def sym_h2d(src, tab, PE, NB, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < NB
    tl.store(peer_i8(tab, PE) + offs, tl.load(src + offs, mask=m), mask=m)


@triton.jit
def sym_d2h(tab, PE, dst, NB, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < NB
    tl.store(dst + offs, tl.load(peer_i8(tab, PE) + offs, mask=m), mask=m)


@triton.jit
def _pick_stream2(x, which: tl.constexpr):
    stream = tl.arange(0, 2)
    return tl.sum(tl.where(stream == which, x, 0), axis=0)


@triton.jit
def dispatch_dual_d8_pull(
    sf_tab,
    w_tab,
    rsum_l,
    recv_l,
    q_l,
    meta_l,
    l1_arrival_count,
    l1_token,
    l1_sf,
    l1_w,
    d8_token_s,
    d8_state_s,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    R_POW2: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    NSF_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
):
    """Two independent descriptorless D8 streams, mirroring CUDA's 2 warps."""
    sm_idx = tl.program_id(0)
    stream = tl.arange(0, 2)
    rank_v = tl.arange(0, R_POW2)
    rank_ok = rank_v < NPES
    offs_sf = tl.arange(0, NSF_POW2)
    sf_ok = offs_sf < NSF

    inactive = tl.full((), 0, tl.int32)
    d8_bytes = tl.full((), K, tl.int32)
    d8_init_op = tl.full((), 0, tl.int32)
    d8_load_op = tl.full((), 1, tl.int32)
    d8_store_op = tl.full((), 2, tl.int32)
    d8_fence_op = tl.full((), 3, tl.int32)
    stage0 = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (0, )))
    stage1 = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (K, )))
    state0 = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (0, )))
    state1 = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (2, )))
    init_addr = tl.cast(l1_token, tl.int64)
    tle_raw.call(d8_tma1d_edsl, [stage0, state0, init_addr, d8_bytes, inactive, d8_init_op])
    tle_raw.call(d8_tma1d_edsl, [stage1, state1, init_addr, d8_bytes, inactive, d8_init_op])
    dispatch_sync(USE_D8_TMA1D)
    tle_raw.call(d8_tma1d_edsl, [stage0, state0, init_addr, d8_bytes, inactive, d8_fence_op])
    tle_raw.call(d8_tma1d_edsl, [stage1, state1, init_addr, d8_bytes, inactive, d8_fence_op])
    dispatch_sync(USE_D8_TMA1D)

    # Logical stream s owns global pool indices sm*2+s, striding by 2*NUM_SMS.
    token_idx = sm_idx * 2 + stream
    cur = tl.full((2, ), -1, tl.int32)
    e_start = tl.zeros((2, ), tl.int32)
    e_end = tl.zeros((2, ), tl.int32)
    pool_blk = tl.zeros((2, ), tl.int32)
    done = tl.zeros((2, ), tl.int32)

    while tl.sum((done == 0).to(tl.int32), axis=0) > 0:
        need = (done == 0) & (token_idx >= e_end)
        while tl.sum(need.to(tl.int32), axis=0) > 0:  # D6, two cursors
            next_cur = cur + 1
            ex = next_cur >= EPR
            safe_cur = tl.where(ex, 0, next_cur)
            n = (tl.load(rsum_l + safe_cur, mask=need & ~ex, other=0) & 0xffffffff).to(tl.int32)
            add_blk = (e_end - e_start + BLOCK_M - 1) // BLOCK_M
            advancing = need & ~ex
            pool_blk = tl.where(advancing, pool_blk + add_blk, pool_blk)
            e_start = tl.where(advancing, e_end, e_start)
            e_end = tl.where(advancing, e_end + n, e_end)
            cur = tl.where(need, next_cur, cur)
            done = tl.where(need & ex, 1, done)
            need = (done == 0) & (token_idx >= e_end)

        live = done == 0
        safe_e = tl.where(live, cur, 0)
        tok_in_e = tl.where(live, token_idx - e_start, 0)

        remaining = tl.load(
            recv_l + rank_v[None, :] * EPR + safe_e[:, None],
            mask=live[:, None] & rank_ok[None, :],
            other=0,
        ).to(tl.int32)
        slot = tok_in_e
        offset = tl.zeros((2, ), tl.int32)
        src_rank = tl.zeros((2, ), tl.int32)
        tir = tl.zeros((2, ), tl.int32)
        found = tl.where(live, 0, 1)
        while tl.sum((found == 0).to(tl.int32), axis=0) > 0:  # D7, two routes
            active = remaining > 0
            num_active = tl.sum(active.to(tl.int32), axis=1)
            length = tl.min(tl.where(active, remaining, 0x7fffffff), axis=1)
            nrt = length * num_active
            hit = ((slot < nrt) | (num_active == 0)) & (found == 0)
            denom = tl.maximum(num_active, 1)
            sir = slot % denom
            order = tl.cumsum(active.to(tl.int32), axis=1) - 1
            sel = active & (order == sir[:, None])
            selected_rank = tl.sum(tl.where(sel, rank_v[None, :], 0), axis=1)
            src_rank = tl.where(hit, selected_rank, src_rank)
            tir = tl.where(hit, offset + slot // denom, tir)
            slot = tl.where(hit, slot, slot - nrt)
            offset = tl.where(hit, offset, offset + length)
            remaining = tl.where(hit[:, None], remaining, remaining - tl.minimum(remaining, length[:, None]))
            found = tl.where(hit, 1, found)

        q_off = tl.where(live, (safe_e * NPES + src_rank) * MAX_RECV + tir, 0)
        stt = tl.load(q_l + q_off, mask=live, other=0)
        src_tok = stt // TOPK
        src_topk = stt % TOPK
        pt = tl.where(live, pool_blk * BLOCK_M + tok_in_e, 0)

        peer_base = tp0
        peer_base = tl.where(src_rank == 1, tp1, peer_base)
        peer_base = tl.where(src_rank == 2, tp2, peer_base)
        peer_base = tl.where(src_rank == 3, tp3, peer_base)
        peer_base = tl.where(src_rank == 4, tp4, peer_base)
        peer_base = tl.where(src_rank == 5, tp5, peer_base)
        peer_base = tl.where(src_rank == 6, tp6, peer_base)
        peer_base = tl.where(src_rank == 7, tp7, peer_base)
        peer_tok = tl.multiple_of(peer_base.to(tl.pointer_type(tl.float8e4nv)), 16)
        src_addr = tl.cast(peer_tok + src_tok * K, tl.int64)
        dst_addr = tl.cast(l1_token + pt * K, tl.int64)
        active_i32 = live.to(tl.int32)

        # Enqueue both remote loads before either stream waits.
        tle_raw.call(d8_tma1d_edsl,
                     [stage0, state0,
                      _pick_stream2(src_addr, 0), d8_bytes,
                      _pick_stream2(active_i32, 0), d8_load_op])
        tle_raw.call(d8_tma1d_edsl,
                     [stage1, state1,
                      _pick_stream2(src_addr, 1), d8_bytes,
                      _pick_stream2(active_i32, 1), d8_load_op])
        dispatch_sync(USE_D8_TMA1D)

        sf_base = peer_f32(sf_tab, src_rank)
        sfv = tl.load(sf_base[:, None] + src_tok[:, None] * NSF + offs_sf[None, :], mask=live[:, None] & sf_ok[None, :],
                      other=0.0)
        tl.store(l1_sf + offs_sf[None, :] * POOL_TOKENS + pt[:, None], sfv, mask=live[:, None] & sf_ok[None, :])
        weight = tl.load(peer_f32(w_tab, src_rank) + stt, mask=live, other=0.0)
        tl.store(l1_w + pt, weight, mask=live)
        dispatch_sync(USE_D8_TMA1D)

        tle_raw.call(d8_tma1d_edsl,
                     [stage0, state0,
                      _pick_stream2(dst_addr, 0), d8_bytes,
                      _pick_stream2(active_i32, 0), d8_store_op])
        tle_raw.call(d8_tma1d_edsl,
                     [stage1, state1,
                      _pick_stream2(dst_addr, 1), d8_bytes,
                      _pick_stream2(active_i32, 1), d8_store_op])
        dispatch_sync(USE_D8_TMA1D)

        tl.store(meta_l + pt * 3 + 0, src_rank, mask=live)  # D9
        tl.store(meta_l + pt * 3 + 1, src_tok, mask=live)
        tl.store(meta_l + pt * 3 + 2, src_topk, mask=live)
        tl.atomic_add(l1_arrival_count + (pool_blk + tok_in_e // BLOCK_M), 1, mask=live, sem="release")
        token_idx = tl.where(live, token_idx + NUM_SMS * 2, token_idx)


@triton.jit
def _resolve_one_d8_route(
    token_idx,
    cur,
    e_start,
    e_end,
    pool_blk,
    done,
    rsum_l,
    recv_l,
    q_l,
    NPES: tl.constexpr,
    EPR: tl.constexpr,
    R_POW2: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TOPK: tl.constexpr,
):
    while (done == 0) & (token_idx >= e_end):
        cur += 1
        ex = cur >= EPR
        done = tl.where(ex, 1, 0)
        safe_cur = tl.where(ex, 0, cur)
        n = (tl.load(rsum_l + safe_cur) & 0xffffffff).to(tl.int32)
        add_blk = (e_end - e_start + BLOCK_M - 1) // BLOCK_M
        pool_blk = tl.where(ex, pool_blk, pool_blk + add_blk)
        e_start = tl.where(ex, e_start, e_end)
        e_end = tl.where(ex, e_end, e_end + n)

    live = done == 0
    safe_e = tl.where(live, cur, 0)
    tok_in_e = tl.where(live, token_idx - e_start, 0)
    rank_v = tl.arange(0, R_POW2)
    rank_ok = rank_v < NPES
    remaining = tl.where(rank_ok & live, tl.load(recv_l + rank_v * EPR + safe_e, mask=rank_ok, other=0).to(tl.int32), 0)
    slot = tok_in_e
    offset = 0
    src_rank = 0
    tir = 0
    found = tl.where(live, 0, 1)
    while found == 0:
        active = remaining > 0
        num_active = tl.sum(active.to(tl.int32), axis=0)
        length = tl.min(tl.where(active, remaining, 0x7fffffff), axis=0)
        nrt = length * num_active
        hit = (slot < nrt) | (num_active == 0)
        denom = tl.maximum(num_active, 1)
        sir = slot % denom
        order = tl.cumsum(active.to(tl.int32), axis=0) - 1
        sel = active & (order == sir)
        src_rank = tl.where(hit, tl.sum(tl.where(sel, rank_v, 0), axis=0), src_rank)
        tir = tl.where(hit, offset + slot // denom, tir)
        slot = tl.where(hit, slot, slot - nrt)
        offset = tl.where(hit, offset, offset + length)
        remaining = tl.where(hit, remaining, remaining - tl.minimum(remaining, length))
        found = tl.where(hit, 1, 0)

    q_off = tl.where(live, (safe_e * NPES + src_rank) * MAX_RECV + tir, 0)
    stt = tl.load(q_l + q_off, mask=live, other=0)
    src_tok = stt // TOPK
    src_topk = stt % TOPK
    pt = tl.where(live, pool_blk * BLOCK_M + tok_in_e, 0)
    return (cur, e_start, e_end, pool_blk, done, live, tok_in_e, src_rank, stt, src_tok, src_topk, pt)


@triton.jit
def _select_d8_peer(src_rank, tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7):
    base = tp0
    base = tl.where(src_rank == 1, tp1, base)
    base = tl.where(src_rank == 2, tp2, base)
    base = tl.where(src_rank == 3, tp3, base)
    base = tl.where(src_rank == 4, tp4, base)
    base = tl.where(src_rank == 5, tp5, base)
    base = tl.where(src_rank == 6, tp6, base)
    base = tl.where(src_rank == 7, tp7, base)
    return base


@triton.jit
def dispatch_dual_d8_pull_scalar(
    sf_tab,
    w_tab,
    rsum_l,
    recv_l,
    q_l,
    meta_l,
    l1_arrival_count,
    l1_token,
    l1_sf,
    l1_w,
    d8_token_s,
    d8_state_s,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    R_POW2: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    NSF_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    offs_sf = tl.arange(0, NSF_POW2)
    sf_ok = offs_sf < NSF
    inactive = tl.full((), 0, tl.int32)
    d8_bytes = tl.full((), K, tl.int32)
    init_op = tl.full((), 0, tl.int32)
    load_op = tl.full((), 1, tl.int32)
    store_op = tl.full((), 2, tl.int32)
    fence_op = tl.full((), 3, tl.int32)
    stage0 = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (0, )))
    stage1 = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (K, )))
    state0 = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (0, )))
    state1 = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (2, )))
    init_addr = tl.cast(l1_token, tl.int64)
    tle_raw.call(d8_tma1d_edsl, [stage0, state0, init_addr, d8_bytes, inactive, init_op])
    tle_raw.call(d8_tma1d_edsl, [stage1, state1, init_addr, d8_bytes, inactive, init_op])
    dispatch_sync(USE_D8_TMA1D)
    tle_raw.call(d8_tma1d_edsl, [stage0, state0, init_addr, d8_bytes, inactive, fence_op])
    tle_raw.call(d8_tma1d_edsl, [stage1, state1, init_addr, d8_bytes, inactive, fence_op])
    dispatch_sync(USE_D8_TMA1D)

    token0 = sm_idx * 2
    token1 = sm_idx * 2 + 1
    cur0 = -1
    cur1 = -1
    start0 = 0
    start1 = 0
    end0 = 0
    end1 = 0
    pool0 = 0
    pool1 = 0
    done0 = 0
    done1 = 0
    while (done0 == 0) | (done1 == 0):
        (cur0, start0, end0, pool0, done0, live0, tie0, rank0, stt0, src_tok0, src_topk0,
         pt0) = _resolve_one_d8_route(token0, cur0, start0, end0, pool0, done0, rsum_l, recv_l, q_l, NPES, EPR, R_POW2,
                                      MAX_RECV, BLOCK_M, TOPK)
        (cur1, start1, end1, pool1, done1, live1, tie1, rank1, stt1, src_tok1, src_topk1,
         pt1) = _resolve_one_d8_route(token1, cur1, start1, end1, pool1, done1, rsum_l, recv_l, q_l, NPES, EPR, R_POW2,
                                      MAX_RECV, BLOCK_M, TOPK)

        base0 = _select_d8_peer(rank0, tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7)
        base1 = _select_d8_peer(rank1, tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7)
        peer0 = tl.multiple_of(base0.to(tl.pointer_type(tl.float8e4nv)), 16)
        peer1 = tl.multiple_of(base1.to(tl.pointer_type(tl.float8e4nv)), 16)
        src_addr0 = tl.cast(peer0 + src_tok0 * K, tl.int64)
        src_addr1 = tl.cast(peer1 + src_tok1 * K, tl.int64)
        active0 = tl.where(live0, 1, 0)
        active1 = tl.where(live1, 1, 0)

        # Both independent loads are in flight before either wait/store path.
        tle_raw.call(d8_tma1d_edsl, [stage0, state0, src_addr0, d8_bytes, active0, load_op])
        tle_raw.call(d8_tma1d_edsl, [stage1, state1, src_addr1, d8_bytes, active1, load_op])
        dispatch_sync(USE_D8_TMA1D)

        sfv0 = tl.load(peer_f32(sf_tab, rank0) + src_tok0 * NSF + offs_sf, mask=live0 & sf_ok, other=0.0)
        sfv1 = tl.load(peer_f32(sf_tab, rank1) + src_tok1 * NSF + offs_sf, mask=live1 & sf_ok, other=0.0)
        tl.store(l1_sf + offs_sf * POOL_TOKENS + pt0, sfv0, mask=live0 & sf_ok)
        tl.store(l1_sf + offs_sf * POOL_TOKENS + pt1, sfv1, mask=live1 & sf_ok)
        weight0 = tl.load(peer_f32(w_tab, rank0) + stt0, mask=live0, other=0.0)
        weight1 = tl.load(peer_f32(w_tab, rank1) + stt1, mask=live1, other=0.0)
        tl.store(l1_w + pt0, weight0, mask=live0)
        tl.store(l1_w + pt1, weight1, mask=live1)
        dispatch_sync(USE_D8_TMA1D)

        dst_addr0 = tl.cast(l1_token + pt0 * K, tl.int64)
        dst_addr1 = tl.cast(l1_token + pt1 * K, tl.int64)
        tle_raw.call(d8_tma1d_edsl, [stage0, state0, dst_addr0, d8_bytes, active0, store_op])
        tle_raw.call(d8_tma1d_edsl, [stage1, state1, dst_addr1, d8_bytes, active1, store_op])
        dispatch_sync(USE_D8_TMA1D)

        tl.store(meta_l + pt0 * 3 + 0, rank0, mask=live0)
        tl.store(meta_l + pt0 * 3 + 1, src_tok0, mask=live0)
        tl.store(meta_l + pt0 * 3 + 2, src_topk0, mask=live0)
        tl.store(meta_l + pt1 * 3 + 0, rank1, mask=live1)
        tl.store(meta_l + pt1 * 3 + 1, src_tok1, mask=live1)
        tl.store(meta_l + pt1 * 3 + 2, src_topk1, mask=live1)
        tl.atomic_add(l1_arrival_count + (pool0 + tie0 // BLOCK_M), 1, mask=live0, sem="release")
        tl.atomic_add(l1_arrival_count + (pool1 + tie1 // BLOCK_M), 1, mask=live1, sem="release")
        token0 = tl.where(live0, token0 + NUM_SMS * 2, token0)
        token1 = tl.where(live1, token1 + NUM_SMS * 2, token1)


@triton.jit
def dispatch_d8_pingpong_pull(
    sf_tab,
    w_tab,
    rsum_l,
    recv_l,
    q_l,
    meta_l,
    l1_arrival_count,
    l1_token,
    l1_sf,
    l1_w,
    d8_token_s,
    d8_state_s,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    R_POW2: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    NSF_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
):
    """One route scheduler feeding two independent TMA stage/mbarrier slots."""
    sm_idx = tl.program_id(0)
    offs_sf = tl.arange(0, NSF_POW2)
    sf_ok = offs_sf < NSF
    inactive = tl.full((), 0, tl.int32)
    d8_bytes = tl.full((), K, tl.int32)
    init_op = tl.full((), 0, tl.int32)
    load_op = tl.full((), 1, tl.int32)
    store_op = tl.full((), 2, tl.int32)
    fence_op = tl.full((), 3, tl.int32)
    stage0 = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (0, )))
    stage1 = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (K, )))
    state0 = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (0, )))
    state1 = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (2, )))
    init_addr = tl.cast(l1_token, tl.int64)
    tle_raw.call(d8_tma1d_edsl, [stage0, state0, init_addr, d8_bytes, inactive, init_op])
    tle_raw.call(d8_tma1d_edsl, [stage1, state1, init_addr, d8_bytes, inactive, init_op])
    dispatch_sync(USE_D8_TMA1D)
    tle_raw.call(d8_tma1d_edsl, [stage0, state0, init_addr, d8_bytes, inactive, fence_op])
    tle_raw.call(d8_tma1d_edsl, [stage1, state1, init_addr, d8_bytes, inactive, fence_op])
    dispatch_sync(USE_D8_TMA1D)

    token_idx = sm_idx
    cur = -1
    e_start = 0
    e_end = 0
    pool_blk = 0
    done = 0
    next_slot = 0
    pending = 0
    pending_stage = stage0
    pending_state = state0
    pending_dst = init_addr
    pending_pt = 0
    pending_rank = 0
    pending_src_tok = 0
    pending_src_topk = 0
    pending_pool = 0
    pending_tie = 0

    # The final iteration has live=0 and drains the last pending store.
    while (done == 0) | (pending != 0):
        (cur, e_start, e_end, pool_blk, done, live, tok_in_e, src_rank, stt, src_tok, src_topk,
         pt) = _resolve_one_d8_route(token_idx, cur, e_start, e_end, pool_blk, done, rsum_l, recv_l, q_l, NPES, EPR,
                                     R_POW2, MAX_RECV, BLOCK_M, TOPK)

        stage = tl.where(next_slot == 0, stage0, stage1)
        state = tl.where(next_slot == 0, state0, state1)
        peer_base = _select_d8_peer(src_rank, tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7)
        peer_tok = tl.multiple_of(peer_base.to(tl.pointer_type(tl.float8e4nv)), 16)
        src_addr = tl.cast(peer_tok + src_tok * K, tl.int64)
        active = tl.where(live, 1, 0)

        # Enqueue current slot before waiting/storing the previous slot.
        tle_raw.call(d8_tma1d_edsl, [stage, state, src_addr, d8_bytes, active, load_op])
        dispatch_sync(USE_D8_TMA1D)
        sfv = tl.load(peer_f32(sf_tab, src_rank) + src_tok * NSF + offs_sf, mask=live & sf_ok, other=0.0)
        tl.store(l1_sf + offs_sf * POOL_TOKENS + pt, sfv, mask=live & sf_ok)
        weight = tl.load(peer_f32(w_tab, src_rank) + stt, mask=live, other=0.0)
        tl.store(l1_w + pt, weight, mask=live)
        dispatch_sync(USE_D8_TMA1D)

        # Always execute the raw call and role barrier; `pending` is a PTX
        # predicate so there is no branch-scoped WS barrier/hang hazard.
        tle_raw.call(d8_tma1d_edsl, [pending_stage, pending_state, pending_dst, d8_bytes, pending, store_op])
        dispatch_sync(USE_D8_TMA1D)
        tl.store(meta_l + pending_pt * 3 + 0, pending_rank, mask=pending != 0)
        tl.store(meta_l + pending_pt * 3 + 1, pending_src_tok, mask=pending != 0)
        tl.store(meta_l + pending_pt * 3 + 2, pending_src_topk, mask=pending != 0)
        tl.atomic_add(l1_arrival_count + (pending_pool + pending_tie // BLOCK_M), 1, mask=pending != 0, sem="release")

        pending = active
        pending_stage = stage
        pending_state = state
        pending_dst = tl.cast(l1_token + pt * K, tl.int64)
        pending_pt = pt
        pending_rank = src_rank
        pending_src_tok = src_tok
        pending_src_topk = src_topk
        pending_pool = pool_blk
        pending_tie = tok_in_e
        next_slot = tl.where(live, next_slot ^ 1, next_slot)
        token_idx = tl.where(live, token_idx + NUM_SMS, token_idx)


# ---------------------------------------------------------------------------
# ROLE: dispatch (worker partition)
# ---------------------------------------------------------------------------
@triton.jit
def dispatch_role(
    topk_local,
    tok_tab,
    sf_tab,
    w_tab,
    q_tab,
    recv_tab,
    rsum_tab,
    sig_tab,
    meta_tab,
    sm_expert_count,
    expert_send_count,
    l1_arrival_count,
    gctr,
    meta_ready,
    l1_token,
    l1_sf,
    l1_w,
    d8_token_s,
    d8_state_s,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,  # v21: NPES peer token bases as kernel-arg ints
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    EPR_POW2: tl.constexpr,
    R_POW2: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    NEXP_POW2: tl.constexpr,
    K_POW2: tl.constexpr,
    NSF_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
    USE_SMEM_EXPERT_COUNT: tl.constexpr,
    FAST_NVLINK_BARRIER: tl.constexpr,
    D8_PULL_STREAMS: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    routes = NTOK * TOPK
    offs_e = tl.arange(0, NEXP_POW2)
    e_ok = offs_e < NEXP
    ec_row = sm_expert_count + sm_idx * NEXP

    if USE_SMEM_EXPERT_COUNT:
        # v12 parity: D1 and D3 share one CTA-local expert-count table, exactly
        # like CUDA's smem_expert_count + atomicAdd_block path.  Keep the
        # global table as the control path so this remains a one-switch A/B.
        ec_s = tle.gpu.alloc([NEXP_POW2], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        ec_vec = tle.gpu.local_ptr(ec_s, (offs_e, ))
        tl.store(ec_vec, tl.zeros((NEXP_POW2, ), dtype=tl.int32))
    else:
        tl.store(ec_row + offs_e, tl.zeros((NEXP_POW2, ), dtype=tl.int32), mask=e_ok)
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):  # D1 count
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        ok1 = m & (e >= 0)
        if USE_SMEM_EXPERT_COUNT:
            tl.atomic_add(tle.gpu.local_ptr(ec_s, (tl.where(ok1, e, 0), )), 1, mask=ok1, sem="relaxed", scope="cta")
        else:
            tl.atomic_add(ec_row + e, 1, mask=ok1, sem="relaxed")
    dispatch_sync(USE_D8_TMA1D)

    if USE_SMEM_EXPERT_COUNT:
        cnt = tl.load(ec_vec)
    else:
        cnt = tl.load(ec_row + offs_e, mask=e_ok, other=0)
    send_value = (tl.full((NEXP_POW2, ), 1, tl.int64) << 32) | cnt.to(tl.int64)
    old = tl.atomic_add(expert_send_count + offs_e, send_value, mask=e_ok, sem="relaxed")
    if USE_SMEM_EXPERT_COUNT:
        tl.store(ec_vec, (old & 0xffffffff).to(tl.int32))
    else:
        tl.store(ec_row + offs_e, (old & 0xffffffff).to(tl.int32), mask=e_ok)
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):  # D3 -> OWNER's queue
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        valid = m & (e >= 0)
        owner = tl.where(valid, e // EPR, 0)
        local_e = tl.where(valid, e % EPR, 0)
        if USE_SMEM_EXPERT_COUNT:
            d3_slot = tl.atomic_add(tle.gpu.local_ptr(ec_s, (tl.where(valid, e, 0), )), 1, mask=valid, sem="relaxed",
                                    scope="cta")
        else:
            d3_slot = tl.atomic_add(ec_row + e, 1, mask=valid, sem="relaxed")
        qb = peer_i32(q_tab, owner)  # VECTOR of peer bases
        tl.store(qb + (local_e * NPES + MY_PE) * MAX_RECV + d3_slot, r, mask=valid)
    dispatch_sync(USE_D8_TMA1D)

    dispatch_rank_barrier(gctr + 0, NUM_SMS, USE_D8_TMA1D)  # comm::grid_sync

    if sm_idx == 0:  # D4 -> every owner
        d4_owner = tl.where(e_ok, offs_e // EPR, 0)
        d4_le = tl.where(e_ok, offs_e % EPR, 0)
        status = tl.load(expert_send_count + offs_e, mask=e_ok, other=0)
        tl.store(peer_i64(recv_tab, d4_owner) + (MY_PE * EPR + d4_le), status & 0xffffffff, mask=e_ok)
        tl.atomic_add(peer_i64(rsum_tab, d4_owner) + d4_le, status, mask=e_ok, sem="relaxed", scope="sys")

    dispatch_nvlink_barrier(
        sig_tab,
        0,
        gctr + 1,
        sm_idx,
        MY_PE,
        NPES,
        NUM_SMS,
        USE_D8_TMA1D,
        FAST_NVLINK_BARRIER,
    )
    tl.atomic_add(meta_ready + sm_idx, 1, sem="release")

    # ---- D6..D9: pull tokens for MY experts from every source rank ----
    rank_v = tl.arange(0, R_POW2)
    rank_ok = rank_v < NPES
    offs_h = tl.arange(0, K_POW2)
    h_ok = offs_h < K
    offs_sf = tl.arange(0, NSF_POW2)
    sf_ok = offs_sf < NSF
    rsum_l = peer_i64(rsum_tab, MY_PE)
    recv_l = peer_i64(recv_tab, MY_PE)
    q_l = peer_i32(q_tab, MY_PE)
    meta_l = peer_i32(meta_tab, MY_PE)

    if D8_PULL_STREAMS == 2:
        dispatch_d8_pingpong_pull(
            sf_tab,
            w_tab,
            rsum_l,
            recv_l,
            q_l,
            meta_l,
            l1_arrival_count,
            l1_token,
            l1_sf,
            l1_w,
            d8_token_s,
            d8_state_s,
            tp0,
            tp1,
            tp2,
            tp3,
            tp4,
            tp5,
            tp6,
            tp7,
            MY_PE,
            NPES,
            NUM_SMS,
            EPR,
            R_POW2,
            MAX_RECV,
            BLOCK_M,
            K,
            NSF,
            POOL_TOKENS,
            TOPK,
            NSF_POW2,
            USE_D8_TMA1D,
        )

    # Keep the single-stream loop's symbols defined even when its runtime loop
    # is constexpr-disabled below; Triton's AST resolver visits that body
    # before dead-code elimination.
    inactive = tl.full((), 0, tl.int32)
    d8_bytes = tl.full((), K, tl.int32)
    d8_init_op = tl.full((), 0, tl.int32)
    d8_load_op = tl.full((), 1, tl.int32)
    d8_store_op = tl.full((), 2, tl.int32)
    d8_fence_op = tl.full((), 3, tl.int32)
    d8_stage_addr = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (0, )))
    d8_state_addr = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (0, )))
    d8_init_addr = tl.cast(l1_token, tl.int64)

    if (USE_D8_TMA1D > 0) & (D8_PULL_STREAMS == 1):
        # The raw owner initializes one mbarrier for the whole dispatch role.
        # TLE's role-scoped barrier has a 128-thread count, unlike the old
        # helper's CTA-wide __syncthreads which deadlocked under WS.
        tle_raw.call(
            d8_tma1d_edsl,
            [d8_stage_addr, d8_state_addr, d8_init_addr, d8_bytes, inactive, d8_init_op],
        )
        dispatch_sync(USE_D8_TMA1D)
        tle_raw.call(
            d8_tma1d_edsl,
            [d8_stage_addr, d8_state_addr, d8_init_addr, d8_bytes, inactive, d8_fence_op],
        )
        dispatch_sync(USE_D8_TMA1D)

    token_idx = sm_idx
    cur = -1
    e_start = 0
    e_end = 0
    pool_blk = 0
    done = D8_PULL_STREAMS - 1
    while done == 0:
        while (done == 0) & (token_idx >= e_end):  # D6 expert cursor
            cur += 1
            ex = cur >= EPR
            done = tl.where(ex, 1, 0)
            safe_cur = tl.where(ex, 0, cur)
            n = (tl.load(rsum_l + safe_cur) & 0xffffffff).to(tl.int32)
            add_blk = (e_end - e_start + BLOCK_M - 1) // BLOCK_M
            pool_blk = tl.where(ex, pool_blk, pool_blk + add_blk)
            e_start = tl.where(ex, e_start, e_end)
            e_end = tl.where(ex, e_end, e_end + n)

        live = done == 0
        safe_e = tl.where(live, cur, 0)
        tok_in_e = tl.where(live, token_idx - e_start, 0)

        remaining = tl.where(rank_ok & live,  # D7 round-robin
                             tl.load(recv_l + rank_v * EPR + safe_e, mask=rank_ok, other=0).to(tl.int32), 0)
        slot = tok_in_e
        offset = 0
        src_rank = 0
        tir = 0
        found = 0
        while found == 0:
            active = remaining > 0
            num_active = tl.sum(active.to(tl.int32), axis=0)
            length = tl.min(tl.where(active, remaining, 0x7fffffff), axis=0)
            nrt = length * num_active
            hit = (slot < nrt) | (num_active == 0)
            denom = tl.maximum(num_active, 1)
            sir = slot % denom
            order = tl.cumsum(active.to(tl.int32), axis=0) - 1
            sel = active & (order == sir)
            src_rank = tl.where(hit, tl.sum(tl.where(sel, rank_v, 0), axis=0), src_rank)
            tir = tl.where(hit, offset + slot // denom, tir)
            slot = tl.where(hit, slot, slot - nrt)
            offset = tl.where(hit, offset, offset + length)
            remaining = tl.where(hit, remaining, remaining - tl.minimum(remaining, length))
            found = tl.where(hit, 1, 0)

        q_off = tl.where(live, (safe_e * NPES + src_rank) * MAX_RECV + tir, 0)
        stt = tl.load(q_l + q_off, mask=live, other=0)
        src_tok = stt // TOPK
        src_topk = stt % TOPK
        pt = tl.where(live, pool_blk * BLOCK_M + tok_in_e, 0)

        # D8: REMOTE pull over NVLink (CUDA: tma_load_1d on the peer pointer)
        # v21: select the peer base among the NPES kernel-arg pointers (tp0..tp7, all
        # divisibility=16 via int-arg specialization). src_rank is a uniform scalar, so
        # the where-chain resolves to one div=16 base -> the fp8 row coalesces into wide
        # LDG (was per-byte LDG.E.U8 = 16x sectors when the base came from a table load).
        _tb = tp0
        _tb = tl.where(src_rank == 1, tp1, _tb)
        _tb = tl.where(src_rank == 2, tp2, _tb)
        _tb = tl.where(src_rank == 3, tp3, _tb)
        _tb = tl.where(src_rank == 4, tp4, _tb)
        _tb = tl.where(src_rank == 5, tp5, _tb)
        _tb = tl.where(src_rank == 6, tp6, _tb)
        _tb = tl.where(src_rank == 7, tp7, _tb)
        _ptok = tl.multiple_of(_tb.to(tl.pointer_type(tl.float8e4nv)), 16)
        src_token = _ptok + src_tok * K
        dst_token = l1_token + pt * K
        if USE_D8_TMA1D > 1:
            active = tl.where(live, 1, 0)
            src_token_addr = tl.cast(src_token, tl.int64)
            # BEGIN only enqueues peer GMEM -> SMEM. The TMA transfer overlaps
            # the unchanged ordinary SF/top-k traffic below.
            tle_raw.call(
                d8_tma1d_edsl,
                [d8_stage_addr, d8_state_addr, src_token_addr, d8_bytes, active, d8_load_op],
            )
            dispatch_sync(USE_D8_TMA1D)
        else:
            x = tl.load(src_token + offs_h, mask=live & h_ok)
            tl.store(dst_token + offs_h, x, mask=live & h_ok)
        sfv = tl.load(peer_f32(sf_tab, src_rank) + src_tok * NSF + offs_sf, mask=live & sf_ok, other=0.0)
        tl.store(l1_sf + offs_sf * POOL_TOKENS + pt, sfv, mask=live & sf_ok)
        w = tl.load(peer_f32(w_tab, src_rank) + stt, mask=live, other=0.0)
        tl.store(l1_w + pt, w, mask=live)
        if USE_D8_TMA1D > 1:
            dispatch_sync(USE_D8_TMA1D)
            # FINISH waits for the load, copies SMEM -> local L1, and waits for
            # the store before D9 metadata and release arrival become visible.
            dst_token_addr = tl.cast(dst_token, tl.int64)
            tle_raw.call(
                d8_tma1d_edsl,
                [d8_stage_addr, d8_state_addr, dst_token_addr, d8_bytes, active, d8_store_op],
            )
            dispatch_sync(USE_D8_TMA1D)

        tl.store(meta_l + pt * 3 + 0, src_rank, mask=live)  # D9
        tl.store(meta_l + pt * 3 + 1, src_tok, mask=live)
        tl.store(meta_l + pt * 3 + 2, src_topk, mask=live)
        tl.atomic_add(l1_arrival_count + (pool_blk + tok_in_e // BLOCK_M), 1, mask=live, sem="release")
        token_idx = tl.where(live, token_idx + NUM_SMS, token_idx)


# ---------------------------------------------------------------------------
# ROLE: independent A+SFA / B producers
# ---------------------------------------------------------------------------
@triton.jit
def loader_role(
    writer,
    rsum_tab,
    l1_arrival_count,
    l2_arrival_mask,
    a1_desc,
    b1_desc,
    a2_desc,
    b2_desc,
    sfa1_desc,
    sfa2_desc,
    loader_blocks,
    LOAD_A: tl.constexpr,
    MY_PE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    EPR_POW2: tl.constexpr,
    EPW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    L1_N: tl.constexpr,
    L2_N: tl.constexpr,
    NK1: tl.constexpr,
    NK2: tl.constexpr,
    NL1N: tl.constexpr,
    NL2N: tl.constexpr,
    RSUM_NPES: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    rsum_l = peer_i64(rsum_tab, MY_PE)

    # D4's high word is the scheduler-readiness contract.  B can schedule and
    # issue before dispatch has finished the later pre-pull barrier.
    expected = NUM_SMS * RSUM_NPES
    ready_e = 0
    while ready_e < EPR:
        status = tl.atomic_add(rsum_l + ready_e, 0, sem="acquire", scope="sys")
        while (status >> 32) != expected:
            status = tl.atomic_add(rsum_l + ready_e, 0, sem="acquire", scope="sys")
        ready_e += 1

    full_mask = ((1 << NL1N) - 1) | (((1 << NL1N) - 1) << 32)

    block_idx = sm_idx
    cur_e = 0
    pool_prefix = 0
    wave_pool_prefix = 0
    phase = 1
    ck = 0
    nblocks = 0
    stop = 0
    while stop == 0:
        got = 0
        while (got == 0) & (stop == 0):
            if cur_e >= EPR:
                stop = 1
            else:
                wave_end = ((cur_e // EPW) + 1) * EPW
                if phase == 1:
                    found = 0
                    while (found == 0) & (cur_e < wave_end):
                        expert_status = tl.atomic_add(rsum_l + cur_e, 0, sem="acquire", scope="sys")
                        expert_tokens = (expert_status & 0xffffffff).to(tl.int32)
                        num_m = (expert_tokens + BLOCK_M - 1) // BLOCK_M
                        if (block_idx // NL1N) < num_m:
                            found = 1
                        else:
                            block_idx -= num_m * NL1N
                            pool_prefix += num_m
                            cur_e += 1
                    if found == 1:
                        got = 1
                    else:
                        phase = 2
                        cur_e = ((cur_e - 1) // EPW) * EPW
                        pool_prefix = wave_pool_prefix
                else:
                    found = 0
                    while (found == 0) & (cur_e < wave_end):
                        expert_status = tl.atomic_add(rsum_l + cur_e, 0, sem="acquire", scope="sys")
                        expert_tokens = (expert_status & 0xffffffff).to(tl.int32)
                        num_m = (expert_tokens + BLOCK_M - 1) // BLOCK_M
                        if block_idx < num_m * NL2N:
                            found = 1
                        else:
                            block_idx -= num_m * NL2N
                            pool_prefix += num_m
                            cur_e += 1
                    if found == 1:
                        got = 1
                    else:
                        phase = 1
                        wave_pool_prefix = pool_prefix

        if stop == 0:
            nbn = NL1N if phase == 1 else NL2N
            m_block = block_idx // nbn
            n_block = block_idx - m_block * nbn
            block_idx += NUM_SMS
            expert_status = tl.atomic_add(rsum_l + cur_e, 0, sem="acquire", scope="sys")
            n_tok = (expert_status & 0xffffffff).to(tl.int32)
            pool_block = pool_prefix + m_block
            valid_m = tl.minimum(n_tok - m_block * BLOCK_M, BLOCK_M)

            if phase == 1:
                if LOAD_A:
                    l1_arrival = tl.atomic_add(l1_arrival_count + pool_block, 0, sem="acquire")
                    while l1_arrival != valid_m:
                        l1_arrival = tl.atomic_add(l1_arrival_count + pool_block, 0, sem="acquire")
                for kb in tl.range(0, NK1):
                    slot = writer.acquire(ck)
                    if LOAD_A:
                        tle.gpu.copy(a1_desc, slot.a, [BLOCK_M, BLOCK_K], [pool_block * BLOCK_M, kb * BLOCK_K])
                        # v220: L1 consumes one activation SF row per K=128.
                        tle.gpu.copy(sfa1_desc, slot.sfa.subslice(0, 1, 0), [1, BLOCK_M], [kb, pool_block * BLOCK_M])
                    else:
                        tle.gpu.copy(b1_desc, slot.b, [BLOCK_N, BLOCK_K],
                                     [cur_e * L1_N + n_block * BLOCK_N, kb * BLOCK_K])
                    writer.commit(ck)
                    ck += 1
            else:
                if LOAD_A:
                    l2_arrival = tl.atomic_add(l2_arrival_mask + pool_block, 0, sem="acquire")
                    while l2_arrival != full_mask:
                        l2_arrival = tl.atomic_add(l2_arrival_mask + pool_block, 0, sem="acquire")
                for kb in tl.range(0, NK2):
                    slot = writer.acquire(ck)
                    if LOAD_A:
                        tle.gpu.copy(a2_desc, slot.a, [BLOCK_M, BLOCK_K], [pool_block * BLOCK_M, kb * BLOCK_K])
                        tle.gpu.copy(sfa2_desc, slot.sfa, [2, BLOCK_M], [2 * kb, pool_block * BLOCK_M])
                    else:
                        tle.gpu.copy(b2_desc, slot.b, [BLOCK_N, BLOCK_K],
                                     [cur_e * L2_N + n_block * BLOCK_N, kb * BLOCK_K])
                    writer.commit(ck)
                    ck += 1
            nblocks += 1
    # Count one producer only so the host coverage check is unchanged.
    if LOAD_A:
        tl.store(loader_blocks + sm_idx, nblocks)


# ---------------------------------------------------------------------------
# ROLE: math (worker partition)
# ---------------------------------------------------------------------------
@triton.jit
def math_role(
    reader,
    cd_s,
    dst_rows_s,
    l1o_s,
    l1_st_desc,
    meta_ready,
    rsum_tab,
    cb_tab,
    sig_tab,
    meta_tab,
    l2out_tab,
    cb_local,
    l1_w,
    l2_acts,
    l2_sf,
    l2_arrival_mask,
    math_blocks,
    topk_local,
    final_y,
    gctr,
    w1_sf,
    w2_sf,
    WRITE_L2_OUT: tl.constexpr,
    USE_L2_TMA: tl.constexpr,
    WG: tl.constexpr,
    USE_L1_STORE_PIPE: tl.constexpr,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    EPR_POW2: tl.constexpr,
    EPW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INTER: tl.constexpr,
    L1_N: tl.constexpr,
    L2_N: tl.constexpr,
    NK1: tl.constexpr,
    NK2: tl.constexpr,
    NL1N: tl.constexpr,
    NL2N: tl.constexpr,
    L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    EARLY_RELEASE: tl.constexpr,
    PARTICIPANT_RELEASE: tl.constexpr,
    MATH_RSUM_EARLY: tl.constexpr = False,
    RSUM_NPES: tl.constexpr = 8,
    PREFETCH_SFB: tl.constexpr = False,
):
    sm_idx = tl.program_id(0)
    rsum_l = peer_i64(rsum_tab, MY_PE)
    if MATH_RSUM_EARLY:
        # v168: D4's high word is the real scheduler-readiness condition.  The
        # later per-CTA meta_ready flag is only published after the pre-pull
        # NVLink barrier.  Once all receive counts are final, math can build its
        # block schedule and sleep on reader.wait() until A/B are delivered.
        expected = NUM_SMS * RSUM_NPES
        ready_e = 0
        while ready_e < EPR:
            ready_status = tl.atomic_add(rsum_l + ready_e, 0, sem="acquire", scope="sys")
            while (ready_status >> 32) != expected:
                ready_status = tl.atomic_add(rsum_l + ready_e, 0, sem="acquire", scope="sys")
            ready_e += 1
    else:
        wait_flag(meta_ready + sm_idx)
    n_vec, blk_vec, off_vec = pool_layout(rsum_l, EPR, EPR_POW2, BLOCK_M)

    WG_M: tl.constexpr = BLOCK_M // 2
    row_base: tl.constexpr = WG * WG_M
    offs_m = tl.arange(0, WG_M)
    offs_n = tl.arange(0, BLOCK_N)
    # v7: gate/up are interleaved every 8 columns along N (see the L1 epilogue's
    # reshape to (BM, NPAIR, 2, 8)). Parity of (col // 8) selects which of the
    # two block-granular weight SFs applies -- mirrors CUDA's `(i & 1u)` on the
    # accumulator index (cuh:1016).
    _is_up = ((offs_n // 8) % 2) == 1
    meta_l = peer_i32(meta_tab, MY_PE)
    l2o_l = peer_f32(l2out_tab, MY_PE)

    block_idx = sm_idx
    cur_e = 0
    phase = 1
    ck = 0
    nblocks = 0
    stop = 0
    while stop == 0:
        got = 0
        while (got == 0) & (stop == 0):
            if cur_e >= EPR:
                stop = 1
            else:
                wave_end = ((cur_e // EPW) + 1) * EPW
                if phase == 1:
                    f = 0
                    while (f == 0) & (cur_e < wave_end):
                        num_m = pick(blk_vec, cur_e, EPR_POW2)
                        if (block_idx // NL1N) < num_m:
                            f = 1
                        else:
                            block_idx -= num_m * NL1N
                            cur_e += 1
                    if f == 1:
                        got = 1
                    else:
                        phase = 2
                        cur_e = ((cur_e - 1) // EPW) * EPW
                else:
                    f = 0
                    while (f == 0) & (cur_e < wave_end):
                        num_m = pick(blk_vec, cur_e, EPR_POW2)
                        if block_idx < num_m * NL2N:
                            f = 1
                        else:
                            block_idx -= num_m * NL2N
                            cur_e += 1
                    if f == 1:
                        got = 1
                    else:
                        phase = 1

        if stop == 0:
            nbn = NL1N if phase == 1 else NL2N
            m_block = block_idx // nbn
            n_block = block_idx - m_block * nbn
            block_idx += NUM_SMS
            n_tok = pick(n_vec, cur_e, EPR_POW2)
            pool_block = pick(off_vec, cur_e, EPR_POW2) + m_block
            valid_m = tl.minimum(n_tok - m_block * BLOCK_M, BLOCK_M)
            m_ok = (row_base + offs_m) < valid_m
            nk = NK1 if phase == 1 else NK2
            rows = pool_block * BLOCK_M + row_base + offs_m

            acc = tl.zeros((WG_M, BLOCK_N), dtype=tl.float32)

            # CUDA preloads k=0 before entering the pipe wait and issues k+1
            # after the current WGMMA, before register scaling.  That gives a
            # weight-SF LDG the current scaling work plus the next full-barrier
            # wait to mature.  The legacy path deliberately remains byte-for-
            # byte selectable for matched A/B experiments.
            if PREFETCH_SFB:
                if phase == 1:
                    _gn = n_block // 2
                    #! gate/up rows occupy the first/second NL1N/2 halves.
                    sfb_g_next = tl.load(w1_sf + (cur_e * NL1N + _gn) * NK1)
                    sfb_u_next = tl.load(w1_sf + (cur_e * NL1N + NL1N // 2 + _gn) * NK1)
                else:
                    sfb_g_next = tl.load(w2_sf + (cur_e * NL2N + n_block) * NK2)
                    sfb_u_next = sfb_g_next
            for kb in tl.range(0, nk):
                sl = reader.wait(ck).slot
                # Weight SF stays in global memory and is loaded once per
                # (BLOCK_N, BLOCK_K) tile.
                # v7: block-granular weight SF. L1 needs TWO scalars (gate/up)
                # because the gran-8 interleave puts both in one BLOCK_N tile;
                # L2 needs ONE. This replaces a BLOCK_N-wide gather + broadcast
                # multiply with 1-2 scalar loads (CUDA: 7.5% vs our 23.2%).
                if PREFETCH_SFB:
                    sfb_g = sfb_g_next
                    sfb_u = sfb_u_next
                else:
                    if phase == 1:
                        _gn = n_block // 2
                        sfb_g = tl.load(w1_sf + (cur_e * NL1N + _gn) * NK1 + kb)
                        sfb_u = tl.load(w1_sf + (cur_e * NL1N + NL1N // 2 + _gn) * NK1 + kb)
                    else:
                        sfb_g = tl.load(w2_sf + (cur_e * NL2N + n_block) * NK2 + kb)
                        sfb_u = sfb_g
                # DeepGEMM-style promotion: FP8 WGMMA into fp32, then apply the
                # per-block scales to the ACCUMULATOR. The activation SF is per-64
                # along K, so BLOCK_K=128 must be two separate K=64 dots (this is
                # exactly why CUDA issues two SFA TMAs for L2).
                # Plain fp8 dot -> ASYNC FP8 WGMMA (qgmma). CRITICAL: do NOT pass
                # max_num_imprecise_acc=0 -- that forces Triton off wgmma onto slow
                # SYNCHRONOUS fp16 mma.sync (proven in PTX: 0 wgmma / 128 mma.sync,
                # fp8->fp16 upconvert). Each dot is a single K<=128 tile scaled
                # separately into the fp32 acc, so wgmma's imprecise accumulation is
                # bounded to one tile: ~1e-4 rel error, 300x below fp8's own ~3.5e-2
                # quantization noise. Async wgmma is also what lets TMA overlap.
                # SF is per-64-K: k-quarters 0,1 -> sf_lo ; 2,3 -> sf_hi
                # v11: split the L1 and L2 paths. The lo/hi accumulator split
                # exists ONLY because L2's activation SF varies per-64-K -- you
                # cannot factor a k-varying scale out of a dot product, so K must
                # be cut at the SF boundary. L1's activation SF is per-row over the
                # WHOLE 128-K block (cuh:312 "L1 (BLOCK_M floats)"; cuh:967 the _hi
                # halves are "Only used in L2"), so for L1 all four K-quarters share
                # one scale: accumulate them into a SINGLE register set and scale
                # once. Halves L1's scaling work, and L1 has 2.7x more k-blocks
                # than L2 (NK1 = 4096/128 = 32 vs NK2 = 1536/128 = 12).
                # v18: ONE K=128 tile. L1 (per-row-128 SF) reads the whole tile in a
                # single wgmma. L2 (per-64-K SF) subslices K into two K=64 halves.
                if phase == 1:
                    #! WG0取前64行，WG1取后64行，共享同一个B tile；BLOCK_M=128，BLOCK_K=128，BLOCK_N=128
                    acc_lo = tle.gpu.wgmma(sl.a.subslice(row_base, WG_M, 0), sl.b, out_dtype=tl.float32, trans_b=True)
                    # v22: SFA is independent of the async accumulator. Load it
                    # while WGMMA is in flight, as the CUDA math warpgroup does.
                    #! L1 activation scale的粒度是K=128，整个WGMMA结果只需要一个scale
                    sf_lo = tl.reshape(tl.load(tle.gpu.local_ptr(sl.sfa.slot(0).subslice(row_base, WG_M, 0))), (WG_M, ))
                    sf_hi = sf_lo
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    acc_hi = acc_lo
                else:
                    # v205: consume the low K=64 fragment before issuing high.
                    HK: tl.constexpr = BLOCK_K // 2
                    a_my = sl.a.subslice(row_base, WG_M, 0)
                    acc_lo = tle.gpu.wgmma(a_my.subslice(0, HK, -1), sl.b.subslice(0, HK, -1), out_dtype=tl.float32,
                                           trans_b=True)
                    sf_lo = tl.reshape(tl.load(tle.gpu.local_ptr(sl.sfa.slot(0).subslice(row_base, WG_M, 0))), (WG_M, ))
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    w_lo_serial = tl.where(_is_up[None, :], (sf_lo * sfb_u)[:, None], (sf_lo * sfb_g)[:, None])
                    acc += acc_lo * w_lo_serial

                    acc_hi = tle.gpu.wgmma(a_my.subslice(HK, HK, -1), sl.b.subslice(HK, HK, -1), out_dtype=tl.float32,
                                           trans_b=True)
                    sf_hi = tl.reshape(tl.load(tle.gpu.local_ptr(sl.sfa.slot(1).subslice(row_base, WG_M, 0))), (WG_M, ))
                    acc_hi = tle.gpu.wgmma_wait(0, acc_hi)
                # UserHopper returns the stage immediately after
                # warpgroup_wait<0>(), before register-only weight-SF scaling.
                # This shortens stage ownership without changing any operand or
                # result.  The legacy path below remains the default for v33-v40.
                #! 提前释放reader，减少stage占用
                if EARLY_RELEASE:
                    if PARTICIPANT_RELEASE == 2:
                        reader.release(ck, warp_arrive=True)
                    elif PARTICIPANT_RELEASE:
                        reader.release(ck, participant_arrive=True)
                    else:
                        reader.release(ck)
                #! 如果PREFETCH_SFB=True，则提前读取下一个k-block的weight SF，减少等待时间
                if PREFETCH_SFB & (kb + 1 < nk):
                    if phase == 1:
                        _gn = n_block // 2
                        sfb_g_next = tl.load(w1_sf + (cur_e * NL1N + _gn) * NK1 + kb + 1)
                        sfb_u_next = tl.load(w1_sf + (cur_e * NL1N + NL1N // 2 + _gn) * NK1 + kb + 1)
                    else:
                        sfb_g_next = tl.load(w2_sf + (cur_e * NL2N + n_block) * NK2 + kb + 1)
                        sfb_u_next = sfb_g_next
                # v9: fold the activation SF into the weight SF FIRST, while both
                # are still cheap: sf_* is a [BLOCK_M] vector, sfb_* a true scalar,
                # so each product is BLOCK_M FMULs -- not BLOCK_M*BLOCK_N.
                #
                # v7 wrote  acc += (acc_lo*sf_lo + acc_hi*sf_hi) * _sfb  which the
                # compiler cannot reassociate (FP mul is not associative), so it
                # emitted three ELEMENTWISE multiplies per accumulator element.
                # SASS proof: v7's FMUL count was byte-identical to v5's
                # (40,181,760) even though v7 made sfb scalar -- the scalar never
                # got hoisted. CUDA does the same folding by hand and lands at
                # 4,374,528 FMUL (9.2x fewer), then one FFMA per element
                # (cuh:1016-1020: `sb = (i&1) ? up_sf : gate_sf;`
                #                 `final_accum[i] += scale_a * sb * accum[i];`).
                # v10: accumulate SEPARATELY. v9 wrote `acc += a*x + b*y`, and the
                # intermediate sum blocked fusion: SASS showed FFMA 75.8M -> 38.0M
                # with FADD 1.4M -> 39.1M (exactly offsetting) -- the compiler
                # UNFUSED into FMUL+FADD. Two independent accumulates give one
                # FFMA per element each, matching CUDA's per-element
                # `final_accum[i] += scale_a * sb * accum[i]` (cuh:1017-1020).
                #! 累加到最后的FP32 accumulator
                if phase == 1:
                    w_lo = tl.where(_is_up[None, :], (sf_lo * sfb_u)[:, None], (sf_lo * sfb_g)[:, None])
                    acc += acc_lo * w_lo  # single scale for all 128 K
                else:
                    # The low fragment was already consumed before high issue.
                    w_hi = tl.where(_is_up[None, :], (sf_hi * sfb_u)[:, None], (sf_hi * sfb_g)[:, None])
                    acc += acc_hi * w_hi
                if not EARLY_RELEASE:
                    if PARTICIPANT_RELEASE == 2:
                        reader.release(ck, warp_arrive=True)
                    elif PARTICIPANT_RELEASE:
                        reader.release(ck, participant_arrive=True)
                    else:
                        reader.release(ck)
                ck += 1

            # v200: WG1 still consumes every GEMM stage and publishes L1
            # readiness, but skips an epilogue with no rows of its own.
            do_epilogue = True
            if WG == 1:
                do_epilogue = valid_m > row_base

            if phase == 1:
                if do_epilogue:
                    # L1 EPILOGUE: SwiGLU on the granularity-8 interleave + UE8M0 + FP8
                    t4 = tl.reshape(acc, (WG_M, NPAIR, 2, 8))
                    t4 = tl.permute(t4, (0, 1, 3, 2))
                    gate, up = tl.split(t4)
                    gate = tl.reshape(gate, (WG_M, L1_OUT_N))
                    up = tl.reshape(up, (WG_M, L1_OUT_N))
                    # v189: match UserHopper's approximate FTZ reciprocal path.
                    sw = libdevice.fast_dividef(gate, 1.0 + tl.exp(-gate)) * up
                    weight = tl.load(l1_w + rows, mask=m_ok, other=0.0)
                    # v198: reduce the unweighted row, then fold top-k weight into
                    # the row scale instead of multiplying every output twice.
                    amax = tl.max(tl.abs(sw), axis=1) * tl.abs(weight)
                    scaled = amax * (1.0 / 448.0)
                    pos = scaled > 0.0
                    e = tl.ceil(tl.log2(tl.where(pos, scaled, 1.0)))
                    sf = tl.where(pos, tl.exp2(e), 1.0)
                    sf_inv = tl.where(pos, tl.exp2(-e), 1.0)
                    row_scale = weight * sf_inv
                    q = (sw * row_scale[:, None]).to(tl.float8e4nv)
                    cols_o = n_block * L1_OUT_N + tl.arange(0, L1_OUT_N)
                    # v32: async TMA store pipeline for L1 output (CUDA "store pipeline").
                    if USE_L1_STORE_PIPE:
                        tl.store(tle.gpu.local_ptr(l1o_s.slot(WG)), q, mask=m_ok[:, None])
                        tle.gpu.copy(l1o_s.slot(WG), l1_st_desc, [BLOCK_M // 2, L1_OUT_N],
                                     [pool_block * BLOCK_M + row_base, n_block * L1_OUT_N])
                    else:
                        tl.store(l2_acts + rows[:, None] * INTER + cols_o[None, :], q, mask=m_ok[:, None])
                    tl.store(l2_sf + n_block * POOL_TOKENS + rows, sf, mask=m_ok)
                # Both WGs publish readiness even when WG1 skipped the heavy
                # epilogue; the L2 producer waits for the complete bit mask.
                tl.debug_barrier()
                tl.atomic_or(l2_arrival_mask + pool_block, (tl.full((), 1, tl.int64) << (n_block + WG * 32)),
                             sem="release")
            else:
                if do_epilogue:
                    # L2 EPILOGUE: BF16 cast + NVLink scatter to the SOURCE rank
                    cols = n_block * BLOCK_N + offs_n
                    # v5: this fp32 local copy is TEST-ONLY (see module docstring) -- the
                    # kernel never reads it back; only the host-side d2h correctness check
                    # does. CUDA has no equivalent. Skip it when benchmarking: it is a full
                    # [BLOCK_M, BLOCK_N] fp32 store, i.e. 2x the bytes of the bf16 scatter.
                    if WRITE_L2_OUT:
                        tl.store(l2o_l + rows[:, None] * L2_N + cols[None, :], acc, mask=m_ok[:, None])
                    md = meta_l + rows * 3
                    dst_rank = tl.load(md + 0, mask=m_ok, other=0)
                    dst_tok = tl.load(md + 1, mask=m_ok, other=0)
                    dst_topk = tl.load(md + 2, mask=m_ok, other=0)
                    cbb = peer_bf16(cb_tab, dst_rank)  # VECTOR of peer bases
                    base = (dst_topk * NTOK + dst_tok) * K
                    # v23: hints cannot change physical ownership of a WGMMA fragment.
                    # Materialize a linear BF16 tile plus one fully resolved remote
                    # destination per row, then let the 4 physical math warps remap it
                    # as 16 lanes/row x 8 BF16/lane.
                    dst_row = tl.cast(cbb + base + n_block * BLOCK_N, tl.uint64)
                    tl.store(tle.gpu.local_ptr(dst_rows_s.slot(WG)), dst_row, mask=m_ok)
                    # v186: scatter consumes only valid_m_wg, so padding rows may
                    # be materialized without a masked SMEM read/modify/write.
                    tl.store(tle.gpu.local_ptr(cd_s.slot(WG)), acc.to(tl.bfloat16))
                    cd_addr = smem_generic_addr(tle.gpu.local_ptr(cd_s.slot(WG), (0, 0)))
                    dst_addr = smem_generic_addr(tle.gpu.local_ptr(dst_rows_s.slot(WG), (0, )))
                    valid_m_wg = tl.maximum(tl.minimum(valid_m - row_base, WG_M), 0)
                    bar_wg = tl.full((), 9 + WG, tl.int32)  # per-WG named barrier (9 / 10)
                    scatter_cols = tl.full((), BLOCK_N, tl.int32)
                    if USE_L2_TMA:
                        tle_raw.call(
                            l2_tma_scatter_edsl,
                            [cd_addr, dst_addr, valid_m_wg, scatter_cols, bar_wg],
                        )
                    else:
                        tle_raw.call(
                            l2_wide_scatter_edsl,
                            [cd_addr, dst_addr, valid_m_wg, scatter_cols, bar_wg],
                        )
            nblocks += 1
    tl.store(math_blocks + sm_idx, nblocks)

    # v31: only WG0 runs the cross-rank barrier + combine (per-token, tiny).
    if WG == 0:
        nvlink_barrier(sig_tab, 1, gctr + 3, sm_idx, MY_PE, NPES, NUM_SMS)
        cb_l = tl.multiple_of(cb_local.to(tl.pointer_type(tl.bfloat16)), 16)
        offs_hid = tl.arange(0, K_POW2)
        hid_ok = offs_hid < K
        t = sm_idx
        while t < NTOK:
            acc_c = tl.zeros((K_POW2, ), dtype=tl.float32)
            for k in tl.static_range(0, TOPK):
                e = tl.load(topk_local + t * TOPK + k).to(tl.int32)
                v = tl.load(cb_l + ((k * NTOK + t) * K) + offs_hid, mask=(e >= 0) & hid_ok, other=0.0)
                acc_c += v.to(tl.float32)
            tl.store(final_y + t * K + offs_hid, acc_c.to(tl.bfloat16), mask=hid_ok)
            t += NUM_SMS


# ---------------------------------------------------------------------------
@triton.jit
def dispatch_frontend_dual_pull(
    topk_local,
    q_tab,
    recv_tab,
    rsum_tab,
    sig_tab,
    sm_expert_count,
    expert_send_count,
    gctr,
    meta_ready,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    NEXP_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
    USE_SMEM_EXPERT_COUNT: tl.constexpr,
    FAST_NVLINK_BARRIER: tl.constexpr,
):
    """Two-warp D1-D5 frontend; publish finalized receive metadata."""
    sm_idx = tl.program_id(0)
    routes = NTOK * TOPK
    offs_e = tl.arange(0, NEXP_POW2)
    e_ok = offs_e < NEXP
    ec_row = sm_expert_count + sm_idx * NEXP

    if USE_SMEM_EXPERT_COUNT:
        ec_s = tle.gpu.alloc([NEXP_POW2], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        ec_vec = tle.gpu.local_ptr(ec_s, (offs_e, ))
        tl.store(ec_vec, tl.zeros((NEXP_POW2, ), dtype=tl.int32))
    else:
        tl.store(ec_row + offs_e, tl.zeros((NEXP_POW2, ), dtype=tl.int32), mask=e_ok)
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        valid = m & (e >= 0)
        if USE_SMEM_EXPERT_COUNT:
            tl.atomic_add(tle.gpu.local_ptr(ec_s, (tl.where(valid, e, 0), )), 1, mask=valid, sem="relaxed", scope="cta")
        else:
            tl.atomic_add(ec_row + e, 1, mask=valid, sem="relaxed")
    dispatch_sync(USE_D8_TMA1D)

    #! ec_vec 从smem中读取expert count数组
    if USE_SMEM_EXPERT_COUNT:
        cnt = tl.load(ec_vec)
    else:
        cnt = tl.load(ec_row + offs_e, mask=e_ok, other=0)
    #! 一个int64保存count和发送状态，high 32bit保存完成统计的CTA数量，low 32bit保存route数量
    send_value = (tl.full((NEXP_POW2, ), 1, tl.int64) << 32) | cnt.to(tl.int64)
    #! 把send_value统计到expert_send_count中，类似于workspace.get_expert_send_count。这里也得到了当前CTA要把route写在expert_send_count的起始位置
    old = tl.atomic_add(expert_send_count + offs_e, send_value, mask=e_ok, sem="relaxed")
    #! 最后保存回到ec_vec中
    if USE_SMEM_EXPERT_COUNT:
        tl.store(ec_vec, (old & 0xffffffff).to(tl.int32))
    else:
        tl.store(ec_row + offs_e, (old & 0xffffffff).to(tl.int32), mask=e_ok)
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        valid = m & (e >= 0)
        owner = tl.where(valid, e // EPR, 0)
        local_e = tl.where(valid, e % EPR, 0)
        #! d3_slot 对应 dst_slot_idx，通过atomic找到expert_count buffer中当前route对应的位置
        if USE_SMEM_EXPERT_COUNT:
            d3_slot = tl.atomic_add(tle.gpu.local_ptr(ec_s, (tl.where(valid, e, 0), )), 1, mask=valid, sem="relaxed",
                                    scope="cta")
        else:
            d3_slot = tl.atomic_add(ec_row + e, 1, mask=valid, sem="relaxed")
        qb = peer_i32(q_tab, owner)
        tl.store(qb + (local_e * NPES + MY_PE) * MAX_RECV + d3_slot, r, mask=valid)
    dispatch_sync(USE_D8_TMA1D)

    dispatch_rank_barrier(gctr + 0, NUM_SMS, USE_D8_TMA1D)

    if sm_idx == 0:
        #! 目标rank
        d4_owner = tl.where(e_ok, offs_e // EPR, 0)
        d4_le = tl.where(e_ok, offs_e % EPR, 0)
        #! 高32bit是完成统计的CTA数量，低32位是发送route总数
        status = tl.load(expert_send_count + offs_e, mask=e_ok, other=0)
        #! recv_tab[owner][MY_PE][local_expert] = route数，保存每个source rank分别发来多少route
        tl.store(peer_i64(recv_tab, d4_owner) + (MY_PE * EPR + d4_le), status & 0xffffffff, mask=e_ok)
        #! rsum_tab[local_expert] 每个 expert 从所有 rank 接收的route总数
        tl.atomic_add(peer_i64(rsum_tab, d4_owner) + d4_le, status, mask=e_ok, sem="relaxed", scope="sys")

    dispatch_nvlink_barrier(sig_tab, 0, gctr + 1, sm_idx, MY_PE, NPES, NUM_SMS, USE_D8_TMA1D, FAST_NVLINK_BARRIER)
    tl.atomic_add(meta_ready + sm_idx, 1, sem="release")


@triton.jit
def dispatch_role_dual_pull(
    topk_local,
    sf_tab,
    w_tab,
    q_tab,
    recv_tab,
    rsum_tab,
    sig_tab,
    meta_tab,
    sm_expert_count,
    expert_send_count,
    l1_arrival_count,
    gctr,
    meta_ready,
    l1_token,
    l1_sf,
    l1_w,
    d8_token0_s,
    d8_state0_s,
    d8_token1_s,
    d8_state1_s,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    NEXP_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
    USE_SMEM_EXPERT_COUNT: tl.constexpr,
    FAST_NVLINK_BARRIER: tl.constexpr,
):
    # D1-D5: both warps cooperate under the same role-scoped barriers.
    dispatch_frontend_dual_pull(
        topk_local,
        q_tab,
        recv_tab,
        rsum_tab,
        sig_tab,
        sm_expert_count,
        expert_send_count,
        gctr,
        meta_ready,
        MY_PE,
        NPES,
        NUM_SMS,
        NTOK,
        TOPK,
        NEXP,
        EPR,
        MAX_RECV,
        BLOCK_R,
        ITERS,
        NEXP_POW2,
        USE_D8_TMA1D,
        USE_SMEM_EXPERT_COUNT,
        FAST_NVLINK_BARRIER,
    )

    # D6-D9: the same two physical warps split into independent streams.
    # Pass integer-encoded addresses across the raw ABI; no addrspace pointer
    # is exposed to Triton's extern linker.
    #! 获取本rank的数据
    rsum_l = peer_i64(rsum_tab, MY_PE)
    recv_l = peer_i64(recv_tab, MY_PE)
    queue_l = peer_i32(q_tab, MY_PE)
    metadata_l = peer_i32(meta_tab, MY_PE)
    #! stage0/1 两个stream格子的FP8 token SMEM缓冲区；state0/1 各自的TMA mbarrier状态
    stage0 = smem_generic_addr(tle.gpu.local_ptr(d8_token0_s, (0, )))
    state0 = smem_generic_addr(tle.gpu.local_ptr(d8_state0_s, (0, )))
    stage1 = smem_generic_addr(tle.gpu.local_ptr(d8_token1_s, (0, )))
    state1 = smem_generic_addr(tle.gpu.local_ptr(d8_state1_s, (0, )))
    # Raw extern operands must be SSA values, not Python-side constexprs.
    npes_v = tl.full((), NPES, tl.int32)
    num_sms_v = tl.full((), NUM_SMS, tl.int32)
    epr_v = tl.full((), EPR, tl.int32)
    max_recv_v = tl.full((), MAX_RECV, tl.int32)
    block_m_v = tl.full((), BLOCK_M, tl.int32)
    k_v = tl.full((), K, tl.int32)
    nsf_v = tl.full((), NSF, tl.int32)
    pool_tokens_v = tl.full((), POOL_TOKENS, tl.int32)
    topk_v = tl.full((), TOPK, tl.int32)
    tle_raw.call(
        d8_unified_dispatch_edsl,
        [
            tl.cast(rsum_l, tl.int64),
            tl.cast(recv_l, tl.int64),
            tl.cast(queue_l, tl.int64),
            tl.cast(metadata_l, tl.int64),
            tl.cast(sf_tab, tl.int64),
            tl.cast(w_tab, tl.int64),
            tl.cast(l1_arrival_count, tl.int64),
            tl.cast(l1_token, tl.int64),
            tl.cast(l1_sf, tl.int64),
            tl.cast(l1_w, tl.int64),
            stage0,
            state0,
            stage1,
            state1,
            tp0,
            tp1,
            tp2,
            tp3,
            tp4,
            tp5,
            tp6,
            tp7,
            npes_v,
            num_sms_v,
            epr_v,
            max_recv_v,
            block_m_v,
            k_v,
            nsf_v,
            pool_tokens_v,
            topk_v,
        ],
    )


@triton.jit
def ws_megakernel(
    topk_local,
    tok_tab,
    sf_tab,
    w_tab,
    q_tab,
    recv_tab,
    rsum_tab,
    sig_tab,
    meta_tab,
    cb_tab,
    l2out_tab,
    cb_local,
    sm_expert_count,
    expert_send_count,
    l1_arrival_count,
    l2_arrival_mask,
    gctr,
    meta_ready,
    l1_token,
    l1_sf,
    l1_w,
    l2_acts,
    l2_sf,
    w1_sf,
    w2_sf,
    loader_blocks,
    math_blocks,
    final_y,
    a1_desc,
    b1_desc,
    a2_desc,
    b2_desc,
    sfa1_desc,
    sfa2_desc,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    l1_st_desc,
    WRITE_L2_OUT: tl.constexpr,
    USE_L2_TMA: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
    USE_L1_STORE_PIPE: tl.constexpr,
    USE_SMEM_EXPERT_COUNT: tl.constexpr,
    FAST_NVLINK_BARRIER: tl.constexpr,
    D8_PULL_STREAMS: tl.constexpr,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    EPR_POW2: tl.constexpr,
    R_POW2: tl.constexpr,
    EPW: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    INTER: tl.constexpr,
    L1_N: tl.constexpr,
    L2_N: tl.constexpr,
    NK1: tl.constexpr,
    NK2: tl.constexpr,
    NL1N: tl.constexpr,
    NL2N: tl.constexpr,
    L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    NEXP_POW2: tl.constexpr,
    K_POW2: tl.constexpr,
    NSF_POW2: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    STAGES: tl.constexpr,
    DISPATCH_WARPS: tl.constexpr,
    MATH_WARPS: tl.constexpr,
):
    # Reject direct launches that try to mutate this immutable snapshot.
    tl.static_assert(USE_L2_TMA == 0, "snapshot fixes USE_L2_TMA=False")
    tl.static_assert(USE_L1_STORE_PIPE == 0, "snapshot fixes USE_L1_STORE_PIPE=False")
    tl.static_assert(USE_D8_TMA1D == 2, "snapshot fixes full raw D8 TMA1D")
    tl.static_assert(USE_SMEM_EXPERT_COUNT, "snapshot fixes SMEM expert counts")
    tl.static_assert(FAST_NVLINK_BARRIER, "snapshot fixes fast NVLink barrier")
    tl.static_assert(D8_PULL_STREAMS == 2, "snapshot fixes two D8 pull streams")
    tl.static_assert(BLOCK_R == 256, "snapshot fixes BLOCK_R=256")
    tl.static_assert(DISPATCH_WARPS == 4, "snapshot fixes dispatch launch warps")
    tl.static_assert(MATH_WARPS == 4, "snapshot fixes math warps")

    a_s = tle.gpu.alloc([STAGES, BLOCK_M, BLOCK_K], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    b_s = tle.gpu.alloc([STAGES, BLOCK_N, BLOCK_K], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    sfa_s = tle.gpu.alloc([STAGES, 2, BLOCK_M], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    cd_s = tle.gpu.alloc([2, BLOCK_M // 2, BLOCK_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)
    dst_rows_s = tle.gpu.alloc([2, BLOCK_M // 2], dtype=tl.uint64, layout=None, scope=tle.gpu.smem,
                               nv_mma_shared_layout=False)
    l1o_s = tle.gpu.alloc([2, BLOCK_M // 2, L1_OUT_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)

    d8_token0_s = tle.gpu.alloc([K], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    d8_token1_s = tle.gpu.alloc([K], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    d8_state0_s = tle.gpu.alloc([1], dtype=tl.uint64, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    d8_state1_s = tle.gpu.alloc([1], dtype=tl.uint64, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)

    p = tle.pipe(capacity=STAGES, scope="cta", name="megamoe_gemm", readers=["m0", "m1"], a=a_s, b=b_s, sfa=sfa_s)

    tle.gpu.warp_specialize(
        [
            (math_role, (p.reader(name="m0"), cd_s, dst_rows_s, l1o_s, l1_st_desc, meta_ready, rsum_tab, cb_tab,
                         sig_tab, meta_tab, l2out_tab, cb_local, l1_w, l2_acts, l2_sf, l2_arrival_mask, math_blocks,
                         topk_local, final_y, gctr, w1_sf, w2_sf, WRITE_L2_OUT, USE_L2_TMA, 0, USE_L1_STORE_PIPE, MY_PE,
                         NPES, NUM_SMS, EPR, EPR_POW2, EPW, BLOCK_M, BLOCK_N, BLOCK_K, INTER, L1_N, L2_N, NK1, NK2,
                         NL1N, NL2N, L1_OUT_N, NPAIR, POOL_TOKENS, NTOK, TOPK, K, K_POW2, False, False)),
            (math_role, (p.reader(name="m1"), cd_s, dst_rows_s, l1o_s, l1_st_desc, meta_ready, rsum_tab, cb_tab,
                         sig_tab, meta_tab, l2out_tab, cb_local, l1_w, l2_acts, l2_sf, l2_arrival_mask, math_blocks,
                         topk_local, final_y, gctr, w1_sf, w2_sf, WRITE_L2_OUT, USE_L2_TMA, 1, USE_L1_STORE_PIPE, MY_PE,
                         NPES, NUM_SMS, EPR, EPR_POW2, EPW, BLOCK_M, BLOCK_N, BLOCK_K, INTER, L1_N, L2_N, NK1, NK2,
                         NL1N, NL2N, L1_OUT_N, NPAIR, POOL_TOKENS, NTOK, TOPK, K, K_POW2, False, False)),
            (dispatch_role_dual_pull,
             (topk_local, sf_tab, w_tab, q_tab, recv_tab, rsum_tab, sig_tab, meta_tab, sm_expert_count,
              expert_send_count, l1_arrival_count, gctr, meta_ready, l1_token, l1_sf, l1_w, d8_token0_s, d8_state0_s,
              d8_token1_s, d8_state1_s, tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7, MY_PE, NPES, NUM_SMS, NTOK, TOPK, NEXP,
              EPR, MAX_RECV, BLOCK_M, K, NSF, POOL_TOKENS, BLOCK_R, ITERS, NEXP_POW2, USE_D8_TMA1D,
              USE_SMEM_EXPERT_COUNT, FAST_NVLINK_BARRIER)),
            (loader_role, (p.writer(), rsum_tab, l1_arrival_count, l2_arrival_mask, a1_desc, b1_desc, a2_desc, b2_desc,
                           sfa1_desc, sfa2_desc, loader_blocks, True, MY_PE, NUM_SMS, EPR, EPR_POW2, EPW, BLOCK_M,
                           BLOCK_N, BLOCK_K, L1_N, L2_N, NK1, NK2, NL1N, NL2N, NPES)),
            (loader_role, (p.writer(), rsum_tab, l1_arrival_count, l2_arrival_mask, a1_desc, b1_desc, a2_desc, b2_desc,
                           sfa1_desc, sfa2_desc, loader_blocks, False, MY_PE, NUM_SMS, EPR, EPR_POW2, EPW, BLOCK_M,
                           BLOCK_N, BLOCK_K, L1_N, L2_N, NK1, NK2, NL1N, NL2N, NPES)),
        ],
        [MATH_WARPS, 2, 1, 1],
        [224, 48, 48, 48],
    )


def rr_select(counts_per_rank, slot):
    remaining = list(counts_per_rank)
    offset = 0
    while True:
        active = [i for i, v in enumerate(remaining) if v > 0]
        length = min(remaining[i] for i in active)
        na = len(active)
        nrt = length * na
        if slot < nrt:
            return active[slot % na], offset + slot // na
        slot -= nrt
        offset += length
        remaining = [v - min(v, length) for v in remaining]


def run_worker() -> int:
    os.environ["NVSHMEM_HOME"] = str(NVSHMEM_HOME)
    os.environ["LD_LIBRARY_PATH"] = f"{NVSHMEM_HOME / 'lib'}:" + os.environ.get("LD_LIBRARY_PATH", "")
    host_so = _compile_nvshmem_host_so(SUPPORT_DIR / "nvshmem_host.cu")
    lib = ctypes.CDLL(str(host_so))
    lib.nvshmem_init_wrapper.restype = None
    lib.nvshmem_team_mype_wrapper.restype = ctypes.c_int
    lib.nvshmem_mype_wrapper.restype = ctypes.c_int
    lib.nvshmem_npes_wrapper.restype = ctypes.c_int
    lib.nvshmem_alloc_bytes_wrapper.argtypes = [ctypes.c_size_t]
    lib.nvshmem_alloc_bytes_wrapper.restype = ctypes.c_void_p
    lib.nvshmem_ptr_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.nvshmem_ptr_wrapper.restype = ctypes.c_void_p
    lib.nvshmemx_barrier_wrapper.argtypes = [ctypes.c_void_p]
    lib.nvshmemx_barrier_wrapper.restype = None
    lib.nvshmem_finalize_wrapper.restype = None

    lib.nvshmem_init_wrapper()
    node_pe = lib.nvshmem_team_mype_wrapper()
    rank = lib.nvshmem_mype_wrapper()
    R = lib.nvshmem_npes_wrapper()
    torch.cuda.set_device(node_pe % torch.cuda.device_count())
    dev = "cuda"
    f8 = torch.float8_e4m3fn

    # ---- shape config (identical on every rank) ----
    NUM_SMS = int(os.environ.get("W_NUM_SMS", "0")) or \
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    NTOK = int(os.environ.get("W_NTOK", "128"))
    TOPK = int(os.environ.get("W_TOPK", "4"))
    NEXP = int(os.environ.get("W_NEXP", "16"))
    K = int(os.environ.get("W_K", "256"))
    INTER_ENV = int(os.environ.get("W_INTER", "128"))
    NL1N = INTER_ENV // 64
    BENCH = int(os.environ.get("W_BENCH", "0"))
    # Immutable adopted choices; environment variables cannot change version semantics.
    USE_L2_TMA = 0
    USE_L1_STORE_PIPE = 0
    USE_D8_TMA1D = 2
    USE_SMEM_EXPERT_COUNT = 1
    FAST_NVLINK_BARRIER = 1
    D8_PULL_STREAMS = 2
    BENCH_ITERS = int(os.environ.get("W_ITERS", "20"))
    BENCH_WARMUP = int(os.environ.get("W_WARMUP", "5"))
    BENCH_REDUCE = os.environ.get("W_BENCH_REDUCE", "median").lower()
    GPU_START_BARRIER = int(os.environ.get("W_GPU_START_BARRIER", "0"))
    assert BENCH_REDUCE in ("median", "mean"), BENCH_REDUCE
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128  # v31: BM128 tile, split across 2 math WGs
    STAGES = int(os.environ.get("W_STAGES", "2"))
    if USE_D8_TMA1D:
        assert 0 < K <= 4096 and K % 16 == 0, \
            f"raw D8 TMA1D requires 16-byte token rows with K<=4096, got K={K}"
    assert D8_PULL_STREAMS in (1, 2), D8_PULL_STREAMS
    assert D8_PULL_STREAMS == 1 or USE_D8_TMA1D > 1, \
        "two D8 pull streams require the full raw TMA1D load+store path"
    assert NEXP % R == 0, f"NEXP={NEXP} must be divisible by npes={R}"
    EPR = NEXP // R
    # v172: UserHopper's MoE-7 heuristic schedules eight experts per wave.
    EPW = min(8, EPR)
    L1_N = NL1N * BLOCK_N
    L1_OUT_N = BLOCK_N // 2
    INTER = NL1N * L1_OUT_N
    L2_N = K
    NL2N = L2_N // BLOCK_N
    NK1 = K // BLOCK_K
    NK2 = INTER // BLOCK_K
    NPAIR = BLOCK_N // 16
    NSF = K // 128
    MAX_RECV = 512
    # Dispatch route tile.  Keep the historical default for old versions, but
    # expose it so newer experiments can match the two-warp CUDA frontend
    # (one route per lane, i.e. 2 * 32 routes per CTA).
    BLOCK_R = 256
    assert BLOCK_R > 0 and (BLOCK_R & (BLOCK_R - 1)) == 0, BLOCK_R
    routes = NTOK * TOPK
    ITERS = (routes + NUM_SMS * BLOCK_R - 1) // (NUM_SMS * BLOCK_R)
    R_POW2 = 1 << (R - 1).bit_length()
    EPR_POW2 = 1 << (EPR - 1).bit_length()
    NEXP_POW2 = 1 << (NEXP - 1).bit_length()
    K_POW2 = 1 << (K - 1).bit_length()
    NSF_POW2 = 1 << (NSF - 1).bit_length()

    # ---- inputs and weights ----
    # When MEGAMOE_SHARED_DATA_DIR is set, CUDA and TLE consume byte-identical
    # Qwen3 activations, scales, routing and checkpoint FP8 expert weights.
    # The random path remains available for legacy regression
    # runs, but it is not a real-data comparison.
    DROP = float(os.environ.get("W_DROP", "0.1"))
    shared_data_dir = os.environ.get("MEGAMOE_SHARED_DATA_DIR", "").strip()
    data_label = "random"
    if shared_data_dir:
        if DROP != 0:
            raise ValueError("real Qwen3 shared data requires W_DROP=0")
        from qwen3_fp8_shared_data import (
            fp8_from_uint8,
            interleave_l1_gate_up_rows,
            load_qwen3_rank_data,
        )

        dataset = load_qwen3_rank_data(
            shared_data_dir,
            rank=rank,
            num_ranks=R,
            num_tokens=NTOK,
            hidden=K,
            intermediate=INTER,
            num_experts=NEXP,
            topk=TOPK,
            torch=torch,
        )
        shared, local = dataset["shared"], dataset["local"]
        topk_all = shared["topk_idx"].reshape(R, routes).contiguous()
        tok_all = fp8_from_uint8(shared["input_fp8_bits"], torch)
        sf_all = shared["input_scales"].contiguous()
        w_all = shared["topk_weights"].reshape(R, routes).contiguous()
        # Checkpoints store [all gate rows | all up rows].  SM90 WGMMA consumes
        # [gate 0:8 | up 0:8 | gate 8:16 | up 8:16 | ...].  This transform is
        # part of the kernel's host preprocessing contract, not a test adapter.
        w1_q = fp8_from_uint8(
            interleave_l1_gate_up_rows(local["l1_weight_fp8_bits"], torch),
            torch,
        ).to(dev)
        w1_sf = local["l1_weight_scale_inv"].to(dev).contiguous()
        w2_q = fp8_from_uint8(local["l2_weight_fp8_bits"], torch).to(dev)
        w2_sf = local["l2_weight_scale_inv"].to(dev).contiguous()
        data_label = (f"{dataset['manifest']['source_model']}@{dataset['manifest']['source_revision']}"
                      f":layer{dataset['manifest']['layer']}")
    else:
        g = torch.Generator(device="cpu").manual_seed(1234)
        # Routing must match the baselines we compare against. `bench_mega_moe_sm90.py`
        # (and the horizontal sweep) use topk-of-random-scores -> TOPK distinct experts.
        _scores = torch.randn(R * NTOK, NEXP, generator=g)
        topk_all = torch.topk(_scores, TOPK, dim=-1, largest=True, sorted=False).indices
        topk_all = topk_all.reshape(R, routes).contiguous()
        if DROP > 0:
            topk_all[torch.rand(R, routes, generator=g) < DROP] = -1
        tok_all = (torch.randn(R, NTOK, K, generator=g) * 0.5).to(f8)
        sf_all = (torch.rand(R, NTOK, NSF, generator=g) * 0.5 + 0.5).contiguous()
        w_all = (torch.rand(R, routes, generator=g) * 0.8 + 0.2).contiguous()
        gg = torch.Generator(device=dev).manual_seed(9000 + rank)
        from qwen3_fp8_shared_data import interleave_l1_gate_up_rows

        w1_q = interleave_l1_gate_up_rows(
            (torch.randn(EPR, L1_N, K, generator=gg, device=dev) * 0.4).to(f8),
            torch,
        )
        w1_sf = (torch.rand(EPR, NL1N, NK1, generator=gg, device=dev) * 0.4 + 0.3).contiguous()
        w2_q = (torch.randn(EPR, L2_N, INTER, generator=gg, device=dev) * 0.4).to(f8)
        w2_sf = (torch.rand(EPR, NL2N, NK2, generator=gg, device=dev) * 0.4 + 0.3).contiguous()
    if tuple(w1_q.shape) != (EPR, L1_N, K):
        raise ValueError(f"invalid W1 weight shape {tuple(w1_q.shape)}; expected {(EPR, L1_N, K)}")
    # Match the checkpoint/UserHopper W1 SF contract exactly. CUDA stores ONE
    # scalar per (128-row weight block, 128-K block); with the gate/up gran-8
    # interleave a BLOCK_N=128 tile spans 64 gate cols + 64 up cols that all
    # share their block's scalar.  The checkpoint layout is [E, NL1N, NK1]:
    # the first NL1N/2 rows are gate SF and the second NL1N/2 rows are up SF.
    # Therefore gate_sf_n=n_block/2 and up_sf_n=NL1N/2+n_block/2.
    if tuple(w1_sf.shape) != (EPR, NL1N, NK1):
        raise ValueError(f"invalid W1 SF shape {tuple(w1_sf.shape)}; expected {(EPR, NL1N, NK1)}")
    # v7: L2 is one scalar per (BLOCK_N, BLOCK_K) tile -- cuh:927.

    # ---- CPU model of the EP layout (same on every rank) ----
    tk = topk_all.tolist()
    q_ref = [[[[] for _ in range(R)] for _ in range(EPR)] for _ in range(R)]
    for src in range(R):
        for r_, e in enumerate(tk[src]):
            if e >= 0:
                q_ref[e // EPR][e % EPR][src].append(r_)
    counts = [[[len(q_ref[o][le][s]) for s in range(R)] for le in range(EPR)] for o in range(R)]
    n_le = [[sum(counts[o][le]) for le in range(EPR)] for o in range(R)]
    blocks = [[(n + BLOCK_M - 1) // BLOCK_M for n in n_le[o]] for o in range(R)]
    pool_off = [[sum(blocks[o][:le]) for le in range(EPR)] for o in range(R)]
    NPB = max(sum(blocks[o]) for o in range(R))
    POOL_TOKENS = max(NPB * BLOCK_M, BLOCK_M)
    EXPECT = sum(blocks[rank]) * (NL1N + NL2N)

    # ---- symmetric heap ----
    def sym(nbytes):
        p = lib.nvshmem_alloc_bytes_wrapper(int(nbytes))
        if not p:
            raise RuntimeError("nvshmem_malloc failed")
        return p

    tok_b = sym(NTOK * K)
    sf_b = sym(NTOK * NSF * 4)
    w_b = sym(routes * 4)
    q_b = sym(EPR * R * MAX_RECV * 4)
    recv_b = sym(R * EPR * 8)
    rsum_b = sym(EPR * 8)
    cb_b = sym(TOPK * NTOK * K * 2)
    sig_b = sym(16 * 4)
    meta_b = sym(POOL_TOKENS * 3 * 4)
    l2out_b = sym(POOL_TOKENS * L2_N * 4)

    def peers(base):
        vals = []
        for pe in range(R):
            p = base if pe == rank else lib.nvshmem_ptr_wrapper(ctypes.c_void_p(base), pe)
            if not p:
                raise RuntimeError(f"nvshmem_ptr NULL for pe={pe}; no NVLink P2P?")
            vals.append(int(p))
        return torch.tensor(vals, device=dev, dtype=torch.int64)

    tok_tab, sf_tab, w_tab = peers(tok_b), peers(sf_b), peers(w_b)
    q_tab, recv_tab, rsum_tab = peers(q_b), peers(recv_b), peers(rsum_b)
    cb_tab, sig_tab, meta_tab, l2out_tab = peers(cb_b), peers(sig_b), peers(meta_b), peers(l2out_b)

    BLK = 1024

    def zero(tab, nb):
        sym_zero[(triton.cdiv(nb, BLK), )](tab, rank, nb, BLOCK=BLK)

    def h2d(t, tab):
        b = t.contiguous().view(-1).view(torch.int8).to(dev)
        sym_h2d[(triton.cdiv(b.numel(), BLK), )](b, tab, rank, b.numel(), BLOCK=BLK)

    def d2h(tab, pe, nb):
        out = torch.empty(nb, device=dev, dtype=torch.int8)
        sym_d2h[(triton.cdiv(nb, BLK), )](tab, pe, out, nb, BLOCK=BLK)
        return out

    for tab, nb in ((q_tab, EPR * R * MAX_RECV * 4), (recv_tab, R * EPR * 8), (rsum_tab,
                                                                               EPR * 8), (cb_tab, TOPK * NTOK * K * 2),
                    (sig_tab, 16 * 4), (meta_tab, POOL_TOKENS * 3 * 4), (l2out_tab, POOL_TOKENS * L2_N * 4)):
        zero(tab, nb)
    h2d(tok_all[rank], tok_tab)
    h2d(sf_all[rank], sf_tab)
    h2d(w_all[rank], w_tab)
    torch.cuda.synchronize()
    lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()

    # ---- local tensors ----
    topk_local = topk_all[rank].to(dev)
    sm_ec = torch.full((NUM_SMS * NEXP, ), -1, device=dev, dtype=torch.int32)
    send = torch.zeros((NEXP, ), device=dev, dtype=torch.int64)
    arrival = torch.zeros((max(NPB, 1), ), device=dev, dtype=torch.int32)
    l2_mask = torch.zeros((max(NPB, 1), ), device=dev, dtype=torch.int64)
    gctr = torch.zeros((8, ), device=dev, dtype=torch.int32)
    meta_ready = torch.zeros((NUM_SMS, ), device=dev, dtype=torch.int32)
    l1_tok = torch.zeros((POOL_TOKENS, K), device=dev, dtype=f8)
    l1_sf = torch.zeros((NSF * POOL_TOKENS, ), device=dev, dtype=torch.float32)
    l1_w = torch.zeros((POOL_TOKENS, ), device=dev, dtype=torch.float32)
    l2_acts = torch.zeros((POOL_TOKENS, INTER), device=dev, dtype=f8)
    l2_sf = torch.zeros((NL1N * POOL_TOKENS, ), device=dev, dtype=torch.float32)
    final_y = torch.zeros((NTOK, K), device=dev, dtype=torch.bfloat16)
    loader_blocks = torch.zeros((NUM_SMS, ), device=dev, dtype=torch.int32)
    math_blocks = torch.zeros((NUM_SMS, ), device=dev, dtype=torch.int32)

    def _alloc_fn(size: int, align: int, stream):
        return torch.empty(size, device=dev, dtype=torch.int8)

    triton.set_allocator(_alloc_fn)

    a1_desc = TensorDescriptor(l1_tok, shape=[POOL_TOKENS, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    # v4: fp8 wgmma wants B N-major, which is w1_q's NATIVE layout -- no permute.
    w1_nk = w1_q.contiguous()  # [EPR, L1_N, K]  N-major
    b1_desc = TensorDescriptor(w1_nk.view(EPR * L1_N, K), shape=[EPR * L1_N, K], strides=[K, 1],
                               block_shape=[BLOCK_N, BLOCK_K])
    a2_desc = TensorDescriptor(l2_acts, shape=[POOL_TOKENS, INTER], strides=[INTER, 1], block_shape=[BLOCK_M, BLOCK_K])
    l1_st_desc = TensorDescriptor(l2_acts, shape=[POOL_TOKENS, INTER], strides=[INTER, 1],
                                  block_shape=[BLOCK_M // 2, BLOCK_N // 2])  # v32: L1 async TMA store
    w2_nk = w2_q.contiguous()  # [EPR, L2_N, INTER]  N-major
    b2_desc = TensorDescriptor(w2_nk.view(EPR * L2_N, INTER), shape=[EPR * L2_N, INTER], strides=[INTER, 1],
                               block_shape=[BLOCK_N, BLOCK_K])
    # SF is MN-major -> a block's SF tile is BLOCK_M contiguous floats: a real TMA box
    sfa1_desc = TensorDescriptor(l1_sf.view(NSF, POOL_TOKENS), shape=[NSF, POOL_TOKENS], strides=[POOL_TOKENS, 1],
                                 block_shape=[1, BLOCK_M])
    sfa2_desc = TensorDescriptor(l2_sf.view(NL1N, POOL_TOKENS), shape=[NL1N, POOL_TOKENS], strides=[POOL_TOKENS, 1],
                                 block_shape=[2, BLOCK_M])

    lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()

    kargs = dict(
        MY_PE=rank,
        NPES=R,
        NUM_SMS=NUM_SMS,
        NTOK=NTOK,
        TOPK=TOPK,
        NEXP=NEXP,
        EPR=EPR,
        EPR_POW2=EPR_POW2,
        R_POW2=R_POW2,
        EPW=EPW,
        MAX_RECV=MAX_RECV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K=K,
        INTER=INTER,
        L1_N=L1_N,
        L2_N=L2_N,
        NK1=NK1,
        NK2=NK2,
        NL1N=NL1N,
        NL2N=NL2N,
        L1_OUT_N=L1_OUT_N,
        NPAIR=NPAIR,
        NSF=NSF,
        POOL_TOKENS=POOL_TOKENS,
        NEXP_POW2=NEXP_POW2,
        K_POW2=K_POW2,
        NSF_POW2=NSF_POW2,
        BLOCK_R=BLOCK_R,
        ITERS=ITERS,
        STAGES=STAGES,
        WRITE_L2_OUT=(BENCH == 0),
        USE_L2_TMA=(USE_L2_TMA != 0),
        USE_L1_STORE_PIPE=(USE_L1_STORE_PIPE != 0),
        # Keep the integer diagnostic level: 0=live v23 control, 1=init-only,
        # 2=full D8 raw load+store. Booleanizing this silently dropped level 2.
        USE_D8_TMA1D=USE_D8_TMA1D,
        USE_SMEM_EXPERT_COUNT=(USE_SMEM_EXPERT_COUNT != 0),
        FAST_NVLINK_BARRIER=(FAST_NVLINK_BARRIER != 0),
        D8_PULL_STREAMS=D8_PULL_STREAMS,
        DISPATCH_WARPS=4,
        MATH_WARPS=4,
        num_warps=LOADER_WARPS,
    )
    kernel_arg_names = getattr(ws_megakernel, "arg_names", ())
    if "EARLY_PIPE_RELEASE" in kernel_arg_names:
        kargs["EARLY_PIPE_RELEASE"] = bool(getattr(EARLY_PIPE_RELEASE, "value", EARLY_PIPE_RELEASE))
    if "PARTICIPANT_PIPE_RELEASE" in kernel_arg_names:
        kargs["PARTICIPANT_PIPE_RELEASE"] = PARTICIPANT_PIPE_RELEASE
    if "LOADER_RSUM_EARLY" in kernel_arg_names:
        kargs["LOADER_RSUM_EARLY"] = LOADER_RSUM_EARLY
    # v21: NPES peer token-buffer bases as direct int kernel-args (div=16 specialized
    # -> wide coalesced LDG), padded to 8; the where-chain in dispatch picks by src_rank.
    _tp = (tok_tab.tolist() + [int(tok_tab[0])] * 8)[:8]
    if "sp0" in kernel_arg_names:
        _sp = (sf_tab.tolist() + [int(sf_tab[0])] * 8)[:8]
        _wp = (w_tab.tolist() + [int(w_tab[0])] * 8)[:8]
        kargs.update({f"sp{i}": int(_sp[i]) for i in range(8)})
        kargs.update({f"wp{i}": int(_wp[i]) for i in range(8)})
    pos = (topk_local, tok_tab, sf_tab, w_tab, q_tab, recv_tab, rsum_tab, sig_tab, meta_tab, cb_tab, l2out_tab, cb_b,
           sm_ec, send, arrival, l2_mask, gctr, meta_ready, l1_tok, l1_sf, l1_w, l2_acts, l2_sf, w1_sf, w2_sf,
           loader_blocks, math_blocks, final_y, a1_desc, b1_desc, a2_desc, b2_desc, sfa1_desc, sfa2_desc, *_tp,
           l1_st_desc)

    def reset():
        send.zero_()
        arrival.zero_()
        l2_mask.zero_()
        gctr.zero_()
        meta_ready.zero_()
        sm_ec.fill_(-1)
        zero(recv_tab, R * EPR * 8)
        zero(rsum_tab, EPR * 8)
        zero(sig_tab, 16 * 4)

    def barrier():
        torch.cuda.synchronize()
        lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()

    reset()
    barrier()
    try:
        compiled = ws_megakernel[(NUM_SMS, )](*pos, **kargs)
    except Exception as _e:
        c = _e
        while getattr(c, '__cause__', None) is not None:
            c = c.__cause__
        print(
            'V31_FULLERR>>> ' + repr(c)[:300] + ' FILENAME=' + str(getattr(c, 'filename', None)) + ' args=' +
            str(getattr(c, 'args', None)), flush=True)
        raise
    barrier()

    bench_us = None
    baseline_us = None
    baseline_kernel = globals().get("BENCH_BASELINE_KERNEL") if BENCH else None
    baseline_kargs = dict(kargs)
    if baseline_kernel is not None:
        baseline_kargs.update(globals().get("BENCH_BASELINE_KWARGS", {}))
        baseline_arg_names = set(getattr(baseline_kernel, "arg_names", ()))
        for _name in [*(f"sp{i}" for i in range(8)), *(f"wp{i}" for i in range(8)), "MATH_PREFETCH_SFB"]:
            if _name not in baseline_arg_names:
                baseline_kargs.pop(_name, None)
    if baseline_kernel is not None:
        reset()
        barrier()
        baseline_kernel[(NUM_SMS, )](*pos, **baseline_kargs)
        barrier()

    if BENCH:
        times = []
        baseline_times = []

        def timed_launch(kernel, launch_kargs):
            reset()
            barrier()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            # The blocking host-side barrier above makes state visible but lets
            # Python process scheduling skew the subsequent launches by hundreds
            # of microseconds.  An optional second barrier is enqueued on the GPU
            # stream; e0 and the kernel sit behind it, so all ranks begin the
            # measured interval only after every stream has arrived.
            if GPU_START_BARRIER:
                lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
            e0.record()
            kernel[(NUM_SMS, )](*pos, **launch_kargs)
            e1.record()
            torch.cuda.synchronize()
            elapsed = e0.elapsed_time(e1) * 1e3
            barrier()
            return elapsed

        for i in range(BENCH_WARMUP + BENCH_ITERS):
            if baseline_kernel is None:
                candidate_elapsed = timed_launch(ws_megakernel, kargs)
                baseline_elapsed = None
            elif i & 1:
                candidate_elapsed = timed_launch(ws_megakernel, kargs)
                baseline_elapsed = timed_launch(baseline_kernel, baseline_kargs)
            else:
                baseline_elapsed = timed_launch(baseline_kernel, baseline_kargs)
                candidate_elapsed = timed_launch(ws_megakernel, kargs)
            if i >= BENCH_WARMUP:
                times.append(candidate_elapsed)
                if baseline_elapsed is not None:
                    baseline_times.append(baseline_elapsed)
        if BENCH_REDUCE == "mean":
            bench_us = sum(times) / len(times)
            if baseline_times:
                baseline_us = sum(baseline_times) / len(baseline_times)
        else:
            times.sort()
            bench_us = times[len(times) // 2]
            if baseline_times:
                baseline_times.sort()
                baseline_us = baseline_times[len(baseline_times) // 2]

    has_ws = "ttg.warp_specialize" in compiled.asm.get("ttgir", "")
    n_tma = compiled.asm.get("ptx", "").count("cp.async.bulk.tensor")

    if BENCH:
        P = sum(n_le[rank])
        flops = 6.0 * P * K * INTER
        tflops = flops / (bench_us * 1e-6) / 1e12
        matched = ""
        if baseline_us is not None:
            speedup = baseline_us / bench_us
            matched = (f" paired_base={baseline_us:.1f}us "
                       f"speedup={speedup:.4f}x")
        print(
            f"[rank {rank}/{R}] BENCH h={K} ih={INTER} E={NEXP} k={TOPK} tokens={NTOK} "
            f"recv={P} experts={EPR} | {bench_us:8.1f} us  {tflops:6.1f} TFLOPS  "
            f"ws={has_ws} tma={n_tma} d8_tma1d_level={USE_D8_TMA1D} "
            f"smem_ec={USE_SMEM_EXPERT_COUNT} fast_nvl={FAST_NVLINK_BARRIER} "
            f"d8_streams={D8_PULL_STREAMS} reduce={BENCH_REDUCE} "
            f"gpu_start_barrier={GPU_START_BARRIER} "
            f"warmup={BENCH_WARMUP} iters={BENCH_ITERS} "
            f"data={data_label}{matched}", flush=True)
        lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
        lib.nvshmem_finalize_wrapper()
        return 0

    # ---- readback ----
    q_c = d2h(q_tab, rank, EPR * R * MAX_RECV * 4).cpu().view(torch.int32)
    recv_c = d2h(recv_tab, rank, R * EPR * 8).cpu().view(torch.int64)
    rsum_c = d2h(rsum_tab, rank, EPR * 8).cpu().view(torch.int64)
    cb_c = d2h(cb_tab, rank, TOPK * NTOK * K * 2).cpu().view(torch.bfloat16).view(TOPK, NTOK, K)
    peer_meta = [d2h(meta_tab, pe, POOL_TOKENS * 3 * 4).cpu().view(torch.int32).view(POOL_TOKENS, 3) for pe in range(R)]
    peer_l2out = [
        d2h(l2out_tab, pe, POOL_TOKENS * L2_N * 4).cpu().view(torch.float32).view(POOL_TOKENS, L2_N) for pe in range(R)
    ]
    torch.cuda.synchronize()

    inject = os.environ.get("W_INJECT", "")
    if inject == "queue":
        q_c[0] = (int(q_c[0]) + 1) % (NTOK * TOPK)  # valid-but-wrong route
    elif inject == "recv":
        recv_c[0] += 1
    elif inject == "meta":
        peer_meta[rank][0, 0] = (int(peer_meta[rank][0, 0]) + 1) % R
    errs = {}

    def bump(k):
        errs[k] = errs.get(k, 0) + 1

    # D3: my queue holds exactly what each source sent me
    for le in range(EPR):
        for s in range(R):
            base = (le * R + s) * MAX_RECV
            got = sorted(int(v) for v in q_c[base:base + counts[rank][le][s]].tolist())
            if got != sorted(q_ref[rank][le][s]):
                bump("queue")
    # D4
    for le in range(EPR):
        for s in range(R):
            if (int(recv_c[s * EPR + le]) & 0xffffffff) != counts[rank][le][s]:
                bump("recv_count")
        if (int(rsum_c[le]) & 0xffffffff) != n_le[rank][le]:
            bump("recv_sum_low")
        if ((int(rsum_c[le]) >> 32) & 0xffffffff) != NUM_SMS * R:
            bump("recv_sum_high")

    # pull: rr source selection, arrival, exact payload
    meta_me = peer_meta[rank]
    arr_c = arrival.cpu()
    l1_tok_c = l1_tok.cpu()
    l1_w_c = l1_w.cpu()
    n_valid = 0
    for le in range(EPR):
        for t in range(n_le[rank][le]):
            pt = pool_off[rank][le] * BLOCK_M + t
            n_valid += 1
            exp_src, exp_tir = rr_select(counts[rank][le], t)
            gs, gt, gk = (int(meta_me[pt, 0]), int(meta_me[pt, 1]), int(meta_me[pt, 2]))
            if gs != exp_src:
                bump("rr_rank")
                continue
            stt = int(q_c[(le * R + exp_src) * MAX_RECV + exp_tir])
            if stt // TOPK != gt or stt % TOPK != gk:
                bump("meta")
            if not torch.equal(l1_tok_c[pt], tok_all[gs, gt]):
                bump("pull_token")
            if abs(float(l1_w_c[pt]) - float(w_all[gs, stt])) > 0:
                bump("pull_weight")
        for b in range(blocks[rank][le]):
            if int(arr_c[pool_off[rank][le] + b]) != min(BLOCK_M, n_le[rank][le] - b * BLOCK_M):
                bump("l1_arrival")

    if int(loader_blocks.cpu().sum()) != EXPECT or int(math_blocks.cpu().sum()) != EXPECT:
        bump("block_coverage")

    # L2 GEMM against a reference built from MY OWN l2_acts / l2_sf
    deq = l2_acts.to(torch.float64).cpu()
    sfg = l2_sf.view(NL1N, POOL_TOKENS).to(torch.float64).cpu()
    for nb in range(NL1N):
        deq[:, nb * L1_OUT_N:(nb + 1) * L1_OUT_N] *= sfg[nb][:, None]
    W2loc = w2_q.to(torch.float64).cpu().clone()
    # v7: w2_sf is now [EPR, NL2N, NK2] -- one scalar per (BLOCK_N, BLOCK_K)
    # tile, so expand it over both the N block and the K block.
    w2sf_c = w2_sf.to(torch.float64).cpu()  # [EPR, NL2N, NK2]
    for nb in range(NL2N):
        for kb in range(NK2):
            W2loc[:, nb * BLOCK_N:(nb + 1) * BLOCK_N, kb * BLOCK_K:(kb + 1) * BLOCK_K] *= w2sf_c[:, nb, kb][:, None,
                                                                                                            None]
    l2o_me = peer_l2out[rank].to(torch.float64)
    l2_err = 0.0
    for le in range(EPR):
        idx = [pool_off[rank][le] * BLOCK_M + t for t in range(n_le[rank][le])]
        if not idx:
            continue
        ref = deq[idx] @ W2loc[le].T
        sc = ref.abs().max().item()
        l2_err = max(l2_err, (l2o_me[idx] - ref).abs().max().item() / max(sc, 1e-12))
    if l2_err > 3e-3:  # async fp8 wgmma ~1-2e-4; bar << fp8 quant noise (3.5e-2)
        bump("l2_gemm")

    if inject == "scatter":
        cb_c[0, 0, 0] = (cb_c[0, 0, 0].float() + 1.0).to(torch.bfloat16)
    elif inject == "combine":
        final_y[0, 0] = (final_y[0, 0].float() + 1.0).to(torch.bfloat16)

    # CROSS-RANK SCATTER, exact: each of my (k,t) partials must equal the owner's
    # l2_out at the pool row whose metadata says (src=me, tok=t, topk=k).
    owner_index = []
    for o in range(R):
        d = {}
        m = peer_meta[o]
        for le in range(EPR):
            for t in range(n_le[o][le]):
                pt = pool_off[o][le] * BLOCK_M + t
                d[(int(m[pt, 0]), int(m[pt, 1]), int(m[pt, 2]))] = pt
        owner_index.append(d)

    scatter_bad = 0
    n_partials = 0
    tk_me = topk_all[rank].view(NTOK, TOPK)
    for t in range(NTOK):
        for k in range(TOPK):
            e = int(tk_me[t, k])
            if e < 0:
                continue
            n_partials += 1
            o = e // EPR
            pt = owner_index[o].get((rank, t, k))
            if pt is None:
                scatter_bad += 1
                continue
            exp = peer_l2out[o][pt].to(torch.bfloat16)
            if not torch.equal(cb_c[k, t], exp):
                scatter_bad += 1
    if scatter_bad:
        bump("cross_rank_scatter")

    # combine, exact
    y_ref = torch.zeros(NTOK, K, dtype=torch.float32)
    for k in range(TOPK):
        sel = tk_me[:, k] >= 0
        y_ref[sel] += cb_c[k][sel].to(torch.float32)
    if int((final_y.cpu() != y_ref.to(torch.bfloat16)).sum()):
        bump("combine")

    ok = not errs and has_ws and n_tma > 0
    print(
        f"[rank {rank}/{R}] EPR={EPR} pool_tok={sum(n_le[rank])} blocks={sum(blocks[rank])} "
        f"gemm_blocks={EXPECT} ws={has_ws} tma={n_tma} l2_gemm={l2_err:.2e} "
        f"partials={n_partials} scatter_bad={scatter_bad} errors={errs or 0} "
        f"d8_tma1d_level={USE_D8_TMA1D} data={data_label} "
        f"-> {'PASS' if ok else 'FAIL'}", flush=True)

    lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    lib.nvshmem_finalize_wrapper()
    return 0 if ok else 1


def main() -> int:
    if "--worker" in sys.argv:
        return run_worker()
    env = os.environ.copy()
    # This parent imported Triton before MPI assigned a rank, so its cache path
    # necessarily ends in rank-0.  Do not leak that path to every worker:
    # each worker must recompute TRITON_CACHE_DIR from OMPI_COMM_WORLD_RANK at
    # import time or cold multi-rank compiles contend on one cache lock.
    env.pop("TRITON_CACHE_DIR", None)
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda-12.8")
    env.update({
        "NVSHMEM_HOME": str(NVSHMEM_HOME),
        "NVSHMEM_BOOTSTRAP": "MPI",
        "LD_LIBRARY_PATH": f"{NVSHMEM_HOME / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
        "CUDA_HOME": cuda_home,
        "CPATH": f"{cuda_home}/targets/x86_64-linux/include:" + env.get("CPATH", ""),
    })
    worker_python = str(TLE_PYTHON if TLE_PYTHON.exists() else Path(sys.executable))
    mpi_launcher = os.environ.get("MEGAMOE_MPIRUN", "/usr/bin/mpirun")
    cmd = [
        mpi_launcher, "--allow-run-as-root", "-np",
        str(NUM_RANKS), worker_python,
        str(Path(__file__).resolve()), "--worker"
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              timeout=int(os.environ.get("W_TIMEOUT", "600")), env=env)
    except subprocess.TimeoutExpired as exc:
        print("TIMEOUT\nstdout:", (exc.stdout or "")[-2500:])
        print("stderr:", (exc.stderr or "")[-2500:])
        return 1
    print(proc.stdout[-5000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-4000:])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
