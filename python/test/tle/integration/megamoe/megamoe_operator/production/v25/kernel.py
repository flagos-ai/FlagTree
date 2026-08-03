"""MegaMoE 3-role WS kernel on NVSHMEM ranks with a narrow raw L2 epilogue.

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

Known PERF gap (not semantic): CUDA pulls the remote token with `ptx::tma_load_1d`
on a peer pointer. Triton cannot build a TensorDescriptor on a peer pointer, so the
remote pull is a plain vectorized load. TMA is used for all LOCAL staging.

V25 EXPERIMENTAL SWITCH: `W_D8_TMA1D=1` replaces only that D8 FP8 token
load/store pair with the same descriptorless `cp.async.bulk` TMA1D path used by
UserHopper. Activation SF and top-k weight deliberately remain ordinary
load/store. TLE allocates one 4 KiB token stage plus mbarrier and retains the
route scheduler, role-scoped barriers, metadata, and release arrival; the raw
helper owns only the descriptorless TMA instructions and their local state.

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

Run:  python run_tle_megamoe_v3_ws_nvshmem_dist_tle.py     (2 ranks)
      MEGAMOE_NP=4 python ...
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

SUPPORT_DIR = Path(__file__).resolve().parent.parent
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
os.environ.setdefault("TRITON_CACHE_DIR",
                      str(cache_root / f"tle-megamoe-{cache_tag}-rank-{rank_env}"))

sys.path.insert(0, str(SUPPORT_DIR))
from production_runtime import NVSHMEM_HOME, _compile_nvshmem_host_so, _import_env  # noqa: E402

_env = _import_env()
torch, triton, tl, tle = _env["torch"], _env["triton"], _env["tl"], _env["tle"]
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: E402
from triton.experimental.tle.raw import dialect  # noqa: E402
import triton.experimental.tle.language.raw as tle_raw  # noqa: E402

# Current FlagTree raises the default raw-CUDA frontend requirement to clang-20.  These
# helpers are validated with clang-17 and use no clang-20-only language feature.
from triton.experimental.tle.raw.cuda import runtime as _tle_raw_cuda_runtime  # noqa: E402
_tle_raw_cuda_runtime._MIN_CLANG_MAJOR = 17
_tle_raw_cuda_runtime._resolve_clang.cache_clear()


HERE = Path(__file__).resolve().parent


@dialect(
    name="cuda",
    file=HERE / "raw_l2_wide_scatter.cu",
    extern_func_name="TleL2WideScatter",
)
def l2_wide_scatter_edsl(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=HERE / "raw_l2_tma_scatter.cu",
    extern_func_name="TleL2TmaScatter",
)
def l2_tma_scatter_edsl(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "raw_d8_tma1d_pull.cu",
    extern=HERE / "raw_d8_tma1d_pull_extern.py",
    extern_func_name="TleD8Tma1d",
)
def d8_tma1d_edsl(*args, **kwargs):
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
    _ = tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, 0;", "=r", [],
                                  dtype=tl.uint32, is_pure=False, pack=1)


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
def dispatch_rank_barrier(counter, NUM_SMS: tl.constexpr,
                          USE_D8_TMA1D: tl.constexpr):
    dispatch_sync(USE_D8_TMA1D)
    tl.atomic_add(counter, 1, sem="release")
    v = tl.atomic_add(counter, 0, sem="acquire")
    while v < NUM_SMS:
        v = tl.atomic_add(counter, 0, sem="acquire")
    dispatch_sync(USE_D8_TMA1D)


@triton.jit
def dispatch_nvlink_barrier(sig_tab, slot, ctr, sm_idx,
                            MY_PE: tl.constexpr, NPES: tl.constexpr,
                            NUM_SMS: tl.constexpr,
                            USE_D8_TMA1D: tl.constexpr):
    fence_sys()
    dispatch_rank_barrier(ctr, NUM_SMS, USE_D8_TMA1D)
    if sm_idx == 0:
        for pe in tl.static_range(0, NPES):
            tl.atomic_add(peer_i32(sig_tab, pe) + slot, 1, sem="release", scope="sys")
        sl = peer_i32(sig_tab, MY_PE)
        v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
        while v < NPES:
            v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
    dispatch_rank_barrier(ctr + 1, NUM_SMS, USE_D8_TMA1D)


@triton.jit
def nvlink_barrier(sig_tab, slot, ctr, sm_idx,
                   MY_PE: tl.constexpr, NPES: tl.constexpr, NUM_SMS: tl.constexpr):
    """comm::nvlink_barrier, barrier.cuh:38-80."""
    fence_sys()
    rank_barrier(ctr, NUM_SMS)
    if sm_idx == 0:
        for pe in tl.static_range(0, NPES):
            tl.atomic_add(peer_i32(sig_tab, pe) + slot, 1, sem="release", scope="sys")
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
    return tl.sum(tl.where(tl.arange(0, EPR_POW2) == idx, vec, 0), axis=0)


# ---------------------------------------------------------------------------
# symmetric-heap byte helpers (host plumbing only)
# ---------------------------------------------------------------------------
@triton.jit
def sym_zero(tab, PE, NB, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(peer_i8(tab, PE) + offs, tl.zeros((BLOCK,), dtype=tl.int8), mask=offs < NB)


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


# ---------------------------------------------------------------------------
# ROLE: dispatch (worker partition)
# ---------------------------------------------------------------------------
@triton.jit
def dispatch_role(
    topk_local, tok_tab, sf_tab, w_tab, q_tab, recv_tab, rsum_tab, sig_tab, meta_tab,
    sm_expert_count, expert_send_count, l1_arrival_count, gctr, meta_ready,
    l1_token, l1_sf, l1_w,
    d8_token_s, d8_state_s,
    tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7,   # v21: NPES peer token bases as kernel-arg ints
    MY_PE: tl.constexpr, NPES: tl.constexpr, NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr, TOPK: tl.constexpr, NEXP: tl.constexpr, EPR: tl.constexpr,
    EPR_POW2: tl.constexpr, R_POW2: tl.constexpr, MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr, K: tl.constexpr, NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr, BLOCK_R: tl.constexpr, ITERS: tl.constexpr,
    NEXP_POW2: tl.constexpr, K_POW2: tl.constexpr, NSF_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    routes = NTOK * TOPK
    offs_e = tl.arange(0, NEXP_POW2)
    e_ok = offs_e < NEXP
    ec_row = sm_expert_count + sm_idx * NEXP

    tl.store(ec_row + offs_e, tl.zeros((NEXP_POW2,), dtype=tl.int32), mask=e_ok)
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):                        # D1 count
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        tl.atomic_add(ec_row + e, 1, mask=m & (e >= 0), sem="relaxed")
    dispatch_sync(USE_D8_TMA1D)

    cnt = tl.load(ec_row + offs_e, mask=e_ok, other=0)          # D2 packed-u64 stake
    send_value = (tl.full((NEXP_POW2,), 1, tl.int64) << 32) | cnt.to(tl.int64)
    old = tl.atomic_add(expert_send_count + offs_e, send_value, mask=e_ok, sem="relaxed")
    tl.store(ec_row + offs_e, (old & 0xffffffff).to(tl.int32), mask=e_ok)
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):                        # D3 -> OWNER's queue
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        valid = m & (e >= 0)
        owner = tl.where(valid, e // EPR, 0)
        local_e = tl.where(valid, e % EPR, 0)
        d3_slot = tl.atomic_add(ec_row + e, 1, mask=valid, sem="relaxed")
        qb = peer_i32(q_tab, owner)                             # VECTOR of peer bases
        tl.store(qb + (local_e * NPES + MY_PE) * MAX_RECV + d3_slot, r, mask=valid)
    dispatch_sync(USE_D8_TMA1D)

    dispatch_rank_barrier(gctr + 0, NUM_SMS, USE_D8_TMA1D)     # comm::grid_sync

    if sm_idx == 0:                                             # D4 -> every owner
        d4_owner = tl.where(e_ok, offs_e // EPR, 0)
        d4_le = tl.where(e_ok, offs_e % EPR, 0)
        status = tl.load(expert_send_count + offs_e, mask=e_ok, other=0)
        tl.store(peer_i64(recv_tab, d4_owner) + (MY_PE * EPR + d4_le), status & 0xffffffff, mask=e_ok)
        tl.atomic_add(peer_i64(rsum_tab, d4_owner) + d4_le, status,
                      mask=e_ok, sem="relaxed", scope="sys")

    dispatch_nvlink_barrier(
        sig_tab, 0, gctr + 1, sm_idx, MY_PE, NPES, NUM_SMS, USE_D8_TMA1D,
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

    if USE_D8_TMA1D > 0:
        # The raw owner initializes one mbarrier for the whole dispatch role.
        # TLE's role-scoped barrier has a 128-thread count, unlike the old
        # helper's CTA-wide __syncthreads which deadlocked under WS.
        inactive = tl.full((), 0, tl.int32)
        d8_bytes = tl.full((), K, tl.int32)
        d8_init_op = tl.full((), 0, tl.int32)
        d8_load_op = tl.full((), 1, tl.int32)
        d8_store_op = tl.full((), 2, tl.int32)
        d8_fence_op = tl.full((), 3, tl.int32)
        d8_stage_addr = smem_generic_addr(tle.gpu.local_ptr(d8_token_s, (0,)))
        d8_state_addr = smem_generic_addr(tle.gpu.local_ptr(d8_state_s, (0,)))
        d8_init_addr = tl.cast(l1_token, tl.int64)
        tle_raw.call(
            d8_tma1d_edsl,
            [d8_stage_addr, d8_state_addr, d8_init_addr,
             d8_bytes, inactive, d8_init_op],
        )
        dispatch_sync(USE_D8_TMA1D)
        tle_raw.call(
            d8_tma1d_edsl,
            [d8_stage_addr, d8_state_addr, d8_init_addr,
             d8_bytes, inactive, d8_fence_op],
        )
        dispatch_sync(USE_D8_TMA1D)

    token_idx = sm_idx
    cur = -1
    e_start = 0
    e_end = 0
    pool_blk = 0
    done = 0
    while done == 0:
        while (done == 0) & (token_idx >= e_end):               # D6 expert cursor
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

        remaining = tl.where(rank_ok & live,                     # D7 round-robin
                             tl.load(recv_l + rank_v * EPR + safe_e, mask=rank_ok,
                                     other=0).to(tl.int32), 0)
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
                [d8_stage_addr, d8_state_addr, src_token_addr,
                 d8_bytes, active, d8_load_op],
            )
            dispatch_sync(USE_D8_TMA1D)
        else:
            x = tl.load(src_token + offs_h, mask=live & h_ok)
            tl.store(dst_token + offs_h, x, mask=live & h_ok)
        sfv = tl.load(peer_f32(sf_tab, src_rank) + src_tok * NSF + offs_sf,
                      mask=live & sf_ok, other=0.0)
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
                [d8_stage_addr, d8_state_addr, dst_token_addr,
                 d8_bytes, active, d8_store_op],
            )
            dispatch_sync(USE_D8_TMA1D)

        tl.store(meta_l + pt * 3 + 0, src_rank, mask=live)       # D9
        tl.store(meta_l + pt * 3 + 1, src_tok, mask=live)
        tl.store(meta_l + pt * 3 + 2, src_topk, mask=live)
        tl.atomic_add(l1_arrival_count + (pool_blk + tok_in_e // BLOCK_M), 1,
                      mask=live, sem="release")
        token_idx = tl.where(live, token_idx + NUM_SMS, token_idx)


# ---------------------------------------------------------------------------
# ROLE: loader (default partition)
# ---------------------------------------------------------------------------
@triton.jit
def loader_role(
    writer, meta_ready, rsum_tab, l1_arrival_count, l2_arrival_mask,
    a1_desc, b1_desc, a2_desc, b2_desc, sfa1_desc, sfa2_desc, loader_blocks,
    MY_PE: tl.constexpr, NUM_SMS: tl.constexpr, EPR: tl.constexpr,
    EPR_POW2: tl.constexpr, EPW: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, L1_N: tl.constexpr,
    L2_N: tl.constexpr, NK1: tl.constexpr, NK2: tl.constexpr,
    NL1N: tl.constexpr, NL2N: tl.constexpr, POOL_TOKENS: tl.constexpr,
    K1: tl.constexpr, K2: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    wait_flag(meta_ready + sm_idx)
    n_vec, blk_vec, off_vec = pool_layout(peer_i64(rsum_tab, MY_PE), EPR, EPR_POW2, BLOCK_M)

    full_mask = (1 << NL1N) - 1

    block_idx = sm_idx
    cur_e = 0
    phase = 1
    ck = 0
    nblocks = 0
    stop = 0
    while stop == 0:
        got = 0
        while (got == 0) & (stop == 0):                          # scheduler.get_next_block
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

            if phase == 1:
                a = tl.atomic_add(l1_arrival_count + pool_block, 0, sem="acquire")
                while a != valid_m:
                    a = tl.atomic_add(l1_arrival_count + pool_block, 0, sem="acquire")
                for kb in tl.range(0, NK1):
                    slot = writer.acquire(ck)                    # empty_barrier.wait
                    tle.gpu.copy(a1_desc, slot.a, [BLOCK_M, BLOCK_K],
                                 [pool_block * BLOCK_M, kb * BLOCK_K])
                    tle.gpu.copy(b1_desc, slot.b, [BLOCK_N, BLOCK_K],
                                 [cur_e * L1_N + n_block * BLOCK_N, kb * BLOCK_K])
                    # per-128 SF: one k-group covers the whole BLOCK_K -> both halves
                    # v7: L1's activation SF is per-row over the WHOLE 128-K block
                    # (cuh:312 "L1 (BLOCK_M floats)"; cuh:967 the _hi halves are
                    # "Only used in L2"). Both TMAs must stay -- every pipe field
                    # needs a commit on every path or TritonTleLowerPipeToNvws
                    # fails. The saving is on the MATH side: L1 reads sfa_lo once
                    # and reuses it, dropping one SMEM read and one multiply-add.
                    # v19: one TMA fetches rows [kb, kb+1]; L1 uses slot(0)=kb only
                    # (per-128 SF), slot(1) is loaded-but-unused (kept for one commit).
                    tle.gpu.copy(sfa1_desc, slot.sfa, [2, BLOCK_M],
                                 [kb, pool_block * BLOCK_M])
                    writer.commit(ck)                            # arrive_and_expect_tx
                    ck += 1
            else:
                am = tl.atomic_add(l2_arrival_mask + pool_block, 0, sem="acquire")
                while am != full_mask:
                    am = tl.atomic_add(l2_arrival_mask + pool_block, 0, sem="acquire")
                for kb in tl.range(0, NK2):
                    slot = writer.acquire(ck)
                    tle.gpu.copy(a2_desc, slot.a, [BLOCK_M, BLOCK_K],
                                 [pool_block * BLOCK_M, kb * BLOCK_K])
                    tle.gpu.copy(b2_desc, slot.b, [BLOCK_N, BLOCK_K],
                                 [cur_e * L2_N + n_block * BLOCK_N, kb * BLOCK_K])
                    # per-64 SF: BLOCK_K=128 spans TWO adjacent k-groups (sm90:794-805)
                    # v19: one TMA fetches both rows [2kb, 2kb+1] = slot(0)/slot(1).
                    tle.gpu.copy(sfa2_desc, slot.sfa, [2, BLOCK_M],
                                 [2 * kb, pool_block * BLOCK_M])
                    writer.commit(ck)
                    ck += 1
            nblocks += 1
    tl.store(loader_blocks + sm_idx, nblocks)


# ---------------------------------------------------------------------------
# ROLE: math (worker partition)
# ---------------------------------------------------------------------------
@triton.jit
def math_role(
    reader, cd_s, dst_rows_s,
    meta_ready, rsum_tab, cb_tab, sig_tab, meta_tab, l2out_tab, cb_local,
    l1_w, l2_acts, l2_sf, l2_arrival_mask, math_blocks, topk_local, final_y, gctr,
    w1_sf, w2_sf,
    WRITE_L2_OUT: tl.constexpr, USE_L2_TMA: tl.constexpr,
    MY_PE: tl.constexpr, NPES: tl.constexpr, NUM_SMS: tl.constexpr,
    EPR: tl.constexpr, EPR_POW2: tl.constexpr, EPW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INTER: tl.constexpr, L1_N: tl.constexpr, L2_N: tl.constexpr,
    NK1: tl.constexpr, NK2: tl.constexpr,
    NL1N: tl.constexpr, NL2N: tl.constexpr, L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr, POOL_TOKENS: tl.constexpr,
    NTOK: tl.constexpr, TOPK: tl.constexpr, K: tl.constexpr, K_POW2: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    wait_flag(meta_ready + sm_idx)
    n_vec, blk_vec, off_vec = pool_layout(peer_i64(rsum_tab, MY_PE), EPR, EPR_POW2, BLOCK_M)

    offs_m = tl.arange(0, BLOCK_M)
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
            m_ok = offs_m < valid_m
            nk = NK1 if phase == 1 else NK2
            rows = pool_block * BLOCK_M + offs_m

            rows_n = n_block * BLOCK_N + offs_n
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kb in tl.range(0, nk):
                sl = reader.wait(ck).slot
                # weight SF straight from global, now K-MAJOR [EPR, NK, N] so this
                # gather over rows_n is CONTIGUOUS (coalesced). Per-row values are
                # identical to before; only the index order changed.
                # v7: block-granular weight SF. L1 needs TWO scalars (gate/up)
                # because the gran-8 interleave puts both in one BLOCK_N tile;
                # L2 needs ONE. This replaces a BLOCK_N-wide gather + broadcast
                # multiply with 1-2 scalar loads (CUDA: 7.5% vs our 23.2%).
                if phase == 1:
                    _gn = n_block // 2
                    sfb_g = tl.load(w1_sf + (cur_e * 2 * NL1N + _gn) * NK1 + kb)
                    sfb_u = tl.load(w1_sf + (cur_e * 2 * NL1N + NL1N + _gn) * NK1 + kb)
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
                    acc_lo = tle.gpu.wgmma(sl.a, sl.b, out_dtype=tl.float32, trans_b=True)
                    # v22: SFA is independent of the async accumulator. Load it
                    # while WGMMA is in flight, as the CUDA math warpgroup does.
                    sf_lo = tl.reshape(
                        tl.load(tle.gpu.local_ptr(sl.sfa.slot(0))), (BLOCK_M,))
                    sf_hi = sf_lo
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    acc_hi = acc_lo
                else:
                    HK: tl.constexpr = BLOCK_K // 2
                    acc_lo = tle.gpu.wgmma(sl.a.subslice(0, HK, -1), sl.b.subslice(0, HK, -1),
                                           out_dtype=tl.float32, trans_b=True)
                    acc_hi = tle.gpu.wgmma(sl.a.subslice(HK, HK, -1), sl.b.subslice(HK, HK, -1),
                                           out_dtype=tl.float32, trans_b=True)
                    sf_lo = tl.reshape(
                        tl.load(tle.gpu.local_ptr(sl.sfa.slot(0))), (BLOCK_M,))
                    sf_hi = tl.reshape(
                        tl.load(tle.gpu.local_ptr(sl.sfa.slot(1))), (BLOCK_M,))
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    acc_hi = tle.gpu.wgmma_wait(0, acc_hi)
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
                w_lo = tl.where(_is_up[None, :],
                                (sf_lo * sfb_u)[:, None], (sf_lo * sfb_g)[:, None])
                w_hi = tl.where(_is_up[None, :],
                                (sf_hi * sfb_u)[:, None], (sf_hi * sfb_g)[:, None])
                # v10: accumulate SEPARATELY. v9 wrote `acc += a*x + b*y`, and the
                # intermediate sum blocked fusion: SASS showed FFMA 75.8M -> 38.0M
                # with FADD 1.4M -> 39.1M (exactly offsetting) -- the compiler
                # UNFUSED into FMUL+FADD. Two independent accumulates give one
                # FFMA per element each, matching CUDA's per-element
                # `final_accum[i] += scale_a * sb * accum[i]` (cuh:1017-1020).
                if phase == 1:
                    acc += acc_lo * w_lo          # single scale for all 128 K
                else:
                    acc += acc_lo * w_lo
                    acc += acc_hi * w_hi
                reader.release(ck)
                ck += 1

            if phase == 1:
                # L1 EPILOGUE: SwiGLU on the granularity-8 interleave + UE8M0 + FP8
                t4 = tl.reshape(acc, (BLOCK_M, NPAIR, 2, 8))
                t4 = tl.permute(t4, (0, 1, 3, 2))
                gate, up = tl.split(t4)
                gate = tl.reshape(gate, (BLOCK_M, L1_OUT_N))
                up = tl.reshape(up, (BLOCK_M, L1_OUT_N))
                sw = (gate / (1.0 + tl.exp(-gate))) * up
                w = tl.load(l1_w + rows, mask=m_ok, other=0.0)
                sw = sw * w[:, None]
                amax = tl.max(tl.abs(sw), axis=1)
                scaled = amax * (1.0 / 448.0)
                pos = scaled > 0.0
                e = tl.ceil(tl.log2(tl.where(pos, scaled, 1.0)))
                sf = tl.where(pos, tl.exp2(e), 1.0)
                sf_inv = tl.where(pos, tl.exp2(-e), 1.0)
                q = (sw * sf_inv[:, None]).to(tl.float8e4nv)
                cols_o = n_block * L1_OUT_N + tl.arange(0, L1_OUT_N)
                tl.store(l2_acts + rows[:, None] * INTER + cols_o[None, :], q, mask=m_ok[:, None])
                tl.store(l2_sf + n_block * POOL_TOKENS + rows, sf, mask=m_ok)
                tl.debug_barrier()
                tl.atomic_or(l2_arrival_mask + pool_block, (tl.full((), 1, tl.int64) << n_block), sem="release")
            else:
                # L2 EPILOGUE: BF16 cast + NVLink scatter to the SOURCE rank
                cols = n_block * BLOCK_N + offs_n
                # v5: this fp32 local copy is TEST-ONLY (see module docstring) -- the
                # kernel never reads it back; only the host-side d2h correctness check
                # does. CUDA has no equivalent. Skip it when benchmarking: it is a full
                # [BLOCK_M, BLOCK_N] fp32 store, i.e. 2x the bytes of the bf16 scatter.
                if WRITE_L2_OUT:
                    tl.store(l2o_l + rows[:, None] * L2_N + cols[None, :], acc,
                             mask=m_ok[:, None])
                md = meta_l + rows * 3
                dst_rank = tl.load(md + 0, mask=m_ok, other=0)
                dst_tok = tl.load(md + 1, mask=m_ok, other=0)
                dst_topk = tl.load(md + 2, mask=m_ok, other=0)
                cbb = peer_bf16(cb_tab, dst_rank)               # VECTOR of peer bases
                base = (dst_topk * NTOK + dst_tok) * K
                # v23: hints cannot change physical ownership of a WGMMA fragment.
                # Materialize a linear BF16 tile plus one fully resolved remote
                # destination per row, then let the 4 physical math warps remap it
                # as 16 lanes/row x 8 BF16/lane.
                dst_row = tl.cast(cbb + base + n_block * BLOCK_N, tl.uint64)
                tl.store(tle.gpu.local_ptr(dst_rows_s), dst_row, mask=m_ok)
                tl.store(tle.gpu.local_ptr(cd_s), acc.to(tl.bfloat16),
                         mask=m_ok[:, None])
                cd_addr = smem_generic_addr(tle.gpu.local_ptr(cd_s, (0, 0)))
                dst_addr = smem_generic_addr(tle.gpu.local_ptr(dst_rows_s, (0,)))
                scatter_cols = tl.full((), BLOCK_N, tl.int32)
                if USE_L2_TMA:
                    tle_raw.call(
                        l2_tma_scatter_edsl,
                        [cd_addr, dst_addr, valid_m, scatter_cols],
                    )
                else:
                    tle_raw.call(
                        l2_wide_scatter_edsl,
                        [cd_addr, dst_addr, valid_m, scatter_cols],
                    )
            nblocks += 1
    tl.store(math_blocks + sm_idx, nblocks)

    nvlink_barrier(sig_tab, 1, gctr + 3, sm_idx, MY_PE, NPES, NUM_SMS)

    # ---- COMBINE: reduce MY tokens over their topk slots ----
    # v20: pass the LOCAL combine buffer as a direct int kernel-arg (cb_local) instead
    # of loading it from the peer table. Triton specializes 16B-divisible int args, so
    # cb_local gets divisibility=16 (unlike the table-loaded div=1 pointer) -> the
    # combine reload coalesces into wide LDG instead of per-2-byte LDG.E.U16.
    cb_l = tl.multiple_of(cb_local.to(tl.pointer_type(tl.bfloat16)), 16)
    offs_hid = tl.arange(0, K_POW2)
    hid_ok = offs_hid < K
    t = sm_idx
    while t < NTOK:
        acc_c = tl.zeros((K_POW2,), dtype=tl.float32)
        for k in tl.static_range(0, TOPK):
            e = tl.load(topk_local + t * TOPK + k).to(tl.int32)
            v = tl.load(cb_l + ((k * NTOK + t) * K) + offs_hid, mask=(e >= 0) & hid_ok, other=0.0)
            acc_c += v.to(tl.float32)
        tl.store(final_y + t * K + offs_hid, acc_c.to(tl.bfloat16), mask=hid_ok)
        t += NUM_SMS


# ---------------------------------------------------------------------------
@triton.jit
def ws_megakernel(
    topk_local, tok_tab, sf_tab, w_tab, q_tab, recv_tab, rsum_tab, sig_tab,
    meta_tab, cb_tab, l2out_tab, cb_local,
    sm_expert_count, expert_send_count, l1_arrival_count, l2_arrival_mask, gctr,
    meta_ready, l1_token, l1_sf, l1_w, l2_acts, l2_sf, w1_sf, w2_sf,
    loader_blocks, math_blocks, final_y, a1_desc, b1_desc, a2_desc, b2_desc,
    sfa1_desc, sfa2_desc,
    tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7,   # v21: NPES peer token bases (kernel-arg ints)
    WRITE_L2_OUT: tl.constexpr, USE_L2_TMA: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
    MY_PE: tl.constexpr, NPES: tl.constexpr, NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr, TOPK: tl.constexpr, NEXP: tl.constexpr, EPR: tl.constexpr,
    EPR_POW2: tl.constexpr, R_POW2: tl.constexpr, EPW: tl.constexpr,
    MAX_RECV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, K: tl.constexpr, INTER: tl.constexpr,
    L1_N: tl.constexpr, L2_N: tl.constexpr, NK1: tl.constexpr, NK2: tl.constexpr,
    NL1N: tl.constexpr, NL2N: tl.constexpr, L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr, NSF: tl.constexpr, POOL_TOKENS: tl.constexpr,
    NEXP_POW2: tl.constexpr, K_POW2: tl.constexpr, NSF_POW2: tl.constexpr,
    BLOCK_R: tl.constexpr, ITERS: tl.constexpr, STAGES: tl.constexpr,
    DISPATCH_WARPS: tl.constexpr, MATH_WARPS: tl.constexpr,
):
    # v4: fp8 Hopper WGMMA is m64nNk32 -- K is FIXED at 32, and B must be N-major.
    # So each BLOCK_K=128 tile becomes 4 wgmma of K=32 (exactly like CUDA's
    # `for k < BLOCK_K/WGMMA::K` loop). Rank-3 buffers let .slot(k) carve out the
    # rank-2 [M,32] / [N,32] operands as pure SMEM views -- no tl.load, no permute.
    # v18: ONE contiguous K=128 tile (was 4x K=32 in v11, 2x K=64 in v17). A single
    # big TMA fills it; wgmma reads K-offset sub-tiles via buffered_tensor.subslice
    # (the SMEM analogue of CUDA's make_smem_desc(addr + k*WGMMA::K)). L1 reads the
    # whole K=128; L2 subslices into two K=64 halves for its per-64-K SF.
    a_s = tle.gpu.alloc([STAGES, BLOCK_M, BLOCK_K], dtype=tl.float8e4nv,
                        layout=None, scope=tle.gpu.smem)
    b_s = tle.gpu.alloc([STAGES, BLOCK_N, BLOCK_K], dtype=tl.float8e4nv,
                        layout=None, scope=tle.gpu.smem)
    # NOTE: nv_mma_shared_layout must stay TRUE (the default). A TMA copy into a
    # pipe field whose smem uses the linear layout (nv_mma_shared_layout=False)
    # makes the NVGPUWarpSpecialization pass fail -- verified by bisection.
    # v19: ONE merged SF field holding both K-halves [2, BLOCK_M]. A single TMA
    # fetches the two adjacent SF rows (L2: k-groups 2kb/2kb+1; L1: kb/kb+1, slot1
    # unused). Halves the SFA TMAs (was 2 fields x 1 copy = 2 -> 1) and drops one
    # pipe field, so tma/kblock 4 -> 3.
    sfa_s = tle.gpu.alloc([STAGES, 2, BLOCK_M], dtype=tl.float32, layout=None,
                          scope=tle.gpu.smem)
    # v23: unlike the TMA pipe fields above, these are deliberately LINEAR.
    # The raw epilogue addresses them by physical row/column byte offset.
    cd_s = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.bfloat16, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    dst_rows_s = tle.gpu.alloc([BLOCK_M], dtype=tl.uint64, layout=None,
                               scope=tle.gpu.smem, nv_mma_shared_layout=False)
    # v25: one descriptorless D8 TMA1D stage plus {mbarrier, phase}. These must
    # be TLE-owned dynamic SMEM: CUDA static shared shifts `global_smem` and
    # invalidates the alignment of the existing local TMA/WGMMA pipe.
    d8_token_s = tle.gpu.alloc([K], dtype=tl.float8e4nv, layout=None,
                               scope=tle.gpu.smem, nv_mma_shared_layout=False)
    d8_state_s = tle.gpu.alloc([2], dtype=tl.uint64, layout=None,
                               scope=tle.gpu.smem, nv_mma_shared_layout=False)
    # every field is TMA -> ONE pipe (no mixed TMA / tl.store commit)
    p = tle.pipe(capacity=STAGES, scope="cta", name="megamoe_gemm",
                 a=a_s, b=b_s, sfa=sfa_s)

    tle.gpu.warp_specialize(
        [
            (loader_role, (
                p.writer(), meta_ready, rsum_tab, l1_arrival_count, l2_arrival_mask,
                a1_desc, b1_desc, a2_desc, b2_desc, sfa1_desc, sfa2_desc, loader_blocks,
                MY_PE, NUM_SMS, EPR, EPR_POW2, EPW, BLOCK_M, BLOCK_N, BLOCK_K,
                L1_N, L2_N, NK1, NK2, NL1N, NL2N, POOL_TOKENS, K, INTER)),
            (dispatch_role, (
                topk_local, tok_tab, sf_tab, w_tab, q_tab, recv_tab, rsum_tab, sig_tab,
                meta_tab, sm_expert_count, expert_send_count, l1_arrival_count, gctr,
                meta_ready, l1_token, l1_sf, l1_w,
                d8_token_s, d8_state_s,
                tp0, tp1, tp2, tp3, tp4, tp5, tp6, tp7,
                MY_PE, NPES, NUM_SMS, NTOK, TOPK, NEXP, EPR, EPR_POW2, R_POW2,
                MAX_RECV, BLOCK_M, K, NSF, POOL_TOKENS, BLOCK_R, ITERS,
                NEXP_POW2, K_POW2, NSF_POW2, USE_D8_TMA1D)),
            (math_role, (
                p.reader(), cd_s, dst_rows_s,
                meta_ready, rsum_tab, cb_tab, sig_tab,
                meta_tab, l2out_tab, cb_local, l1_w, l2_acts, l2_sf, l2_arrival_mask,
                math_blocks, topk_local, final_y, gctr, w1_sf, w2_sf,
                WRITE_L2_OUT, USE_L2_TMA,
                MY_PE, NPES, NUM_SMS, EPR, EPR_POW2, EPW, BLOCK_M, BLOCK_N, BLOCK_K,
                INTER, L1_N, L2_N, NK1, NK2, NL1N, NL2N, L1_OUT_N, NPAIR, POOL_TOKENS,
                NTOK, TOPK, K, K_POW2)),
        ],
        [DISPATCH_WARPS, MATH_WARPS],
        [48, 232],
    )


# ---------------------------------------------------------------------------
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
    USE_L2_TMA = int(os.environ.get("W_L2_TMA", "0"))
    USE_D8_TMA1D = int(os.environ.get(
        "W_D8_TMA1D_LEVEL",
        "2" if int(os.environ.get("W_D8_TMA1D", "0")) else "0",
    ))
    BENCH_ITERS = int(os.environ.get("W_ITERS", "20"))
    BENCH_WARMUP = int(os.environ.get("W_WARMUP", "5"))
    BENCH_REDUCE = os.environ.get("W_BENCH_REDUCE", "median").lower()
    assert BENCH_REDUCE in ("median", "mean"), BENCH_REDUCE
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
    STAGES = int(os.environ.get("W_STAGES","2"))
    if USE_D8_TMA1D:
        assert 0 < K <= 4096 and K % 16 == 0, \
            f"raw D8 TMA1D requires 16-byte token rows with K<=4096, got K={K}"
    assert NEXP % R == 0, f"NEXP={NEXP} must be divisible by npes={R}"
    EPR = NEXP // R
    EPW = min(2, EPR)
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
    BLOCK_R = 256
    routes = NTOK * TOPK
    ITERS = (routes + NUM_SMS * BLOCK_R - 1) // (NUM_SMS * BLOCK_R)
    R_POW2 = 1 << (R - 1).bit_length()
    EPR_POW2 = 1 << (EPR - 1).bit_length()
    NEXP_POW2 = 1 << (NEXP - 1).bit_length()
    K_POW2 = 1 << (K - 1).bit_length()
    NSF_POW2 = 1 << (NSF - 1).bit_length()

    # ---- global inputs, generated identically on every rank ----
    g = torch.Generator(device="cpu").manual_seed(1234)
    # Routing must match the baselines we compare against. `bench_mega_moe_sm90.py`
    # (and the horizontal sweep) use topk-of-random-scores -> TOPK *distinct* experts,
    # then mask a `masked_ratio` fraction to -1. The old `randint` here sampled WITH
    # replacement (a token could pick the same expert twice), which is not real topk.
    # W_DROP mirrors CUDA's --masked-ratio; the sweep's baselines ran with 0.0.
    DROP = float(os.environ.get("W_DROP", "0.1"))
    _scores = torch.randn(R * NTOK, NEXP, generator=g)
    topk_all = torch.topk(_scores, TOPK, dim=-1, largest=True, sorted=False).indices
    topk_all = topk_all.reshape(R, routes).contiguous()
    if DROP > 0:
        topk_all[torch.rand(R, routes, generator=g) < DROP] = -1
    tok_all = (torch.randn(R, NTOK, K, generator=g) * 0.5).to(f8)
    sf_all = (torch.rand(R, NTOK, NSF, generator=g) * 0.5 + 0.5).contiguous()
    w_all = (torch.rand(R, routes, generator=g) * 0.8 + 0.2).contiguous()
    gg = torch.Generator(device=dev).manual_seed(9000 + rank)
    w1_q = (torch.randn(EPR, L1_N, K, generator=gg, device=dev) * 0.4).to(f8)
    # weight SF stored K-MAJOR [EPR, NK, N] (was N-major [EPR, N, NK]). Per-row
    # semantics are unchanged; only the memory layout differs, so that for a fixed
    # k-block the math role's `sfb` load over rows_n hits CONTIGUOUS addresses and
    # coalesces. The old N-major layout made adjacent lanes 128 B apart -> 32x
    # sector waste (ncu: 16.5M excessive sectors, single largest offender @L540).
    # v7: match CUDA's weight-SF granularity (cuh:919-928). CUDA stores ONE
    # scalar per (128-row weight block, 128-K block); with the gate/up gran-8
    # interleave a BLOCK_N=128 tile spans 64 gate cols + 64 up cols that all
    # share their block's scalar. Layout [E, 2*NL1N, NK1]: N axis is
    # [gate(NL1N), up(NL1N)], matching `gate_sf_n = n_block/2`,
    # `up_sf_n = NL1N + n_block/2`.
    w1_sf = (torch.rand(EPR, 2 * NL1N, NK1, generator=gg, device=dev) * 0.4 + 0.3).contiguous()
    w2_q = (torch.randn(EPR, L2_N, INTER, generator=gg, device=dev) * 0.4).to(f8)
    # v7: L2 is one scalar per (BLOCK_N, BLOCK_K) tile -- cuh:927.
    w2_sf = (torch.rand(EPR, NL2N, NK2, generator=gg, device=dev) * 0.4 + 0.3).contiguous()

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
        sym_zero[(triton.cdiv(nb, BLK),)](tab, rank, nb, BLOCK=BLK)

    def h2d(t, tab):
        b = t.contiguous().view(-1).view(torch.int8).to(dev)
        sym_h2d[(triton.cdiv(b.numel(), BLK),)](b, tab, rank, b.numel(), BLOCK=BLK)

    def d2h(tab, pe, nb):
        out = torch.empty(nb, device=dev, dtype=torch.int8)
        sym_d2h[(triton.cdiv(nb, BLK),)](tab, pe, out, nb, BLOCK=BLK)
        return out

    for tab, nb in ((q_tab, EPR * R * MAX_RECV * 4), (recv_tab, R * EPR * 8),
                    (rsum_tab, EPR * 8), (cb_tab, TOPK * NTOK * K * 2),
                    (sig_tab, 16 * 4), (meta_tab, POOL_TOKENS * 3 * 4),
                    (l2out_tab, POOL_TOKENS * L2_N * 4)):
        zero(tab, nb)
    h2d(tok_all[rank], tok_tab)
    h2d(sf_all[rank], sf_tab)
    h2d(w_all[rank], w_tab)
    torch.cuda.synchronize()
    lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()

    # ---- local tensors ----
    topk_local = topk_all[rank].to(dev)
    sm_ec = torch.full((NUM_SMS * NEXP,), -1, device=dev, dtype=torch.int32)
    send = torch.zeros((NEXP,), device=dev, dtype=torch.int64)
    arrival = torch.zeros((max(NPB, 1),), device=dev, dtype=torch.int32)
    l2_mask = torch.zeros((max(NPB, 1),), device=dev, dtype=torch.int64)
    gctr = torch.zeros((8,), device=dev, dtype=torch.int32)
    meta_ready = torch.zeros((NUM_SMS,), device=dev, dtype=torch.int32)
    l1_tok = torch.zeros((POOL_TOKENS, K), device=dev, dtype=f8)
    l1_sf = torch.zeros((NSF * POOL_TOKENS,), device=dev, dtype=torch.float32)
    l1_w = torch.zeros((POOL_TOKENS,), device=dev, dtype=torch.float32)
    l2_acts = torch.zeros((POOL_TOKENS, INTER), device=dev, dtype=f8)
    l2_sf = torch.zeros((NL1N * POOL_TOKENS,), device=dev, dtype=torch.float32)
    final_y = torch.zeros((NTOK, K), device=dev, dtype=torch.bfloat16)
    loader_blocks = torch.zeros((NUM_SMS,), device=dev, dtype=torch.int32)
    math_blocks = torch.zeros((NUM_SMS,), device=dev, dtype=torch.int32)

    def _alloc_fn(size: int, align: int, stream):
        return torch.empty(size, device=dev, dtype=torch.int8)
    triton.set_allocator(_alloc_fn)

    a1_desc = TensorDescriptor(l1_tok, shape=[POOL_TOKENS, K], strides=[K, 1],
                               block_shape=[BLOCK_M, BLOCK_K])
    # v4: fp8 wgmma wants B N-major, which is w1_q's NATIVE layout -- no permute.
    w1_nk = w1_q.contiguous()                           # [EPR, L1_N, K]  N-major
    b1_desc = TensorDescriptor(w1_nk.view(EPR * L1_N, K), shape=[EPR * L1_N, K],
                               strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K])
    a2_desc = TensorDescriptor(l2_acts, shape=[POOL_TOKENS, INTER], strides=[INTER, 1],
                               block_shape=[BLOCK_M, BLOCK_K])
    w2_nk = w2_q.contiguous()                           # [EPR, L2_N, INTER]  N-major
    b2_desc = TensorDescriptor(w2_nk.view(EPR * L2_N, INTER), shape=[EPR * L2_N, INTER],
                               strides=[INTER, 1], block_shape=[BLOCK_N, BLOCK_K])
    # SF is MN-major -> a block's SF tile is BLOCK_M contiguous floats: a real TMA box
    sfa1_desc = TensorDescriptor(l1_sf.view(NSF, POOL_TOKENS), shape=[NSF, POOL_TOKENS],
                                 strides=[POOL_TOKENS, 1], block_shape=[2, BLOCK_M])
    sfa2_desc = TensorDescriptor(l2_sf.view(NL1N, POOL_TOKENS), shape=[NL1N, POOL_TOKENS],
                                 strides=[POOL_TOKENS, 1], block_shape=[2, BLOCK_M])

    lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()

    kargs = dict(
        MY_PE=rank, NPES=R, NUM_SMS=NUM_SMS, NTOK=NTOK, TOPK=TOPK, NEXP=NEXP, EPR=EPR,
        EPR_POW2=EPR_POW2, R_POW2=R_POW2, EPW=EPW, MAX_RECV=MAX_RECV,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, K=K, INTER=INTER,
        L1_N=L1_N, L2_N=L2_N, NK1=NK1, NK2=NK2, NL1N=NL1N, NL2N=NL2N,
        L1_OUT_N=L1_OUT_N, NPAIR=NPAIR, NSF=NSF, POOL_TOKENS=POOL_TOKENS,
        NEXP_POW2=NEXP_POW2, K_POW2=K_POW2, NSF_POW2=NSF_POW2,
        BLOCK_R=BLOCK_R, ITERS=ITERS, STAGES=STAGES,
        WRITE_L2_OUT=(BENCH == 0),
        USE_L2_TMA=(USE_L2_TMA != 0),
        # Keep the integer diagnostic level: 0=live v23 control, 1=init-only,
        # 2=full D8 raw load+store. Booleanizing this silently dropped level 2.
        USE_D8_TMA1D=USE_D8_TMA1D,
        DISPATCH_WARPS=4, MATH_WARPS=4, num_warps=4,
    )
    # v21: NPES peer token-buffer bases as direct int kernel-args (div=16 specialized
    # -> wide coalesced LDG), padded to 8; the where-chain in dispatch picks by src_rank.
    _tp = (tok_tab.tolist() + [int(tok_tab[0])] * 8)[:8]
    pos = (topk_local, tok_tab, sf_tab, w_tab, q_tab, recv_tab, rsum_tab, sig_tab,
           meta_tab, cb_tab, l2out_tab, cb_b, sm_ec, send, arrival, l2_mask, gctr, meta_ready,
           l1_tok, l1_sf, l1_w, l2_acts, l2_sf, w1_sf, w2_sf, loader_blocks, math_blocks,
           final_y, a1_desc, b1_desc, a2_desc, b2_desc, sfa1_desc, sfa2_desc, *_tp)

    def reset():
        send.zero_(); arrival.zero_(); l2_mask.zero_(); gctr.zero_(); meta_ready.zero_()
        sm_ec.fill_(-1)
        zero(recv_tab, R * EPR * 8); zero(rsum_tab, EPR * 8); zero(sig_tab, 16 * 4)

    def barrier():
        torch.cuda.synchronize()
        lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()

    reset(); barrier()
    compiled = ws_megakernel[(NUM_SMS,)](*pos, **kargs)
    barrier()

    bench_us = None
    if BENCH:
        times = []
        for i in range(BENCH_WARMUP + BENCH_ITERS):
            reset(); barrier()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); ws_megakernel[(NUM_SMS,)](*pos, **kargs); e1.record()
            torch.cuda.synchronize()
            if i >= BENCH_WARMUP:
                times.append(e0.elapsed_time(e1) * 1e3)   # us
            barrier()
        if BENCH_REDUCE == "mean":
            bench_us = sum(times) / len(times)
        else:
            times.sort()
            bench_us = times[len(times) // 2]

    has_ws = "ttg.warp_specialize" in compiled.asm.get("ttgir", "")
    n_tma = compiled.asm.get("ptx", "").count("cp.async.bulk.tensor")

    if BENCH:
        P = sum(n_le[rank])
        flops = 6.0 * P * K * INTER
        tflops = flops / (bench_us * 1e-6) / 1e12
        print(f"[rank {rank}/{R}] BENCH h={K} ih={INTER} E={NEXP} k={TOPK} tokens={NTOK} "
              f"recv={P} experts={EPR} | {bench_us:8.1f} us  {tflops:6.1f} TFLOPS  "
              f"ws={has_ws} tma={n_tma} d8_tma1d_level={USE_D8_TMA1D}", flush=True)
        lib.nvshmemx_barrier_wrapper(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize(); lib.nvshmem_finalize_wrapper()
        return 0

    # ---- readback ----
    q_c = d2h(q_tab, rank, EPR * R * MAX_RECV * 4).cpu().view(torch.int32)
    recv_c = d2h(recv_tab, rank, R * EPR * 8).cpu().view(torch.int64)
    rsum_c = d2h(rsum_tab, rank, EPR * 8).cpu().view(torch.int64)
    cb_c = d2h(cb_tab, rank, TOPK * NTOK * K * 2).cpu().view(torch.bfloat16).view(TOPK, NTOK, K)
    peer_meta = [d2h(meta_tab, pe, POOL_TOKENS * 3 * 4).cpu().view(torch.int32).view(POOL_TOKENS, 3)
                 for pe in range(R)]
    peer_l2out = [d2h(l2out_tab, pe, POOL_TOKENS * L2_N * 4).cpu().view(torch.float32).view(POOL_TOKENS, L2_N)
                  for pe in range(R)]
    torch.cuda.synchronize()

    inject = os.environ.get("W_INJECT", "")
    if inject == "queue":
        q_c[0] = (int(q_c[0]) + 1) % (NTOK * TOPK)   # valid-but-wrong route
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
            got = sorted(int(v) for v in q_c[base: base + counts[rank][le][s]].tolist())
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
    w2sf_c = w2_sf.to(torch.float64).cpu()          # [EPR, NL2N, NK2]
    for nb in range(NL2N):
        for kb in range(NK2):
            W2loc[:, nb * BLOCK_N:(nb + 1) * BLOCK_N,
                  kb * BLOCK_K:(kb + 1) * BLOCK_K] *= w2sf_c[:, nb, kb][:, None, None]
    l2o_me = peer_l2out[rank].to(torch.float64)
    l2_err = 0.0
    for le in range(EPR):
        idx = [pool_off[rank][le] * BLOCK_M + t for t in range(n_le[rank][le])]
        if not idx:
            continue
        ref = deq[idx] @ W2loc[le].T
        sc = ref.abs().max().item()
        l2_err = max(l2_err, (l2o_me[idx] - ref).abs().max().item() / max(sc, 1e-12))
    if l2_err > 3e-3:   # async fp8 wgmma ~1-2e-4; bar << fp8 quant noise (3.5e-2)
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
    print(f"[rank {rank}/{R}] EPR={EPR} pool_tok={sum(n_le[rank])} blocks={sum(blocks[rank])} "
          f"gemm_blocks={EXPECT} ws={has_ws} tma={n_tma} l2_gemm={l2_err:.2e} "
          f"partials={n_partials} scatter_bad={scatter_bad} errors={errs or 0} "
          f"d8_tma1d_level={USE_D8_TMA1D} "
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
    env.update({
        "NVSHMEM_HOME": str(NVSHMEM_HOME),
        "NVSHMEM_BOOTSTRAP": "MPI",
        "LD_LIBRARY_PATH": f"{NVSHMEM_HOME / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "CPATH": "/usr/local/cuda-12.8/targets/x86_64-linux/include:" + env.get("CPATH", ""),
    })
    worker_python = str(TLE_PYTHON if TLE_PYTHON.exists() else Path(sys.executable))
    cmd = ["/usr/bin/mpirun", "--allow-run-as-root", "-np", str(NUM_RANKS),
           worker_python, str(Path(__file__).resolve()), "--worker"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              timeout=int(os.environ.get("W_TIMEOUT","600")), env=env)
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
