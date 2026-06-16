import argparse
import contextlib

import torch
import triton
import triton.language as tl
import triton.experimental.dl as dl


# =============================================================================
#  Extended-Triton / dl.dsa mock primitives
#
#  Same "executable mock" idiom as fa_triton_zcx.py: each dl.dsa.* call is a
#  light Python stub so the kernel parses as ordinary Triton, while the real
#  Ascend `dl` backend lowers it to the corresponding hardware instruction.
#
#  This file translates the SINGLE-STREAM 3-task software pipeline described in
#  ref_flash_attn.py (§C-2). Unlike fa_triton_zcx.py (warp-specialised MIX_1_2
#  with two async_task blocks talking over cross-core semaphores), the arch22
#  variant keeps ONE instruction stream per AI core: it issues *asynchronous*
#  Cube matmuls (cube_launch -> token, cube_wait(token)) and runs the Vector
#  softmax / rescale inline between launch and wait, exactly like an AscendC MIX
#  kernel. Cross-engine ordering uses HardEvent flags (set_flag / wait_flag).
#
#  Per tick (one flattened sub-task k):
#     Cube  : Bmm1(k)        ||  Bmm2(k-1)
#     Vector: Vec1(k-1)      ||  Vec2(k-2)
#  i.e. every sub-task flows  Bmm1 -> [Vec1, Bmm2] -> Vec2  over 3 consecutive
#  ticks => a 3-deep task ring (taskId % 3) with 2-deep ping-pong GM buffers
#  (taskId % 2) for mm1Res / stage1Res / mm2Res.
# =============================================================================
class _DL_DSA:
    # ---- memory spaces ----
    L1, L0A, L0B, L0C, UB = "L1", "L0A", "L0B", "L0C", "UB"

    # ---- cross-engine HardEvent ids (== AscendC HardEvent enum) ------------
    class Event:
        MTE3_MTE2 = "MTE3_MTE2"   # Vec1 wrote P (UB->GM) -> Bmm2 may read P
        MTE2_V    = "MTE2_V"      # GM->UB done          -> Vector may compute
        V_MTE2    = "V_MTE2"      # Vector done          -> release UB for next GM->UB
        V_MTE3    = "V_MTE3"      # Vector done          -> UB->GM may start
        MTE3_V    = "MTE3_V"      # UB->GM done          -> Vector may reuse UB
        MTE2_MTE3 = "MTE2_MTE3"   # GM->UB -> UB->GM (Vec2 sub-pipe)

    # ---- intra-engine pipe kinds (RAW barrier) -----------------------------
    class Pipe:
        V    = "PIPE_V"
        MTE2 = "PIPE_MTE2"
        MTE3 = "PIPE_MTE3"

    # ---- explicit memory hierarchy + DMA -----------------------------------
    @staticmethod
    def local_alloc(shape, dtype, space):
        """Allocate an on-chip buffer (L1/L0*/UB) that persists across loop
        iterations (== tilelang T.alloc_*)."""
        return tl.zeros(shape, dtype=dtype)

    @staticmethod
    def copy(src, dst, transpose=False):
        """DMA between memory levels (GM<->L1/UB, L1<->L0, L0C->GM, UB<->GM).
        Lowers to MTE2/MTE1/FIX/MTE3 depending on src/dst."""
        return src.T if transpose else src

    # ---- asynchronous Cube matmul + token double buffer --------------------
    # == AscendC matmul::Matmul<...>::IterateAll<async> + WaitIterateAll.
    @staticmethod
    def cube_launch(a_l0a, b_l0b, c_l0c, c_gm, init=True):
        """Fire one async Cube matmul: iterate a_l0a @ b_l0b accumulating in the
        L0C buffer c_l0c, then FIX-copy the result to GM workspace c_gm -- the
        whole mma->L0C->FIX->GM chain as a single async unit. Returns a *token*
        (its GM destination) that the matching cube_wait() drains a tick later.

        Token double buffer: BMM1 owns L0 slot 0, BMM2 owns L0 slot 1, and each
        GM workspace is 2-deep ping-pong (taskId % 2). So Bmm1(k) (slot k%2) can
        be in flight while Bmm1(k-1)'s result (slot (k-1)%2) is being consumed --
        the wait one tick later both frees l0c for reuse and guarantees the GM
        region is complete before the Vector reads it."""
        tl.store(c_gm, tl.dot(a_l0a, b_l0b))   # mock: backend lowers to async iterate+FIX
        return c_gm                            # token == the GM region being filled

    @staticmethod
    def cube_wait(token):
        """Block until the async cube_launch that targeted `token` (a GM region)
        has fully landed (== AscendC WaitBmm1Result / WaitBmm2Result). Issuing
        this one tick AFTER the launch is what overlaps the Cube with the Vector."""
        pass

    # ---- cross-engine sync (Cube <-> Vector, via HardEvent) ---------------
    @staticmethod
    def set_flag(event):   # == AscendC::SetFlag<HardEvent::X>(id)
        pass

    @staticmethod
    def wait_flag(event):  # == AscendC::WaitFlag<HardEvent::X>(id)
        pass

    @staticmethod
    def pipe_barrier(pipe):  # == AscendC::PipeBarrier<PIPE_X>()
        pass


dl.dsa = _DL_DSA()


# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_CORES = 24
BLOCK_M = 128
BLOCK_N = 128
DIM = 128

# ---- 3-task pipeline constants ----
#  PIPE_DEPTH = 2 : a sub-task's Bmm1 (tick k), [Vec1+Bmm2] (tick k+1) and Vec2
#  (tick k+2) span 3 ticks, so 2 extra "cooldown" ticks drain the tail.
PIPE_DEPTH = 2
RING = PIPE_DEPTH + 1   # = 3 task-metadata slots (taskId % 3)
PP = 2                  # ping-pong GM buffers (taskId % 2)


# =============================================================================
#  The single-stream 3-task scheduler kernel
#
#  grid = (NUM_CORES,). program_id(0) = cid. Each program drives one Cube + one
#  Vector engine from a single instruction stream (AscendC MIX_1_1).
#
#  GM ping-pong workspaces (per core, indexed by taskId % 2):
#     mm1Res    [NUM_CORES, 2, BLOCK_M, BLOCK_N] fp32  : S = Q·Kᵀ      (Cube Bmm1 out)
#     stage1Res [NUM_CORES, 2, BLOCK_M, BLOCK_N] fp16  : P = softmax(S) (Vec1 out)
#     mm2Res    [NUM_CORES, 2, BLOCK_M, DIM]     fp32  : O_part = P·V   (Cube Bmm2 out)
# =============================================================================
@triton.jit
def flash_attention_fwd_3task_kernel(
    Q, K, V, Out,
    mm1Res, stage1Res, mm2Res,           # GM ping-pong workspaces
    sm_scale,
    B, Hq, Hkv, S,                       # shapes
    sQb, sQh, sQs, sQd,                  # Q strides
    sKb, sKh, sKs, sKd,                  # K/V strides
    sOb, sOh, sOs, sOd,                  # Out strides
    num_seq_blocks, heads_q, gqa_group,  # task-decode helpers
    n_iters,                             # KV blocks (s2 loop) per output tile
    q_tasks, r_tasks,                    # static task split
    NUM_ITERS: tl.constexpr,             # = n_iters as constexpr
    IS_CAUSAL: tl.constexpr,
):
    cid = tl.program_id(0)

    # ---- static task distribution (== AICPU GetFASectionInfo metadata) ----
    my_start = cid * q_tasks + tl.where(cid < r_tasks, cid, r_tasks)
    my_count = q_tasks + tl.where(cid < r_tasks, 1, 0)
    n_sub = my_count * n_iters            # flattened (output-tile x KV-block) sub-tasks

    # =========================================================================
    #  ① on-chip working set
    # =========================================================================
    # -- Cube side: L1 staging + L0 double buffer (slot 0 = Bmm1, slot 1 = Bmm2) --
    q_l1 = dl.dsa.local_alloc((BLOCK_M, DIM), Q.dtype.element_ty, dl.dsa.L1)
    k_l1 = dl.dsa.local_alloc((BLOCK_N, DIM), Q.dtype.element_ty, dl.dsa.L1)
    v_l1 = dl.dsa.local_alloc((BLOCK_N, DIM), Q.dtype.element_ty, dl.dsa.L1)
    p_l1 = dl.dsa.local_alloc((BLOCK_M, BLOCK_N), Q.dtype.element_ty, dl.dsa.L1)
    l0a = dl.dsa.local_alloc((2, BLOCK_M, DIM), Q.dtype.element_ty, dl.dsa.L0A)
    l0b = dl.dsa.local_alloc((2, DIM, BLOCK_N), Q.dtype.element_ty, dl.dsa.L0B)
    l0c = dl.dsa.local_alloc((2, BLOCK_M, BLOCK_N), tl.float32, dl.dsa.L0C)

    # -- Vector side: online-softmax state (accumulated across the KV loop) ----
    acc_o  = dl.dsa.local_alloc((BLOCK_M, DIM), tl.float32, dl.dsa.UB)   # P·V accumulator
    sumexp = dl.dsa.local_alloc((BLOCK_M, 1), tl.float32, dl.dsa.UB)     # running denom
    neg_sm = dl.dsa.local_alloc((2, BLOCK_M, 1), tl.float32, dl.dsa.UB)  # running (neg,scaled) max ping-pong
    # per-sub-task carry  Vec1(g) -> Vec2(g), indexed by task ring slot g % RING
    r_factors = dl.dsa.local_alloc((RING, BLOCK_M, 1), tl.float32, dl.dsa.UB)
    sumexp_is = dl.dsa.local_alloc((RING, BLOCK_M, 1), tl.float32, dl.dsa.UB)

    # =========================================================================
    #  ② init cross-engine flags: pre-arm so the first producer's wait passes.
    # =========================================================================
    dl.dsa.set_flag(dl.dsa.Event.MTE3_MTE2)   # pretend last P already consumed

    # =========================================================================
    #  ③ 3-task pipeline. One tick = one flattened sub-task g; +PIPE_DEPTH ticks
    #     drain the tail (cooldown). Per tick:
    #        Bmm1(g)  [if g < n_sub]
    #        Vec1(g-1) + Bmm2(g-1)  [if 1 <= g <= n_sub]
    #        Vec2(g-2)              [if 2 <= g, g-2 < n_sub]
    # =========================================================================
    for g in range(n_sub + PIPE_DEPTH):

        # ---------------------------------------------------------------------
        # ─── 1) WaitBmm1Result(k-1): drain the async Bmm1 launched last tick ──
        #     so ProcessVec1(g-1) reads a *complete* S from mm1Res[(g-1)%2], and
        #     l0c[0] + that GM slot are free to be reused by Bmm1(g) below.
        #     (== AscendC WaitBmm1Result(extraInfo[(taskId+2)%3]).)
        # ---------------------------------------------------------------------
        if g >= 1 and (g - 1) < n_sub:
            prev_pp = (g - 1) % PP
            m1_done = tl.make_block_ptr(mm1Res + cid * (PP * BLOCK_M * BLOCK_N) + prev_pp * (BLOCK_M * BLOCK_N),
                                        (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
            dl.dsa.cube_wait(m1_done)   # token == the GM region Bmm1(g-1) targeted

        # ─── 2) IterateBmm1(k): async Q·Kᵀ -> mm1Res[g%2] ────────────────────
        if g < n_sub:
            t = g // n_iters
            j = g % n_iters
            cur_pp = g % PP
            task_id = my_start + t
            bx = task_id % num_seq_blocks
            by = (task_id // num_seq_blocks) % heads_q
            bz = task_id // (num_seq_blocks * heads_q)
            kv_by = by // gqa_group

            if j == 0:                              # new output tile: (re)load Q
                q_bp = tl.make_block_ptr(Q + bz * sQb + by * sQh, (S, DIM), (sQs, sQd),
                                         (bx * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
                dl.dsa.copy(tl.load(q_bp), q_l1)
            # K[j] -> k_l1 -> L0 slot 0
            k_bp = tl.make_block_ptr(K + bz * sKb + kv_by * sKh, (S, DIM), (sKs, sKd),
                                     (j * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
            dl.dsa.copy(tl.load(k_bp), k_l1)
            dl.dsa.copy(q_l1, l0a[0])
            dl.dsa.copy(k_l1, l0b[0], transpose=True)
            # async S = Q·Kᵀ : mma -> l0c[0] -> FIX -> mm1Res[cur_pp]. l0b[0]
            # already holds Kᵀ (copied transposed), so the iterate is a plain dot.
            # The token is drained next tick by step ①; meanwhile the Vector runs
            # Vec1(g-1) over mm1Res[(g-1)%2] while this Bmm1 is still iterating.
            m1_bp = tl.make_block_ptr(mm1Res + cid * (PP * BLOCK_M * BLOCK_N) + cur_pp * (BLOCK_M * BLOCK_N),
                                      (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
            dl.dsa.cube_launch(l0a[0], l0b[0], l0c[0], m1_bp, init=True)

        # ---------------------------------------------------------------------
        # ─── 3) ProcessVec1(k-1): softmax(mm1Res[prev]) -> stage1Res[prev] ───
        #     Runs on Vector while the Cube is busy with Bmm1(k). Online-softmax
        #     running max + P = exp(scale*S - m); records the per-sub-task carry.
        # ---------------------------------------------------------------------
        if g >= 1:
            g1 = g - 1
            if g1 < n_sub:
                j1   = g1 % n_iters
                pp1  = g1 % PP
                slot = g1 % RING
                cur = j1 % 2
                prv = 1 - cur

                if j1 == 0:                         # first KV block: reset running max
                    neg_sm[cur] = tl.full((BLOCK_M, 1), 2 ** 30, tl.float32)
                    neg_sm[prv] = tl.full((BLOCK_M, 1), 2 ** 30, tl.float32)

                # GM mm1Res[pp1] -> registers (MTE2)
                m1r_bp = tl.make_block_ptr(mm1Res + cid * (PP * BLOCK_M * BLOCK_N) + pp1 * (BLOCK_M * BLOCK_N),
                                           (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                s_tile = tl.load(m1r_bp).to(tl.float32)
                dl.dsa.set_flag(dl.dsa.Event.MTE2_V)
                dl.dsa.wait_flag(dl.dsa.Event.MTE2_V)

                if IS_CAUSAL:
                    t1  = g1 // n_iters
                    tid = my_start + t1
                    bx1 = tid % num_seq_blocks
                    q_idx  = bx1 * BLOCK_M + tl.arange(0, BLOCK_M)
                    kv_idx = j1 * BLOCK_N + tl.arange(0, BLOCK_N)
                    valid  = q_idx[:, None] >= kv_idx[None, :]
                    s_tile = tl.where(valid, s_tile, float("-inf"))

                # running max (stored negated & scaled: neg = -scale*max) + P = exp
                row_max = tl.max(s_tile, axis=-1, keep_dims=True)        # [BLOCK_M,1]
                neg_cur = -sm_scale * row_max
                neg_cur = tl.minimum(neg_cur, neg_sm[prv])               # most-neg == largest max
                neg_sm[cur] = neg_cur
                p_tile = tl.exp(sm_scale * s_tile + neg_cur)

                dl.dsa.pipe_barrier(dl.dsa.Pipe.V)

                # P -> GM stage1Res[pp1] (UB->GM, MTE3); signal Bmm2 it may read
                s1_bp = tl.make_block_ptr(stage1Res + cid * (PP * BLOCK_M * BLOCK_N) + pp1 * (BLOCK_M * BLOCK_N),
                                          (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                dl.dsa.set_flag(dl.dsa.Event.V_MTE3)
                dl.dsa.wait_flag(dl.dsa.Event.V_MTE3)
                tl.store(s1_bp, p_tile.to(stage1Res.dtype.element_ty))
                dl.dsa.set_flag(dl.dsa.Event.MTE3_MTE2)   # P[pp1] ready for Bmm2

                # per-sub-task carry: rowsum(P) and log-rescale factor for Vec2
                sumexp_is[slot] = tl.sum(p_tile, axis=-1, keep_dims=True)
                r_factors[slot] = neg_sm[cur] - neg_sm[prv]               # = m_prev - m_new

        # ---------------------------------------------------------------------
        # ─── 4) WaitBmm2Result(k-2): drain the async Bmm2 launched last tick ──
        #     so ProcessVec2(g-2) reads a *complete* O_partial from mm2Res[(g-2)%2]
        #     and l0c[1] is free for Bmm2(g-1) below. (== AscendC WaitBmm2Result().)
        # ---------------------------------------------------------------------
        if g >= 2 and (g - 2) < n_sub:
            prev2_pp = (g - 2) % PP
            m2_done = tl.make_block_ptr(mm2Res + cid * (PP * BLOCK_M * DIM) + prev2_pp * (BLOCK_M * DIM),
                                        (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
            dl.dsa.cube_wait(m2_done)   # token == the GM region Bmm2(g-2) targeted

        # ---------------------------------------------------------------------
        # ─── 5) IterateBmm2(k-1): async stage1Res[prev]·V -> mm2Res[prev] ────
        #     Issued AFTER Vec1(k-1) wrote P (wait MTE3_MTE2). Overlaps with
        #     Bmm1(k) on the Cube and Vec2(k-2) on the Vector.
        # ---------------------------------------------------------------------
        if g >= 1:
            g1 = g - 1
            if g1 < n_sub:
                t1   = g1 // n_iters
                j1   = g1 % n_iters
                pp1  = g1 % PP
                tid1 = my_start + t1
                by1  = (tid1 // num_seq_blocks) % heads_q
                bz1  = tid1 // (num_seq_blocks * heads_q)
                kv_by1 = by1 // gqa_group

                # P from stage1Res[pp1] -> p_l1 -> L0 slot 1
                dl.dsa.wait_flag(dl.dsa.Event.MTE3_MTE2)
                s1r_bp = tl.make_block_ptr(stage1Res + cid * (PP * BLOCK_M * BLOCK_N) + pp1 * (BLOCK_M * BLOCK_N),
                                           (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                dl.dsa.copy(tl.load(s1r_bp), p_l1)
                # V[j1] -> v_l1 -> L0 slot 1
                v_bp = tl.make_block_ptr(V + bz1 * sKb + kv_by1 * sKh, (S, DIM), (sKs, sKd),
                                         (j1 * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
                dl.dsa.copy(tl.load(v_bp), v_l1)
                dl.dsa.copy(p_l1, l0a[1])
                dl.dsa.copy(v_l1, l0b[1])
                # async O_part = P·V : mma -> l0c[1] -> FIX -> mm2Res[pp1].
                # Token drained next tick by step ④.
                m2_bp = tl.make_block_ptr(mm2Res + cid * (PP * BLOCK_M * DIM) + pp1 * (BLOCK_M * DIM),
                                          (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
                dl.dsa.cube_launch(l0a[1], l0b[1], l0c[1], m2_bp, init=True)

        # ---------------------------------------------------------------------
        # ─── 6) ProcessVec2(k-2): rescale acc_o + add mm2Res[prev2], finalize ─
        #     Runs on Vector while the Cube runs Bmm1(k) + Bmm2(k-1).
        # ---------------------------------------------------------------------
        if g >= 2:
            g2 = g - 2
            if g2 < n_sub:
                t2   = g2 // n_iters
                j2   = g2 % n_iters
                pp2  = g2 % PP
                slot2 = g2 % RING
                is_first = j2 == 0
                is_last  = j2 == NUM_ITERS - 1
                tid2 = my_start + t2
                bx2  = tid2 % num_seq_blocks
                by2  = (tid2 // num_seq_blocks) % heads_q
                bz2  = tid2 // (num_seq_blocks * heads_q)

                if is_first:                        # init O accumulator at tile start
                    acc_o  = tl.zeros((BLOCK_M, DIM), tl.float32)
                    sumexp = tl.zeros((BLOCK_M, 1), tl.float32)

                # apply this sub-task's rescale factor recorded by its Vec1
                rf = tl.exp(r_factors[slot2])       # exp(m_prev - m_new) in (0,1]
                sumexp = sumexp * rf + sumexp_is[slot2]
                acc_o  = acc_o * rf

                # O_partial from mm2Res[pp2] -> add (MTE2)
                m2r_bp = tl.make_block_ptr(mm2Res + cid * (PP * BLOCK_M * DIM) + pp2 * (BLOCK_M * DIM),
                                           (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
                o_part = tl.load(m2r_bp).to(tl.float32)
                dl.dsa.set_flag(dl.dsa.Event.MTE2_V)
                dl.dsa.wait_flag(dl.dsa.Event.MTE2_V)
                acc_o = acc_o + o_part

                if is_last:                         # last KV block: normalize + write O
                    out_tile = (acc_o / sumexp).to(Out.dtype.element_ty)
                    o_bp = tl.make_block_ptr(Out + bz2 * sOb + by2 * sOh, (S, DIM), (sOs, sOd),
                                             (bx2 * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
                    dl.dsa.set_flag(dl.dsa.Event.V_MTE3)
                    dl.dsa.wait_flag(dl.dsa.Event.V_MTE3)
                    tl.store(o_bp, out_tile)


# =============================================================================
#  Host launcher
# =============================================================================
def flash_attention_fwd(q, k, v, is_causal=False):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and S % BLOCK_N == 0 and Hq % Hkv == 0
    num_seq_blocks = S // BLOCK_M
    block_num = num_seq_blocks * Hq * B
    n_iters = S // BLOCK_N
    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    out = torch.empty_like(q)
    # GM ping-pong workspaces (taskId % 2), one slice per core.
    mm1Res    = torch.empty((NUM_CORES, PP, BLOCK_M, BLOCK_N), dtype=torch.float32, device=q.device)
    stage1Res = torch.empty((NUM_CORES, PP, BLOCK_M, BLOCK_N), dtype=q.dtype,        device=q.device)
    mm2Res    = torch.empty((NUM_CORES, PP, BLOCK_M, DIM),     dtype=torch.float32, device=q.device)
    sm_scale = (1.0 / D) ** 0.5

    grid = (NUM_CORES,)  # one program per AI core; one Cube + one Vector stream (MIX_1_1)
    flash_attention_fwd_3task_kernel[grid](
        q, k, v, out, mm1Res, stage1Res, mm2Res, sm_scale,
        B, Hq, Hkv, S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_seq_blocks, Hq, Hq // Hkv,
        n_iters, q_tasks, r_tasks,
        NUM_ITERS=n_iters,
        IS_CAUSAL=is_causal,
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D
    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    q = torch.randn((B, Q_H, S, D), dtype=torch.float16, device=device)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)

    out = flash_attention_fwd(q, k, v, is_causal=args.causal)

    if not args.no_check:
        def ref(q, k, v):
            if k.shape[1] != q.shape[1]:
                n_rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=args.causal).to(torch.float16)

        torch.testing.assert_close(ref(q, k, v), out, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
