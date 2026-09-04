import argparse
import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

import torch

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

L0_MAX_SIZE = 64 * 1024  # 64KB
NUM_CORES = 24  # 910B has 24 AI Cores

# ------------------------------------------------------------------------------
# arch22 "3-task" (Cube 双发) FlashAttention schedule, ported to TileLang-Ascend,
# with arch22's GLOBAL pipeline (§4.2) AND nRatio big-block tiling (§3) applied.
#
#   * 3-task pipeline (taskId % 3): three tasks kept in flight; the cube 双发
#     (dual-issues) MM1(g) and MM2(g-1); the vector dual-issues Vec1(g) softmax
#     and Vec2(g-1) rescale+accumulate.
#   * GLOBAL pipeline: the (tile × KV) loops are flattened into one task stream so
#     the pipeline never drains at tile boundaries -> the Cube stays continuously
#     fed (达成 MTE2/Cube bound).
#   * nRatio (§3): a *task* now covers N_RATIO consecutive KV blocks (S2 = 128*NR),
#     so a single cross-core handshake covers N_RATIO blocks -- "Cube一次发射大块的
#     数据，避免因为小块数据不断发射带来的通信开销". This cuts the number of CV
#     (cross-core) syncs by N_RATIO×. The Cube still issues 128×128 MMAs internally
#     (looped nr), but READY/FREE flags are raised once per task.
#
# A *task* = one (output S1-tile, S2 chunk of N_RATIO KV blocks).  tpt = num_kv //
# N_RATIO tasks per tile; GT = my_count * tpt global tasks per core.  The online
# softmax threads through (task, nr) in order (running -m*scale in neg_sm ping-pong,
# per-(task,nr) rescale r_factors and row-sum sumexp_is); state resets at the tile's
# first KV block (kv_global == 0) and finalizes (÷l, write Output) at the last
# (kv_global == num_kv-1).  Running sum/accumulator are updated only in Vec2, so the
# state stays single-buffered.  Q is resident per tile (loaded at the tile's first
# task, guarded by SIG_Q) so the cube never pipe-drains across tiles.
# ------------------------------------------------------------------------------

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

RING = 3  # depth of the task ring -- the "3-task" of the schedule


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(batch, seq_len, heads_q, heads_kv, dim, cross_interval=1,  # kept for CLI compatibility
                        n_ratio=8,  # arch22 nRatio: KV blocks per task (S2 = 128 * n_ratio)
                        ):
    assert heads_q % heads_kv == 0, "heads_q must be a multiple of heads_kv"
    block_M, block_N = 128, 128
    assert dim == 128, "dim must be 128"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim)**0.5

    shape_q = [batch, heads_q, seq_len, dim]
    shape_kv = [batch, heads_kv, seq_len, dim]

    num_seq_blocks = seq_len // block_M
    block_num = num_seq_blocks * heads_q * batch
    num_kv = seq_len // block_N  # KV blocks per output tile

    NR = n_ratio
    if num_kv < NR:
        NR = num_kv
    assert num_kv % NR == 0, f"num_kv ({num_kv}) must be divisible by n_ratio ({NR})"
    tpt = num_kv // NR  # tasks per output tile

    # Static task distribution across NUM_CORES (identical to the baseline).
    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # ---- Cross-core semaphores (one "ready" + one "free" per workspace) -------
    SEM_WS1_READY = 0  # C -> V : ws1 (=S) has data
    SEM_WS1_FREE = 1  # V -> C : ws1 slot free
    SEM_WS2_READY = 2  # V -> C : ws2 (=P) has data
    SEM_WS2_FREE = 3  # C -> V : ws2 slot free
    SEM_WS3_READY = 4  # C -> V : ws3 (=P·V) has data
    SEM_WS3_FREE = 5  # V -> C : ws3 slot free

    # ---- Intra-core signal IDs (C / cube scope) -------------------------------
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_L0AB = 3  # double-buffer base: slot s = SIG_L0AB + s   (uses 3, 4)
    SIG_L0C = 5  # double-buffer base: slot s = SIG_L0C  + s   (uses 5, 6)
    SIG_Q = 7  # resident-Q L1 reuse guard across tiles

    # ---- Intra-core signal IDs (V / vector scope) -----------------------------
    SIG_IO_UB = 0
    SIG_S_HALF = 1

    half_M = block_M // 2

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(Q: T.Tensor(shape_q, dtype),  # type: ignore
             K: T.Tensor(shape_kv, dtype),  # type: ignore
             V: T.Tensor(shape_kv, dtype),  # type: ignore
             Output: T.Tensor(shape_q, dtype),  # type: ignore
             workspace_1: T.Tensor([NUM_CORES, RING, NR, block_M, block_N], dtype),  # S
             workspace_2: T.Tensor([NUM_CORES, RING, NR, block_M, block_N], dtype),  # P
             workspace_3: T.Tensor([NUM_CORES, RING, NR, block_M, dim], dtype),  # P·V
             ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # ---- L1 buffers (cube) ----
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout({
                q_l1: make_zn_layout(q_l1),
                k_l1: make_nz_layout(k_l1),
                p_l1: make_zn_layout(p_l1),
                v_l1: make_zn_layout(v_l1),
            })

            # ---- L0 buffers (cube), 2-deep double buffer (slot = nr % 2) ----
            l0a = T.alloc_L0A([2, block_M, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            # ---- UB buffers (vector) ----  (single-buffered running state)
            acc_o = T.alloc_ub([half_M, dim], accum_dtype)

            # per-(task,nr) rescale factor and partial row-sum
            r_factors = T.alloc_ub([RING, NR, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([RING, NR, half_M, 1], accum_dtype)

            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, half_M, 1], accum_dtype)  # running -m*scale, ping-pong

            io_buf = T.alloc_ub([half_M, block_N], dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)
            out_half = T.alloc_ub([half_M, dim], dtype)

            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)

            my_start, my_count = task_range(cid)
            GT = my_count * tpt  # total global tasks on this core

            # ==================================================================
            # CUBE scope: MM1(g) and MM2(g-1) dual-issued; each task = NR MMAs.
            # ==================================================================
            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_WS2_FREE)
                T.set_cross_flag("MTE2", SEM_WS2_FREE)
                T.set_cross_flag("MTE2", SEM_WS2_FREE)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)
                T.set_flag("MTE1", "MTE2", SIG_Q)

                for g in T.serial(GT + 1):
                    # ===== MM1(g): S = Q · K_kvᵀ for NR blocks -> ws1[cid, g%3, :] =====
                    if g < GT:
                        tit = g % tpt
                        tl = g // tpt
                        task_id = my_start + tl
                        bx = task_id % num_seq_blocks
                        by = (task_id // num_seq_blocks) % heads_q
                        bz = task_id // (num_seq_blocks * heads_q)
                        kv_by = by // (heads_q // heads_kv)
                        r1 = g % RING

                        T.wait_cross_flag(SEM_WS1_FREE)  # ws1[r1] task slot free

                        if tit == 0:  # resident Q reload at the first task of each tile
                            T.wait_flag("MTE1", "MTE2", SIG_Q)
                            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)
                            T.set_flag("MTE2", "MTE1", SIG_Q)
                            T.wait_flag("MTE2", "MTE1", SIG_Q)

                        for nr in T.serial(NR):
                            kv = tit * NR + nr
                            s1 = nr % 2

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, kv_by, kv * block_N:(kv + 1) * block_N, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            T.wait_flag("M", "MTE1", SIG_L0AB + s1)
                            T.copy(q_l1, l0a[s1, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b[s1, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + s1)

                            T.wait_flag("MTE1", "M", SIG_L0AB + s1)
                            T.wait_flag("FIX", "M", SIG_L0C + s1)
                            T.mma(l0a[s1, :, :], l0b[s1, :, :], l0c[s1, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + s1)
                            T.set_flag("M", "FIX", SIG_L0C + s1)

                            T.wait_flag("M", "FIX", SIG_L0C + s1)
                            T.copy(l0c[s1, :, :], workspace_1[cid, r1, nr, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + s1)

                        T.set_cross_flag("FIX", SEM_WS1_READY)  # ws1[r1] (all NR) ready

                        if tit == tpt - 1:  # release resident Q at the last task of the tile
                            T.set_flag("MTE1", "MTE2", SIG_Q)

                    # ===== MM2(g-1): O_partial = P · V for NR blocks -> ws3[cid, (g-1)%3, :] =====
                    if g >= 1:
                        gm = g - 1
                        tit2 = gm % tpt
                        tl2 = gm // tpt
                        task_id2 = my_start + tl2
                        by2 = (task_id2 // num_seq_blocks) % heads_q
                        bz2 = task_id2 // (num_seq_blocks * heads_q)
                        kv_by2 = by2 // (heads_q // heads_kv)
                        r2 = gm % RING

                        T.wait_cross_flag(SEM_WS2_READY)  # P (all NR) ready in ws2[r2]
                        for nr in T.serial(NR):
                            kv = tit2 * NR + nr
                            s2 = nr % 2

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz2, kv_by2, kv * block_N:(kv + 1) * block_N, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            T.copy(workspace_2[cid, r2, nr, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + s2)
                            T.copy(v_l1, l0b[s2, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[s2, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + s2)

                            T.wait_flag("MTE1", "M", SIG_L0AB + s2)
                            T.wait_flag("FIX", "M", SIG_L0C + s2)
                            T.mma(l0a[s2, :, :], l0b[s2, :, :], l0c[s2, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + s2)
                            T.set_flag("M", "FIX", SIG_L0C + s2)

                            T.wait_flag("M", "FIX", SIG_L0C + s2)
                            T.copy(l0c[s2, :, :], workspace_3[cid, r2, nr, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + s2)

                        T.set_cross_flag("FIX", SEM_WS3_READY)  # ws3[r2] (all NR) ready
                        T.set_cross_flag("MTE2", SEM_WS2_FREE)  # ws2[r2] task slot free

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)
                T.wait_flag("MTE1", "MTE2", SIG_Q)

            # ==================================================================
            # VECTOR scope: Vec1(g) softmax and Vec2(g-1) accumulate; NR sub-blocks.
            # ==================================================================
            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_WS1_FREE)
                T.set_cross_flag("MTE2", SEM_WS1_FREE)
                T.set_cross_flag("MTE2", SEM_WS1_FREE)
                T.set_cross_flag("MTE2", SEM_WS3_FREE)
                T.set_cross_flag("MTE2", SEM_WS3_FREE)
                T.set_cross_flag("MTE2", SEM_WS3_FREE)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for g in T.serial(GT + 1):
                    # ===== Vec1(g): softmax of NR blocks ws1[g%3,:] -> ws2[g%3,:] =====
                    if g < GT:
                        tit = g % tpt
                        r1 = g % RING

                        T.wait_cross_flag(SEM_WS1_READY)  # S (all NR) ready in ws1[r1]
                        if tit == 0:  # reset running max at the tile's first KV block
                            T.tile.fill(neg_sm, 2**30)

                        for nr in T.serial(NR):
                            kv = tit * NR + nr
                            cur = kv % 2
                            prv = 1 - cur

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            T.copy(workspace_1[cid, r1, nr, vid * half_M:vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)  # work_ub = P

                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, workspace_2[cid, r1, nr, vid * half_M:vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)

                            T.reduce_sum(work_ub, sumexp_is[r1, nr, :, :], dim=-1)
                            T.tile.sub(r_factors[r1, nr, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                        T.set_cross_flag("MTE2", SEM_WS1_FREE)  # ws1[r1] task slot free
                        T.set_cross_flag("MTE3", SEM_WS2_READY)  # ws2[r1] (all NR) ready

                    # ===== Vec2(g-1): acc_o = acc_o*α + P·V for NR blocks =====
                    if g >= 1:
                        gm = g - 1
                        tit2 = gm % tpt
                        tl2 = gm // tpt
                        r2 = gm % RING

                        T.wait_cross_flag(SEM_WS3_READY)  # P·V (all NR) ready in ws3[r2]
                        for nr in T.serial(NR):
                            kv = tit2 * NR + nr

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            T.copy(workspace_3[cid, r2, nr, vid * half_M:vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)  # work_ub = P·V
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            if kv == 0:
                                T.copy(work_ub, acc_o)  # acc_o = P·V
                                T.copy(sumexp_is[r2, nr, :, :], sumexp)  # sumexp = rowsum
                            else:
                                T.tile.exp(r_factors[r2, nr, :, :], r_factors[r2, nr, :, :])
                                T.tile.mul(sumexp, sumexp, r_factors[r2, nr, :, :])
                                T.tile.add(sumexp, sumexp, sumexp_is[r2, nr, :, :])
                                T.tile.broadcast(buf_2d, r_factors[r2, nr, :, :])
                                T.tile.mul(acc_o, acc_o, buf_2d)
                                T.tile.add(acc_o, acc_o, work_ub)

                            if kv == num_kv - 1:
                                task_id2 = my_start + tl2
                                bx2 = task_id2 % num_seq_blocks
                                by2 = (task_id2 // num_seq_blocks) % heads_q
                                bz2 = task_id2 // (num_seq_blocks * heads_q)
                                T.tile.broadcast(buf_2d, sumexp)
                                T.tile.div(acc_o, acc_o, buf_2d)
                                T.copy(acc_o, out_half)
                                T.barrier_all()
                                T.copy(
                                    out_half,
                                    Output[bz2, by2,
                                           bx2 * block_M + vid * half_M:bx2 * block_M + vid * half_M + half_M, :],
                                )

                        T.set_cross_flag("MTE2", SEM_WS3_FREE)  # ws3[r2] task slot free

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4, help="batch size")
    parser.add_argument("--S", type=int, default=4096, help="seq len")
    parser.add_argument("--H", type=int, default=16, help="attention head size")
    parser.add_argument("--q-heads", type=int, default=None, help="query head size")
    parser.add_argument("--kv-heads", type=int, default=None, help="kv head size")
    parser.add_argument("--D", type=int, default=128, help="hidden dim")
    parser.add_argument("--no-check", action="store_true", help="disable reference check")
    parser.add_argument("--cross-interval", type=int, default=2, help="cross-core signal interval")
    parser.add_argument("--n-ratio", type=int, default=8, help="KV blocks per task (arch22 nRatio)")
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D
    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    func = flash_attention_fwd(
        batch=B,
        seq_len=S,
        heads_q=Q_H,
        heads_kv=KV_H,
        dim=D,
        cross_interval=args.cross_interval,
        n_ratio=args.n_ratio,
    )
    print(func.get_kernel_source())

    def ref_flash_attn(q, k, v):
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q = q.float()
        k = k.float()
        v = v.float()

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                                                  is_causal=False)
        return output.to(torch.float16)

    q = torch.randn((B, Q_H, S, D), dtype=torch.float16)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16)

    torch.npu.synchronize()
    print("init successful!")

    output = func(q, k, v)
    torch.npu.synchronize()

    if not args.no_check:
        ref_output = ref_flash_attn(q, k, v)
        torch.npu.synchronize()
        torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
