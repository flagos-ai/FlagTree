# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import torch_npu
import triton
import triton.language as tl
import numpy as np
from datetime import datetime
from triton.backends.ascend.testing import do_bench_npu
import triton.experimental.tle as tle
# import random

np.random.seed(21)
DEVICE = "npu"
DEVICE_ID = 0
torch.manual_seed(20)
torch_npu.npu.set_device(int(DEVICE_ID))
torch.set_printoptions(sci_mode=False, precision=4, linewidth=300)

ascend_aiv_core_nums = triton.language.constexpr(24)


# ===== Fused PA + Rope Concat + BNSD + Gather Kernel =====
@triton.jit
def fused_pa_rope_to_sparse_kernel(
    k_pa_ptr,
    k_rope_pa_ptr,
    v_pa_ptr,  # PA_BSND input [block_num, block_size, n, d]
    block_table_ptr,  # block_table [B, max_blocks]
    sparse_indices_ptr,  # sparse_indices [B, N, TOPK]
    k_sparse_out_ptr,
    v_sparse_out_ptr,  # BNSD output [B, N, TOPK, d]
    stride_k_pa_bn,
    stride_k_pa_bs,
    stride_k_pa_n,
    stride_k_pa_d,  # K PA strides
    stride_k_rope_pa_bn,
    stride_k_rope_pa_bs,
    stride_k_rope_pa_n,
    stride_k_rope_pa_d,  # K_rope PA strides
    stride_v_pa_bn,
    stride_v_pa_bs,
    stride_v_pa_n,
    stride_v_pa_d,  # V PA strides
    stride_bt_b,
    stride_bt_blk,  # block_table strides
    stride_si_b,
    stride_si_n,
    stride_si_topk,  # sparse_indices strides
    stride_out_b,
    stride_out_n,
    stride_out_topk,
    stride_out_d,  # output strides
    stride_v_b,
    stride_v_n,
    stride_v_topk,
    stride_v_d,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DK_ROPE: tl.constexpr,  # 0 if no rope
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    """
    Fused kernel: PA_BSND + Rope Concat -> BNSD Sparse
    Input: K/V in PA_BSND format, K_rope in PA_BSND format
    Output: K/V_sparse in BNSD format
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Process (b, n, topk) combinations
    for b_idx in range(B):
        b = b_idx  # sparse_indices is [B, N, TOPK], assume B=1 for now
        for idx in range(pid, TOPK, num_programs):
            # Get batch and sparse index from sparse_indices
            n = 0  # KV_N = 1

            # Load sparse index
            sparse_idx = tl.load(sparse_indices_ptr + b * stride_si_b + n * stride_si_n + idx * stride_si_topk)

            # Map sparse_idx to PA_BSND position
            block_id = sparse_idx // BLOCK_SIZE  # Which block
            bs_offset = sparse_idx % BLOCK_SIZE  # Offset within block

            # Get actual block ID from block_table
            actual_block_id = tl.load(block_table_ptr + b * stride_bt_b + block_id * stride_bt_blk)

            # Compute PA_BSND offset for K
            k_pa_offset = (actual_block_id * stride_k_pa_bn + bs_offset * stride_k_pa_bs + n * stride_k_pa_n)

            # Compute PA_BSND offset for K_rope
            k_rope_pa_offset = (actual_block_id * stride_k_rope_pa_bn + bs_offset * stride_k_rope_pa_bs +
                                n * stride_k_rope_pa_n)

            # Compute PA_BSND offset for V
            v_pa_offset = (actual_block_id * stride_v_pa_bn + bs_offset * stride_v_pa_bs + n * stride_v_pa_n)
            # Load K vector (no rope part)
            k_vec = tl.load(k_pa_ptr + k_pa_offset + tl.arange(0, BLOCK_DK) * stride_k_pa_d)

            # Load V vector
            v_vec = tl.load(v_pa_ptr + v_pa_offset + tl.arange(0, BLOCK_DV) * stride_v_pa_d)
            # Output to BNSD format: [B, N, TOPK, D]
            out_offset = b * stride_out_b + n * stride_out_n + idx * stride_out_topk
            out_offset_v = b * stride_v_b + n * stride_v_n + idx * stride_v_topk

            if BLOCK_DK_ROPE > 0:
                # Load K_rope vector
                full_k = tl.full((BLOCK_DK + BLOCK_DK_ROPE, ), 0.0, dtype=tl.float16)
                k_rope_vec = tl.load(k_rope_pa_ptr + k_rope_pa_offset +
                                     tl.arange(0, BLOCK_DK_ROPE) * stride_k_rope_pa_d)
                full_k = tle.dsa.insert_slice(full_k, k_vec, offsets=(0, ), sizes=(BLOCK_DK, ), strides=(1, ))
                full_k = tle.dsa.insert_slice(full_k, k_rope_vec, offsets=(BLOCK_DK, ), sizes=(BLOCK_DK_ROPE, ),
                                              strides=(1, ))
                tl.store(k_sparse_out_ptr + out_offset + tl.arange(0, BLOCK_DK + BLOCK_DK_ROPE) * stride_out_d, full_k)
            else:
                # No rope, store K directly
                tl.store(k_sparse_out_ptr + out_offset + tl.arange(0, BLOCK_DK) * stride_out_d, k_vec)

            # Store V
            tl.store(v_sparse_out_ptr + out_offset_v + tl.arange(0, BLOCK_DV) * stride_v_d, v_vec)


def triton_fused_pa_rope_to_sparse(k_pa, k_rope_pa, v_pa, block_table, sparse_indices, block_size):
    """
    Fused PA_BSND + Rope Concat -> BNSD Sparse conversion

    Args:
        k_pa: Key in PA_BSND format [block_num, block_size, n, dk]
        k_rope_pa: Key rope in PA_BSND format [block_num, block_size, n, d_rope], None if no rope
        v_pa: Value in PA_BSND format [block_num, block_size, n, dv]
        block_table: Block table [B, max_blocks]
        sparse_indices: Sparse indices [B, N, TOPK]
        block_size: Block size for PA format

    Returns:
        k_sparse: Sparse key in BNSD format [B, N, TOPK, dk+d_rope]
        v_sparse: Sparse value in BNSD format [B, N, TOPK, dv]
    """
    block_num, _, n, dk = k_pa.shape
    B = block_table.shape[0]
    TOPK = sparse_indices.size(-1)
    N = 1  # KV_N = 1
    _, _, _, dv = v_pa.shape

    has_rope = k_rope_pa is not None
    dk_rope = k_rope_pa.shape[-1] if has_rope else 0
    dk_total = dk + dk_rope

    # Output BNSD format [B, N, TOPK, D]
    k_sparse = torch.empty((B, N, TOPK, dk_total), dtype=k_pa.dtype, device=DEVICE)
    v_sparse = torch.empty((B, N, TOPK, dv), dtype=v_pa.dtype, device=DEVICE)

    # Grid: use 48 programs for parallelism
    grid = (min(48, TOPK), )

    # sparse_indices input format: [T, N, TOPK] or [B, N, TOPK]
    # No squeeze needed - kernel expects [B, N, TOPK] format
    sparse_indices_input = sparse_indices
    if sparse_indices.dim() == 2:
        # If already 2D [B, TOPK], reshape to [B, 1, TOPK]
        sparse_indices_input = sparse_indices.unsqueeze(1)

    # Set k_rope_pa to k_pa if no rope (dummy pointer, won't be accessed)
    k_rope_pa_input = k_rope_pa if has_rope else k_pa
    fused_pa_rope_to_sparse_kernel[grid](k_pa, k_rope_pa_input, v_pa, block_table, sparse_indices_input,
                                         k_sparse, v_sparse, k_pa.stride(0), k_pa.stride(1), k_pa.stride(2),
                                         k_pa.stride(3), k_rope_pa_input.stride(0), k_rope_pa_input.stride(1),
                                         k_rope_pa_input.stride(2), k_rope_pa_input.stride(3), v_pa.stride(0),
                                         v_pa.stride(1), v_pa.stride(2), v_pa.stride(3), block_table.stride(0),
                                         block_table.stride(1), sparse_indices_input.stride(0),
                                         sparse_indices_input.stride(1), sparse_indices_input.stride(2),
                                         k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
                                         v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
                                         BLOCK_DK=dk, BLOCK_DV=dv, BLOCK_DK_ROPE=dk_rope, TOPK=TOPK,
                                         BLOCK_SIZE=block_size, B=B)

    return k_sparse, v_sparse


@triton.jit
def gather_kv_bnsd_vec_kernel(
    k_ptr,
    v_ptr,
    ind_ptr,
    k_out_ptr,
    v_out_ptr,
    stride_kb,
    stride_kn,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_on,
    stride_os,
    stride_od,
    stride_ovb,
    stride_ovn,
    stride_ovs,
    stride_ovd,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    TOPK: tl.constexpr,
    B: tl.constexpr,
):
    end = TOPK // 48 * 48
    for b_idx in range(B):
        # 分批处理所有TOPK个索引，每次48个
        for batch_start in range(0, end, 48):
            pid_k = tl.program_id(0) + batch_start

            # 读 index
            idx = tl.load(ind_ptr + pid_k)

            # 加载 K 向量 [BLOCK_DK] - 直接线性加载
            k_src_off = idx * stride_ks + b_idx * stride_kb
            k_val = tl.load(k_ptr + k_src_off + tl.arange(0, BLOCK_DK) * stride_kd)

            # 加载 V 向量 [BLOCK_DV] - 直接线性加载
            v_src_off = idx * stride_vs + b_idx * stride_vb
            v_val = tl.load(v_ptr + v_src_off + tl.arange(0, BLOCK_DV) * stride_vd)

            # 写回 K: [B, N, TOPK, Dk]
            k_dst_off = pid_k * stride_os + b_idx * stride_ob
            tl.store(k_out_ptr + k_dst_off + tl.arange(0, BLOCK_DK) * stride_od, k_val)

            # 写回 V: [B, N, TOPK, Dv]
            v_dst_off = pid_k * stride_ovs + b_idx * stride_ovb
            tl.store(v_out_ptr + v_dst_off + tl.arange(0, BLOCK_DV) * stride_ovd, v_val)

        # 处理余数部分（end到TOPK）
        for batch_start in range(end, TOPK, 48):
            pid_k = tl.program_id(0) + batch_start

            # 必须在计算pid_k之后检查边界
            if pid_k < TOPK:
                idx = tl.load(ind_ptr + pid_k)

                # 加载 K 向量 [BLOCK_DK] - 直接线性加载
                k_src_off = idx * stride_ks + b_idx * stride_kb
                k_val = tl.load(k_ptr + k_src_off + tl.arange(0, BLOCK_DK) * stride_kd)

                # 加载 V 向量 [BLOCK_DV] - 直接线性加载
                v_src_off = idx * stride_vs + b_idx * stride_vb
                v_val = tl.load(v_ptr + v_src_off + tl.arange(0, BLOCK_DV) * stride_vd)

                # 写回 K: [B, N, TOPK, Dk]
                k_dst_off = pid_k * stride_os + b_idx * stride_ob
                tl.store(k_out_ptr + k_dst_off + tl.arange(0, BLOCK_DK) * stride_od, k_val)

                # 写回 V: [B, N, TOPK, Dv]
                v_dst_off = pid_k * stride_ovs + b_idx * stride_ovb
                tl.store(v_out_ptr + v_dst_off + tl.arange(0, BLOCK_DV) * stride_ovd, v_val)


def triton_gather_kv_bnsd_vec(k, v, indices):
    B, N, SK, Dk = k.shape  # N=1
    B, N, SK, Dv = v.shape  # N=1
    TOPK = indices.size(-1)

    # 输出保持 bnsd [B, N, TOPK, D]
    k_sparse = torch.empty((B, N, TOPK, Dk), dtype=k.dtype, device=DEVICE)
    v_sparse = torch.empty((B, N, TOPK, Dv), dtype=v.dtype, device=DEVICE)

    grid = (48, )  # TOPK 个 program，每个搬 Dk/Dv 元素
    gather_kv_bnsd_vec_kernel[grid](
        k,
        v,
        indices.squeeze(0),  # [B, N, SK, D] -> [N, SK, D]
        k_sparse,
        v_sparse,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        k_sparse.stride(0),
        k_sparse.stride(1),
        k_sparse.stride(2),
        k_sparse.stride(3),
        v_sparse.stride(0),
        v_sparse.stride(1),
        v_sparse.stride(2),
        v_sparse.stride(3),
        BLOCK_DK=Dk,
        BLOCK_DV=Dv,
        TOPK=TOPK,
        B=B,
    )
    return k_sparse, v_sparse


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    O,
    scale_value,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_on: tl.constexpr,
    stride_od: tl.constexpr,
    B: tl.constexpr,
    Q_N: tl.constexpr,
    Q_D: tl.constexpr,
    Q_S: tl.constexpr,
    KV_S: tl.constexpr,
    K_D: tl.constexpr,
    V_D: tl.constexpr,
    sparse_mode: tl.constexpr,  # 0 or 3
    O_N: tl.constexpr,
    O_D: tl.constexpr,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    blk_size: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
):
    # total b * n tasks
    BLOCK_QN_NUM = Q_N // Q_BLOCK_SIZE
    NUM_BLOCKS = B * Q_S * BLOCK_QN_NUM
    pid = tl.program_id(0)
    num_cores = min(ascend_aiv_core_nums, NUM_BLOCKS)

    #最外层循环，沿b*n切
    for block_idx in range(pid, NUM_BLOCKS, num_cores):  # 并行
        off_b = (block_idx // (Q_S * BLOCK_QN_NUM)).to(tl.int32)  #当前任务在第几个b块中
        off_s = ((block_idx // BLOCK_QN_NUM) % Q_S).to(tl.int32)  #当前任务在第几个s块中
        off_n = (block_idx % BLOCK_QN_NUM).to(tl.int32)  #当前任务在第几个n块中
        # off_n = 0

        q_offset = off_b * stride_qb + off_s * stride_qs
        o_offset = off_b * stride_ob + off_s * stride_os
        k_offset = off_b * stride_kb  # KV_N = 1
        v_offset = off_b * stride_vb

        cur_act_s_q = tl.load(actual_seq_lengths_query + off_b)

        for i in range(cur_act_s_q):
            cur_max = tl.full((Q_BLOCK_SIZE, ), float('-inf'), dtype=tl.float32)
            logSum = tl.zeros((Q_BLOCK_SIZE, ), dtype=tl.float32)
            acc = tl.zeros((Q_BLOCK_SIZE, V_D), dtype=tl.float32)  # 升维到[q_block_size, V_D]

            # load q
            q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(Q_N, Q_D), strides=(stride_qn, stride_qd),
                                            offsets=(off_n * Q_BLOCK_SIZE, 0), block_shape=(Q_BLOCK_SIZE, Q_D),
                                            order=(1, 0))
            q_vec = tl.load(q_block_ptr, boundary_check=(0, 1))  # [q_block_size, K_D]
            k_block_ptr = tl.make_block_ptr(
                base=K + k_offset,
                shape=(KV_S, K_D),
                strides=(stride_ks, stride_kd),
                offsets=(0, 0),
                block_shape=(blk_size, K_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(KV_S, V_D), strides=(stride_vs, stride_vd),
                                            offsets=(0, 0), block_shape=(blk_size, V_D), order=(1, 0))

            for k_idx in range(KV_S // blk_size):
                # load k
                k_vec = tl.load(k_block_ptr, boundary_check=(0, 1))

                # 使用dot加速：[blk_size, K_D] @ [K_D] -> [q_block_size, blk_size]
                qk = tl.dot(q_vec.to(tl.float16),
                            tl.trans(k_vec).to(tl.float16)) * scale_value  # [q_block_size, blk_size]
                # online softmax update
                # Triton's tl.max doesn't accept keyword 'dim'; use positional axis.
                block_max = tl.max(qk, axis=1)  # [q_block_size]
                # align shapes to (q_block_size, 1) for broadcasting
                # block_max = block_max[:, None]            # [q_block_size, 1]
                new_max = tl.maximum(cur_max, block_max)  # [q_block_size, 1]
                coeff = tl.math.exp(cur_max - new_max)  # [q_block_size, 1]
                p = tl.math.exp(qk - new_max[:, None])  # [q_block_size, blk_size]
                # logsum per row
                logSum = logSum * coeff + tl.sum(p, axis=1)  # [q_block_size, 1]

                # update accumulator: compute per-row pv by summing over block dim
                v_vec = tl.load(v_block_ptr, boundary_check=(0, 1))  # [blk_size, V_D]
                pv = tl.dot(p.to(tl.float16), v_vec)  # [q_block_size, V_D]
                acc = acc * coeff[:, None] + pv  # [q_block_size, V_D]
                cur_max = new_max

                k_block_ptr = k_block_ptr.advance((blk_size, 0))
                v_block_ptr = v_block_ptr.advance((blk_size, 0))

            o_block_ptr = tl.make_block_ptr(base=O + o_offset, shape=(O_N, O_D), strides=(stride_on, stride_od),
                                            offsets=(off_n * Q_BLOCK_SIZE, 0), block_shape=(Q_BLOCK_SIZE, O_D),
                                            order=(1, 0))
            # final normalize
            acc = acc / logSum[:, None]  # [q_block_size, V_D] / [q_block_size,1] -> [q_block_size, V_D]
            tl.store(o_block_ptr, acc)


@triton.jit
def _attn_fwd_fused_bsnd_to_tnd(
    Q,
    K,
    V,
    O,
    scale_value,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_on: tl.constexpr,
    stride_od: tl.constexpr,
    B: tl.constexpr,
    Q_N: tl.constexpr,
    Q_D: tl.constexpr,
    Q_S: tl.constexpr,
    KV_S: tl.constexpr,
    K_D: tl.constexpr,
    V_D: tl.constexpr,
    sparse_mode: tl.constexpr,  # 0 or 3
    O_N: tl.constexpr,
    O_D: tl.constexpr,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    blk_size: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
):
    # total b * n tasks
    BLOCK_QN_NUM = Q_N // Q_BLOCK_SIZE
    NUM_BLOCKS = B * Q_S * BLOCK_QN_NUM
    pid = tl.program_id(0)
    num_cores = min(ascend_aiv_core_nums, NUM_BLOCKS)

    #最外层循环，沿b*n切
    for block_idx in range(pid, NUM_BLOCKS, num_cores):  # 并行
        off_b = (block_idx // (Q_S * BLOCK_QN_NUM)).to(tl.int32)  #当前任务在第几个b块中
        off_s = ((block_idx // BLOCK_QN_NUM) % Q_S).to(tl.int32)  #当前任务在第几个s块中
        off_n = (block_idx % BLOCK_QN_NUM).to(tl.int32)  #当前任务在第几个n块中

        q_offset = off_b * stride_qb + off_s * stride_qs
        o_offset = off_b * stride_ot
        k_offset = off_b * stride_kb  # KV_N = 1
        v_offset = off_b * stride_vb

        cur_act_s_q = tl.load(actual_seq_lengths_query + off_b)

        for i in range(cur_act_s_q):
            cur_max = tl.full((Q_BLOCK_SIZE, ), float('-inf'), dtype=tl.float32)
            logSum = tl.zeros((Q_BLOCK_SIZE, ), dtype=tl.float32)
            acc = tl.zeros((Q_BLOCK_SIZE, V_D), dtype=tl.float32)  # 升维到[q_block_size, V_D]

            # load q
            q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(Q_N, Q_D), strides=(stride_qn, stride_qd),
                                            offsets=(off_n * Q_BLOCK_SIZE, 0), block_shape=(Q_BLOCK_SIZE, Q_D),
                                            order=(1, 0))
            q_vec = tl.load(q_block_ptr, boundary_check=(0, 1))  # [q_block_size, K_D]
            k_block_ptr = tl.make_block_ptr(
                base=K + k_offset,
                shape=(KV_S, K_D),
                strides=(stride_ks, stride_kd),
                offsets=(0, 0),
                block_shape=(blk_size, K_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(KV_S, V_D), strides=(stride_vs, stride_vd),
                                            offsets=(0, 0), block_shape=(blk_size, V_D), order=(1, 0))

            for k_idx in range(KV_S // blk_size):
                # load k
                k_vec = tl.load(k_block_ptr, boundary_check=(0, 1))

                # 使用dot加速：[blk_size, K_D] @ [K_D] -> [q_block_size, blk_size]
                qk = tl.dot(q_vec.to(tl.float16),
                            tl.trans(k_vec).to(tl.float16)) * scale_value  # [q_block_size, blk_size]
                # online softmax update
                # Triton's tl.max doesn't accept keyword 'dim'; use positional axis.
                block_max = tl.max(qk, axis=1)  # [q_block_size]
                # align shapes to (q_block_size, 1) for broadcasting
                # block_max = block_max[:, None]            # [q_block_size, 1]
                new_max = tl.maximum(cur_max, block_max)  # [q_block_size, 1]
                coeff = tl.math.exp(cur_max - new_max)  # [q_block_size, 1]
                p = tl.math.exp(qk - new_max[:, None])  # [q_block_size, blk_size]
                # logsum per row
                logSum = logSum * coeff + tl.sum(p, axis=1)  # [q_block_size, 1]

                # update accumulator: compute per-row pv by summing over block dim
                v_vec = tl.load(v_block_ptr, boundary_check=(0, 1))  # [blk_size, V_D]
                pv = tl.dot(p.to(tl.float16), v_vec)  # [q_block_size, V_D]
                acc = acc * coeff[:, None] + pv  # [q_block_size, V_D]
                cur_max = new_max

                k_block_ptr = k_block_ptr.advance((blk_size, 0))
                v_block_ptr = v_block_ptr.advance((blk_size, 0))

            o_block_ptr = tl.make_block_ptr(base=O + o_offset, shape=(O_N, O_D), strides=(stride_on, stride_od),
                                            offsets=(off_n * Q_BLOCK_SIZE, 0), block_shape=(Q_BLOCK_SIZE, O_D),
                                            order=(1, 0))
            # final normalize
            acc = acc / logSum[:, None]  # [q_block_size, V_D] / [q_block_size,1] -> [q_block_size, V_D]
            tl.store(o_block_ptr, acc)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query, key, value, sparse_indices, scale_value, sparse_block_size=1, actual_seq_lengths_query=None,
                actual_seq_lengths_kv=None, query_rope=None, key_rope=None, layout_query='BSND', layout_kv='BSND',
                sparse_mode=0, block_table=None):
        # Save original sparse_indices for PA_BSND case
        sparse_indices_orig = sparse_indices.clone()
        total_len = 0
        # Handle query layout transformation (TND -> BSND)
        if layout_query == 'TND':
            actual_seq_lengths_query, total_len = trans_tnd_actseq(actual_seq_lengths_query)
            # ✅ 融合版本：一次 kernel 调用处理所有 tensor + concat
            query, sparse_indices = trans_tnd_to_bsnd_fused(query, query_rope, sparse_indices, query.shape,
                                                            actual_seq_lengths_query)
        else:
            if query_rope is not None:
                query = torch.cat([query, query_rope], dim=-1)

        # Handle KV layout and gather sparse K/V
        if layout_kv == 'PA_BSND':
            # Fused PA -> BNSD + rope concat + sparse gather
            block_size = key.shape[1]  # Get block_size from PA shape
            # Use original sparse_indices [T, N, TOPK] for fused kernel
            k_sparse, v_sparse = triton_fused_pa_rope_to_sparse(key, key_rope, value, block_table, sparse_indices_orig,
                                                                block_size)
            # sparse_indices is already in BSND, needs permute to BNSD for downstream use
            sparse_indices_bnsd = sparse_indices.permute(0, 2, 1, 3).contiguous()
        else:
            # Original path for non-PA layouts
            if key_rope is not None:
                key = torch.cat([key, key_rope], dim=-1)
            key_bnsd = key.permute(0, 2, 1, 3).contiguous()
            value_bnsd = value.permute(0, 2, 1, 3).contiguous()
            sparse_indices_bnsd = sparse_indices.permute(0, 2, 1, 3).contiguous()

            k_sparse, v_sparse = triton_gather_kv_bnsd_vec(key_bnsd, value_bnsd, sparse_indices_bnsd)

        k_sparse = k_sparse.contiguous()
        v_sparse = v_sparse.contiguous()
        enable_check_kv_sparse = 0
        if enable_check_kv_sparse:
            key = pa_to_bsnd(key, block_table, actual_seq_lengths_kv)
            key_rope = pa_to_bsnd(key_rope, block_table, actual_seq_lengths_kv)
            value = pa_to_bsnd(value, block_table, actual_seq_lengths_kv)
            if key_rope is not None:
                key = torch.cat([key, key_rope], dim=-1)
            key_bnsd = key.permute(0, 2, 1, 3).contiguous()
            value_bnsd = value.permute(0, 2, 1, 3).contiguous()
            k_sparse_ref, v_sparse_ref = triton_gather_kv_bnsd_vec(key_bnsd, value_bnsd, sparse_indices_bnsd)
            print(f"k_sparse={k_sparse}")
            print(f"k_sparse_ref={k_sparse_ref}")
            print(f"v_sparse={v_sparse}")
            print(f"v_sparse_ref={v_sparse_ref}")
            assert torch.allclose(k_sparse, k_sparse_ref, rtol=1e-5, atol=1e-5), "K_sparse mismatch!"
            assert torch.allclose(v_sparse, v_sparse_ref, rtol=1e-5, atol=1e-5), "V_sparse mismatch!"

        # expected_k = key_bnsd[:, :, :sparse_size, :].contiguous()
        # assert torch.allclose(k_sparse, expected_k, rtol=1e-5, atol=1e-5), "K_sparse mismatch!"
        # expected_v = value_bnsd[:, :, :sparse_size, :].contiguous()
        # assert torch.allclose(v_sparse, expected_v, rtol=1e-5, atol=1e-5), "V_sparse mismatch!"
        num_cores = ascend_aiv_core_nums
        # sparse_size = sparse_indices_bnsd.shape[-1] # 4
        out_shape_bsnd = list(query.shape)
        if query_rope is not None:
            out_shape_bsnd[-1] = out_shape_bsnd[-1] - query_rope.shape[-1]
        B, Q_S, Q_N, Q_D = query.shape
        _, _, KV_S, K_D = k_sparse.shape

        if layout_query == 'TND':
            # t = B*act_q_s
            output = torch.empty((total_len, out_shape_bsnd[2], out_shape_bsnd[3]), device=query.device,
                                 dtype=torch.float32)
            _attn_fwd_fused_bsnd_to_tnd[(num_cores, )](
                query, k_sparse, v_sparse, output, scale_value, query.stride(0), query.stride(1), query.stride(2),
                query.stride(3), k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
                v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3), output.stride(0),
                output.stride(1), output.stride(2), B=B, Q_N=Q_N, Q_D=Q_D, Q_S=Q_S, KV_S=KV_S, K_D=K_D,
                V_D=v_sparse.shape[3], sparse_mode=sparse_mode, O_N=output.shape[1], O_D=output.shape[2],
                actual_seq_lengths_query=actual_seq_lengths_query, actual_seq_lengths_kv=actual_seq_lengths_kv,
                blk_size=128, Q_BLOCK_SIZE=16, limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit")

        else:
            output = torch.empty(out_shape_bsnd, device=query.device, dtype=torch.float32)
            _attn_fwd[(num_cores, )](query, k_sparse, v_sparse, output, scale_value, query.stride(0), query.stride(1),
                                     query.stride(2), query.stride(3), k_sparse.stride(0), k_sparse.stride(1),
                                     k_sparse.stride(2), k_sparse.stride(3), v_sparse.stride(0), v_sparse.stride(1),
                                     v_sparse.stride(2), v_sparse.stride(3), output.stride(0), output.stride(1),
                                     output.stride(2), output.stride(3), B=B, Q_N=Q_N, Q_D=Q_D, Q_S=Q_S, KV_S=KV_S,
                                     K_D=K_D, V_D=v_sparse.shape[3], sparse_mode=sparse_mode, O_N=output.shape[2],
                                     O_D=output.shape[3], actual_seq_lengths_query=actual_seq_lengths_query,
                                     actual_seq_lengths_kv=actual_seq_lengths_kv, blk_size=128, Q_BLOCK_SIZE=16,
                                     limit_auto_multi_buffer_only_for_local_buffer=False,
                                     limit_auto_multi_buffer_of_local_buffer="no-limit")
            output = output.permute(0, 2, 1, 3).contiguous()

        ctx.save_for_backward(query, k_sparse, v_sparse, output)
        ctx.scale_value = scale_value
        return output


def pa_to_bsnd(pa_in, block_table, actual_seq_lengths):
    block_num, block_size, n, d = pa_in.shape
    b = len(actual_seq_lengths)
    output = torch.empty((b, block_num * block_size // b, 1, d), dtype=pa_in.dtype).to(DEVICE)
    for i in range(b):
        for j in range(20):
            output[i, j * block_size: (j + 1) * block_size, 0, :] = \
                pa_in[block_table[i][j], :, 0, :].reshape(block_size, d)
    return output


@triton.jit
def trans_tnd_to_bsnd_fused_kernel(
    query_ptr,
    query_rope_ptr,
    sparse_ptr,
    query_out_ptr,
    sparse_out_ptr,  # query_out 已经拼接了 rope
    act_s,
    stride_q_t,
    stride_q_tn,
    stride_q_td,
    stride_qr_t,
    stride_qr_tn,
    stride_qr_td,
    stride_s_t,
    stride_s_tn,
    stride_s_td,
    stride_qob,
    stride_qobs,
    stride_qon,
    stride_qod,  # query_out strides
    stride_sb,
    stride_sbs,
    stride_sbn,
    stride_sbd,
    B: tl.constexpr,
    N: tl.constexpr,
    D_QUERY: tl.constexpr,
    D_ROPE: tl.constexpr,
    D_SPARSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_QUERY: tl.constexpr,
    BLOCK_D_ROPE: tl.constexpr,
    BLOCK_D_SPARSE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # 计算 head 的总块数
    num_head_blocks = (N + BLOCK_N - 1) // BLOCK_N
    t_idx = tl.full((), 0, dtype=tl.int64)  # TODO: 需要正确的 token 映射
    # 每个 pid 负责处理特定的 (batch, head_block) 组合
    for tn_id in range(B):
        # sparse_indices 是单头的，只在第一个 head_block 处理一次
        if pid == 0:
            sparse_block_ptr = tl.make_block_ptr(base=sparse_ptr + t_idx * stride_s_t, shape=(1, D_SPARSE),
                                                 strides=(stride_s_tn, stride_s_td), offsets=(0, 0),
                                                 block_shape=(1, D_SPARSE), order=(1, 0))
            sparse = tl.load(sparse_block_ptr)

            sparse_out_block_ptr = tl.make_block_ptr(base=sparse_out_ptr + t_idx * stride_sb, shape=(1, D_SPARSE),
                                                     strides=(stride_sbn, stride_sbd), offsets=(0, 0),
                                                     block_shape=(1, D_SPARSE), order=(1, 0))
            tl.store(sparse_out_block_ptr, sparse)

        # query 和 query_rope 是多头的，需要在 head 维度上分块处理
        for head_block_id in range(pid, num_head_blocks, num_programs):
            n_offset = head_block_id * BLOCK_N

            # Load q and q_ro
            q_block_ptr = tl.make_block_ptr(base=query_ptr + t_idx * stride_q_t, shape=(N, D_QUERY),
                                            strides=(stride_q_tn, stride_q_td), offsets=(n_offset, 0),
                                            block_shape=(BLOCK_N, D_QUERY), order=(1, 0))
            q_ro_block_ptr = tl.make_block_ptr(base=query_rope_ptr + t_idx * stride_qr_t, shape=(N, D_ROPE),
                                               strides=(stride_qr_tn, stride_qr_td), offsets=(n_offset, 0),
                                               block_shape=(BLOCK_N, D_ROPE), order=(1, 0))
            q = tl.load(q_block_ptr)
            q_ro = tl.load(q_ro_block_ptr)

            # Combine query and query_rope using insert_slice, then store in one operation
            full_q = tl.zeros((BLOCK_N, D_QUERY + D_ROPE), dtype=query_out_ptr.dtype.element_ty)
            full_q = tle.dsa.insert_slice(full_q, q, offsets=(0, 0), sizes=(BLOCK_N, D_QUERY), strides=(1, 1))
            full_q = tle.dsa.insert_slice(full_q, q_ro, offsets=(0, D_QUERY), sizes=(BLOCK_N, D_ROPE), strides=(1, 1))

            q_out_block_ptr = tl.make_block_ptr(base=query_out_ptr + t_idx * stride_qob, shape=(N, D_QUERY + D_ROPE),
                                                strides=(stride_qon, stride_qod), offsets=(n_offset, 0),
                                                block_shape=(BLOCK_N, D_QUERY + D_ROPE), order=(1, 0))
            tl.store(q_out_block_ptr, full_q)
        t_idx = t_idx + tl.load(act_s + tn_id)


def trans_tnd_to_bsnd_fused(query, query_rope, sparse_indices, shape, act_seq, grid=(16, )):
    """
    融合版本的 TND -> BSND 转换（包含 concat）
    一次性处理 query, query_rope, sparse_indices，并拼接 query + query_rope
    """
    t, n, d_query = shape
    b = len(act_seq)
    s = max(act_seq)

    # 获取各个 tensor 的维度
    d_rope = query_rope.shape[2] if query_rope is not None else 0
    d_sparse = sparse_indices.shape[2]
    d_query_out = d_query + d_rope  # 拼接后的维度

    # 分配输出（query_out 已经包含 rope）
    query_out = torch.empty((b, s, n, d_query_out), dtype=query.dtype, device=query.device)
    sparse_out = torch.empty((b, s, 1, d_sparse), dtype=sparse_indices.dtype, device=sparse_indices.device)
    assert sparse_indices.shape[1] == 1, "sparse_indices second dim must be 1 when MLA"
    # 启动 fused kernel
    # 使用较小的 BLOCK_N 避免内存溢出
    block_n = min(16, n)
    # 计算需要的核心数：使用多核心并行处理不同的头
    num_head_blocks = (n + block_n - 1) // block_n
    num_programs = min(ascend_aiv_core_nums, num_head_blocks)  # 最多使用24个核心

    trans_tnd_to_bsnd_fused_kernel[
        num_programs,
    ](
        query,
        query_rope,
        sparse_indices,
        query_out,
        sparse_out,
        act_seq,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_rope.stride(0),
        query_rope.stride(1),
        query_rope.stride(2),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        sparse_indices.stride(2),
        query_out.stride(0),
        query_out.stride(1),
        query_out.stride(2),
        query_out.stride(3),
        sparse_out.stride(0),
        sparse_out.stride(1),
        sparse_out.stride(2),
        sparse_out.stride(3),
        B=b,
        N=n,
        D_QUERY=d_query,
        D_ROPE=d_rope,
        D_SPARSE=d_sparse,
        BLOCK_N=block_n,
        BLOCK_D_QUERY=d_query,
        BLOCK_D_ROPE=d_rope,
        BLOCK_D_SPARSE=d_sparse,
    )
    return query_out, sparse_out


def trans_tnd_actseq(seq):
    if isinstance(seq, torch.Tensor):
        seq = seq.cpu().tolist()
    list_len = len(seq)
    output = []
    output = [seq[0]]
    total_len = seq[0]
    for i in range(list_len - 1):
        new_item = seq[i + 1] - seq[i]
        if new_item >= 0:
            output.append(new_item)
            total_len += new_item
        else:
            print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{seq}, in loop {i}, item {new_item} < 0")
    return torch.tensor(output).to(DEVICE), total_len


def sparse_attention(query, key, value, sparse_indices, scale_value, sparse_block_size=1, actual_seq_lengths_query=None,
                     actual_seq_lengths_kv=None, query_rope=None, key_rope=None, layout_query='BSND', layout_kv='BSND',
                     sparse_mode=0, block_table=None):
    return _attention.apply(query, key, value, sparse_indices, scale_value, sparse_block_size, actual_seq_lengths_query,
                            actual_seq_lengths_kv, query_rope, key_rope, layout_query, layout_kv, sparse_mode,
                            block_table)


def test_op(T, B, KV_S, Q_N, KV_N, D, D_rope, sparse_size, scale_value, sparse_block_size, sparse_mode, block_size,
            act_kv_s):
    assert sparse_size <= KV_S
    assert KV_N == 1
    assert sparse_mode == 0 or 3
    assert sparse_block_size == 1
    assert (B * KV_S) % block_size == 0
    assert D == 512
    assert D_rope == 0 or 64
    print("*batch_size=", B)
    qkv_dtype = torch.float16
    #sparse_size = KV_S
    query = torch.empty((T, Q_N, D), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    key = torch.empty((B * KV_S // block_size, block_size, KV_N, D), dtype=qkv_dtype,
                      device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    value = key.clone()

    # act_q_s = T // B # step
    # rand_vals = torch.rand(T, KV_N, act_kv_s, device=DEVICE)
    # _, indices = torch.topk(rand_vals, sparse_size, dim=-1) #sparse_indices不重复
    # sparse_indices = indices.to(torch.int32)
    sparse_indices = torch.arange(sparse_size, device=DEVICE, dtype=torch.int32).view(1, 1, -1).expand(T, KV_N, -1)
    sparse_indices = sparse_indices.to(torch.int32)
    # print("sparse_indices=", sparse_indices)
    actual_seq_lengths_query = torch.arange(1, B + 1, dtype=torch.int32, device=DEVICE)
    # actual_seq_lengths_query = torch.tensor([1]).reshape(B).to(torch.int32).to(DEVICE)
    actual_seq_lengths_kv = torch.tensor([act_kv_s] * B, dtype=torch.int32, device=DEVICE)
    print(actual_seq_lengths_kv)
    block_table = torch.tensor([range(B * KV_S // block_size)], dtype=torch.int32, device=DEVICE).reshape(B, -1)

    if D_rope == 0:
        query_rope = None
        key_rope = None
    else:
        query_rope = torch.empty((T, Q_N, D_rope), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0,
                                                                                           std=0.5).requires_grad_()
        key_rope = torch.empty((B * KV_S // block_size, block_size, KV_N, D_rope), dtype=qkv_dtype,
                               device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()

    print("q.shape=", query.shape)
    print("k.shape=", key.shape)
    print("v.shape=", value.shape)
    print("sparse_indices.shape=", sparse_indices.shape)
    print("act_seq_query=", actual_seq_lengths_query)
    print("act_seq_kv=", actual_seq_lengths_kv)

    triton_out = sparse_attention(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=scale_value,
        sparse_block_size=sparse_block_size,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        layout_query='TND',
        layout_kv='PA_BSND',
        sparse_mode=sparse_mode,
        block_table=block_table,
    )
    npu_out, _, _ = torch_npu.npu_sparse_flash_attention(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=scale_value,
        sparse_block_size=sparse_block_size,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        layout_query='TND',
        layout_kv='PA_BSND',
        sparse_mode=sparse_mode,
        block_table=block_table,
        attention_mode=2,
    )
    triton_out = triton_out.to(npu_out.dtype)
    torch.testing.assert_close(triton_out, npu_out, rtol=1e-2, atol=1e-2, equal_nan=True)
    print("[PASSED]")

    # benchmarking
    triton_time = do_bench_npu(
        lambda: sparse_attention(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            scale_value=scale_value,
            sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=key_rope,
            layout_query='TND',
            layout_kv='PA_BSND',
            sparse_mode=sparse_mode,
            block_table=block_table,
        ), clear_l2_cache=True, collect_prof=False)
    print(f"[Triton SFA] Time: {triton_time:.4f} us")

    npu_time = do_bench_npu(
        lambda: torch_npu.npu_sparse_flash_attention(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            scale_value=scale_value,
            sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=key_rope,
            layout_query='TND',
            layout_kv='PA_BSND',
            sparse_mode=sparse_mode,
            block_table=block_table,
            attention_mode=2,
        ), clear_l2_cache=True, collect_prof=False)
    print(f"[Torch-NPU SFA] Time: {npu_time:.4f} us")


if __name__ == "__main__":
    print(torch_npu.__version__)
    print("Test Real Case in DS-v3.2-Exp")
    print(f"time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    i = 1
    print(f"====================第{i}次测试=================")
    test_op(T=1, B=1, KV_S=2560, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
            sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=2560)
    test_op(T=1, B=1, KV_S=5120, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
            sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=5120)
    test_op(T=1, B=1, KV_S=10240, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
            sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=10240)
    test_op(T=1, B=1, KV_S=20480, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
            sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=20480)
    # i += 1
    # print(f"====================第{i}次测试=================")
    # test_op(T=4, B=4, KV_S=6400, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
    #         sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=2560)
    # i += 1
    # print(f"====================第{i}次测试=================")
    # test_op(T=8, B=8, KV_S=48000, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
    #         sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=2560)
    # i += 1
    # print(f"====================第{i}次测试=================")
    # test_op(T=16, B=16, KV_S=48000, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
    #         sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=2560)
