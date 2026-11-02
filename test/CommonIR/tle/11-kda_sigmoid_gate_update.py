"""sigmoid gating + delta rule 递推算子（MetaX TLE / CommonIR 实现）。

算子语义（每 (batch, value-head) 一条独立递推）：
    h_t = h_{t-1} * decay_t + k_t^T * beta_t * (v_t - h_{t-1} * k_t)
    o_t = (h_t^T * q_t) * scale
其中 decay_t 由 a / dt_bias / A_log 经 softplus 或 sigmoid(lower_bound) 得到；
KDA 模式下 decay 是 [K] 向量（逐 k 列），普通模式下是标量。

实现分三块：
  1) 预计算 kernel：把 gate（exp(g)）与 beta=sigmoid(b) 并行算进全局表，递推核内只
     做一次查表——这是 KDA 模式性能的关键（超越函数不进递推主循环）。
  2) 递推 kernel A（bare，默认路径）：逐 token 从 global load k/q/v，h tile 常驻寄存器。
  3) 递推 kernel B（SMEM 双缓冲）：k/q/v 用 async_copy_global_to_local + gvm_arrive
     双缓冲预取，`force_path="smem"` 时使用。

实现要点：
  * bare 路径必须把 h tile 钉成 BlockEncoding([1,1],[4,16],[warps,1],[1,0])：否则
    autolayout 会由 sum(axis=1) 反推，把整行 BK 列摊到所有 lane 上，跨 lane 归约代价
    极高（实测同一形状 263ms → 22.6ms）。
  * 默认 num_warps 按 BK 分档（≤64 → 2，≥128 → 4）。K=128 用 2 warps 时归约爆炸
    （18.1ms → 2.82ms）。
  * TLE 的 local_load 没有 slice/offset 参数，且 extract_tile 作用在 local_load 结果
    上会在 CommonIRLowerTensorTiles 阶段编译失败，故 SMEM 路径逐行提取用
    tl.max(tl.where(...)) 实现。
  * SMEM 路径无尾块掩码，TT 必须整除 T 且 ≥2，否则自动退回 bare。

用法：
    python kda_sigmoid_gating_tle.py
"""

from __future__ import annotations

import argparse
import traceback

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED

TT = 16
_DEFAULT_TT = TT
BS = 128


def _resolve_bv(V):
    if V <= 64:
        return min(triton.next_power_of_2(V), 32)
    return min(triton.next_power_of_2(V), 64)


# ============================= 预计算 kernels =============================


@triton.layout_tune()
@triton.jit
def _gate_precompute_kernel(
    a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    g_pre_ptr, beta_pre_ptr, total,
    HV: tl.constexpr, BS: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BS + tl.arange(0, BS)
    mask = offs < total
    i_hv = offs % HV

    A_log = tl.load(A_log_ptr + i_hv, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + i_hv, mask=mask, other=0.0).to(tl.float32)
    aa = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    x = aa + dt_bias
    beta_x = SOFTPLUS_BETA * x
    sp_small = (1.0 / SOFTPLUS_BETA) * tl.log(1.0 + tl.exp(beta_x))
    softplus_x = tl.where(beta_x <= SOFTPLUS_THRESHOLD, sp_small, x)
    g = -tl.exp(A_log) * softplus_x

    tl.store(g_pre_ptr + offs, tl.exp(g), mask=mask)
    tl.store(beta_pre_ptr + offs, 1.0 / (1.0 + tl.exp(-bb)), mask=mask)


@triton.layout_tune()
@triton.jit
def _beta_precompute_kernel(
    b_ptr, beta_pre_ptr, total,
    HV: tl.constexpr, BS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BS + tl.arange(0, BS)
    mask = offs < total
    bb = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(beta_pre_ptr + offs, 1.0 / (1.0 + tl.exp(-bb)), mask=mask)


@triton.layout_tune()
@triton.jit
def _kda_gate_precompute_kernel(
    a_ptr, dt_bias_ptr, A_log_ptr, g_pre_ptr, total,
    HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr, LOWER_BOUND: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
):
    # KDA 逐 token 逐 k 列衰减：写 exp(g) 到 g_pre[(b*T+t)*HV+hv, :]
    i = tl.program_id(axis=0)
    i_hv = i % HV
    o_k = tl.arange(0, BK)
    mask = o_k < K

    A_v = tl.load(A_log_ptr + i_hv).to(tl.float32)
    aa = tl.load(a_ptr + i * K + o_k, mask=mask, other=0.0).to(tl.float32)
    dt = tl.load(dt_bias_ptr + i_hv * K + o_k, mask=mask, other=0.0).to(tl.float32)

    x = aa + dt
    if USE_LOWER_BOUND:
        g = LOWER_BOUND * (1.0 / (1.0 + tl.exp(-(tl.exp(A_v) * x))))
    else:
        beta_x = SOFTPLUS_BETA * x
        sp = (1.0 / SOFTPLUS_BETA) * tl.log(1.0 + tl.exp(beta_x))
        g = -tl.exp(A_v) * tl.where(beta_x <= SOFTPLUS_THRESHOLD, sp, x)
    tl.store(g_pre_ptr + i * K + o_k, tl.exp(g), mask=mask)


@triton.layout_tune()
@triton.jit
def _qk_l2norm_precompute_kernel(
    q_ptr, k_ptr, q_inv_ptr, k_inv_ptr, total,
    K: tl.constexpr, BK: tl.constexpr,
):
    i = tl.program_id(axis=0)
    offs = tl.arange(0, BK)
    mask = offs < K
    q_row = tl.load(q_ptr + i * K + offs, mask=mask, other=0.0).to(tl.float32)
    k_row = tl.load(k_ptr + i * K + offs, mask=mask, other=0.0).to(tl.float32)
    q_inv = 1.0 / (tl.sqrt(tl.sum(q_row * q_row, axis=0)) + 1e-6)
    k_inv = 1.0 / (tl.sqrt(tl.sum(k_row * k_row, axis=0)) + 1e-6)
    tl.store(q_inv_ptr + i, q_inv)
    tl.store(k_inv_ptr + i, k_inv)


# ==================== 递推 kernel A：逐行 load（无 SMEM） ====================


@triton.layout_tune()
@triton.jit
def _sigmoid_gating_delta_rule_bare_kernel(
    g_pre_ptr, a_ptr, dt_bias_ptr, A_log_ptr, beta_pre_ptr, q_inv_ptr, k_inv_ptr,
    q_ptr, k_ptr, v_ptr, o_ptr,
    h0_source_ptr, h0_indices_ptr, h_final_ptr,
    scale, T,
    B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr, IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr, LOWER_BOUND: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_STATE_POOL: tl.constexpr,
    UPDATE_POOL: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    H_ENC: tl.constexpr,
):
    i_v = tl.program_id(axis=0)
    i_nh = tl.program_id(axis=1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    o_k_row = tl.arange(0, BK)[None, :]
    o_vc = tl.arange(0, BV)[:, None]
    o_v_row = tl.arange(0, BV)[None, :]

    # h tile [BV, BK]；state pool 里存的是 [K, V]
    if USE_STATE_POOL:
        idx = tl.load(h0_indices_ptr + i_n).to(tl.int32)
        base_h0 = (idx * HV + i_hv) * K * V + i_v * BV
        if idx >= 0:
            h = tl.load(h0_source_ptr + base_h0 + o_k_row * V + o_vc).to(tl.float32)
        else:
            h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        h = tl.zeros([BV, BK], dtype=tl.float32)

    # h tile 必须钉布局：否则 autolayout 会由 sum(axis=1) 反推，把整行 BK 列摊到所有
    # lane 上，跨 lane 归约代价极高（B85T256 KDA：263ms → 22.6ms）。
    h = tle.gpu.set_layout(h, H_ENC)

    base_qk = (i_n * T * H + i_h) * K
    base_v = (i_n * T * HV + i_hv) * V + i_v * BV

    for t in range(T):
        k_row = tl.load(k_ptr + base_qk + t * (H * K) + o_k_row).to(tl.float32)
        q_row = tl.load(q_ptr + base_qk + t * (H * K) + o_k_row).to(tl.float32)
        v_row = tl.load(v_ptr + base_v + t * (HV * V) + o_v_row).to(tl.float32)

        if USE_QK_L2NORM:
            kinv_val = tl.load(k_inv_ptr + (i_n * T + t) * H + i_h).to(tl.float32)
            qinv_val = tl.load(q_inv_ptr + (i_n * T + t) * H + i_h).to(tl.float32)
            k_row = k_row * kinv_val
            q_row = q_row * qinv_val

        v_col = tl.permute(v_row, (1, 0))
        beta_pre = tl.load(beta_pre_ptr + (i_n * T + t) * HV + i_hv).to(tl.float32)

        if IS_KDA:
            decay_row = tl.load(
                g_pre_ptr + ((i_n * T + t) * HV + i_hv) * K + o_k_row,
                mask=o_k_row < K, other=1.0).to(tl.float32)
        else:
            g_pre = tl.load(g_pre_ptr + (i_n * T + t) * HV + i_hv).to(tl.float32)

        h = h * (decay_row if IS_KDA else g_pre)
        delta_2d = tl.sum(h * k_row, axis=1, keep_dims=True)
        v_new_2d = (v_col - delta_2d) * beta_pre
        h = h + v_new_2d * k_row
        bo_2d = tl.permute(tl.sum(h * q_row, axis=1, keep_dims=True), (1, 0)) * scale

        base_o = ((i_n * T + t) * HV + i_hv) * V + i_v * BV
        tl.store(o_ptr + base_o + o_v_row, bo_2d.to(o_ptr.dtype.element_ty))

    if STORE_FINAL:
        base_hf = (i_n * HV + i_hv) * K * V + i_v * BV
        tl.store(h_final_ptr + base_hf + o_k_row * V + o_vc,
                 h.to(h_final_ptr.dtype.element_ty))

    # 回写 state pool：decode 语义是每步原地推进 cache
    if UPDATE_POOL:
        if idx >= 0:
            tl.store(h0_source_ptr + base_h0 + o_k_row * V + o_vc,
                     h.to(h0_source_ptr.dtype.element_ty))


# ==================== 递推 kernel B：SMEM 双缓冲（metax 原语） ====================


@triton.layout_tune()
@triton.jit
def _sigmoid_gating_delta_rule_pc_kernel(
    g_pre_ptr, a_ptr, dt_bias_ptr, A_log_ptr, beta_pre_ptr, q_inv_ptr, k_inv_ptr,
    q_ptr, k_ptr, v_ptr, o_ptr,
    h0_source_ptr, h0_indices_ptr, h_final_ptr,
    scale, T,
    B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    TT: tl.constexpr, USE_QK_L2NORM: tl.constexpr, IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr, LOWER_BOUND: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_STATE_POOL: tl.constexpr, UPDATE_POOL: tl.constexpr,
    STORE_FINAL: tl.constexpr,
):
    i_v = tl.program_id(axis=0)
    i_nh = tl.program_id(axis=1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    o_tt_col = tl.arange(0, TT)[:, None]
    o_tt_r = tl.arange(0, TT)
    o_k_row = tl.arange(0, BK)[None, :]
    o_v_row = tl.arange(0, BV)[None, :]

    k_smem = tle.gpu.alloc((2, TT, BK), k_ptr.dtype.element_ty)
    q_smem = tle.gpu.alloc((2, TT, BK), q_ptr.dtype.element_ty)
    v_smem = tle.gpu.alloc((2, TT, BV), v_ptr.dtype.element_ty)

    if USE_STATE_POOL:
        idx = tl.load(h0_indices_ptr + i_n).to(tl.int32)
        base_h0 = (idx * HV + i_hv) * K * V + i_v * BV
        o_h = tl.arange(0, BK)[None, :] * V + tl.arange(0, BV)[:, None]
        if idx >= 0:
            h = tl.load(h0_source_ptr + base_h0 + o_h).to(tl.float32)
        else:
            h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        h = tl.zeros([BV, BK], dtype=tl.float32)

    base_qk = (i_n * T * H + i_h) * K
    base_v = (i_n * T * HV + i_hv) * V + i_v * BV
    k_ptrs = k_ptr + base_qk + o_tt_col * (H * K) + o_k_row
    q_ptrs = q_ptr + base_qk + o_tt_col * (H * K) + o_k_row
    v_ptrs = v_ptr + base_v + o_tt_col * (HV * V) + o_v_row

    # 预取块 0（async_copy 的参数顺序是 (源指针, 目标 buffer)）
    tle.gpu.metax.async_copy_global_to_local(k_ptrs, k_smem.slot(0))
    tle.gpu.metax.async_copy_global_to_local(q_ptrs, q_smem.slot(0))
    tle.gpu.metax.async_copy_global_to_local(v_ptrs, v_smem.slot(0))
    tle.gpu.metax.gvm_arrive(3)
    tle.gpu.metax.barrier()

    num_blocks = T // TT

    for blk in range(num_blocks):
        cur = blk % 2
        nxt = (blk + 1) % 2

        # 预取下一块
        next_blk = blk + 1
        if next_blk < num_blocks:
            t0 = next_blk * TT
            base_qk = ((i_n * T + t0) * H + i_h) * K
            base_v = ((i_n * T + t0) * HV + i_hv) * V + i_v * BV
            k_ptrs = k_ptr + base_qk + o_tt_col * (H * K) + o_k_row
            q_ptrs = q_ptr + base_qk + o_tt_col * (H * K) + o_k_row
            v_ptrs = v_ptr + base_v + o_tt_col * (HV * V) + o_v_row
            tle.gpu.metax.async_copy_global_to_local(k_ptrs, k_smem.slot(nxt))
            tle.gpu.metax.async_copy_global_to_local(q_ptrs, q_smem.slot(nxt))
            tle.gpu.metax.async_copy_global_to_local(v_ptrs, v_smem.slot(nxt))
            tle.gpu.metax.gvm_arrive(3)
        tle.gpu.metax.barrier_shared()

        # local_load 只能整块装载（TLE 没有 slice-load）
        k_tile = tle.gpu.metax.local_load(
            k_smem.slot(cur), intrinsic=True, is_constant_offs=True).to(tl.float32)
        q_tile = tle.gpu.metax.local_load(
            q_smem.slot(cur), intrinsic=True, is_constant_offs=True).to(tl.float32)
        v_tile = tle.gpu.metax.local_load(
            v_smem.slot(cur), intrinsic=True, is_constant_offs=True).to(tl.float32)

        # 消费当前块：逐行提取（extract_tile 在该 layout 上编译不过，故用 where/max）
        for i in tl.static_range(TT):
            t = blk * TT + i

            sel2 = (o_tt_r == i)[:, None]
            NEG = -1.0e30
            k_row = tl.max(tl.where(sel2, k_tile, NEG), axis=0, keep_dims=True)
            q_row = tl.max(tl.where(sel2, q_tile, NEG), axis=0, keep_dims=True)
            v_row = tl.max(tl.where(sel2, v_tile, NEG), axis=0, keep_dims=True)

            v_col = tl.permute(v_row, (1, 0))

            if USE_QK_L2NORM:
                k_row = k_row * tl.load(k_inv_ptr + (i_n * T + t) * H + i_h).to(tl.float32)
                q_row = q_row * tl.load(q_inv_ptr + (i_n * T + t) * H + i_h).to(tl.float32)

            beta_pre = tl.load(beta_pre_ptr + (i_n * T + t) * HV + i_hv).to(tl.float32)

            if IS_KDA:
                decay_row = tl.load(
                    g_pre_ptr + ((i_n * T + t) * HV + i_hv) * K + o_k_row,
                    mask=o_k_row < K, other=1.0).to(tl.float32)
            else:
                g_pre = tl.load(g_pre_ptr + (i_n * T + t) * HV + i_hv).to(tl.float32)

            h = h * (decay_row if IS_KDA else g_pre)
            delta_2d = tl.sum(h * k_row, axis=1, keep_dims=True)
            v_new_2d = (v_col - delta_2d) * beta_pre
            h = h + v_new_2d * k_row
            bo_2d = tl.permute(tl.sum(h * q_row, axis=1, keep_dims=True), (1, 0)) * scale

            base_o = ((i_n * T + t) * HV + i_hv) * V + i_v * BV
            tl.store(o_ptr + base_o + o_v_row, bo_2d.to(o_ptr.dtype.element_ty))

            tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=2)

    tle.gpu.metax.gvm_arrive(0)
    tle.gpu.metax.barrier_shared()

    if STORE_FINAL:
        base_hf = (i_n * HV + i_hv) * K * V + i_v * BV
        o_hf = tl.arange(0, BK)[None, :] * V + tl.arange(0, BV)[:, None]
        tl.store(h_final_ptr + base_hf + o_hf, h.to(h_final_ptr.dtype.element_ty))

    # 回写 state pool（decode 语义：每步原地推进 cache）
    if UPDATE_POOL:
        if idx >= 0:
            tl.store(h0_source_ptr + base_h0 + o_h,
                     h.to(h0_source_ptr.dtype.element_ty))


# ============================= 入口 =============================


def _choose_tt(T, requested):
    """能整除 T 的最大 2 的幂，且不超过 requested（SMEM 路径无尾块掩码）。"""
    if T <= 0:
        return 1
    tt = min(requested, 1 << (T.bit_length() - 1))
    while tt > 1 and T % tt != 0:
        tt >>= 1
    return max(tt, 1)


def sigmoid_gating_delta_rule_pc(A_log, a, dt_bias, softplus_beta, softplus_threshold,
                                 q, k, v, b, scale=None, num_warps=None, tt=None,
                                 force_path=None, qk_l2norm=False,
                                 initial_state_source=None, initial_state_indices=None,
                                 output_final_state=False, disable_state_update=False,
                                 is_kda=False, lower_bound=None):
    """sigmoid gating + delta rule（decode / chunked recurrent）。

    参数：
        A_log                : [HV]            softplus/sigmoid 的通道缩放
        a                    : [B, T, HV] 或 [B, T, HV, K]（KDA）
        dt_bias              : [HV] 或 [HV, K]（KDA）
        q                    : [B, T, H, K]
        k                    : [B, T, H, K]
        v                    : [B, T, HV, V]
        b                    : [B, T, HV]
        scale                : 默认 K ** -0.5
        qk_l2norm            : 是否在核内对 q/k 做 L2 归一化
        initial_state_source : [num_slots, HV, K, V]，配合 initial_state_indices 使用
        initial_state_indices: [B] int32，负值表示该 batch 不读不写 state
        output_final_state   : True 时额外返回 h_final [B, HV, K, V]
        disable_state_update : True 时不回写 state pool
        is_kda               : 逐 k 列衰减（a/dt_bias 带 K 维）
        lower_bound          : 非 None 时 KDA 走 sigmoid(lower_bound) 分支

    返回：
        o [B, T, HV, V]（fp32），output_final_state 时再返回 h_final。
    """
    assert ENABLED, "Requires a MetaX FLAGTREE_COMMON_IR build"
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    BK = triton.next_power_of_2(K)
    BV = _resolve_bv(V)
    NV = triton.cdiv(V, BV)
    TT = _choose_tt(T, tt if tt is not None else _DEFAULT_TT)

    if num_warps is None:
        # K=128 用 2 warps 时 h tile 的归约会爆（B32T512 K128V64：18.1ms vs 2.82ms），
        # 故按 BK 分档而不是按 V。
        num_warps = 2 if BK <= 64 else 4

    total = B * T * HV
    beta_pre = torch.empty(B * T * HV, device=q.device, dtype=torch.float32)
    if is_kda:
        if a.dim() != 4 or a.shape[-1] != K:
            raise ValueError(f"KDA 模式 a 期望 [B, T, HV, K={K}]，got {tuple(a.shape)}")
        if dt_bias.dim() != 2 or tuple(dt_bias.shape) != (HV, K):
            raise ValueError(f"KDA 模式 dt_bias 期望 [HV={HV}, K={K}]，got {tuple(dt_bias.shape)}")
        g_pre = torch.empty(B * T * HV * K, device=q.device, dtype=torch.float32)
        _beta_precompute_kernel[(triton.cdiv(total, BS),)](
            b, beta_pre, total, HV=HV, BS=BS, num_warps=2)
        _kda_gate_precompute_kernel[(total,)](
            a, dt_bias, A_log, g_pre, total,
            HV=HV, K=K, BK=BK,
            USE_LOWER_BOUND=lower_bound is not None,
            LOWER_BOUND=float(lower_bound) if lower_bound is not None else 0.0,
            SOFTPLUS_BETA=float(softplus_beta),
            SOFTPLUS_THRESHOLD=float(softplus_threshold),
            num_warps=1,
        )
    else:
        g_pre = torch.empty(B * T * HV, device=q.device, dtype=torch.float32)
        _gate_precompute_kernel[(triton.cdiv(total, BS),)](
            a, b, A_log, dt_bias, g_pre, beta_pre, total,
            HV=HV, BS=BS,
            SOFTPLUS_BETA=float(softplus_beta),
            SOFTPLUS_THRESHOLD=float(softplus_threshold),
            num_warps=2,
        )

    if qk_l2norm:
        n_rows = B * T * H
        q_inv = torch.empty(n_rows, device=q.device, dtype=torch.float32)
        k_inv = torch.empty(n_rows, device=q.device, dtype=torch.float32)
        _qk_l2norm_precompute_kernel[(n_rows,)](
            q, k, q_inv, k_inv, n_rows,
            K=K, BK=BK,
            num_warps=1,
        )
    else:
        q_inv = beta_pre
        k_inv = beta_pre

    o = torch.empty(B, T, HV, V, device=q.device, dtype=torch.float32)

    use_pool = initial_state_source is not None and initial_state_indices is not None
    if use_pool:
        if initial_state_source.dim() != 4 or initial_state_source.shape[1:] != (HV, K, V):
            raise ValueError(
                f"initial_state_source 期望 [num_slots, HV, K, V]，got {tuple(initial_state_source.shape)}")
        h0_source = initial_state_source
        h0_indices = initial_state_indices
        if initial_state_indices.dtype != torch.int32:
            h0_indices = initial_state_indices.to(torch.int32)
    else:
        h0_source = beta_pre
        h0_indices = beta_pre
    store_final = 1 if output_final_state else 0
    if store_final:
        h_final = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32)
    else:
        h_final = beta_pre

    use_bare = (force_path != "smem") if force_path else True
    # SMEM 路径无尾块掩码：TT 必须整除 T 且至少为 2
    if TT < 2 or T % TT != 0:
        use_bare = True

    if use_bare:
        _sigmoid_gating_delta_rule_bare_kernel[(NV, B * HV)](
            g_pre, a, dt_bias, A_log, beta_pre, q_inv, k_inv,
            q, k, v, o,
            h0_source, h0_indices, h_final,
            float(scale), T,
            B, H, HV, K, V, BK, BV,
            USE_QK_L2NORM=qk_l2norm, IS_KDA=is_kda,
            USE_LOWER_BOUND=lower_bound is not None,
            LOWER_BOUND=float(lower_bound) if lower_bound is not None else 0.0,
            SOFTPLUS_BETA=float(softplus_beta),
            SOFTPLUS_THRESHOLD=float(softplus_threshold),
            USE_STATE_POOL=use_pool,
            UPDATE_POOL=use_pool and not disable_state_update,
            STORE_FINAL=store_final,
            H_ENC=tle.gpu.BlockEncoding([1, 1], [4, 16], [int(num_warps), 1], [1, 0]),
            num_warps=int(num_warps),
        )
    else:
        _sigmoid_gating_delta_rule_pc_kernel[(NV, B * HV)](
            g_pre, a, dt_bias, A_log, beta_pre, q_inv, k_inv,
            q, k, v, o,
            h0_source, h0_indices, h_final,
            float(scale), T,
            B, H, HV, K, V, BK, BV, TT,
            USE_QK_L2NORM=qk_l2norm, IS_KDA=is_kda,
            USE_LOWER_BOUND=lower_bound is not None,
            LOWER_BOUND=float(lower_bound) if lower_bound is not None else 0.0,
            SOFTPLUS_BETA=float(softplus_beta),
            SOFTPLUS_THRESHOLD=float(softplus_threshold),
            USE_STATE_POOL=use_pool,
            UPDATE_POOL=use_pool and not disable_state_update,
            STORE_FINAL=store_final,
            num_warps=int(num_warps),
        )
    if store_final:
        return o, h_final
    return o


# ============================= fp32 参考 & 自测 =============================


def _softplus(x, beta, threshold):
    bx = beta * x
    return torch.where(bx <= threshold, torch.log1p(torch.exp(bx)) / beta, x)


def reference_sigmoid_delta(A_log, a, dt_bias, softplus_beta, softplus_threshold,
                            q, k, v, b, scale, qk_l2norm=False,
                            h0_source=None, h0_indices=None,
                            output_final_state=False,
                            is_kda=False, lower_bound=None):
    """纯 torch 的 fp32 逐 token 参考实现。"""
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    group = HV // H
    o = torch.zeros(B, T, HV, V, device=q.device, dtype=torch.float32)
    h_out = (torch.zeros(B, HV, K, V, device=q.device, dtype=torch.float32)
             if output_final_state else None)
    for n in range(B):
        for hv in range(HV):
            if h0_source is not None and h0_indices is not None and h0_indices[n] >= 0:
                h = h0_source[h0_indices[n], hv].float().clone()
            else:
                h = torch.zeros(K, V, device=q.device, dtype=torch.float32)
            i_h = hv // group
            for t in range(T):
                qq = q[n, t, i_h].float()
                kk = k[n, t, i_h].float()
                if qk_l2norm:
                    qq = qq / (qq.norm() + 1e-6)
                    kk = kk / (kk.norm() + 1e-6)
                qq = qq * scale
                vv = v[n, t, hv].float()
                if is_kda:
                    x = a[n, t, hv].float() + dt_bias[hv].float()
                    if lower_bound is not None:
                        g = lower_bound * torch.sigmoid(torch.exp(A_log[hv].float()) * x)
                    else:
                        g = -torch.exp(A_log[hv].float()) * _softplus(
                            x, softplus_beta, softplus_threshold)
                    decay = torch.exp(g)
                else:
                    g = -torch.exp(A_log[hv].float()) * _softplus(
                        a[n, t, hv].float() + dt_bias[hv].float(),
                        softplus_beta, softplus_threshold)
                    decay = torch.exp(g)
                beta = torch.sigmoid(b[n, t, hv].float())
                h = h * (decay[:, None] if is_kda else decay)
                v_new = vv - h.T @ kk
                v_new = v_new * beta
                h = h + kk[:, None] * v_new[None, :]
                o[n, t, hv] = h.T @ qq
            if output_final_state:
                h_out[n, hv] = h
    if output_final_state:
        return o, h_out
    return o


def _make_inputs(B=1, T=4, H=1, HV=1, K=16, V=16, seed=0, kda=False):
    torch.manual_seed(seed)
    device = "cuda"
    s = 0.1
    if kda:
        a = torch.randn(B, T, HV, K, device=device, dtype=torch.float16) * s
        dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * s
    else:
        a = torch.randn(B, T, HV, device=device, dtype=torch.float16) * s
        dt_bias = torch.randn(HV, device=device, dtype=torch.float32)
    return (
        torch.randn(HV, device=device, dtype=torch.float32),
        a, dt_bias,
        torch.randn(B, T, H, K, device=device, dtype=torch.float16) * s,
        torch.randn(B, T, H, K, device=device, dtype=torch.float16) * s,
        torch.randn(B, T, HV, V, device=device, dtype=torch.float16) * s,
        torch.randn(B, T, HV, device=device, dtype=torch.float16) * s,
    )


ACC_CASES = [
    dict(B=2, T=16, H=2, HV=2, K=64, V=64),
    dict(B=2, T=32, H=2, HV=2, K=64, V=64),
    dict(B=2, T=32, H=2, HV=2, K=128, V=64),
    dict(B=2, T=32, H=2, HV=2, K=128, V=128),
    dict(B=2, T=16, H=2, HV=2, K=64, V=64, kda=True),
    dict(B=2, T=32, H=2, HV=2, K=128, V=128, kda=True),
    # T 不被 16 整除（SMEM 尾块保护）、GQA 分组、T < TT
    dict(B=2, T=24, H=2, HV=4, K=64, V=64),
    dict(B=2, T=8, H=2, HV=2, K=64, V=128),
    dict(B=2, T=17, H=2, HV=2, K=64, V=64),
]


def run_accuracy(atol=2e-2, rtol=2e-2):
    """与文件内 fp32 参考实现逐个 case 对比（bare / smem / state pool 全覆盖）。"""
    softplus_beta, softplus_threshold = 1.0, 20.0
    B_POOL = 2
    print("| case | variant | max_abs | ok |")
    print("| --- | --- | --- | --- |")
    for c in ACC_CASES:
        is_kda = c.get("kda", False)
        lower_bound = -5.0 if is_kda else None
        tag = " kda" if is_kda else ""
        A_log, a, dt_bias, q, k, v, b = _make_inputs(**c)
        scale = c["K"] ** -0.5
        name = f"B{c['B']}T{c['T']}H{c['H']}HV{c['HV']}K{c['K']}V{c['V']}{tag}"
        paths = [(None, "auto"), ("bare", "bare")]
        if _choose_tt(c["T"], _DEFAULT_TT) >= 2:
            paths.append(("smem", "smem"))
        for qk_l2norm in (False, True):
            ref = None
            for force_path, pname in paths:
                variant = f"{pname}/l2norm:{'on' if qk_l2norm else 'off'}"
                try:
                    if ref is None:
                        ref = reference_sigmoid_delta(
                            A_log, a, dt_bias, softplus_beta, softplus_threshold,
                            q, k, v, b, scale, qk_l2norm=qk_l2norm,
                            is_kda=is_kda, lower_bound=lower_bound)
                    got = sigmoid_gating_delta_rule_pc(
                        A_log, a, dt_bias, softplus_beta, softplus_threshold,
                        q, k, v, b, scale=scale, qk_l2norm=qk_l2norm,
                        force_path=force_path,
                        is_kda=is_kda, lower_bound=lower_bound)
                    diff = (got - ref).abs().max().item()
                    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
                    print(f"| {name} | {variant} | {diff:.3g} | {ok} |")
                except Exception:
                    traceback.print_exc()
                    print(f"| {name} | {variant} | ERR | - |")
        # state pool + final state
        try:
            pool = torch.randn(B_POOL, c["HV"], c["K"], c["V"],
                               device="cuda", dtype=torch.float32) * 0.1
            idx = torch.arange(c["B"], device="cuda", dtype=torch.int32) % B_POOL
            idx[0] = -1
            ref_o, ref_h = reference_sigmoid_delta(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b, scale, qk_l2norm=True,
                h0_source=pool, h0_indices=idx, output_final_state=True,
                is_kda=is_kda, lower_bound=lower_bound)
            got_o, got_h = sigmoid_gating_delta_rule_pc(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b, scale=scale, qk_l2norm=True,
                initial_state_source=pool, initial_state_indices=idx,
                output_final_state=True,
                is_kda=is_kda, lower_bound=lower_bound)
            diff_o = (got_o - ref_o).abs().max().item()
            diff_h = (got_h - ref_h).abs().max().item()
            ok = (torch.allclose(got_o, ref_o, atol=atol, rtol=rtol)
                  and torch.allclose(got_h, ref_h, atol=atol, rtol=rtol))
            print(f"| {name} | pool | o:{diff_o:.3g} h:{diff_h:.3g} | {ok} |")
        except Exception:
            traceback.print_exc()
            print(f"| {name} | pool | ERR | - |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()
    run_accuracy(atol=args.atol, rtol=args.rtol)