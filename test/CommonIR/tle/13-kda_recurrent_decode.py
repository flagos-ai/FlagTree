"""KDA recurrent decode（MetaX TLE / CommonIR 实现）。

    算子名称   : kda_recurrent_decode
    Kernel 名称: fused_recurrent_gated_delta_rule_fwd_kernel
    算子接口名 : fused_recurrent_gated_delta_rule_fwd（KDA decode adapter）

语义（每个 (batch, value-head) 一条独立递推，单 token decode）：
    g_t     = -exp(A_log) * softplus(a_t + dt_bias)
    beta_t  = sigmoid(b_t)
    h_t     = h_{t-1} * exp(g_t)
    v_new   = (v_t - h_t k_t^T) * beta_t
    h_t    += v_new^T k_t
    o_t     = h_t q_t^T * scale

q|k|v 打包在 `mixed_qkv`（`[B, 2*H*K + HV*V]`），gate/beta 在核内由 a/b/A_log/dt_bias 现算。

契约：
  * state 布局 `[num_slots, HV, V, K]`（K 连续），slot 步长可以是任意值
    （padded slot stride），按传入 stride 读，不假设 HV*V*K。
  * `ssm_state_indices`：`idx < 0` 表示无效 slot —— 输出写 0，state 不读写；
    `idx == 0` 是合法 slot。
  * beta 先按 b 的 dtype 舍入再转 fp32。
  * K 不要求是 2 的幂（BK 向上取整 + mask），V 不要求是 BV 的整数倍。

实现要点：
  * TLE 原语（本文件用到的）：
    - `tle.gpu.set_layout` + `tle.gpu.BlockEncoding`：把 h tile `[BV, BK]` 钉成
      「D 维连续、每 lane 连续 16 列」的布局（默认开）。不钉时 layout_tune 由
      `sum(axis=1)` 反推，B128/HV32 慢 1.0%，HV24 慢 3.0%。
    - `tle.gpu.metax.iglp`：收尾的取指/发射调度提示（默认开），K=128 两档 +1~3%。
    - `tle.gpu.alloc` + `tle.gpu.metax.async_copy_global_to_local` +
      `gvm_arrive` + `barrier` + `local_load`：h tile 走 SMEM 的异步搬运路径
      （`use_smem=True`，**默认关**）。设计上是在搬 4KB h tile 的同时算门控，
      但实测大形状只打平（1.01~1.02）、B32/K=V=64 反而 0.93~0.97——tile 太小，
      SMEM 往返 + 一次 barrier 盖过了重叠收益。
  * q/k/v/out/v_new 全部用 1-D 行/列向量，h 之外不构造 2-D 中间量：写成
    `[1,BV]` + `tl.permute` 后端要插 layout 转换，实测慢 2.5~3 倍。
  * 先发 q/k/v 三个小 load，再 load h tile（实测比先 load h 快 ~15%）。
  * BV 与 num_warps 配套：每 warp 管 16 行（w1→16, w2→32, w4→64, w8→128）。
  * stride 参数不要加 `do_not_specialize`：关掉 specialization 后编译器不知道
    stride 是 16 的倍数，地址运算多占几十个寄存器，BV>=32 直接溢出（256 regs /
    124 spills，慢 7 倍）；K=V=64 时也慢 1.5 倍。默认 bv=16 + 1 warp。

用法（两条路径都自包含，不依赖任何外部算子库）：
    python 13-kda_recurrent_decode.py              # 精度自测：8 个形状 × 3 条 TLE 路径 = 24 行
    python 13-kda_recurrent_decode.py --benchmark  # 延迟对比（mcoplib 基线已内联在第 4 节）
"""

from __future__ import annotations

import argparse
import traceback

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED

SOFTPLUS_THRESHOLD = 20.0


# =============================================================================
# 1. Packed decode kernel（单 token，q|k|v 打包进 mixed_qkv）
# =============================================================================


@triton.layout_tune()
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    mixed_qkv_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    o_ptr, h0_ptr, ht_ptr, indices_ptr,
    scale,
    stride_mixed, stride_a, stride_b, stride_h, stride_idx,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    ALIGNED: tl.constexpr,
    H_ENC: tl.constexpr,
    USE_SMEM: tl.constexpr,
    USE_IGLP: tl.constexpr,
):
    i_v = tl.program_id(axis=0)
    i_nh = tl.program_id(axis=1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    # 全部用 1-D 行/列向量：[BV] 的 v、out，[BK] 的 q、k。
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # TLE：h tile 走 SMEM。最大的一块（4KB/CTA）先异步搬，搬运期间算门控。
    if USE_SMEM:
        h_smem = tle.gpu.alloc((BV, BK), h0_ptr.dtype.element_ty)

    state_idx = tl.load(indices_ptr + i_n * stride_idx).to(tl.int64)
    p_o = o_ptr + (i_n * HV + i_hv) * V + o_v
    if state_idx < 0:                                     # idx<0: 输出 0，state 不动
        zero = tl.zeros([BV], dtype=o_ptr.dtype.element_ty)
        if ALIGNED:
            tl.store(p_o, zero)
        else:
            tl.store(p_o, zero, mask=mask_v)
        return

    o_h = o_v[:, None] * K + o_k[None, :]                 # state tile [BV, BK] 的偏移
    p_h = h0_ptr + state_idx * stride_h + i_hv * V * K

    # 1) 先把 h tile 的异步搬运发出去（global -> SMEM）
    if USE_SMEM:
        if ALIGNED:
            tle.gpu.metax.async_copy_global_to_local(p_h + o_h, h_smem)
        else:
            tle.gpu.metax.async_copy_global_to_local(p_h + o_h, h_smem, mask=mask_h)
        tle.gpu.metax.gvm_arrive(1)

    # 2) 搬运在飞：q/k/v 小 load + 标量门控计算
    p_mixed = mixed_qkv_ptr + i_n * stride_mixed
    if ALIGNED:
        q_row = tl.load(p_mixed + i_h * K + o_k).to(tl.float32)
        k_row = tl.load(p_mixed + H * K + i_h * K + o_k).to(tl.float32)
        v_row = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v).to(tl.float32)
    else:
        q_row = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        k_row = tl.load(p_mixed + H * K + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        v_row = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0.0).to(tl.float32)

    if USE_QK_L2NORM:
        q_row = q_row * tl.rsqrt(tl.sum(q_row * q_row) + 1e-6)
        k_row = k_row * tl.rsqrt(tl.sum(k_row * k_row) + 1e-6)
    q_row = q_row * scale

    a_val = tl.load(a_ptr + i_n * stride_a + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * stride_b + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD,
                          tl.log(1.0 + tl.exp(tl.minimum(x, SOFTPLUS_THRESHOLD))), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = (1.0 / (1.0 + tl.exp(-b_val))).to(b_ptr.dtype.element_ty).to(tl.float32)

    # 3) 等 h 落地，再进寄存器
    if USE_SMEM:
        tle.gpu.metax.barrier()
        h = tle.gpu.metax.local_load(
            h_smem, intrinsic=True, is_constant_offs=True).to(tl.float32)
    else:
        if ALIGNED:
            h = tl.load(p_h + o_h).to(tl.float32)
        else:
            h = tl.load(p_h + o_h, mask=mask_h, other=0.0).to(tl.float32)
    if H_ENC is not None:
        h = tle.gpu.set_layout(h, H_ENC)

    h = h * tl.exp(g_val)
    v_row = (v_row - tl.sum(h * k_row[None, :], axis=1)) * beta_val
    h = h + v_row[:, None] * k_row[None, :]

    out = tl.sum(h * q_row[None, :], axis=1)
    if ALIGNED:
        tl.store(p_o, out.to(o_ptr.dtype.element_ty))
        tl.store(ht_ptr + state_idx * stride_h + i_hv * V * K + o_h,
                 h.to(ht_ptr.dtype.element_ty))
    else:
        tl.store(p_o, out.to(o_ptr.dtype.element_ty), mask=mask_v)
        tl.store(ht_ptr + state_idx * stride_h + i_hv * V * K + o_h,
                 h.to(ht_ptr.dtype.element_ty), mask=mask_h)
    if USE_IGLP:
        tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=2)


def fused_recurrent_gated_delta_rule_fwd(
    mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out, ssm_state_indices,
    use_qk_l2norm_in_kernel=False, num_warps=None, bv=16,
    use_smem=False, pin_layout=True, use_iglp=True,
):
    """KDA decode adapter：单 token decode。state 原地更新，返回 (out, initial_state)。"""
    assert ENABLED, "Requires a MetaX FLAGTREE_COMMON_IR build"
    if mixed_qkv.dim() != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv 必须是 [B, 2*H*K + HV*V] 且最后一维连续")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a/b 必须是 2D")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("a/b 最后一维必须连续")
    if initial_state.dim() != 4:
        raise ValueError("initial_state 必须是 4D")
    if initial_state.stride(-1) != 1:
        raise ValueError("initial_state 最后一维必须连续")
    if ssm_state_indices.dim() != 1:
        raise ValueError("ssm_state_indices 必须是 1D")

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    if out.shape != (B, 1, HV, V):
        raise ValueError(f"out 期望 {(B, 1, HV, V)}，got {tuple(out.shape)}")
    qkv_dim = mixed_qkv.shape[1]
    q_dim = (qkv_dim - HV * V) // 2
    if q_dim <= 0 or q_dim % K != 0:
        raise ValueError(f"mixed_qkv 宽度 {qkv_dim} 与 HV={HV}/V={V}/K={K} 不匹配")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(f"推不出合法 head 配置：H={H}, HV={HV}")

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"仅支持 NK=1（K={K}, BK={BK}）")
    # BV 跟随 warp 数：每 warp 管 16 行（w1→BV16, w2→32, w4→64）。显式给 num_warps
    # 时按它反推 BV，否则用 bv（默认 16，必须和 1 warp 配套，错配会慢一个量级）。
    vmax = triton.next_power_of_2(V)
    if num_warps is None:
        BV = min(vmax, int(bv))
        nw = 1
    else:
        nw = int(num_warps)
        BV = min(vmax, 16 * nw)
    NV = triton.cdiv(V, BV)

    fused_recurrent_gated_delta_rule_fwd_kernel[(NV, B * HV)](
        mixed_qkv, a, b, A_log, dt_bias, out, initial_state, initial_state,
        ssm_state_indices,
        float(scale),
        mixed_qkv.stride(0), a.stride(0), b.stride(0),
        initial_state.stride(0), ssm_state_indices.stride(0),
        H, HV, K, V, BK, BV,
        SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
        USE_QK_L2NORM=bool(use_qk_l2norm_in_kernel),
        ALIGNED=(K == BK and V % BV == 0),
        H_ENC=(tle.gpu.BlockEncoding([1, 1], [4, 16], [nw, 1], [1, 0])
               if pin_layout else None),
        USE_SMEM=bool(use_smem),
        USE_IGLP=bool(use_iglp),
        num_warps=nw,
    )
    return out, initial_state


# 算子名称（注册用别名）
kda_recurrent_decode = fused_recurrent_gated_delta_rule_fwd


# =============================================================================
# 2. 参考实现
# =============================================================================


def reference(q, k, v, g, beta, state, normalize=False):
    q, k, v = q.float(), k.float(), v.float()
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    group = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(group, dim=2) * q.shape[-1] ** -0.5
    k = k.repeat_interleave(group, dim=2)
    state = state.float().clone()
    outputs = []
    for t in range(q.shape[1]):
        decay = g[:, t].float().exp()
        state = state * (decay[:, :, None, :] if g.dim() == 4 else decay[..., None, None])
        b_beta = beta[:, t].float()
        if b_beta.dim() == 2:
            b_beta = b_beta[..., None]
        delta = (v[:, t] - (state * k[:, t, :, None]).sum(-1)) * b_beta
        state = state + delta[..., None] * k[:, t, :, None]
        outputs.append((state * q[:, t, :, None]).sum(-1))
    return torch.stack(outputs, 1), state


def _packed_gate(a, b, A_log, dt_bias):
    x = a.float() + dt_bias.float()
    sp = torch.where(x <= SOFTPLUS_THRESHOLD,
                     torch.log(1.0 + torch.exp(x.clamp(max=SOFTPLUS_THRESHOLD))), x)
    g = -A_log.exp() * sp
    beta = b.float().sigmoid().to(b.dtype).float()
    return g, beta


# =============================================================================
# 3. 精度自测
# =============================================================================


def _check(name, got, exp, atol, rtol):
    diff = (got.float() - exp.float()).abs().max().item()
    ok = torch.allclose(got.float(), exp.float(), atol=atol, rtol=rtol)
    print(f"| {name} | {diff:.3g} | {ok} |")
    return ok


def _packed_inputs(B, H, HV, K, V, idx=None, padded_slot=False,
                   state_dtype=torch.float32):
    dev = "cuda"
    torch.manual_seed(11)
    mixed = (torch.randn(B, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16) * 0.1)
    a = torch.randn(B, HV, device=dev, dtype=torch.bfloat16) * 0.5 - 1
    b = torch.randn(B, HV, device=dev, dtype=torch.bfloat16) * 0.5
    A_log = torch.randn(HV, device=dev) * 0.1
    dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16) * 0.1

    n_slot = B + 2
    if padded_slot:
        # 每个 slot 尾部留 64 个 fp32 的 padding，验证不依赖 HV*V*K 步长
        backing = torch.randn(n_slot, HV * V * K + 64, device=dev, dtype=state_dtype) * 0.01
        state = backing[:, : HV * V * K].view(n_slot, HV, V, K)
    else:
        state = torch.randn(n_slot, HV, V, K, device=dev, dtype=state_dtype) * 0.01

    if idx is None:
        idx = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
    else:
        idx = torch.tensor(idx, device=dev, dtype=torch.int32)
    return dict(mixed=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
                state=state, idx=idx)


def _packed_case(name, B, H, HV, K, V, l2norm=True, idx=None,
                 padded_slot=False, state_dtype=torch.float32, bv=16,
                 kern_kw=None):
    dev = "cuda"
    inp = _packed_inputs(B, H, HV, K, V, idx, padded_slot, state_dtype)
    mixed, a, b = inp["mixed"], inp["a"], inp["b"]
    A_log, dt_bias, idx = inp["A_log"], inp["dt_bias"], inp["idx"]
    state = inp["state"]
    before = state.clone()

    q, k, v = mixed.split([H * K, H * K, HV * V], dim=-1)
    g, beta = _packed_gate(a, b, A_log, dt_bias)
    sel = idx.clamp_min(0).long()
    exp_o, exp_s = reference(
        q.reshape(B, 1, H, K), k.reshape(B, 1, H, K), v.reshape(B, 1, HV, V),
        g[:, None], beta[:, None], before[sel], normalize=l2norm)
    exp_o = exp_o.clone()
    exp_o[idx < 0] = 0

    out = torch.empty(B, 1, HV, V, device=dev, dtype=torch.bfloat16)
    kda_recurrent_decode(
        mixed, a, b, A_log, dt_bias, K ** -0.5, state, out, idx,
        use_qk_l2norm_in_kernel=l2norm, bv=bv, **(kern_kw or {}))

    ok_o = _check(name, out, exp_o, 3e-3, 3e-2)
    # 有效 slot 必须被更新；无效 slot 必须原样不动
    ok_s = True
    for i in range(B):
        si = int(idx[i])
        if si < 0:
            ok_s &= bool(torch.equal(state[si], before[si]))
        else:
            ok_s &= bool(torch.allclose(state[si].float(), exp_s[i], atol=3e-4, rtol=3e-3))
    print(f"| {name} | state | - | {ok_s} |")
    return ok_o and ok_s


PACKED_CASES = [
    # 主形状：Qwen3.5 4B/9B 与 27B TP=2
    ("B128 H16 HV32 K128 V128", dict(B=128, H=16, HV=32, K=128, V=128)),
    ("B8 H8 HV24 K128 V128 (27B)", dict(B=8, H=8, HV=24, K=128, V=128)),
    ("B128 H8 HV24 K128 V128 (27B)", dict(B=128, H=8, HV=24, K=128, V=128)),
    # l2norm 关
    ("B32 H16 HV32 K128 V128 l2off", dict(B=32, H=16, HV=32, K=128, V=128, l2norm=False)),
    # 边界：V 不是 BV 的倍数 / K 不是 2 的幂 / bf16 state
    ("B16 H4 HV8 K64 V48", dict(B=16, H=4, HV=8, K=64, V=48)),
    ("B16 H4 HV8 K96 V64", dict(B=16, H=4, HV=8, K=96, V=64)),
    ("B16 H4 HV8 K64 V64 bf16state", dict(B=16, H=4, HV=8, K=64, V=64,
                                         state_dtype=torch.bfloat16)),
    # idx 语义：有效/无效混合 + padded slot stride
    ("idx=[2,-1,0] padded", dict(B=3, H=2, HV=4, K=128, V=128,
                                 idx=[2, -1, 0], padded_slot=True)),
]


# TLE 开关的正确性覆盖：每个形状都要在三条 kernel 路径上过一遍
#   默认      = 钉布局（set_layout）+ iglp
#   smem      = h tile 走 SMEM 异步搬运（async_copy + gvm_arrive + barrier）
#   nopin     = 两个原语全关（裸 Triton 路径，作对照）
VARIANTS = [
    ("", {}),
    (" smem", dict(use_smem=True)),
    (" nopin", dict(pin_layout=False, use_iglp=False)),
]


def run_accuracy():
    print("### packed decode（每个形状 × 3 条 TLE 路径）")
    print("| case | max_abs | ok |")
    print("| --- | --- | --- |")
    total = 0
    ok = 0
    for base, kw in PACKED_CASES:
        for suffix, kk in VARIANTS:
            total += 1
            name = base + suffix
            try:
                ok += bool(_packed_case(name, kern_kw=kk, **kw))
            except Exception:
                traceback.print_exc()
                print(f"| {name} | ERR | False |")
                print(f"| {name} | state | - | False |")
    print(f"\n结果：{ok}/{total} 行通过")


# =============================================================================
# 4. mcoplib 基线（内联，仅供 --benchmark 对比）
#
# 摘自 FlagTree-metax-gluon-auto-layout/test/CommonIR/all-test/mcoplib_recurrent.py
# 的 packed decode 部分（该文件改编自 flash-linear-attention，MIT，
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang）。这里只保留 benchmark 用到的
# 两个 kernel 和同款形状→配置选择，入参校验从略，kernel 代码未改动。
# =============================================================================


@triton.jit
def _mc_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    LOAD_ORDER: tl.constexpr = 0,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    # Experimental source schedules; the backend may still reorder instructions.
    if LOAD_ORDER == 0 or LOAD_ORDER == 2:
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    if LOAD_ORDER < 2:
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        if LOAD_ORDER < 2:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    if LOAD_ORDER < 2:
        b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    if LOAD_ORDER == 1 or LOAD_ORDER == 3:
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    if LOAD_ORDER >= 2:
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.jit
def _mc_packed_decode_grouped_hv_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    REUSE_Q: tl.constexpr,
    GROUPED_SCHEDULE: tl.constexpr,
):
    """Process all value heads sharing one Q/K head in one program.

    The stable kernel launches one program per value head and therefore repeats
    the K load and K reduction HV/H times.  Qwen3.5-27B has HV/H=3.  This
    try schedule keeps normalized K live, processes those three value
    heads sequentially, and optionally keeps normalized Q live as well.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32)
        for i_g in tl.static_range(0, GROUP_SIZE):
            i_hv = i_h * GROUP_SIZE + i_g
            p_o = o + (i_n * HV + i_hv) * V + o_v
            tl.store(p_o, zero, mask=mask_v)
        return

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = H * K + i_h * K + o_k
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

    if REUSE_Q:
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale

    for i_g in tl.static_range(0, GROUP_SIZE):
        i_hv = i_h * GROUP_SIZE + i_g
        v_off = 2 * H * K + i_hv * V + o_v
        # Experimental schedules: bit 0 loads state before gate computation;
        # bit 1 writes updated state before output reduction. Default stays 0.
        if GROUPED_SCHEDULE & 1:
            p_h = h0 + state_idx * stride_init_state_token
            p_h = p_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

        if not (GROUPED_SCHEDULE & 1):
            p_h = h0 + state_idx * stride_init_state_token
            p_h = p_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        b_h *= tl.exp(g_val)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]

        if GROUPED_SCHEDULE & 2:
            p_ht = ht + state_idx * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        if not REUSE_Q:
            b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
        b_o = tl.sum(b_h * b_q[None, :], 1)
        p_o = o + (i_n * HV + i_hv) * V + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if not (GROUPED_SCHEDULE & 2):
            p_ht = ht + state_idx * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def _mcoplib_packed_decode(mixed_qkv, a, b, A_log, dt_bias, scale,
                           initial_state, out, ssm_state_indices,
                           use_qk_l2norm_in_kernel=False):
    """mcoplib packed decode：BV/LOAD_ORDER/grouped-hv 的选法与上游一致。"""
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    H = ((mixed_qkv.shape[1] - HV * V) // 2) // K
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    num_warps, num_stages = 1, 3

    # 0: original; 1: late state; 2: late Q; 3: late state and Q
    measured = (((H == 16 and HV == 32) or (H == 8 and HV == 24))
                and K == 128 and V == 128 and BV == 32 and num_warps == 1)
    load_order = (1 if B * HV <= 128 else 3) if measured else 0

    group_value_heads, reuse_q, grouped_num_warps = False, False, num_warps
    if H == 8 and HV == 24 and K == 128 and V == 128 and BV == 32:
        if B == 8 or (16 <= B <= 128 and B % 8 == 0):
            group_value_heads, reuse_q, BV, grouped_num_warps = True, True, 16, 1
        elif 4 <= B <= 8:
            group_value_heads, reuse_q, grouped_num_warps = True, True, 4
        elif 16 <= B <= 128:
            group_value_heads, reuse_q, grouped_num_warps = True, True, 2

    strides = dict(
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
    )
    common = dict(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        o=out, h0=initial_state, ht=initial_state,
        ssm_state_indices=ssm_state_indices, scale=scale,
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        **strides,
    )
    NV = triton.cdiv(V, BV)
    if group_value_heads:
        _mc_packed_decode_grouped_hv_kernel[(NV, B * H)](
            GROUP_SIZE=HV // H, REUSE_Q=reuse_q, GROUPED_SCHEDULE=0,
            num_warps=grouped_num_warps, num_stages=num_stages, **common)
    else:
        _mc_packed_decode_kernel[(NV, B * HV)](
            LOAD_ORDER=load_order, num_warps=num_warps,
            num_stages=num_stages, **common)
    return out, initial_state


# =============================================================================
# 5. 延迟对比（--benchmark）
# =============================================================================


def _bench(fn, warmup=50, rep=200, rounds=3):
    """取 rounds 次 do_bench 的最小值（ms）：共享卡上别的租户一抢占，
    均值就会被抬高，最小值对抢占不敏感。"""
    return min(triton.testing.do_bench(fn, warmup=warmup, rep=rep)
               for _ in range(rounds))


def _bench_packed(B, H, HV, K, V, bv=16, warps=None):
    vmax = triton.next_power_of_2(V)
    eff_w = 1 if warps is None else int(warps)
    eff_bv = min(vmax, int(bv)) if warps is None else min(vmax, 16 * eff_w)

    inp = _packed_inputs(B, H, HV, K, V)
    mixed, a, b = inp["mixed"], inp["a"], inp["b"]
    A_log, dt_bias, idx = inp["A_log"], inp["dt_bias"], inp["idx"]
    state = inp["state"]
    out = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)

    ours = _bench(lambda: kda_recurrent_decode(
        mixed, a, b, A_log, dt_bias, K ** -0.5, state, out, idx,
        use_qk_l2norm_in_kernel=True, bv=bv, num_warps=warps))

    mc = _bench(lambda: _mcoplib_packed_decode(
        mixed, a, b, A_log, dt_bias, K ** -0.5, state, out, idx, True))

    name = f"B{B} H{H} HV{HV} K{K} V{V} bv={eff_bv} w={eff_w}"
    print(f"| {name} | {ours * 1e3:.1f} | {mc * 1e3:.1f} | {mc / ours:.3f} |")


def run_benchmark():
    print("### packed decode benchmark（us，spd = mcoplib/ours，>1 我方快）")
    print("| case | ours_us | mc_us | spd(mc) |")
    print("| --- | --- | --- | --- |")
    for B in (1, 8, 32, 128):
        _bench_packed(B, 16, 32, 128, 128)
    for B in (8, 32, 128):
        _bench_packed(B, 8, 24, 128, 128)
    # 补充尺寸：更大 batch / K=V=64 / 无 GQA（HV==H）
    _bench_packed(256, 16, 32, 128, 128)
    _bench_packed(16, 8, 24, 128, 128)
    _bench_packed(32, 16, 32, 64, 64)
    _bench_packed(128, 8, 24, 64, 64)
    _bench_packed(64, 16, 16, 128, 128)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true",
                        help="只跑延迟对比，不做精度自测")
    args = parser.parse_args()
    if args.benchmark:
        run_benchmark()
    else:
        run_accuracy()
