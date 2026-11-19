"""TLE (MetaX) chunk_kda_prefill —— chunked KDA prefill，v1.0。

两个 kernel，BT=16 / BV=64（MetaX C550）：

  kernel1 intra  grid (NT, B*HV)：chunk 内 gate cumsum → Aqk/Akk →
                 (I+Akk)^-1（三次平方）→ w/qg/kg。写 Aqk、Akk、ws、g_last。
  kernel2 state_output  grid (NV, B*HV)：按 chunk 顺序推进，h 常驻寄存器。
                 v_new = A^-1(vb) - w@h
                 o     = scale*h@q + Aqk@v_new
                 h     = h*exp2(g_last) + kg^T@v_new
                 六个 per-chunk tile 走双缓冲 async copy。

改这个文件之前要知道的三件事：

  * A^-1 的作用域是 v_new = A^-1(vb) - w@h，不是 A^-1(vb - w@h)。
    h=0 时两式相同，所以写错时单 chunk 测试全绿、从第 2 个 chunk 开始崩。
  * async copy 只留在验证过确定性的位置。kernel1 入口那三块已退回普通
    tl.load：本构建上 3 路 async copy + gvm_arrive(3) 会让 Aqk/Akk/ws 在
    重复调用间漂移，而这三块每个 CTA 只读一次，本来也没有重叠收益。
  * g_last 走全局而非 smem：3-D (1,1,K) 的 smem 往返带布局嫌疑且会漂。

支持范围：K == V == 128、bf16、HV == H（无 GVA）、chunk_size=16、BV=64。
超范围组合在 chunk_kda_prefill_tle 里 assert。

============================== 两条基线 ==============================

都内联在本文件内，bench 一次打印两列 speedup（gems/TLE、mcoplib/TLE）。

  flaggems  FlagGems-vllm 自带的整条 chunk_kda（BT=16，plain-Triton 后端，
            L2norm / 门控 / beta 全部融在 kernel 内），与 TLE 同构，是两列里
            最 apples-to-apples 的一条。
  mcoplib   intra_token_parallel(BT=64, BC=16) → inter_solve_fused(
            FUSE_DIAGONAL + FUSE_RECOMPUTE，出 w/u/kg) → fwd_h(use_exp2) →
            flaggems chunk_gla_fwd_o_gk。
            mcoplib 库里没有 o kernel，也没有融合前处理的 python 入口，所以 o
            段借 flaggems 同源实现、前处理用 torch 拼（sglang 系三家此处本就
            同构）。这两处都只会让该列更慢，不会更快。

flaggems 列需要一处在内存里打的 patch（不落盘）：
  它的 triangular inverse 用 tl.dot(..., out_dtype=tl.float16) 做 fp16 累加，
  本 MACA 后端没有 fp16 输出 MMA，原样编译直接失败（PassManager::run failed /
  "Unsupported mma type found"，并把整段 MLIR dump 冲到 stdout）。加载时把 3 处
  fp16-accum 块换成 fp32-accum 再 round 回 fp16（FP32_FP16_FP16_FP32，后端支持
  的路径）。该 patch 只会让基线变慢。
  关键细节：patch 后的源码必须同时塞进 linecache.cache（mtime 置 None，避免
  checkcache 丢弃）——Triton JIT 编译时按 co_filename 用 inspect.getsource 重新
  读磁盘，只 exec 内存版本的话 JIT 照样拿到未 patch 的原文然后编译失败。

flaggems 列是软依赖：源码树缺失、上游改动导致 patch 点为 0、patch 后残留
  out_dtype=tl.float16、或编译失败 —— 任何一环出问题都只让该列打印 n/a，
  不影响 TLE / mcoplib 两列，也不让整个文件崩。

============================== 精度口径 ==============================

safe_gate 的 chunk 内 cumsum 会 clamp_min(-16)，而每步门控 ≈ -3.6，于是 BT=16
的 chunk 从第 5 个 token 起就压到地板、BT=64 不会 —— chunk 大小本身改变数学
语义（同输入下 ref(BT=16) 与 ref(BT=64) 差 5.1e-2，|o| 量级 1.4e-2）。flaggems
的 kernel 根本不 clamp，是第三个函数（同输入与 clamp 版差 3.4e-2，|o| 量级
2.3e-2）。所以每列只能对比同 chunk 大小、同 clamp 语义的 fp32 参考，三方输出
不可直接互比：

    TLE → BT=16+clamp    mcoplib → BT=64+clamp    flaggems → BT=16 无 clamp

本文件有两个 fp32 参考，不是重复：
  reference_chunk_kda       逐 token、逐 slot 的标量循环，慢但直白，BT=16 固定。
                            它是 oracle。
  reference_chunk_kda_fast  按 chunk 向量化，BT 与 clamp 可参数化，用来扫档。
test_reference_equivalence 先用它俩在小形状上互锁（差 ~1e-6），再让快的那个去
扫大 T。否则"精度通过"只说明两个实现一样快地错。

计时口径：三列都带各自的 wrapper / 前处理开销，do_bench 同参
  （warmup=10, rep=50），计时函数内各自 _clone_inputs，host 侧开销对称。
  轮转交替测量（每轮把所有候选各测一次，跨 rounds 轮取中位数），不是单次
  do_bench —— 本卡单次摆动可达 ±10%，与要分辨的差异同量级。
  speedup = 基线耗时 / TLE 耗时，>1 为 TLE 快。

============================== 实测 ==============================

2026-11-18，C550，同一份代码连跑 3 轮：

  精度: tle 2.17e-4~2.21e-4 / mc 2.24e-4~3.73e-4 / fg 7.0e-5~1.4e-4

  | shape      | tle_ms        | mc/tle        | fg/tle        |
  | B1T512H16  | 0.364~0.372   | 1.77~2.03x    | 0.97~1.09x    |
  | B1T2048H16 | 0.883~0.926   | 2.10~2.14x    | 1.03~1.15x    |
  | B4T2048H16 | 2.004~2.054   | 2.63~2.71x    | 1.18~1.22x    |
  | B1T8192H16 | 3.436~3.489   | 1.74~1.80x    | 1.03~1.06x    |

  区间是多次运行的全部观测。怎么读：
    mcoplib 列 1.77–2.71x，跨轮稳定，是本文件的可信结论。
    flaggems 列只有 B4T2048 那档（1.18~1.22x）是稳定赢面；T512 与 T8192 的
    区间跨过 1.0，两档实际是打平，别把 1.06x 当成优化成果引用。

============================== 运行 ==============================

  python3 15-chunk_kda_prefill.py                # 精度 + 三方 A/B
  python3 15-chunk_kda_prefill.py --accuracy     # 参考互锁 + 形状扫描
  python3 15-chunk_kda_prefill.py --perf         # 仅 TLE 延迟
  python3 15-chunk_kda_prefill.py --ab           # flaggems / mcoplib speedup
  python3 15-chunk_kda_prefill.py --ab --rounds 5 --shapes 512,2048

  不带参数时跑精度 + A/B，不单列 --perf 那张表（A/B 已经报过同样形状的
  TLE 延迟）。基线是可选的：mcoplib / FlagGems-vllm 源码树不在时对应列打
  n/a，精度与 TLE 延迟照常。
"""

import argparse
import contextlib
import importlib.util
import io
import linecache
import os
import re
import sys
import traceback

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED

LOG2E = 1.4426950408889634
L2NORM_EPS = 1e-6
GC_CLAMP = -16.0  # 保守 clamp：exp2(16)=65536，bf16 完全安全
LOWER_BOUND = -5.0


# =============================================================================
# Kernel 1: intra
# =============================================================================


@triton.layout_tune()
@triton.jit
def _kda_fwd_intra_tle_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr,
    A_log_ptr, dt_bias_ptr,
    Aqk_ptr, Akk_ptr, ws_ptr, g_last_ptr,
    lower_bound, g_scale, scale, l2norm_eps, gc_clamp,
    T_orig, T_pad,
    H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    NT = T_pad // BT

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)

    q_base = q_ptr + ((i_b * T_pad + i_t * BT) * H + i_h) * K
    k_base = k_ptr + ((i_b * T_pad + i_t * BT) * H + i_h) * K
    g_base = g_ptr + ((i_b * T_pad + i_t * BT) * HV + i_hv) * K

    q_ptrs = q_base + o_i[:, None] * (H * K) + o_k[None, :]
    k_ptrs = k_base + o_i[:, None] * (H * K) + o_k[None, :]
    g_ptrs = g_base + o_i[:, None] * (HV * K) + o_k[None, :]

    # 这三块用普通 tl.load，不要换成 async copy：3 路 async copy +
    # gvm_arrive(3)/barrier 在本构建上完成语义不可靠，重复调用时 Aqk/Akk/ws
    # 会漂。它们每个 CTA 只在入口读一次，本来也没有重叠收益。
    b_q = tl.load(q_ptrs)
    b_k = tl.load(k_ptrs)
    b_g = tl.load(g_ptrs).to(tl.float32)

    beta_base = beta_ptr + (i_b * T_pad + i_t * BT) * HV + i_hv
    b_beta = tl.load(beta_base + o_i * HV).to(tl.float32)
    b_beta = 1.0 / (1.0 + tl.exp(-b_beta))
    b_A = tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32))
    b_dt = tl.load(dt_bias_ptr + i_hv * K + o_k).to(tl.float32)

    b_qf = b_q.to(tl.float32)
    b_kf = b_k.to(tl.float32)

    b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
    b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

    b_g = b_g + b_dt[None, :]
    b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)

    in_dt = q_ptr.dtype.element_ty

    lower_tri = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_gc = tl.dot(lower_tri.to(in_dt), b_g.to(in_dt), input_precision="tf32")
    b_gc = tl.maximum(b_gc, gc_clamp)

    b_gq = tl.exp2(b_gc)
    b_gk = tl.exp2(-b_gc)

    # 三个归一化-门控乘积各算一次：b_kgp 供 Akk / w / kg 共用，b_qg_bf 供
    # Aqk 与 qg store 共用。乘数结合顺序与逐位精度都和展开写法一致。
    b_qgp = b_qf * b_q_rstd[:, None]
    b_kgp = b_kf * b_k_rstd[:, None]
    b_qg_bf = (b_qgp * b_gq).to(in_dt)
    b_kg_bf = (b_kgp * b_gk).to(in_dt)
    b_kk_bf = (b_kgp * b_gq).to(in_dt)

    b_Aqk_raw = tl.dot(b_qg_bf, tl.trans(b_kg_bf), input_precision="tf32")
    b_Akk_raw = tl.dot(b_kk_bf, tl.trans(b_kg_bf), input_precision="tf32")

    # 防御：NaN 检测（NaN != NaN → 清零）
    b_Aqk_raw = tl.where(b_Aqk_raw == b_Aqk_raw, b_Aqk_raw, 0.0)
    b_Akk_raw = tl.where(b_Akk_raw == b_Akk_raw, b_Akk_raw, 0.0)

    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk_raw * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk_raw * b_beta[:, None], 0.0)

    Aqk_offs = ((i_b * T_pad + i_t * BT) * HV + i_hv) * BT
    Aqk_ptrs = Aqk_ptr + Aqk_offs + o_i[:, None] * (HV * BT) + o_i[None, :]
    tl.store(Aqk_ptrs, b_Aqk.to(Aqk_ptr.dtype.element_ty))

    b_I_dt = tl.where(m_I, 1.0, 0.0).to(in_dt)
    b_L = b_Akk.to(in_dt)
    b_Ai = b_I_dt - b_L
    b_L2 = tl.dot(b_L, b_L).to(in_dt)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2).to(in_dt)
    b_L4 = tl.dot(b_L2, b_L2).to(in_dt)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4).to(in_dt)
    b_L8 = tl.dot(b_L4, b_L4).to(in_dt)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8).to(in_dt)

    # 防御：清除 NaN
    b_Ai_f = b_Ai.to(tl.float32)
    b_Ai_f = tl.where(b_Ai_f == b_Ai_f, b_Ai_f, 0.0)
    b_Ai = b_Ai_f.to(in_dt)

    Akk_ptrs = Akk_ptr + Aqk_offs + o_i[:, None] * (HV * BT) + o_i[None, :]
    tl.store(Akk_ptrs, b_Ai.to(Akk_ptr.dtype.element_ty))

    # 这里原有一个 g_cs 缓冲区的 store：fp32、只写不读（kernel2 从不读它），
    # 单次调用写 16 MiB = kernel1 全部 store 的 37%。该核 MTE 受限（duty 51%，
    # MMA 仅 4.4%），这块浪费的写直接落在瓶颈上，已删除。不要加回来。
    last_local = tl.minimum(BT, T_orig - i_t * BT) - 1
    b_g_last = tl.sum(tl.where(o_i[:, None] == last_local, b_gc, 0.0), axis=0)
    g_last_offs = ((i_b * NT + i_t) * HV + i_hv) * K
    tl.store(g_last_ptr + g_last_offs + o_k, b_g_last)

    b_kb = (b_kgp * b_beta[:, None] * b_gq).to(in_dt)
    b_w_raw = tl.dot(b_Ai, b_kb, input_precision="tf32")
    b_w = tl.where(b_w_raw == b_w_raw, b_w_raw, 0.0).to(ws_ptr.dtype.element_ty)
    b_qg_val = b_qg_bf
    b_kg_val = (b_kgp * tl.exp2(b_g_last[None, :] - b_gc)).to(in_dt)

    ws_offs = ((i_b * T_pad + i_t * BT) * HV + i_hv) * (3 * K)
    w_ptrs = ws_ptr + ws_offs + o_i[:, None] * (HV * 3 * K) + o_k[None, :]
    qg_ptrs = ws_ptr + ws_offs + o_i[:, None] * (HV * 3 * K) + K + o_k[None, :]
    kg_ptrs = ws_ptr + ws_offs + o_i[:, None] * (HV * 3 * K) + 2 * K + o_k[None, :]
    tl.store(w_ptrs, b_w)
    tl.store(qg_ptrs, b_qg_val)
    tl.store(kg_ptrs, b_kg_val)

    tle.gpu.metax.iglp(config_2=1, config_5=1, config_6=2)


# =============================================================================
# Kernel 2: state_output
# =============================================================================


@triton.jit
def _async_copy_chunk(
    ws_ptr, Aqk_ptr, Akk_ptr, v_ptr,
    i_b, i_hv, i_t,
    T, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, ALIGNED_V: tl.constexpr,
    o_i, o_k, o_v, mask_v,
    w_slot, qg_slot, kg_slot, Aqk_slot, Akk_slot, v_slot,
):
    t_offset = i_t * BT
    ws_offs = ((i_b * T + t_offset) * HV + i_hv) * (3 * K)
    w_ptrs = ws_ptr + ws_offs + o_i[:, None] * (HV * 3 * K) + o_k[None, :]
    qg_ptrs = ws_ptr + ws_offs + o_i[:, None] * (HV * 3 * K) + K + o_k[None, :]
    kg_ptrs = ws_ptr + ws_offs + o_i[:, None] * (HV * 3 * K) + 2 * K + o_k[None, :]
    tle.gpu.metax.async_copy_global_to_local(w_ptrs, w_slot)
    tle.gpu.metax.async_copy_global_to_local(qg_ptrs, qg_slot)
    tle.gpu.metax.async_copy_global_to_local(kg_ptrs, kg_slot)

    Aqk_offs = ((i_b * T + t_offset) * HV + i_hv) * BT
    Aqk_ptrs = Aqk_ptr + Aqk_offs + o_i[:, None] * (HV * BT) + o_i[None, :]
    Akk_ptrs = Akk_ptr + Aqk_offs + o_i[:, None] * (HV * BT) + o_i[None, :]
    tle.gpu.metax.async_copy_global_to_local(Aqk_ptrs, Aqk_slot)
    tle.gpu.metax.async_copy_global_to_local(Akk_ptrs, Akk_slot)

    # g_last 不在这里搬：3-D (1,1,K) 的 smem 往返会漂，kernel2 循环内直接
    # tl.load 全局。
    # v 搬原始值，sigmoid(beta) 在 kernel2 里算。把 vb 在 host 侧预算会变成
    # (v.float()*sigmoid(beta.float())).to(bf16)，v 被读 3 写 3、20 B/elem，
    # 实测单这一笔就占 T512 端到端 21%。
    v_offs = ((i_b * T + t_offset) * HV + i_hv) * V
    v_ptrs = v_ptr + v_offs + o_i[:, None] * (HV * V) + o_v[None, :]
    if ALIGNED_V:
        tle.gpu.metax.async_copy_global_to_local(v_ptrs, v_slot)
    else:
        tle.gpu.metax.async_copy_global_to_local(v_ptrs, v_slot, mask=mask_v[None, :])


@triton.layout_tune()
@triton.jit
def _kda_fwd_state_output_tle_kernel(
    ws_ptr, Aqk_ptr, Akk_ptr, g_last_ptr,
    v_ptr, beta_ptr, h0_ptr, ht_ptr, o_ptr,
    scale, T_pad,
    HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    ALIGNED_V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HV
    i_hv = i_bh % HV

    NT = T_pad // BT

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V

    # 六个 per-chunk tile，各两槽双缓冲（autolayout 下 smem 装得下）。
    w_smem = tle.gpu.alloc((2, BT, K), ws_ptr.dtype.element_ty)
    qg_smem = tle.gpu.alloc((2, BT, K), ws_ptr.dtype.element_ty)
    kg_smem = tle.gpu.alloc((2, BT, K), ws_ptr.dtype.element_ty)
    Aqk_smem = tle.gpu.alloc((2, BT, BT), Aqk_ptr.dtype.element_ty)
    Akk_smem = tle.gpu.alloc((2, BT, BT), Akk_ptr.dtype.element_ty)
    v_smem = tle.gpu.alloc((2, BT, BV), v_ptr.dtype.element_ty)

    # h 初始值（fp32 普通寄存器值，与 v1 同形态，无布局冲突）
    if USE_INITIAL_STATE:
        h0_base = h0_ptr + i_bh * K * V
        h_ptrs = h0_base + o_k[:, None] * V + o_v[None, :]
        if ALIGNED_V:
            h = tl.load(h_ptrs).to(tl.float32)
        else:
            h = tl.load(h_ptrs, mask=mask_v[None, :], other=0.0).to(tl.float32)
    else:
        h = tl.zeros((K, BV), tl.float32)

    _async_copy_chunk(
        ws_ptr, Aqk_ptr, Akk_ptr, v_ptr,
        i_b, i_hv, 0, T_pad,
        HV=HV, K=K, V=V, BT=BT, ALIGNED_V=ALIGNED_V,
        o_i=o_i, o_k=o_k, o_v=o_v, mask_v=mask_v,
        w_slot=w_smem.slot(0), qg_slot=qg_smem.slot(0), kg_slot=kg_smem.slot(0),
        Aqk_slot=Aqk_smem.slot(0), Akk_slot=Akk_smem.slot(0),
        v_slot=v_smem.slot(0),
    )
    tle.gpu.metax.gvm_arrive(6)
    tle.gpu.metax.barrier()

    for i_t in range(NT):
        cur = i_t % 2
        nxt = (i_t + 1) % 2

        if i_t + 1 < NT:
            _async_copy_chunk(
                ws_ptr, Aqk_ptr, Akk_ptr, v_ptr,
                i_b, i_hv, i_t + 1, T_pad,
                HV=HV, K=K, V=V, BT=BT, ALIGNED_V=ALIGNED_V,
                o_i=o_i, o_k=o_k, o_v=o_v, mask_v=mask_v,
                w_slot=w_smem.slot(nxt), qg_slot=qg_smem.slot(nxt), kg_slot=kg_smem.slot(nxt),
                Aqk_slot=Aqk_smem.slot(nxt), Akk_slot=Akk_smem.slot(nxt),
                v_slot=v_smem.slot(nxt),
            )
            tle.gpu.metax.gvm_arrive(6)

        b_w = tle.gpu.metax.local_load(w_smem.slot(cur), intrinsic=True, is_constant_offs=True)
        b_qg = tle.gpu.metax.local_load(qg_smem.slot(cur), intrinsic=True, is_constant_offs=True)
        b_kg = tle.gpu.metax.local_load(kg_smem.slot(cur), intrinsic=True, is_constant_offs=True)
        b_Aqk = tle.gpu.metax.local_load(Aqk_smem.slot(cur), intrinsic=True, is_constant_offs=True)
        b_Akk = tle.gpu.metax.local_load(Akk_smem.slot(cur), intrinsic=True, is_constant_offs=True)
        # g_last 直接全局 load（走 3-D smem 会漂，见 _async_copy_chunk）
        b_gk = tl.load(g_last_ptr + (i_b * NT + i_t) * HV * K + i_hv * K + o_k)
        b_v = tle.gpu.metax.local_load(v_smem.slot(cur), intrinsic=True, is_constant_offs=True)
        b_beta = tl.load(beta_ptr + (i_b * T_pad + i_t * BT) * HV + i_hv + o_i * HV)
        b_vb = (b_v.to(tl.float32)
                * tl.sigmoid(b_beta.to(tl.float32))[:, None]).to(v_ptr.dtype.element_ty)

        # v_new = A^-1(vb) - w@h。不要把 A^-1 提到 (vb - w@h) 外面：
        # A^-1vb - A^-1(w@h) != A^-1vb - w@h，而 h=0 时两式相同，所以写错时
        # 单 chunk 测试全绿、多 chunk 才崩。
        in_dt = ws_ptr.dtype.element_ty
        h_hi = h.to(in_dt)

        b_kh = tl.dot(b_w, h_hi).to(tl.float32)
        b_u = tl.dot(b_Akk, b_vb).to(tl.float32)
        b_v_new_f = b_u - b_kh
        b_v_new = b_v_new_f.to(in_dt)

        b_qh = tl.dot(b_qg, h_hi).to(tl.float32)
        b_o = scale * b_qh + tl.dot(b_Aqk, b_v_new.to(in_dt)).to(tl.float32)

        o_offs = ((i_b * T_pad + i_t * BT) * HV + i_hv) * V
        o_ptrs = o_ptr + o_offs + o_i[:, None] * (HV * V) + o_v[None, :]
        if ALIGNED_V:
            tl.store(o_ptrs, b_o.to(o_ptr.dtype.element_ty))
        else:
            tl.store(o_ptrs, b_o.to(o_ptr.dtype.element_ty), mask=mask_v[None, :])

        b_h_decay = h * tl.exp2(b_gk)[:, None]
        b_h_upd = tl.dot(tl.trans(b_kg), b_v_new.to(in_dt)).to(tl.float32)
        h = b_h_decay + b_h_upd

        tle.gpu.metax.iglp(config_2=1, config_5=1, config_6=2)

        if i_t + 1 < NT:
            tle.gpu.metax.barrier()

    if STORE_FINAL_STATE:
        ht_base = ht_ptr + i_bh * K * V
        ht_ptrs = ht_base + o_k[:, None] * V + o_v[None, :]
        if ALIGNED_V:
            tl.store(ht_ptrs, h.to(ht_ptr.dtype.element_ty))
        else:
            tl.store(ht_ptrs, h.to(ht_ptr.dtype.element_ty), mask=mask_v[None, :])


# =============================================================================
# Host wrapper
# =============================================================================


def _pad_to(x, target_T):
    """右补零到 target_T 行；输出必须新开张量（timestep 维不连续）。"""
    if x.shape[1] == target_T:
        return x
    out = torch.zeros(x.shape[0], target_T, *x.shape[2:], device=x.device, dtype=x.dtype)
    out[:, :x.shape[1]] = x
    return out


def chunk_kda_prefill_tle(
    q, k, v, g, beta,
    scale=None, lower_bound=-5.0,
    A_log=None, dt_bias=None,
    initial_state=None, output_final_state=False,
    chunk_size=16, BV=64, num_warps=4,
):
    assert ENABLED, "Requires a MetaX FLAGTREE_COMMON_IR build"
    B, T_orig, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    BT = chunk_size
    assert K == 128 and V == 128, "先只支持 K=V=128"
    assert q.dtype == torch.bfloat16, "先只支持 bf16"
    assert HV == H, "先只支持无 GVA（HV==H）"
    assert A_log is not None and dt_bias is not None
    assert beta.ndim == 3 and beta.shape == (B, T_orig, HV)

    if scale is None:
        scale = K ** -0.5

    NT = triton.cdiv(T_orig, BT)
    T_pad = NT * BT
    q_p, k_p, g_p = _pad_to(q, T_pad), _pad_to(k, T_pad), _pad_to(g, T_pad)
    v_p, beta_p = _pad_to(v, T_pad), _pad_to(beta, T_pad)

    Aqk = torch.empty(B, T_pad, HV, BT, device=q.device, dtype=q.dtype)
    Akk = torch.empty(B, T_pad, HV, BT, device=q.device, dtype=q.dtype)
    ws = torch.empty(B, T_pad, HV, 3 * K, device=q.device, dtype=q.dtype)
    g_last = torch.empty(B * NT, HV, K, device=q.device, dtype=torch.float32)

    grid1 = (NT, B * HV)
    _kda_fwd_intra_tle_kernel[grid1](
        q_p, k_p, g_p, beta_p,
        A_log, dt_bias,
        Aqk, Akk, ws, g_last,
        float(lower_bound), float(LOG2E), float(scale), float(L2NORM_EPS), float(GC_CLAMP),
        T_orig, T_pad,
        H=H, HV=HV, K=K, BT=BT,
        num_warps=num_warps, num_stages=2, pipeline="cpasync",
    )

    o_pad = torch.empty(B, T_pad, HV, V, device=q.device, dtype=v.dtype)
    NV = triton.cdiv(V, BV)
    if output_final_state:
        h_final = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32)
    else:
        h_final = o_pad
    h0_arg = initial_state if initial_state is not None else o_pad

    grid2 = (NV, B * HV)
    _kda_fwd_state_output_tle_kernel[grid2](
        ws, Aqk, Akk, g_last,
        v_p, beta_p, h0_arg, h_final, o_pad,
        float(scale), T_pad,
        HV=HV, K=K, V=V, BT=BT, BV=BV,
        ALIGNED_V=(V % BV == 0),
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=output_final_state,
        num_warps=num_warps, num_stages=2, pipeline="cpasync",
    )

    o = o_pad[:, :T_orig]
    if output_final_state:
        return o, h_final
    return o

# =============================================================================
# Torch fp32 参考（oracle）
# =============================================================================


def reference_chunk_kda(q, k, v, g, beta, A_log, dt_bias, lower_bound, scale=None,
                       initial_state=None):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    device = q.device

    qf = q.float() / (q.float().norm(dim=-1, keepdim=True) + L2NORM_EPS)
    kf = k.float() / (k.float().norm(dim=-1, keepdim=True) + L2NORM_EPS)

    A = torch.exp(A_log.float())
    x = g.float() + dt_bias.float().unsqueeze(0).unsqueeze(0)
    b_g = (lower_bound * LOG2E) * torch.sigmoid(A.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * x)
    b_beta = torch.sigmoid(beta.float())

    BT = 16
    NT = triton.cdiv(T, BT)
    T_pad = NT * BT
    if T_pad != T:
        def pad(x):
            if x.ndim == 4:
                y = torch.zeros(x.shape[0], T_pad, *x.shape[2:], device=device, dtype=x.dtype)
            else:
                y = torch.zeros(x.shape[0], T_pad, x.shape[2], device=device, dtype=x.dtype)
            y[:, :T] = x
            return y
        qf, kf = pad(qf), pad(kf)
        v_padded = pad(v.float())
        b_g = pad(b_g)
        b_beta = pad(b_beta)
    else:
        v_padded = v.float()

    o = torch.zeros(B, T_pad, HV, V, device=device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.clone().float()
    else:
        h = torch.zeros(B, HV, K, V, device=device, dtype=torch.float32)

    for i_t in range(NT):
        t0, t1 = i_t * BT, (i_t + 1) * BT
        gc = torch.cumsum(b_g[:, t0:t1], dim=1)
        gc = torch.clamp(gc, min=GC_CLAMP)
        gq = torch.exp2(gc)
        gk = torch.exp2(-gc)

        for hv in range(HV):
            h_i = hv // (HV // H)
            q_i = qf[:, t0:t1, h_i]
            k_i = kf[:, t0:t1, h_i]
            v_i = v_padded[:, t0:t1, hv]
            beta_i = b_beta[:, t0:t1, hv]
            gc_i = gc[:, :, hv]
            gq_i = gq[:, :, hv]
            gk_i = gk[:, :, hv]

            qg_i = q_i * gq_i
            kg_i = k_i * gk_i
            kk_i = k_i * gq_i
            Aqk_i = torch.matmul(qg_i, kg_i.transpose(-1, -2)) * scale
            Aqk_i = torch.tril(Aqk_i, diagonal=0)
            Akk_i = torch.matmul(kk_i, kg_i.transpose(-1, -2)) * beta_i[:, :, None]
            Akk_i = torch.tril(Akk_i, diagonal=-1)

            I = torch.eye(BT, device=device, dtype=torch.float32).expand(B, -1, -1)
            Akk_inv_i = torch.linalg.inv(I + Akk_i)

            w_i = torch.matmul(Akk_inv_i, k_i * beta_i[:, :, None] * gq_i)
            u_i = torch.matmul(Akk_inv_i, v_i * beta_i[:, :, None])

            gc_last = gc_i[:, -1]
            kg_last_i = k_i * torch.exp2(gc_last[:, None, :] - gc_i)

            h_hv = h[:, hv]
            v_new = u_i - torch.matmul(w_i, h_hv)
            o[:, t0:t1, hv] = scale * torch.matmul(qg_i, h_hv) + torch.matmul(Aqk_i, v_new)
            h[:, hv] = h_hv * torch.exp2(gc_last)[:, :, None] + torch.matmul(
                kg_last_i.transpose(-1, -2), v_new
            )

    o = o[:, :T]
    return o, h



# =============================================================================
# Fast fp32 reference (parameterised BT / clamp) — what the baselines are
# checked against.  Use it only after test_reference_equivalence has locked it
# to the token-loop oracle above.
# =============================================================================


def reference_chunk_kda_fast(x, scale, BT=64, clamp=GC_CLAMP):
    """fp32 参考；BT 必须与待测实现的 chunk 大小一致（见文件头的精度口径）。

    clamp=None 给出"无 gate 下界保护"的变体（flaggems 的语义）；TLE/mcoplib
    路径用 clamp=GC_CLAMP。两者是两个不同函数，只能各自与自己的参考比。
    """
    q, k, v = x["q"], x["k"], x["v"]
    g, beta = x["g"], x["beta"]
    A_log, dt_bias = x["A_log"], x["dt_bias"]
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert T % BT == 0, f"参考实现要求 T%{BT}==0，当前 T={T}"

    qf = F.normalize(q.float(), dim=-1).permute(0, 2, 1, 3)   # [B,H,T,K]
    kf = F.normalize(k.float(), dim=-1).permute(0, 2, 1, 3)
    vf = v.float().permute(0, 2, 1, 3)                        # [B,H,T,V]
    A = torch.exp(A_log.float())
    xx = g.float().permute(0, 2, 1, 3) + dt_bias.float()[None, :, None, :]  # [B,H,T,K]
    bg = (LOWER_BOUND * LOG2E) * torch.sigmoid(A[None, :, None, None] * xx)
    bb = torch.sigmoid(beta.float()).permute(0, 2, 1)          # [B,H,T]

    o = torch.zeros(B, H, T, V, device=q.device)
    h = torch.zeros(B, H, K, V, device=q.device)
    for i in range(T // BT):
        s = slice(i * BT, (i + 1) * BT)
        gc = bg[:, :, s].cumsum(-2)
        if clamp is not None:
            gc = gc.clamp_min(clamp)                          # [B,H,BT,K]
        gq, gk2 = torch.exp2(gc), torch.exp2(-gc)
        qg = qf[:, :, s] * gq
        kg_ = kf[:, :, s] * gk2
        kk = kf[:, :, s] * gq

        Aqk = torch.tril((qg @ kg_.transpose(-1, -2)) * scale)
        M = torch.tril((kk @ kg_.transpose(-1, -2)) * bb[:, :, s, None], -1)
        eye = torch.eye(BT, device=q.device)
        Ai = torch.linalg.inv(eye + M)

        w = Ai @ (kf[:, :, s] * bb[:, :, s, None] * gq)
        u = Ai @ (vf[:, :, s] * bb[:, :, s, None])

        gl = gc[:, :, -1:]                                     # [B,H,1,K]
        kgl = kf[:, :, s] * torch.exp2(gl - gc)

        v_new = u - w @ h
        o[:, :, s] = scale * (qg @ h) + Aqk @ v_new
        h = h * torch.exp2(gl).squeeze(2).unsqueeze(-1) + kgl.transpose(-1, -2) @ v_new

    return o.permute(0, 2, 1, 3).contiguous(), h


# =============================================================================
# Baselines: flaggems and mcoplib, loaded from their source trees at runtime
# =============================================================================


def _load_mcoplib():
    """mcoplib 三段：intra / inter_solve_fused / fwd_h（另有 o 段见 _load_fg_o）。

    mcoplib 的 __init__ 会往 stdout print 一行 version 文件 WARNING，这里屏蔽掉。
    """
    try:
        for p in ("/home/flagos-ly/mcoplib",):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        with contextlib.redirect_stdout(io.StringIO()):
            from mcoplib.triton_sglang_chunk_intra_token_parallel import (
                chunk_kda_fwd_intra_token_parallel as mc_intra)
            from mcoplib.triton_chunk_kda_fwd_kernel_inter_solve_fused import (
                chunk_kda_fwd_kernel_inter_solve_fused as mc_inter)
            from mcoplib.triton_sglang_chunk_delta_h import (
                chunk_gated_delta_rule_fwd_h as mc_fwd_h)
        return dict(intra=mc_intra, inter=mc_inter, fwd_h=mc_fwd_h)
    except Exception:
        traceback.print_exc()
        return None


def _load_fg_o():
    """只取 flaggems 的 o kernel（chunk_gla_fwd_o_gk），给 mcoplib 收尾用。

    放在 ops/FLA/chunk_gla.py，不在 chunk_kda.py。包 __init__ 会 import
    flag_gems，这里把 FlagGems 源码树挂进 sys.path。
    """
    for p in ("/home/flagos-ly/FlagGems-vllm/src", "/home/flagos-ly/FlagGems/src"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    # TLE 变体在本构建上会抛 "backend 'metax' does not support TLE primitive
    # 'load'"，切回其 plain-triton kernel。
    os.environ.setdefault("FLA_GLA_TLE", "0")
    try:
        from flaggems_vllm.ops.FLA.chunk_gla import chunk_gla_fwd_o_gk
        return chunk_gla_fwd_o_gk
    except Exception:
        traceback.print_exc()
        return None


# -----------------------------------------------------------------------------
# flaggems-vllm 整条流水线（软依赖）：运行时读上游 ops/FLA/chunk_kda.py，在内存里
# 打 MACA patch。为什么必须 patch、为什么 patch 后还要塞 linecache，见文件头。
#
# 三个 patch 点是 triangular inverse 的并行前缀（6 行块），上游若改过形状就会
# 匹配不到 —— 那种情况下拒绝编译，而不是拿未 patch 的源码去撞编译错误。
# -----------------------------------------------------------------------------

FG_VLLM_SRC = "/home/flagos-ly/FlagGems-vllm/src"
FG_KDA_REL = "flaggems_vllm/ops/FLA/chunk_kda.py"

_FP16_DOT_RE = re.compile(
    r"^(?P<i>[ \t]*)b_L2 = tl\.dot\(b_L, b_L, out_dtype=tl\.float16\)\n"
    r"(?P=i)b_Ai = b_Ai \+ tl\.dot\(b_Ai, b_L2, out_dtype=tl\.float16\)\n"
    r"(?P=i)b_L4 = tl\.dot\(b_L2, b_L2, out_dtype=tl\.float16\)\n"
    r"(?P=i)b_Ai = b_Ai \+ tl\.dot\(b_Ai, b_L4, out_dtype=tl\.float16\)\n"
    r"(?P=i)b_L8 = tl\.dot\(b_L4, b_L4, out_dtype=tl\.float16\)\n"
    r"(?P=i)b_Ai = b_Ai \+ tl\.dot\(b_Ai, b_L8, out_dtype=tl\.float16\)",
    re.MULTILINE,
)

_FP16_DOT_NEW = """{i}b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float32).to(tl.float16)
{i}b_Ai = (b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float32)).to(tl.float16)
{i}b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float32).to(tl.float16)
{i}b_Ai = (b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float32)).to(tl.float16)
{i}b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float32).to(tl.float16)
{i}b_Ai = (b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float32)).to(tl.float16)"""


def _patch_fg_source(src):
    """把上游 chunk_kda.py 源码里的 fp16-accum 前缀换成 fp32-accum。

    tri-inverse 块在不同 kernel 里缩进不同（4/8 空格），故按行首缩进回填。
    返回 (patched_src, n_sites)。n_sites=0 表示上游已变更，调用方据此拒绝编译。
    """
    patched, n = _FP16_DOT_RE.subn(
        lambda m: _FP16_DOT_NEW.format(i=m.group("i")), src)
    return patched, n


def _load_fg_full():
    """加载（内存 patch 后的）flaggems 全流水线 chunk_kda；失败返回 None。

    必须在 import flaggems_vllm 之前设置 FLAGGEMS_CHUNK_KDA_BACKEND=triton，
    否则 _tle_primitives_available() 会选 TLE 路径并在本后端抛错。
    """
    os.environ.setdefault("FLAGGEMS_CHUNK_KDA_BACKEND", "triton")
    if not os.path.isdir(FG_VLLM_SRC):
        print(f"[load] fg-full : 缺少 FlagGems-vllm 源码树 {FG_VLLM_SRC} -> 该列 n/a")
        return None
    if FG_VLLM_SRC not in sys.path:
        sys.path.insert(0, FG_VLLM_SRC)

    path = os.path.join(FG_VLLM_SRC, FG_KDA_REL)
    if not os.path.isfile(path):
        print(f"[load] fg-full : 缺少 {path} -> 该列 n/a")
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        patched, n_sites = _patch_fg_source(src)
        if n_sites == 0:
            print(f"[load] fg-full : 上游 {FG_KDA_REL} 未找到 fp16-accum patch 点"
                  "（上游可能已改），拒绝在未 patch 状态下编译 -> 该列 n/a")
            return None
        residual = len(re.findall(r"out_dtype=tl\.float16", patched))
        if residual:
            print(f"[load] fg-full : patch 后仍残留 {residual} 处 out_dtype=tl.float16"
                  "（上游改动超出预期），未 patch 的 kernel 会编译失败 -> 该列 n/a")
            return None

        spec = importlib.util.spec_from_loader("_fg_chunk_kda_patched", loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = path
        # 关键：Triton 的 JIT 在编译时用 inspect.getsource(fn) 按 co_filename
        # **重新从磁盘**读源码，exec 一个内存 patch 版本不会被它看到（实测会继续
        # 编未 patch 的 kernel 然后死于 Unsupported mma type）。
        # 因此把 patch 后的源码塞进 linecache（mtime 置 None 时 checkcache 会跳过
        # 条目，不会被磁盘版本覆盖），inspect/inspect.getsourcelines 便会取到
        # patched 文本。全程不落盘。
        patched_lines = [ln + "\n" for ln in patched.splitlines(keepends=False)]
        linecache.cache[path] = (len(patched), None, patched_lines, path)
        # 编译期/加载期 noise（WARNING、可能的 MLIR dump）不污染 bench 输出
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(patched, path, "exec"), mod.__dict__)
        print(f"[load] fg-full : patch {n_sites} 处 fp16-accum -> fp32-accum")
        return mod.chunk_kda
    except Exception:
        traceback.print_exc()
        return None



# =============================================================================
# 三方流水线：TLE / mcoplib / flaggems，同一份输入，端到端出 o
# =============================================================================


def tle_pipeline(x, scale):
    """TLE 端到端：wrapper + 两个 kernel，无额外前处理。"""
    return chunk_kda_prefill_tle(
        x["q"], x["k"], x["v"], x["g"], x["beta"],
        scale=scale, lower_bound=LOWER_BOUND,
        A_log=x["A_log"], dt_bias=x["dt_bias"],
        output_final_state=False,
    )


def mcoplib_pipeline(mc, fg_o, x, scale):
    """intra -> inter_solve_fused(fuse_recompute) -> fwd_h -> flaggems o。

    前处理用 torch（mcoplib 未暴露融合前处理入口），对该基线偏保守。
    """
    q, k, v, g, beta = x["q"], x["k"], x["v"], x["g"], x["beta"]
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT, BC = 64, 16
    assert T % BT == 0, f"mcoplib 路径要求 T%{BT}==0，当前 T={T}"

    # ---- torch 前处理（与 TLE kernel 内语义一致）----
    qn = F.normalize(q.float(), dim=-1).to(q.dtype).contiguous()
    kn = F.normalize(k.float(), dim=-1).to(k.dtype).contiguous()
    A = torch.exp(x["A_log"].float())
    xx = g.float() + x["dt_bias"].float()
    bg = (LOWER_BOUND * LOG2E) * torch.sigmoid(A[None, None, :, None] * xx)
    NT = T // BT
    gc = bg.view(B, NT, BT, H, K).cumsum(2).clamp_min(GC_CLAMP)
    gc = gc.view(B, T, H, K).contiguous()                      # fp32, log2 空间
    beta_s = torch.sigmoid(beta.float()).contiguous()

    # ---- intra：出对角块 Aqk / Akkd ----
    Aqk = torch.zeros(B, T, H, BT, device=q.device, dtype=q.dtype)
    Akkd = torch.empty(B, T, H, BC, device=q.device, dtype=torch.float32)
    mc["intra"](qn, kn, gc, beta_s, Aqk, Akkd, scale,
                None, chunk_size=BT, sub_chunk_size=BC)

    # ---- inter_solve_fused：非对角块 + 求逆 + w/u/kg 重算 ----
    w = torch.empty_like(kn)
    u = torch.empty_like(v)
    kg = torch.empty_like(kn)
    Akk = torch.empty(B, T, H, BT, device=q.device, dtype=q.dtype)
    mc["inter"][(NT, B * H)](
        q=qn, k=kn, g=gc, beta=beta_s,
        Aqk=Aqk, Akkd=Akkd, Akk=Akk, scale=scale,
        v_in=v, w_out=w, u_out=u, kg_out=kg,
        cu_seqlens=None, chunk_indices=None,
        T=T, H=H, K=K, BT=BT, BC=BC, V=V,
        USE_SAFE_GATE=False, FUSE_DIAGONAL=True, FUSE_RECOMPUTE=True,
    )

    # ---- fwd_h：跨 chunk 状态递推（in-place 更新 h0 槽位）----
    h0 = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    idx = torch.arange(B, device=q.device, dtype=torch.int32)
    h, v_new = mc["fwd_h"](
        k=kg, w=w, u=u, gk=gc,
        initial_state=h0, initial_state_indices=idx,
        cu_seqlens=None, chunk_indices=None, use_exp2=True,
    )

    # ---- o（mcoplib 无独立 o kernel，借用 flaggems 同源实现）----
    # fwd_h 返回 h = [B, NT, H, V, K]（state_v_first），o kernel 内部按此布局取
    o = fg_o(q=qn, v=v_new, g=gc, A=Aqk, h=h, scale=scale,
             chunk_size=BT, cu_seqlens=None, state_v_first=True)
    return o


def fg_pipeline(fg_full, x, scale):
    """flaggems-vllm 全流水线（BT=16, safe_gate=True），与 TLE 同构。

    safe_gate=True 让 cuDNN/gate 的激活与 cumsum 融在 kernel 内；注意 fg 的
    safe_gate 不含 clamp_min(-16)，语义见文件头的精度口径。返回 o（[B,T,H,V] bf16）。
    """
    with torch.no_grad():
        return fg_full(
            x["q"], x["k"], x["v"], x["g"], x["beta"],
            scale=scale, lower_bound=LOWER_BOUND,
            A_log=x["A_log"], dt_bias=x["dt_bias"],
            chunk_size=16, safe_gate=True, output_final_state=False,
        )[0]


# =============================================================================
# 测试：输入、参考互锁、形状扫描
# =============================================================================


def _make_inputs(B, T, H, K, V, seed=0, dtype=torch.bfloat16):
    """精度自测用的输入（q/k/v 幅度 0.1）。

    与 _make_ab_inputs 分开是有意的：0.1 上 TLE 与参考的相对差更容易看清，
    而三方 A/B 用 0.5 才能把各实现的量化误差分出层次。
    """
    torch.manual_seed(seed)
    q = (torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1)
    k = (torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1)
    v = (torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1)
    g = (torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.5)
    beta = (torch.randn(B, T, H, device="cuda", dtype=dtype) * 0.5)
    A_log = (torch.randn(H, device="cuda", dtype=torch.float32) * 0.1)
    dt_bias = (torch.randn(H, K, device="cuda", dtype=torch.float32) * 0.1)
    return q, k, v, g, beta, A_log, dt_bias


def _make_ab_inputs(B, T, H, K=128, V=128, seed=0):
    """三方 A/B 共用的一份输入（q/k/v 幅度 0.5）。"""
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16) * 0.5
    g = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
    beta = torch.randn(B, T, H, device=dev, dtype=torch.bfloat16) * 0.5
    A_log = torch.randn(H, device=dev, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device=dev, dtype=torch.float32) * 0.1
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias)


def _clone_inputs(x):
    return {k: v.clone() for k, v in x.items()}


def test_reference_equivalence():
    """两个 fp32 参考互锁。

    reference_chunk_kda_fast 是后面所有精度判定的基准，所以先证明它和逐 token
    的 oracle 在 fp32 舍入内一致。三个小形状足够：它俩的差异只跟 BT / clamp
    的语义有关，跟 T 无关。
    """
    print("== 两个 fp32 参考互锁（fast vs token-loop oracle）==")
    ok = True
    for (B, T, H) in [(1, 16, 2), (1, 48, 2), (2, 64, 2)]:
        q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H, 128, 128, seed=11)
        o_slow, h_slow = reference_chunk_kda(q, k, v, g, beta, A_log, dt_bias,
                                             LOWER_BOUND)
        x = dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias)
        o_fast, h_fast = reference_chunk_kda_fast(x, 128 ** -0.5, BT=16,
                                                  clamp=GC_CLAMP)
        d_o = (o_slow.float() - o_fast).abs().max().item()
        d_h = (h_slow - h_fast).abs().max().item()
        mag = o_slow.abs().max().item()
        h_mag = h_slow.abs().max().item()
        rel_o, rel_h = d_o / mag, d_h / h_mag
        # 两条不同的 fp32 归约顺序，千分之一量级的相对差是正常的；
        # 大于 1e-2 就说明某个参考写错了。
        good = rel_o < 1e-2 and rel_h < 1e-2
        ok &= good
        print(f"  B{B}T{T}H{H}: |o|max={mag:.4g} rel_o={rel_o:.1e} "
              f"|h|max={h_mag:.4g} rel_h={rel_h:.1e} {'ok' if good else 'FAIL'}")
    print(f"test_reference_equivalence: {'passed' if ok else 'FAILED'}")
    return ok


def test_correctness():
    """TLE vs fp32 参考，含 initial_state 与 T 非 BT 整数倍的边界。"""
    print()
    print("| shape | max_abs_o | max_abs_h | h_max(ref) | allclose |")
    print("| --- | --- | --- | --- | --- |")
    ok = True
    for (B, T, H, K, V) in [
        (1, 16, 2, 128, 128),   # NT=1，单 chunk
        (1, 32, 2, 128, 128),   # NT=2：误差是逐 chunk 累积还是一步到位
        (1, 48, 2, 128, 128),   # NT=3
        (2, 64, 2, 128, 128),   # NT=4，多 batch
        (2, 63, 2, 128, 128),   # T 不是 BT 整数倍
        (1, 17, 1, 128, 128),   # H=1，T=BT+1
    ]:
        # T16/T32 加随机初始态：单 chunk 下让 decay 第一次作用于非零 h。
        # o 若仍对 -> h0 载入 / w / v_new 全对；h 若崩 -> decay 或 kg_last。
        use_h0 = T in (16, 32)
        if use_h0:
            torch.manual_seed(1234)
            h0 = (torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.05)
        else:
            h0 = None
        q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H, K, V)
        o_tle, h_tle = chunk_kda_prefill_tle(
            q, k, v, g, beta, lower_bound=LOWER_BOUND,
            A_log=A_log, dt_bias=dt_bias, output_final_state=True,
            initial_state=h0,
        )
        o_ref, h_ref = reference_chunk_kda(q, k, v, g, beta, A_log, dt_bias,
                                           LOWER_BOUND, initial_state=h0)
        diff_o = (o_tle.float() - o_ref).abs().max().item()
        diff_h = (h_tle - h_ref).abs().max().item()
        h_mag = h_ref.abs().max().item()
        good = (torch.allclose(o_tle.float(), o_ref, atol=1e-1, rtol=1e-1)
                and torch.allclose(h_tle, h_ref, atol=1e-1, rtol=1e-1))
        ok &= good
        print(f"| B{B}T{T}H{H}K{K}V{V} | {diff_o:.4g} | {diff_h:.4g} "
              f"| {h_mag:.4g} | {good} |")
    print(f"test_correctness: {'all passed' if ok else 'FAILED'}")
    return ok


def test_ab_accuracy(mc, fg_o, fg_full):
    """三方各自对比同 chunk / 同 clamp 的 fp32 参考（见文件头精度口径）。"""
    print()
    print("== 三方精度（各自 vs 同 chunk / 同 clamp 的 fp32 参考）==")
    print("| shape | ref_max(BT16) | tle_abs | ref_max(BT64) | mc_abs "
          "| ref_max(no-clamp) | fg_abs |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for (B, T, H) in [(1, 64, 2), (2, 128, 2), (1, 256, 4), (1, 512, 4)]:
        K = V = 128
        scale = K ** -0.5
        x = _make_ab_inputs(B, T, H, K, V, seed=7)
        o_ref, _ = reference_chunk_kda_fast(x, scale, BT=16)                # TLE
        o_ref64, _ = reference_chunk_kda_fast(x, scale, BT=64)              # mcoplib
        o_ref_nc, _ = reference_chunk_kda_fast(x, scale, BT=16, clamp=None)  # fg

        o = tle_pipeline(_clone_inputs(x), scale)
        if torch.isnan(o.float()).any():
            nf = int(o.isnan().sum())
            bi = tuple(torch.nonzero(o.isnan())[0].tolist())
            am = o.float().abs().nan_to_num(0.0).max().item()
            print(f"  [tle-nan] B{B}T{T}H{H}: o 含 {nf} 个 nan，"
                  f"首位置(b,t,h,v)={bi}，非nan abs_max={am}")
        d_tle = (o.float() - o_ref).abs().max().item()

        d_mc = float("nan")
        if mc is not None and fg_o is not None:
            o = mcoplib_pipeline(mc, fg_o, _clone_inputs(x), scale)
            d_mc = (o.float() - o_ref64).abs().max().item()

        d_fg = float("nan")
        if fg_full is not None:
            try:
                o = fg_pipeline(fg_full, _clone_inputs(x), scale)
                d_fg = (o.float() - o_ref_nc).abs().max().item()
            except Exception:
                traceback.print_exc()
                print(f"  [fg-fail] B{B}T{T}H{H}: 该行 fg 列记为 n/a")

        print(f"| B{B}T{T}H{H} | {o_ref.abs().max().item():.3g} "
              f"| {d_tle:.3g} "
              f"| {o_ref64.abs().max().item():.3g} | {d_mc:.3g} "
              f"| {o_ref_nc.abs().max().item():.3g} | {d_fg:.3g} |")
    print("  注：tle/mc 对带 clamp 参考，fg 对无 clamp 参考；三方输出不互比"
          "（见文件头精度口径）。")


BENCH_SHAPES = [(1, 512, 16), (1, 2048, 16), (4, 2048, 16), (1, 8192, 16)]


def _pick_shapes(only_t):
    """按 T 过滤 BENCH_SHAPES；过滤空了就说清可用值，别打一张空表。"""
    if not only_t:
        return list(BENCH_SHAPES)
    picked = [s for s in BENCH_SHAPES if s[1] in only_t]
    if not picked:
        avail = sorted({s[1] for s in BENCH_SHAPES})
        print(f"\n--shapes {','.join(map(str, only_t))} 没有匹配：可用 T 值 {avail}")
    return picked


def bench(only_t=None):
    """TLE 端到端延迟（不带基线）。"""
    print()
    print("| shape | TLE ms |")
    print("| --- | --- |")
    shapes = _pick_shapes(only_t)
    for (B, T, H) in shapes:
        K, V = 128, 128
        scale = K ** -0.5
        q, k, v, g, beta, A_log, dt_bias = _make_inputs(B, T, H, K, V, seed=7)
        x = dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias)
        fn = lambda: tle_pipeline(_clone_inputs(x), scale)  # noqa: E731
        fn()
        ms = triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")
        print(f"| B{B}T{T}H{H}K{K}V{V} | {ms:.4f} |")


def _bench_interleaved(fns, rounds=3, warmup=10, rep=50):
    """交替测量：每轮把所有候选各测一次，跨轮取中位数。

    顺序测量（每个候选连着 do_bench）会吃到时钟阶梯和顺序偏置；本卡单次
    do_bench 摆动可达 ±10%，比要分辨的差异还大。
    """
    acc = [[] for _ in fns]
    for _ in range(rounds):
        for i, fn in enumerate(fns):
            acc[i].append(
                triton.testing.do_bench(fn, warmup=warmup, rep=rep,
                                        return_mode="median"))
    return [sorted(a)[len(a) // 2] for a in acc]


def bench_ab(mc, fg_o, fg_full, only_t=None, rounds=3):
    """三方 speedup：flaggems 与 mcoplib 各自对 TLE。"""
    shapes = _pick_shapes(only_t)

    print()
    print(f"{'shape':>12} {'TLE ms':>9} {'flaggems ms':>12} {'gems/TLE':>9} "
          f"{'mcop ms':>9} {'mcp/TLE':>8}")
    print("-" * 64)
    for (B, T, H) in shapes:
        K, V = 128, 128
        scale = K ** -0.5
        x = _make_ab_inputs(B, T, H, K, V, seed=7)

        fns = [lambda: tle_pipeline(_clone_inputs(x), scale)]
        labels = ["tle"]
        if fg_full is not None:
            def fn_fg(x=x, scale=scale, fg_full=fg_full):
                return fg_pipeline(fg_full, _clone_inputs(x), scale)
            fns.append(fn_fg)
            labels.append("fg")
        if mc is not None and fg_o is not None:
            def fn_mc(x=x, scale=scale, mc=mc, fg_o=fg_o):
                return mcoplib_pipeline(mc, fg_o, _clone_inputs(x), scale)
            fns.append(fn_mc)
            labels.append("mc")

        try:
            times = _bench_interleaved(fns, rounds=rounds)
        except Exception:
            traceback.print_exc()
            print(f"  [bench-fail] B{B}T{T}H{H}: 跳过该行")
            continue
        ms = dict(zip(labels, times))

        def fmt(ratio):
            return f"{ratio:.2f}" if ratio == ratio else "n/a"

        t_tle = ms["tle"]
        s_g = ms["fg"] / t_tle if "fg" in ms else float("nan")
        s_m = ms["mc"] / t_tle if "mc" in ms else float("nan")
        tag = ("g" if s_g > 1 else "") + ("m" if s_m > 1 else "")
        print(f"{f'B{B}T{T}H{H}':>12} {t_tle:9.4f} "
              f"{ms.get('fg', float('nan')):12.4f} {fmt(s_g):>9} "
              f"{ms.get('mc', float('nan')):9.4f} {fmt(s_m):>8}  {tag or '-'}")
    print(f"\n交替测量 {rounds} 轮取中位数；两列 speedup 都 >1 才算该形状收敛。")
    print("mcoplib 列补了 torch 前处理 + flaggems o kernel，fg 列打了 fp32-accum "
          "patch —— 两处都只会让基线更慢。")


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="chunk_kda_prefill: 精度 + TLE 延迟 + 三方 A/B")
    parser.add_argument("--accuracy", action="store_true",
                        help="只跑精度（参考互锁 + 形状扫描）")
    parser.add_argument("--perf", action="store_true", help="只跑 TLE 延迟")
    parser.add_argument("--ab", action="store_true",
                        help="只跑三方 speedup（flaggems、mcoplib vs TLE）")
    parser.add_argument("--shapes", default=None,
                        help="逗号分隔的 T 值，用于 --perf / --ab"
                             "（可选 512,2048,8192）")
    parser.add_argument("--rounds", type=int, default=3,
                        help="--ab 交替测量轮数（默认 3）")
    args = parser.parse_args()
    only_t = [int(s) for s in args.shapes.split(",")] if args.shapes else None

    run_all = not (args.accuracy or args.perf or args.ab)
    if run_all or args.accuracy:
        test_reference_equivalence()
        test_correctness()

    if run_all or args.ab:
        # 先打印再加载：_load_fg_full 自己会往 stdout 写 patch 日志，
        # 放在 [load] 行之后会串序。
        print()
        print("[load] mcoplib ...")
        mc = _load_mcoplib()
        print(f"[load] mcoplib : {'ok' if mc else 'FAILED -> 该列 n/a'}")
        print("[load] fg-o ...")
        fg_o = _load_fg_o()
        print(f"[load] fg-o    : {'ok' if fg_o else 'FAILED -> mcoplib 列 n/a'}")
        print("[load] fg-full ...")
        fg_full = _load_fg_full()
        print(f"[load] fg-full : "
              f"{'ok (MACA-patched fp32-accum)' if fg_full else 'FAILED -> 该列 n/a'}")
        bench_ab(mc, fg_o, fg_full, only_t, rounds=args.rounds)
        test_ab_accuracy(mc, fg_o, fg_full)

    if args.perf:
        # 单独 --perf 才打这张表：--ab 已经报过同样四个形状的 TLE 延迟。
        bench(only_t)
