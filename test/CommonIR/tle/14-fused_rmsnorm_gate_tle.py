"""fused RMSNorm + sigmoid gate kernel：TLE / CommonIR 版。

    算子名称   : fused_rmsnorm_gate
    Kernel 名称: fused RMSNorm + sigmoid gate kernel
    算子接口名 : FusedRMSNormGated
    模型调用点 : GLM-5-Next `Glm5NextLinearAttention.o_norm`（线性注意力输出逐 head 归一化）
    参考实现   : glm-5-next-sgl.patch: kernels/ops/attention/fla/fused_norm_gate.py

算子语义（GLM-5-Next 线性注意力的 o_norm，逐 head 归一化 + gate）：
    out = (x * rsqrt(mean(x^2) + eps)) * weight * sigmoid(gate)   # activation="sigmoid"
    out = (x * rsqrt(mean(x^2) + eps)) * weight * silu(gate)      # activation="swish" / "silu"
x/gate 形状 [M, D]（M = tokens * value heads，D = head dim，典型 128），bf16/fp16。
x/gate 必须 contiguous；activation 只接受 sigmoid / swish / silu，未知取值报错。

对照实现与基线口径：
* gems : flaggems_vllm 库 2-launch 融合组合基线——add_rms_norm（KernelGen 生成，
         rmsnorm(x1+x2)·w，N≤4096 整行单遍 / 超出走两遍 streaming + 逆序写）
         + gate_sigmoid_mul（Kimi-K3 output gate 专用单算子，FP32 sigmoid 中间
         精度，广播 stride），共 2 kernel。这是库内能做到的最强形态：add 与
         sigmoid·mul 均已各自融合，只差最后一步（add_rms_norm 的输出写回显存
         再被 gate 读一次 + 两次 launch）。
         语义对齐说明：本算子无 residual，add_rms_norm 的 x2 喂缓存的 zeros
         占位（x+0 ≡ x，数学等价，不进 bench 计时），基线因此多读一份全零
         张量，带宽略吃亏于它——口径已在此注明。
         swish 档无对应实现（gate_sigmoid_mul 仅支持 sigmoid），本 bench 全档
         为 sigmoid 不受影响。kernel 代码完整嵌入本文件（仅 import flaggems_vllm
         的 runtime/utils 基础件），加载失败打印堆栈，gems 列显示 n/a。
         官方背书链：库内 baseline/gate_sigmoid_mul.py pin 的 native 是
         vllm.models.kimi_k3.nvidia.mla._gate_sigmoid_mul（带 commit/sha256
         溯源）——vLLM 官方 Kimi-K3 gate 实现 → flaggems_vllm 移植版 →
         本文件嵌入版；gate 后乘同族结构，语义与本算子对口。
         注意：本环境库内无 gate_sigmoid_mul 的 tuned config（调参 yaml
         缺失，get_tuned_config 返回空），BLOCK_SIZE/warps 取保守定值
         1024/4——基线未调优，spd_gems 数字相对保守。
* mcop : mcoplib 库 ops 组合基线——pybind 导出的 vLLM 血统裸 rms_norm
         （7 参签名 rms_norm(result, input, weight, eps, residual, inv_rms,
         fuse_params)，后三参可 None/False）+ ATen sigmoid + ATen mul，共 3 kernel。
         库内 sigmoid 无独立 C++ 单算子（只内联在 recurrent kernel /
         gate 预计算 kernel 里），走 ATen 通用实现。

与朴素单行 Triton 写法的差异（TLE 改写点）：
1. 窄 D / 大 M 时一个 program 处理 BM 行，x/gate 整块 load，rstd 沿行内归约，
   输出整块 store——prefill 档 1.6~3.4x 的来源。
2. x tile 显式钉布局（BlockEncoding，D 维连续）——与 sigmoid_gating TLE 版的
   h tile 同一教训：不钉则 layout_tune 由 sum(axis=1) 反推，整行摊到所有 lane，
   跨 lane 归约代价极高。只在多行块路径上钉。
3. weight 每行相同，只读一次 [BD] 向量，广播乘。
4. 按 D/M 分三路（由 MC550 64KB smem 硬限倒推，详见下方路径表）。

用法：
    python 14-fused_rmsnorm_gate_tle.py               # 精度
    python 14-fused_rmsnorm_gate_tle.py --benchmark   # 延迟（flaggems_vllm 2-launch + mcoplib 组合基线）

============================ 路径分发（wrapper） ============================

| 条件                    | 路径         | 形态                                    |
| ----------------------- | ------------ | --------------------------------------- |
| D > 4096                | loop kernel  | BD=1024 分趟循环，num_stages=1          |
| D ≥ 512 或 M ≤ 1024     | 单行 kernel  | 每 program 一行，grid=M，1-D tile       |
| 其余（窄 D、大 M）      | 多行块 kernel| BM 行/program（8/16 自适应），钉布局    |

为什么单行路径不能用多行块：D ≥ 512 时 [BM, BD] 的二维 tile 在 C500 上被
layout_tune 摊成跨 lane 归约，且寄存器随 BM 线性涨；退化成 1-D [BD] 后
（每行独立归约，行内连续访存）实测在 D512/D4096/M 小时全面领先，
D4096 档 116→97us，D512 档 34→28.6us。窄 D（128/256）大 M 时反过来：
二维块摊薄了 CTA 数，D128 档 21.9→18.6us。

============================ 版本流水 ============================

| 版本 | 改动 | 结果 |
| ---- | ---- | ---- |
| v1→v2 | 修三处 bug（weight 占位顺序 / D96 列掩码 / loop 超 smem） | 精度 18/18 |
| v2.1~v2.3 | wide-D4096 三连修（loop→BM=1→放掉钉布局） | 5978→111.9us |
| v2.4 | warps 4→8 | 反慢（137.6us），回退 |
| v2.5 | 定稿 4 warps | wide 111.9us（0.89x） |
| v2.6 | BM=2 | 打平不赢，保留（无害） |
| v2.7 | evict_first 缓存提示 | 负优化，撤销 |
| v3 | block_ptr 重写访存 | 全线小输，回退 |
| v3.1 | = v2.8 代码 | 上一版定稿 |
| v4 | 新增 1-D 单行 kernel（D≥512 / M≤1024 走它）；wrapper 去掉热路径上的
       BlockEncoding 构造、next_power_of_2 与 weight 占位 view | wide 116→97us、
       D512 34→28.6us、decode 15.5→14.2us |
| v4.1 | gems 基线列转正：sigmoid 改走 flag_gems metax 后端实现（直接 import，
       不做全局 enable——enable 会劫持 ATen 污染 aten 列，实测 6-op 从 ~60us
       炸到 ~900us），基线=库内 3-op 组合 |
| v4.2 | 基线列调整：删 ATen 列，新增 mcop 列（mcoplib 裸 rms_norm + sigmoid +
       mul 3-op 组合）；基准=sglang / FlagGems / mcoplib 三条
| v4.4 | gems 基线列升级：3-op 组合（rms_norm+sigmoid+mul）→ flaggems_vllm
       2-launch 融合组合（add_rms_norm + gate_sigmoid_mul，kernel 嵌入本文件）。
       基线变强后 spd_gems 预期整体回落——新口径下赢的才是真赢。x2 喂缓存
       zeros 占位对齐无 residual 语义（见文件头口径说明）。
| v4.5 | 修 v4.4 两个启动错误：①删多余的 `import triton.autotune`（该子模块
       不存在，首跑即 ModuleNotFoundError）；②gate_sigmoid_mul 的 BLOCK_SIZE
       改为 launch 显式传 1024——本环境 libtuner 不把 tuned config 的
       BLOCK_SIZE 注入 kwargs，缺省报 "missing BLOCK_SIZE"。add_rms_norm
       单遍 kernel 本就显式传 BLOCK_SIZE，loop kernel 走标准 triton.autotune
       注入正常（bench D≤4096 全走单遍，不触）。 |
| v4.6 | Claude Code 交叉审查后两处加固：①gate_sigmoid_mul 加同秩 assert
       （嵌入版未移植库的跨 rank 广播 pad，低秩 gate 会被 zip 静默截断——
       bench 输入恒 [M,D] 同秩，钉死语义）；②BLOCK_SIZE 改条件传（仅当
       tuned config 为空时显式传 1024，防未来补 yaml 后与 libtuner 注入
       撞 multiple values）；③文件头口径注释改为"库内无 tuned config，
       基线未调优"（原"Kimi-K3 workload 域外泛化"说法失实）。测量口径
       零改动，v4.5 数字不作废。 |

最终战绩（v4.6）：vs FlagGems 2-launch 融合组合 1.22~3.76x、vs mcoplib 3-op
组合 1.51~11.12x，全档 >1x；精度 18/18。

小 shape 档（decode）的绝对时间被启动开销按住：本机空 kernel 一次 launch 的
CPU 提交就要 ~12.6us（`torch.empty(1)` 都要 9us），因此 decode 档 15.5→14.2us
里真正的 kernel 只占 ~1.6us，剩下全是 launch 地板，天花板就是基线那条线。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED


# ============================= TLE kernel =============================


@triton.layout_tune()
@triton.jit
def _rmsnorm_gate_kernel(
    x_ptr, g_ptr, w_ptr, o_ptr,
    M, eps,
    D: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ACT_SWISH: tl.constexpr,
    X_ENC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BM + tl.arange(0, BM)
    row_mask = rows < M
    d = tl.arange(0, BD)
    # 列掩码：D 非 2 的幂（如 96）时 BD > D，越界列必须挡住，
    # 否则 load 垃圾 x 污染 rstd
    m = row_mask[:, None] & (d < D)[None, :]

    # x tile 钉布局（D 维连续，行归约不跨 lane）；X_ENC=None 时自扫
    # （单行/双行路径，钉多行布局反而灾难级变慢）
    x_ptrs = x_ptr + rows[:, None] * D + d[None, :]
    x = tl.load(x_ptrs, mask=m, other=0.0).to(tl.float32)
    if X_ENC is not None:
        x = tle.gpu.set_layout(x, X_ENC)

    ss = tl.sum(x * x, axis=1, keep_dims=True) / D
    rstd = 1.0 / tl.sqrt(ss + eps)
    y = x * rstd

    d = tl.arange(0, BD)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + d, mask=d < D, other=0.0).to(tl.float32)
        y = y * w[None, :]

    g_ptrs = g_ptr + rows[:, None] * D + d[None, :]
    g = tl.load(g_ptrs, mask=m, other=0.0).to(tl.float32)
    if ACT_SWISH:
        gate = g * (1.0 / (1.0 + tl.exp(-g)))
    else:
        gate = 1.0 / (1.0 + tl.exp(-g))
    y = y * gate

    o_ptrs = o_ptr + rows[:, None] * D + d[None, :]
    tl.store(o_ptrs, y.to(o_ptr.dtype.element_ty), mask=m)


# ==================== D ≥ 512 / 小 M：单行版本 ====================
# 每 program 一行，tile 是 1-D [BD]：行内归约沿 lane 展开，寄存器只随 BD 线性
# 增长（不随 BM 翻倍）。D ≥ 512 与 M 小时全面快于二维块版本。


@triton.layout_tune()
@triton.jit
def _rmsnorm_gate_row_kernel(
    x_ptr, g_ptr, w_ptr, o_ptr,
    M, eps,
    D: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ACT_SWISH: tl.constexpr,
):
    row = tl.program_id(axis=0)
    d = tl.arange(0, BD)
    mask = d < D
    x = tl.load(x_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    y = x * rstd
    if HAS_WEIGHT:
        y = y * tl.load(w_ptr + d, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)
    if ACT_SWISH:
        gate = g * (1.0 / (1.0 + tl.exp(-g)))
    else:
        gate = 1.0 / (1.0 + tl.exp(-g))
    tl.store(o_ptr + row * D + d, (y * gate).to(o_ptr.dtype.element_ty), mask=mask)


# ==================== D > 4096：循环归约版本 ====================


@triton.layout_tune()
@triton.jit
def _rmsnorm_gate_loop_kernel(
    x_ptr, g_ptr, w_ptr, o_ptr,
    M, eps,
    D: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ACT_SWISH: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BM + tl.arange(0, BM)
    row_mask = rows < M
    d = tl.arange(0, BD)

    # 第一趟：累加平方和（布局交给 layout_tune 自动扫——大 D 非主战场，
    # 循环体内 set_layout 反而触发 llir codegen 错误，v1 实测踩坑后移除）
    ss = tl.zeros([BM, 1], dtype=tl.float32)
    for off in range(0, D, BD):
        cols = off + d
        m = row_mask[:, None] & (cols[None, :] < D)
        x = tl.load(x_ptr + rows[:, None] * D + cols[None, :],
                    mask=m, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1, keep_dims=True)
    rstd = 1.0 / tl.sqrt(ss / D + eps)

    # 第二趟：归一化 + gate + 写回
    for off in range(0, D, BD):
        cols = off + d
        m = row_mask[:, None] & (cols[None, :] < D)
        x = tl.load(x_ptr + rows[:, None] * D + cols[None, :],
                    mask=m, other=0.0).to(tl.float32)
        y = x * rstd
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
            y = y * w[None, :]
        g = tl.load(g_ptr + rows[:, None] * D + cols[None, :],
                    mask=m, other=0.0).to(tl.float32)
        if ACT_SWISH:
            gate = g * (1.0 / (1.0 + tl.exp(-g)))
        else:
            gate = 1.0 / (1.0 + tl.exp(-g))
        y = y * gate
        tl.store(o_ptr + rows[:, None] * D + cols[None, :],
                 y.to(o_ptr.dtype.element_ty), mask=m)


# ============================= 入口 =============================


def _resolve_bm(M):
    if M <= 8192:
        return 8
    return 16


# weight=None 时的占位指针。缓存一份，避免每次调用 o.view(-1)[:1] 的
# 两个 ATen 视图（实测 ~2.8us/次，小 shape 档全在启动开销上，省下来直接可见）。
_PLACEHOLDER = {}


def _placeholder_weight(device):
    w = _PLACEHOLDER.get(device)
    if w is None:
        w = torch.zeros(1, device=device, dtype=torch.float32)
        _PLACEHOLDER[device] = w
    return w


def _act_swish(activation):
    """归一化 activation：'sigmoid' → False；'swish'/'silu'（同义）→ True。

    未知取值直接报错——v4.2 及以前只判 == "swish"，传 "silu" 会静默按 sigmoid
    算，结果错但无任何提示。
    """
    a = str(activation).lower()
    if a == "sigmoid":
        return False
    if a in ("swish", "silu"):
        return True
    raise ValueError(f"activation 仅支持 'sigmoid'/'swish'/'silu'，got {activation!r}")


def fused_rmsnorm_gated(x, gate, weight=None, eps=1e-6, activation="sigmoid",
                        bm=None, num_warps=None):
    """FusedRMSNormGated：out = rmsnorm(x) * weight * act(gate)。

    x/gate: [M, D]，D > 4096 走 loop 版（BD=1024 分趟），其余走单块版本。
    **x/gate/weight 必须是 contiguous**（kernel 全按 row*D+d / +d 平铺取数，
    切片或 permute 出来的张量会读到错位数据；三条检测开销实测 0.18us）。
    bm / num_warps 给定时强制走多行块 kernel 并按其调参（仅该路径生效；
    D > 4096 仍走 loop 路径，二者忽略）。None 时按路径表自动分发。
    返回 [M, D]，dtype 同 x。
    """
    assert ENABLED, "Requires a MetaX FLAGTREE_COMMON_IR build"
    M, D = x.shape
    assert gate.shape == x.shape, "gate 形状必须与 x 一致"
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"仅支持 fp16/bf16，got {x.dtype}")

    o = torch.empty_like(x)
    has_w = weight is not None          # 必须在占位前判：占位后恒非 None，
    if has_w and weight.numel() != D:   # v1 曾因此把输出内存当 weight 乘
        raise ValueError(f"weight 期望 [{D}]，got {tuple(weight.shape)}")
    # 静默错值防线：kernel 全部按平铺取数，非紧排张量读到的数据错位且不会报错
    # （实测非连续 x/gate max_abs 4.8、非连续 weight max_abs 14.3，全部 False）
    if not (x.is_contiguous() and gate.is_contiguous()
            and (not has_w or weight.is_contiguous())):
        raise ValueError(
            f"x/gate/weight 必须 contiguous（got x.stride={tuple(x.stride())}, "
            f"gate.stride={tuple(gate.stride())}"
            + (f", weight.stride={tuple(weight.stride())}" if has_w else "")
            + "）；请先 .contiguous()")
    # = triton.next_power_of_2(D)，内联省掉一次 python 调用（~1us/次）
    BD = min(1 << max(D - 1, 0).bit_length(), 4096)
    if not has_w:
        weight = _placeholder_weight(x.device)   # kernel 内 HAS_WEIGHT=False 不读
    act_swish = _act_swish(activation)

    # 三路分发见文件头路径表
    if D > 4096:
        BM = bm if bm is not None else _resolve_bm(M)
        BD_LOOP = min(BD, 1024)
        grid = (triton.cdiv(M, BM),)
        _rmsnorm_gate_loop_kernel[grid](
            x, gate, weight, o, M, float(eps),
            D=D, BD=BD_LOOP, BM=BM,
            HAS_WEIGHT=has_w,
            ACT_SWISH=act_swish,
            num_warps=4, num_stages=1,
        )
    elif bm is None and (D >= 512 or M <= 1024):
        # 单行版：宽 D 或小 M。BD ≤ 1024 时 1 warp 够，再宽加到 2 warp
        _rmsnorm_gate_row_kernel[(M,)](
            x, gate, weight, o, M, float(eps),
            D=D, BD=BD,
            HAS_WEIGHT=has_w,
            ACT_SWISH=act_swish,
            num_warps=1 if BD <= 1024 else 2,
        )
    else:
        BM = bm if bm is not None else _resolve_bm(M)
        nw = num_warps if num_warps is not None else (4 if BD >= 1024 else 2)
        # 钉布局：[BM, BD]，D 维（axis=1）连续，每 lane 拿连续 16 列，行归约沿
        # lane 内进行——与 sigmoid_gating TLE 版 h tile 的钉法同源。
        x_enc = tle.gpu.BlockEncoding([1, 1], [4, 16], [int(nw), 1], [1, 0])
        grid = (triton.cdiv(M, BM),)
        _rmsnorm_gate_kernel[grid](
            x, gate, weight, o, M, float(eps),
            D=D, BD=BD, BM=BM,
            HAS_WEIGHT=has_w,
            ACT_SWISH=act_swish, X_ENC=x_enc,
            num_warps=int(nw),
        )
    return o


# 算子接口名（注册用别名）
FusedRMSNormGated = fused_rmsnorm_gated


# =============================================================================
# mcoplib 库 ops 组合基线
#
# 口径：pybind 导出的 vLLM 血统裸 rms_norm（正式 Python API 全是缝合形态
# ——residual/fp8 量化/RoPE，裸版只有 pybind 层直接暴露）：
#   rms_norm(result, input, weight, eps, residual=None, inv_rms=None,
#            fuse_params=False) -> None
# weight 必须传 Tensor；组合 ATen sigmoid + ATen mul，共 3 kernel。
# 库内 sigmoid 无独立 C++ 单算子（triton_fused_sigmoid_gating_delta_rule_update.py
# 里只内联在 recurrent/gate 预计算 kernel 中），走 ATen 通用实现。
# 精度已验证（2026-11-10）：max_rel 0.4%，纯 bf16 舍入。
# =============================================================================

_MCOP = None
_MCOP_LOADED = False


def _load_mcoplib():
    global _MCOP, _MCOP_LOADED
    if _MCOP_LOADED:
        return _MCOP
    _MCOP_LOADED = True
    try:
        import mcoplib.op as mcop
        _MCOP = mcop
    except Exception:
        traceback.print_exc()
    return _MCOP


def _rmsnorm_gate_mcop(x, gate, weight=None, eps=1e-6):
    """mcoplib 组合：mcop.rms_norm + ATen sigmoid + ATen mul。"""
    mcop = _load_mcoplib()
    if mcop is None:
        raise RuntimeError("mcoplib 不可用")
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    r = torch.empty_like(x)
    mcop.rms_norm(r, x, weight, eps, None, None, False)
    return r * torch.sigmoid(gate)


# =============================================================================
# flaggems_vllm 库 2-launch 融合组合基线
#
# 口径：add_rms_norm（KernelGen 生成，rmsnorm(x1+x2)·w）+ gate_sigmoid_mul
# （Kimi-K3 output gate 专用单算子），共 2 kernel——库内能做到的最强形态。
# 本算子无 residual：add_rms_norm 的 x2 喂缓存的 zeros 占位（x+0 ≡ x，
# 数学等价；zeros 在 wrapper 外缓存，分配不进计时），基线因此多读一份
# 全零张量，带宽略吃亏于它。
# 两个 kernel 均带 autotune/libtuner（warmup=5/rep=20 与 tuned config），
# bench 的 warmup 轮数（默认 25）足以覆盖其调参轮。
# swish 档无对应实现（gate_sigmoid_mul 仅支持 sigmoid）；bench 全档 sigmoid。
# kernel 代码完整嵌入（来源：flaggems_vllm 仓库，FlagOS Apache-2.0），仅
# import 其 runtime/utils 基础件——不依赖库内 ops 的导出路径，加载失败
# 打印堆栈，gems 列显示 n/a，不影响其他列。
# =============================================================================

_GEMS2 = None
_GEMS2_LOADED = False


def _load_flaggems_vllm():
    """加载 flaggems_vllm 基础件并定义 2-launch 组合 kernel；失败返回 None。"""
    global _GEMS2, _GEMS2_LOADED
    if _GEMS2_LOADED:
        return _GEMS2
    _GEMS2_LOADED = True
    try:
        _GEMS2 = _build_flaggems_vllm()
        return _GEMS2
    except Exception:
        traceback.print_exc()
    src = os.environ.get("FLAGGEMS_VLLM_SRC", "")
    if src and os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
        try:
            _GEMS2 = _build_flaggems_vllm()
        except Exception:
            traceback.print_exc()
    return _GEMS2


def _build_flaggems_vllm():
    from flaggems_vllm import runtime
    from flaggems_vllm.runtime import torch_device_fn
    from flaggems_vllm.utils import libentry, libtuner
    from flaggems_vllm.utils import triton_lang_extension as tle

    @triton.jit
    def prev_multiple_of(a, b):
        return tl.cdiv(a, b) * b - b

    @libentry()
    @triton.jit(do_not_specialize=["eps"])
    def add_rms_norm_kernel(
        out_ptr, in_ptr1, in_ptr2, w_ptr,
        y_stride_r, y_stride_c, x1_stride_r, x1_stride_c,
        x2_stride_r, x2_stride_c, N, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        if tl.constexpr(in_ptr1.dtype.element_ty == tl.float16) or tl.constexpr(
            in_ptr1.dtype.element_ty == tl.bfloat16
        ):
            cdtype = tl.float32
        else:
            cdtype = in_ptr1.dtype.element_ty

        pid = tl.program_id(0)
        out_ptr += pid * y_stride_r
        in_ptr1 += pid * x1_stride_r
        in_ptr2 += pid * x2_stride_r

        mask = tl.arange(0, BLOCK_SIZE) < N
        cols = tl.arange(0, BLOCK_SIZE)
        x1 = tl.load(in_ptr1 + cols * x1_stride_c, mask, other=0.0).to(cdtype)
        x2 = tl.load(in_ptr2 + cols * x2_stride_c, mask, other=0.0).to(cdtype)

        x = x1 + x2

        var = tl.sum(x * x, axis=0) / N
        rrms = 1 / tl.sqrt(var + eps)

        w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        y = (x * rrms * w).to(cdtype)
        tl.store(out_ptr + cols * y_stride_c, y, mask=mask)

    @libentry()
    @triton.autotune(
        configs=runtime.get_tuned_config("add_rms_norm_loop"),
        key=["N"],
    )
    @triton.jit(do_not_specialize=["eps"])
    def add_rms_norm_loop_kernel(
        out_ptr, in_ptr1, in_ptr2, w_ptr, N, eps, TILE_N: tl.constexpr,
    ):
        if tl.constexpr(in_ptr1.dtype.element_ty == tl.float16) or tl.constexpr(
            in_ptr1.dtype.element_ty == tl.bfloat16
        ):
            cdtype = tl.float32
        else:
            cdtype = in_ptr1.dtype.element_ty

        pid = tle.program_id(0)

        # Pass 1: compute sum(x^2) in chunks
        acc = tl.zeros((TILE_N,), dtype=tl.float32)
        num_steps = tl.cdiv(N, TILE_N)

        for step in range(0, num_steps - 1):
            start_n = step * TILE_N
            n_offsets = start_n + tl.arange(0, TILE_N)
            x1 = tl.load(in_ptr1 + pid * N + n_offsets).to(tl.float32)
            x2 = tl.load(in_ptr2 + pid * N + n_offsets).to(tl.float32)
            x = x1 + x2
            acc += x * x

        start_n = (num_steps - 1) * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x1 = tl.load(in_ptr1 + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(in_ptr2 + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        x = x1 + x2
        acc += x * x

        var = tl.sum(acc) / N
        rrms = 1 / tl.sqrt(var + eps)

        # Pass 2: normalize in reverse order (better L2 cache reuse)
        prev_multiple = prev_multiple_of(N, TILE_N)

        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x1 = tl.load(
                in_ptr1 + pid * N + n_offsets,
                mask=mask, other=0.0, eviction_policy="evict_first",
            ).to(cdtype)
            x2 = tl.load(
                in_ptr2 + pid * N + n_offsets,
                mask=mask, other=0.0, eviction_policy="evict_first",
            ).to(cdtype)
            x = x1 + x2
            w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
            y = (x * rrms * w).to(cdtype)
            tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
            x1 = tl.load(
                in_ptr1 + pid * N + n_offsets,
                eviction_policy="evict_first",
            ).to(cdtype)
            x2 = tl.load(
                in_ptr2 + pid * N + n_offsets,
                eviction_policy="evict_first",
            ).to(cdtype)
            x = x1 + x2
            w = tl.load(w_ptr + n_offsets)
            y = (x * rrms * w).to(cdtype)
            tl.store(out_ptr + pid * N + n_offsets, y)

    def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-5):
        dim = x1.ndim - len(normalized_shape)
        M = math.prod(x1.shape[:dim])
        N = math.prod(normalized_shape)

        assert x1.shape == x2.shape, (
            f"Input shapes must match: {tuple(x1.shape)} vs {tuple(x2.shape)}"
        )

        x1 = x1.contiguous()
        x2 = x2.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x1)

        with torch_device_fn.device(x1.device):
            if N <= 4096:
                BLOCK_SIZE = triton.next_power_of_2(N)
                add_rms_norm_kernel[M,](
                    y, x1, x2, weight, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
                )
            else:
                add_rms_norm_loop_kernel[M,](y, x1, x2, weight, N, eps)

        return y

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("gate_sigmoid_mul"),
        key=["N", "WIDTH", "LAYOUT_KEY"],
        warmup=5,
        rep=20,
        use_cuda_graph=False,
    )
    @triton.jit
    def _gate_sigmoid_mul_kernel(
        A, G, O, N,
        WIDTH: tl.constexpr, SHAPE: tl.constexpr, A_STRIDES: tl.constexpr,
        G_STRIDES: tl.constexpr, FLAT: tl.constexpr, LAYOUT_KEY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        index = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = index < N
        if FLAT:
            a_offset = index
            g_offset = index
        else:
            remaining = index
            a_offset = tl.full((BLOCK_SIZE,), 0, tl.int64)
            g_offset = tl.full((BLOCK_SIZE,), 0, tl.int64)
            for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
                coordinate = remaining % SHAPE[dim]
                remaining = remaining // SHAPE[dim]
                a_offset += coordinate * A_STRIDES[dim]
                g_offset += coordinate * G_STRIDES[dim]
        a = tl.load(A + a_offset, mask=mask, other=0).to(tl.float32)
        g = tl.load(G + g_offset, mask=mask, other=0).to(tl.float32)
        # The pinned function slice compiles with an FP32 sigmoid intermediate.
        sigmoid = 1.0 / (1.0 + tl.exp(-g))
        tl.store(O + index, a * sigmoid, mask=mask)

    def gate_sigmoid_mul(attn_out, gate):
        # 嵌入版不实现库的跨 rank 广播 pad（低秩 gate 会被 zip 静默截断）——
        # 本 bench 输入恒为同秩 [M, D]，用 assert 钉死语义，不触发即零开销。
        assert attn_out.ndim == gate.ndim, (
            "嵌入版 gate_sigmoid_mul 仅支持同秩输入（广播 pad 未移植）"
        )
        a_shape = tuple(attn_out.shape)
        g_shape = tuple(gate.shape)
        a_stride = attn_out.stride()
        g_stride = gate.stride()
        shape, a_strides, g_strides = [], [], []
        n = 1
        for a_size, g_size, a_step, g_step in zip(a_shape, g_shape, a_stride, g_stride):
            if a_size != g_size and a_size != 1 and g_size != 1:
                raise ValueError("attn_out and gate shapes are not broadcastable")
            size = g_size if a_size == 1 else a_size
            shape.append(size)
            a_strides.append(0 if a_size == 1 else a_step)
            g_strides.append(0 if g_size == 1 else g_step)
            n *= size
        output = torch.empty(
            tuple(shape), device=attn_out.device,
            dtype=torch.result_type(attn_out, gate),
        )
        if n == 0:
            return output
        flat = (
            attn_out.shape == gate.shape
            and attn_out.is_contiguous()
            and gate.is_contiguous()
        )
        width = shape[-1] if shape else 1
        layout_key = f"{shape}:{a_strides}:{g_strides}:{flat}"
        # BLOCK_SIZE 条件传（1024）：本环境 libtuner 的 tuned config 为空
        # （yaml 缺失），不会把 BLOCK_SIZE 注入 kwargs，缺省报 "missing
        # BLOCK_SIZE"。仅当库内无 config 时显式传，防未来补了 yaml 后
        # 与 libtuner 注入值撞车（multiple values TypeError）。
        tuned = runtime.get_tuned_config("gate_sigmoid_mul")
        block = 1024 if not tuned else None
        with torch_device_fn.device(attn_out.device):
            _gate_sigmoid_mul_kernel[
                lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"] if block is None
                                           else block),)
            ](
                attn_out, gate, output, n, width,
                tuple(shape), tuple(a_strides), tuple(g_strides),
                flat, layout_key,
                **({} if block is None else {"BLOCK_SIZE": block}),
            )
        return output

    return type("Gems2", (), {
        "add_rms_norm": staticmethod(add_rms_norm),
        "gate_sigmoid_mul": staticmethod(gate_sigmoid_mul),
    })()


# zeros 占位缓存：add_rms_norm 的 x2（无 residual 语义对齐，x+0 ≡ x）。
# 按 (shape, dtype, device) 缓存，分配不进 bench 计时。
_GEMS2_ZEROS = {}


def _rmsnorm_gate_gems(x, gate, weight=None, eps=1e-6, activation="sigmoid"):
    """flaggems_vllm 2-launch 组合：add_rms_norm(x, 0, w) + gate_sigmoid_mul。"""
    gems2 = _load_flaggems_vllm()
    if gems2 is None:
        raise RuntimeError("flaggems_vllm 不可用")
    if _act_swish(activation):
        raise ValueError("gems 2-launch 基线不支持 swish（gate_sigmoid_mul 仅 sigmoid）")
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    key = (tuple(x.shape), x.dtype, x.device.index)
    zeros = _GEMS2_ZEROS.get(key)
    if zeros is None:
        zeros = torch.zeros_like(x)
        _GEMS2_ZEROS[key] = zeros
    y = gems2.add_rms_norm(x, zeros, [x.shape[-1]], weight, eps)
    return gems2.gate_sigmoid_mul(y, gate)


# ============================= fp32 参考 & 自测 =============================


def reference(x, gate, weight=None, eps=1e-6, activation="sigmoid"):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    if _act_swish(activation):
        return (y * torch.nn.functional.silu(gate.float())).to(x.dtype)
    return (y * torch.sigmoid(gate.float())).to(x.dtype)


ACC_CASES = [
    (16, 128),          # decode 单 token 16 v-heads
    (4096, 128),        # prefill 4k tokens
    (8192, 128),        # prefill 8k tokens
    (32768, 128),
    (65536, 128),
    (4096, 96),         # 非 2 幂 head dim
    (4096, 8192),       # 触发 loop 路径
    (1000, 128),        # M / BM 双尾块
]


def run_accuracy(atol=1e-2, rtol=1e-2):
    print("| case | dtype | act | weight | max_abs | ok |")
    print("| --- | --- | --- | --- | --- | --- |")
    for M, D in ACC_CASES:
        torch.manual_seed(0)
        x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16) * 0.5
        g = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
        for dtype in (torch.bfloat16, torch.float16):
            x = x.to(dtype)
            g = g.to(dtype)
            for act in ("sigmoid", "swish"):
                for use_w in (False, True):
                    w = (torch.randn(D, device="cuda", dtype=dtype) * 0.1
                         if use_w else None)
                    ref = reference(x, g, w, activation=act)
                    # None = 自动分发（含单行 kernel），其余强制多行块 kernel
                    for bm in (None, 4, 8, 16):
                        try:
                            got = fused_rmsnorm_gated(x, g, w, activation=act, bm=bm)
                            diff = (got.float() - ref.float()).abs().max().item()
                            ok = torch.allclose(got.float(), ref.float(),
                                                atol=atol, rtol=rtol)
                        except Exception:
                            traceback.print_exc()
                            diff, ok = float("nan"), False
                        name = f"M{M}D{D}/bm{'auto' if bm is None else bm}"
                        print(f"| {name} | {str(dtype)[6:]} | {act} "
                              f"| {'on' if use_w else 'off'} | {diff:.3g} | {ok} |")


BENCH_CASES = [
    # (name, M, D)：GLM-5-Next Glm5NextLinearAttention.o_norm 的调用量级
    # D = head_dim = 128，M = tokens × value heads
    ("decode-1tok", 16, 128),
    ("decode-4tok", 64, 128),
    ("prefill-4k", 4096, 128),
    ("prefill-8k", 8192, 128),
    ("prefill-16k", 16384, 128),
    ("prefill-32k", 32768, 128),
    ("wide-D512", 4096, 512),
    ("wide-D4096", 4096, 4096),
]


def run_benchmark(warmup=25, rep=100):
    print("### fused_rmsnorm_gate TLE benchmark"
          "（vs flaggems_vllm 2-launch 融合组合 + mcoplib 库 ops 组合）")
    print("| case | tle_us | gems_us | spd_gems | mcop_us | spd_mcop |")
    print("| --- | --- | --- | --- | --- | --- |")
    gems = _load_flaggems_vllm()
    mcop = _load_mcoplib()
    if gems is None:
        print("（flaggems_vllm 加载失败，gems 列为 n/a；堆栈见上方输出）")
    for name, M, D in BENCH_CASES:
        torch.manual_seed(0)
        x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16) * 0.5
        g = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1

        # 先验精度
        ref = reference(x, g, w)
        got = fused_rmsnorm_gated(x, g, w)
        assert torch.allclose(got.float(), ref.float(), atol=1e-2, rtol=1e-2), name

        tle_us = triton.testing.do_bench(
            lambda: fused_rmsnorm_gated(x, g, w), warmup=warmup, rep=rep) * 1e3
        gems_us = float("nan")
        if gems is not None:
            gems_us = triton.testing.do_bench(
                lambda: _rmsnorm_gate_gems(x, g, w), warmup=warmup, rep=rep) * 1e3
        mcop_us = float("nan")
        if mcop is not None:
            mcop_us = triton.testing.do_bench(
                lambda: _rmsnorm_gate_mcop(x, g, w), warmup=warmup, rep=rep) * 1e3
        gems_s = f"{gems_us:.1f}" if gems_us == gems_us else "n/a"
        gems_r = f"{gems_us / tle_us:.2f}" if gems_us == gems_us else "-"
        mcop_s = f"{mcop_us:.1f}" if mcop_us == mcop_us else "n/a"
        mcop_r = f"{mcop_us / tle_us:.2f}" if mcop_us == mcop_us else "-"
        print(f"| {name} | {tle_us:.1f} | {gems_s} | {gems_r} | {mcop_s} | {mcop_r} |")
    print("\n列义：gems = flaggems_vllm 库 2-launch 融合组合"
          "（add_rms_norm + gate_sigmoid_mul；本算子无 residual，x2 喂缓存"
          " zeros 占位 x+0≡x，基线多读一份零张量），mcop = mcoplib 库 ops"
          " 组合（裸 rms_norm + sigmoid + mul，库内 sigmoid 无独立单算子走"
          " ATen），spd = 基线 / TLE，>1 我方快。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bm", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    if args.benchmark:
        run_benchmark(warmup=args.warmup, rep=args.rep)
    else:
        run_accuracy()
