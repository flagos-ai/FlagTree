"""Single-kernel raw-NVSHMEM + Tensor Core L1 smoke.

This is a narrow stage-4 capability test: one TLE warp-specialized kernel runs
UserHopper-like raw-NVSHMEM dispatch, raw-NVSHMEM receiver, and an L1/L2 FP8
``tl.dot`` worker in the same ``warp_specialize`` region.

The test intentionally fixes the local shape to the smallest useful cases:
up to 2 experts per rank, I=128, H=128 with <=16 received token-topk entries
per local expert, or H=256 with <=8 entries per local expert.  That lets one
CTA prove the communication-to-TensorCore handoff without introducing a
multi-CTA scheduler.
"""

from __future__ import annotations

import ctypes
import os
import site
from pathlib import Path

cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_HOME", cuda_home)
os.environ.setdefault("USERHOPPER_WS_NUM_RANKS", "2")
os.environ.setdefault("USERHOPPER_WS_NUM_EXPERTS", "2")
os.environ.setdefault("USERHOPPER_WS_NUM_TOPK", "1")
os.environ.setdefault("USERHOPPER_WS_NUM_TOKENS", "4")
os.environ.setdefault("USERHOPPER_WS_HIDDEN", "128")
os.environ.setdefault("USERHOPPER_WS_INTERMEDIATE_HIDDEN", "128")
os.environ.setdefault("USERHOPPER_WS_NUM_WARPS", "4")
os.environ.setdefault("USERHOPPER_WS_NUM_DISPATCH_WARPS", "1")
os.environ.setdefault("USERHOPPER_WS_COMPUTE_FULL_HIDDEN", "1")
os.environ.setdefault("USERHOPPER_WS_ALLOW_EXPERIMENTAL_H256_TLDOT", "0")
os.environ.setdefault("USERHOPPER_WS_CLEANUP", "0")
os.environ.setdefault("USERHOPPER_WS_REPEAT_LAUNCHES", "1")
os.environ.setdefault("USERHOPPER_WS_EXPERT_WAVE_SINGLE_KERNEL", "0")
os.environ.setdefault("USERHOPPER_WS_EXPERT_WAVE_COMPUTE_WARPS", "4")
os.environ.setdefault("USERHOPPER_WS_L2_H_TILES", "0")
os.environ.setdefault("USERHOPPER_WS_L1_I_TILES", "0")
os.environ.setdefault("USERHOPPER_WS_SPLIT_L1_L2_WORKERS", "0")
os.environ.setdefault("USERHOPPER_WS_L1_TILE_LOOP", "0")
os.environ.setdefault("USERHOPPER_WS_L2_BLOCK_N", "64")
os.environ.setdefault("USERHOPPER_WS_L2_SCALAR", "0")
os.environ.setdefault("USERHOPPER_WS_MULTI_CTA_EXPERT_WAVE", "0")
os.environ.setdefault("USERHOPPER_WS_MULTI_CTA_L1_ONLY", "0")
os.environ.setdefault("USERHOPPER_WS_MULTI_CTA_L1_TILE_SPLIT", "0")
os.environ.setdefault("USERHOPPER_WS_MULTI_CTA_L2_TILE_SPLIT", "0")
os.environ.setdefault("USERHOPPER_WS_SKIP_COMBINE", "0")
os.environ.setdefault("USERHOPPER_WS_SKIP_REDUCE", "0")
os.environ["CPATH"] = (
    f"{cuda_home}/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
)
os.environ["LD_LIBRARY_PATH"] = (
    f"{cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)

import triton
import triton.language as tl
import triton.experimental.tle.language as tle

torch_site_packages = os.environ.get("MEGAMOE_TORCH_SITE_PACKAGES")
if torch_site_packages and Path(torch_site_packages).exists():
    site.addsitedir(torch_site_packages)

import torch
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
import triton_tle_ws_userhopper_dispatch_receiver_smoke as uh


@dialect(
    name="cuda",
    compiler="nvcc",
    file=uh.HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=uh.HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_tldot_combine_write_partition",
    libs={"nvshmem": uh.NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_tldot_combine_write(*args, **kwargs):
    ...


DEFAULT_PY_BLOCK_M = 8 if uh.HIDDEN == 256 else 16
PY_BLOCK_M = int(os.environ.get("USERHOPPER_WS_BLOCK_M", str(DEFAULT_PY_BLOCK_M)))
JIT_BLOCK_M = tl.constexpr(PY_BLOCK_M)
JIT_BLOCK_I = tl.constexpr(64)
JIT_BLOCK_K = tl.constexpr(64)
PY_BLOCK_N = int(os.environ["USERHOPPER_WS_L2_BLOCK_N"])
JIT_BLOCK_N = tl.constexpr(PY_BLOCK_N)
DEFAULT_PY_L1_I_TILES = uh.INTERMEDIATE_HIDDEN // 64
PY_L1_I_TILES = int(os.environ["USERHOPPER_WS_L1_I_TILES"])
if PY_L1_I_TILES == 0:
    PY_L1_I_TILES = DEFAULT_PY_L1_I_TILES
JIT_L1_I_TILES = tl.constexpr(PY_L1_I_TILES)
DEFAULT_PY_L2_H_TILES = uh.HIDDEN // PY_BLOCK_N
PY_L2_H_TILES = int(os.environ["USERHOPPER_WS_L2_H_TILES"])
if PY_L2_H_TILES == 0:
    PY_L2_H_TILES = DEFAULT_PY_L2_H_TILES
JIT_L2_H_TILES = tl.constexpr(PY_L2_H_TILES)
CLEANUP_WORKSPACE = int(os.environ["USERHOPPER_WS_CLEANUP"])
REPEAT_LAUNCHES = int(os.environ["USERHOPPER_WS_REPEAT_LAUNCHES"])
EXPERT_WAVE_SINGLE_KERNEL = int(os.environ["USERHOPPER_WS_EXPERT_WAVE_SINGLE_KERNEL"])
EXPERT_WAVE_COMPUTE_WARPS = int(os.environ["USERHOPPER_WS_EXPERT_WAVE_COMPUTE_WARPS"])
SPLIT_L1_L2_WORKERS = int(os.environ["USERHOPPER_WS_SPLIT_L1_L2_WORKERS"])
L1_TILE_LOOP = int(os.environ["USERHOPPER_WS_L1_TILE_LOOP"])
JIT_L1_TILE_LOOP = tl.constexpr(L1_TILE_LOOP)
L2_SCALAR = int(os.environ["USERHOPPER_WS_L2_SCALAR"])
JIT_L2_SCALAR = tl.constexpr(L2_SCALAR)
MULTI_CTA_EXPERT_WAVE = int(os.environ["USERHOPPER_WS_MULTI_CTA_EXPERT_WAVE"])
MULTI_CTA_L1_ONLY = int(os.environ["USERHOPPER_WS_MULTI_CTA_L1_ONLY"])
JIT_MULTI_CTA_L1_ONLY = tl.constexpr(MULTI_CTA_L1_ONLY)
MULTI_CTA_L1_TILE_SPLIT = int(os.environ["USERHOPPER_WS_MULTI_CTA_L1_TILE_SPLIT"])
JIT_MULTI_CTA_L1_TILE_SPLIT = tl.constexpr(MULTI_CTA_L1_TILE_SPLIT)
MULTI_CTA_L2_TILE_SPLIT = int(os.environ["USERHOPPER_WS_MULTI_CTA_L2_TILE_SPLIT"])
JIT_MULTI_CTA_L2_TILE_SPLIT = tl.constexpr(MULTI_CTA_L2_TILE_SPLIT)
SKIP_COMBINE = int(os.environ["USERHOPPER_WS_SKIP_COMBINE"])
JIT_SKIP_COMBINE = tl.constexpr(SKIP_COMBINE)
SKIP_REDUCE = int(os.environ["USERHOPPER_WS_SKIP_REDUCE"])
JIT_SKIP_REDUCE = tl.constexpr(SKIP_REDUCE)

if CLEANUP_WORKSPACE not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_CLEANUP must be 0 or 1, got {CLEANUP_WORKSPACE}")
if REPEAT_LAUNCHES <= 0:
    raise ValueError(f"USERHOPPER_WS_REPEAT_LAUNCHES must be positive, got {REPEAT_LAUNCHES}")
if REPEAT_LAUNCHES > 1 and CLEANUP_WORKSPACE == 0:
    raise ValueError("USERHOPPER_WS_REPEAT_LAUNCHES>1 requires USERHOPPER_WS_CLEANUP=1")
if EXPERT_WAVE_SINGLE_KERNEL not in (0, 1):
    raise ValueError(
        "USERHOPPER_WS_EXPERT_WAVE_SINGLE_KERNEL must be 0 or 1, "
        f"got {EXPERT_WAVE_SINGLE_KERNEL}"
    )
if EXPERT_WAVE_COMPUTE_WARPS not in (2, 4):
    raise ValueError(
        "USERHOPPER_WS_EXPERT_WAVE_COMPUTE_WARPS must be 2 or 4, "
        f"got {EXPERT_WAVE_COMPUTE_WARPS}"
    )
if SPLIT_L1_L2_WORKERS not in (0, 1):
    raise ValueError(
        "USERHOPPER_WS_SPLIT_L1_L2_WORKERS must be 0 or 1, "
        f"got {SPLIT_L1_L2_WORKERS}"
    )
if L1_TILE_LOOP not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_L1_TILE_LOOP must be 0 or 1, got {L1_TILE_LOOP}")
if L2_SCALAR not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_L2_SCALAR must be 0 or 1, got {L2_SCALAR}")
if MULTI_CTA_EXPERT_WAVE not in (0, 1):
    raise ValueError(
        "USERHOPPER_WS_MULTI_CTA_EXPERT_WAVE must be 0 or 1, "
        f"got {MULTI_CTA_EXPERT_WAVE}"
    )
if MULTI_CTA_L1_ONLY not in (0, 1):
    raise ValueError(
        "USERHOPPER_WS_MULTI_CTA_L1_ONLY must be 0 or 1, "
        f"got {MULTI_CTA_L1_ONLY}"
    )
if MULTI_CTA_L1_ONLY != 0 and MULTI_CTA_EXPERT_WAVE == 0:
    raise ValueError("USERHOPPER_WS_MULTI_CTA_L1_ONLY requires USERHOPPER_WS_MULTI_CTA_EXPERT_WAVE=1")
if MULTI_CTA_L1_TILE_SPLIT not in (0, 1):
    raise ValueError(
        "USERHOPPER_WS_MULTI_CTA_L1_TILE_SPLIT must be 0 or 1, "
        f"got {MULTI_CTA_L1_TILE_SPLIT}"
    )
if MULTI_CTA_L1_TILE_SPLIT != 0 and (MULTI_CTA_EXPERT_WAVE == 0 or MULTI_CTA_L1_ONLY == 0):
    raise ValueError(
        "USERHOPPER_WS_MULTI_CTA_L1_TILE_SPLIT requires "
        "USERHOPPER_WS_MULTI_CTA_EXPERT_WAVE=1 and USERHOPPER_WS_MULTI_CTA_L1_ONLY=1"
    )
if MULTI_CTA_L2_TILE_SPLIT not in (0, 1):
    raise ValueError(
        "USERHOPPER_WS_MULTI_CTA_L2_TILE_SPLIT must be 0 or 1, "
        f"got {MULTI_CTA_L2_TILE_SPLIT}"
    )
if MULTI_CTA_L2_TILE_SPLIT != 0 and (MULTI_CTA_EXPERT_WAVE == 0 or MULTI_CTA_L1_TILE_SPLIT == 0):
    raise ValueError(
        "USERHOPPER_WS_MULTI_CTA_L2_TILE_SPLIT requires "
        "USERHOPPER_WS_MULTI_CTA_EXPERT_WAVE=1 and USERHOPPER_WS_MULTI_CTA_L1_TILE_SPLIT=1"
    )
if SKIP_COMBINE not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_SKIP_COMBINE must be 0 or 1, got {SKIP_COMBINE}")
if SKIP_COMBINE != 0 and MULTI_CTA_L2_TILE_SPLIT == 0:
    raise ValueError("USERHOPPER_WS_SKIP_COMBINE requires USERHOPPER_WS_MULTI_CTA_L2_TILE_SPLIT=1")
if SKIP_COMBINE != 0 and CLEANUP_WORKSPACE != 0:
    raise ValueError("USERHOPPER_WS_SKIP_COMBINE=1 requires USERHOPPER_WS_CLEANUP=0")
if SKIP_REDUCE not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_SKIP_REDUCE must be 0 or 1, got {SKIP_REDUCE}")
if SKIP_REDUCE != 0 and MULTI_CTA_L2_TILE_SPLIT == 0:
    raise ValueError("USERHOPPER_WS_SKIP_REDUCE requires USERHOPPER_WS_MULTI_CTA_L2_TILE_SPLIT=1")
if SKIP_REDUCE != 0 and CLEANUP_WORKSPACE != 0:
    raise ValueError("USERHOPPER_WS_SKIP_REDUCE=1 requires USERHOPPER_WS_CLEANUP=0")


@triton.jit
def _l1_single_cta_two_tile_body(
    acts,
    acts_sf,
    topk_weights,
    weights,
    weights_sf,
    l2_acts,
    l2_acts_sf,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    I_OFFSET_C: tl.constexpr,
):
    offs_m = tl.arange(0, JIT_BLOCK_M)
    offs_i = I_OFFSET_C + tl.arange(0, JIT_BLOCK_I)
    group = offs_i // 8
    lane = offs_i - group * 8
    gate_rows = group * 16 + lane
    up_rows = gate_rows + 8

    gate_acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_I), dtype=tl.float32)
    for k0 in range(0, H_C, JIT_BLOCK_K):
        offs_k = k0 + tl.arange(0, JIT_BLOCK_K)
        a = tl.load(
            acts + offs_m[:, None] * H_C + offs_k[None, :],
            mask=offs_m[:, None] < M_C,
            other=0.0,
        )
        gate_w = tl.load(
            weights + gate_rows[None, :] * H_C + offs_k[:, None],
            mask=offs_i[None, :] < I_C,
            other=0.0,
        )
        up_w = tl.load(
            weights + up_rows[None, :] * H_C + offs_k[:, None],
            mask=offs_i[None, :] < I_C,
            other=0.0,
        )
        act_scale = tl.load(
            acts_sf + (k0 // 128) * NUM_PADDED_M_C + offs_m,
            mask=offs_m < M_C,
            other=0.0,
        )
        gate_scale = tl.load(
            weights_sf + (offs_i // 128) * (H_C // 128) + (k0 // 128),
            mask=offs_i < I_C,
            other=0.0,
        )
        up_scale = tl.load(
            weights_sf + ((I_C + offs_i) // 128) * (H_C // 128) + (k0 // 128),
            mask=offs_i < I_C,
            other=0.0,
        )
        gate_acc += tl.dot(a, gate_w, out_dtype=tl.float32) * act_scale[:, None] * gate_scale[None, :]
        up_acc += tl.dot(a, up_w, out_dtype=tl.float32) * act_scale[:, None] * up_scale[None, :]

    topk = tl.load(topk_weights + offs_m, mask=offs_m < M_C, other=0.0)
    swiglu = gate_acc * tl.sigmoid(gate_acc) * up_acc * topk[:, None]
    max_abs = tl.max(tl.abs(swiglu), axis=1)
    scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scaled = swiglu / scale[:, None]
    pid_i: tl.constexpr = I_OFFSET_C // JIT_BLOCK_I

    tl.store(
        l2_acts_sf + pid_i * NUM_PADDED_M_C + offs_m,
        scale,
        mask=offs_m < M_C,
    )
    tl.store(
        l2_acts + offs_m[:, None] * I_C + offs_i[None, :],
        scaled,
        mask=(offs_m[:, None] < M_C) & (offs_i[None, :] < I_C),
    )
    tl.store(marker, 0x4C1107)


@triton.jit
def _l1_single_cta_runtime_i_body(
    acts,
    acts_sf,
    topk_weights,
    weights,
    weights_sf,
    l2_acts,
    l2_acts_sf,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    i_offset,
):
    offs_m = tl.arange(0, JIT_BLOCK_M)
    offs_i = i_offset + tl.arange(0, JIT_BLOCK_I)
    group = offs_i // 8
    lane = offs_i - group * 8
    gate_rows = group * 16 + lane
    up_rows = gate_rows + 8

    gate_acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_I), dtype=tl.float32)
    for k0 in range(0, H_C, JIT_BLOCK_K):
        offs_k = k0 + tl.arange(0, JIT_BLOCK_K)
        a = tl.load(
            acts + offs_m[:, None] * H_C + offs_k[None, :],
            mask=offs_m[:, None] < M_C,
            other=0.0,
        )
        gate_w = tl.load(
            weights + gate_rows[None, :] * H_C + offs_k[:, None],
            mask=offs_i[None, :] < I_C,
            other=0.0,
        )
        up_w = tl.load(
            weights + up_rows[None, :] * H_C + offs_k[:, None],
            mask=offs_i[None, :] < I_C,
            other=0.0,
        )
        act_scale = tl.load(
            acts_sf + (k0 // 128) * NUM_PADDED_M_C + offs_m,
            mask=offs_m < M_C,
            other=0.0,
        )
        gate_scale = tl.load(
            weights_sf + (offs_i // 128) * (H_C // 128) + (k0 // 128),
            mask=offs_i < I_C,
            other=0.0,
        )
        up_scale = tl.load(
            weights_sf + ((I_C + offs_i) // 128) * (H_C // 128) + (k0 // 128),
            mask=offs_i < I_C,
            other=0.0,
        )
        gate_acc += tl.dot(a, gate_w, out_dtype=tl.float32) * act_scale[:, None] * gate_scale[None, :]
        up_acc += tl.dot(a, up_w, out_dtype=tl.float32) * act_scale[:, None] * up_scale[None, :]

    topk = tl.load(topk_weights + offs_m, mask=offs_m < M_C, other=0.0)
    swiglu = gate_acc * tl.sigmoid(gate_acc) * up_acc * topk[:, None]
    max_abs = tl.max(tl.abs(swiglu), axis=1)
    scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scaled = swiglu / scale[:, None]
    pid_i = i_offset // JIT_BLOCK_I

    tl.store(
        l2_acts_sf + pid_i * NUM_PADDED_M_C + offs_m,
        scale,
        mask=offs_m < M_C,
    )
    tl.store(
        l2_acts + offs_m[:, None] * I_C + offs_i[None, :],
        scaled,
        mask=(offs_m[:, None] < M_C) & (offs_i[None, :] < I_C),
    )
    tl.store(marker, 0x4C1108)


@triton.jit
def _l1_single_cta_runtime_m_i_body(
    acts,
    acts_sf,
    topk_weights,
    weights,
    weights_sf,
    l2_acts,
    l2_acts_sf,
    marker,
    M,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    i_offset,
):
    offs_m = tl.arange(0, JIT_BLOCK_M)
    offs_i = i_offset + tl.arange(0, JIT_BLOCK_I)
    group = offs_i // 8
    lane = offs_i - group * 8
    gate_rows = group * 16 + lane
    up_rows = gate_rows + 8

    gate_acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_I), dtype=tl.float32)
    for k0 in range(0, H_C, JIT_BLOCK_K):
        offs_k = k0 + tl.arange(0, JIT_BLOCK_K)
        a = tl.load(
            acts + offs_m[:, None] * H_C + offs_k[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        gate_w = tl.load(
            weights + gate_rows[None, :] * H_C + offs_k[:, None],
            mask=offs_i[None, :] < I_C,
            other=0.0,
        )
        up_w = tl.load(
            weights + up_rows[None, :] * H_C + offs_k[:, None],
            mask=offs_i[None, :] < I_C,
            other=0.0,
        )
        act_scale = tl.load(
            acts_sf + (k0 // 128) * NUM_PADDED_M_C + offs_m,
            mask=offs_m < M,
            other=0.0,
        )
        gate_scale = tl.load(
            weights_sf + (offs_i // 128) * (H_C // 128) + (k0 // 128),
            mask=offs_i < I_C,
            other=0.0,
        )
        up_scale = tl.load(
            weights_sf + ((I_C + offs_i) // 128) * (H_C // 128) + (k0 // 128),
            mask=offs_i < I_C,
            other=0.0,
        )
        gate_acc += tl.dot(a, gate_w, out_dtype=tl.float32) * act_scale[:, None] * gate_scale[None, :]
        up_acc += tl.dot(a, up_w, out_dtype=tl.float32) * act_scale[:, None] * up_scale[None, :]

    topk = tl.load(topk_weights + offs_m, mask=offs_m < M, other=0.0)
    swiglu = gate_acc * tl.sigmoid(gate_acc) * up_acc * topk[:, None]
    max_abs = tl.max(tl.abs(swiglu), axis=1)
    scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scaled = swiglu / scale[:, None]
    pid_i = i_offset // JIT_BLOCK_I

    tl.store(
        l2_acts_sf + pid_i * NUM_PADDED_M_C + offs_m,
        scale,
        mask=offs_m < M,
    )
    tl.store(
        l2_acts + offs_m[:, None] * I_C + offs_i[None, :],
        scaled,
        mask=(offs_m[:, None] < M) & (offs_i[None, :] < I_C),
    )
    tl.store(marker, 0x4C1109)


@triton.jit
def _l2_single_cta_tile_body(
    acts,
    acts_sf,
    weights,
    weights_sf,
    out,
    marker,
    M_C: tl.constexpr,
    N_C: tl.constexpr,
    K_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    N_OFFSET_C: tl.constexpr,
):
    offs_m = tl.arange(0, JIT_BLOCK_M)
    offs_n = N_OFFSET_C + tl.arange(0, JIT_BLOCK_N)

    acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_C, JIT_BLOCK_K):
        offs_k = k0 + tl.arange(0, JIT_BLOCK_K)
        a = tl.load(
            acts + offs_m[:, None] * K_C + offs_k[None, :],
            mask=offs_m[:, None] < M_C,
            other=0.0,
        )
        b = tl.load(
            weights + offs_n[None, :] * K_C + offs_k[:, None],
            mask=offs_n[None, :] < N_C,
            other=0.0,
        )
        partial = tl.dot(a, b, out_dtype=tl.float32)
        act_scale = tl.load(
            acts_sf + (k0 // 64) * NUM_PADDED_M_C + offs_m,
            mask=offs_m < M_C,
            other=0.0,
        )
        weight_scale = tl.load(
            weights_sf + (offs_n // 128) * (K_C // 128) + (k0 // 128),
            mask=offs_n < N_C,
            other=0.0,
        )
        acc += partial * act_scale[:, None] * weight_scale[None, :]

    tl.store(
        out + offs_m[:, None] * N_C + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M_C) & (offs_n[None, :] < N_C),
    )
    tl.store(marker, 0x4C2207)


@triton.jit
def _l2_single_cta_scalar_tile_body(
    acts,
    acts_sf,
    weights,
    weights_sf,
    out,
    marker,
    M_C: tl.constexpr,
    N_C: tl.constexpr,
    K_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    N_OFFSET_C: tl.constexpr,
):
    offs_m = tl.arange(0, JIT_BLOCK_M)
    offs_n = N_OFFSET_C + tl.arange(0, JIT_BLOCK_N)
    acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), dtype=tl.float32)

    for kk in range(0, K_C):
        a = tl.load(
            acts + offs_m * K_C + kk,
            mask=offs_m < M_C,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            weights + offs_n * K_C + kk,
            mask=offs_n < N_C,
            other=0.0,
        ).to(tl.float32)
        act_scale = tl.load(
            acts_sf + (kk // 64) * NUM_PADDED_M_C + offs_m,
            mask=offs_m < M_C,
            other=0.0,
        )
        weight_scale = tl.load(
            weights_sf + (offs_n // 128) * (K_C // 128) + (kk // 128),
            mask=offs_n < N_C,
            other=0.0,
        )
        acc += (a * act_scale)[:, None] * (b * weight_scale)[None, :]

    tl.store(
        out + offs_m[:, None] * N_C + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M_C) & (offs_n[None, :] < N_C),
    )
    tl.store(marker, 0x4C2307)


@triton.jit
def _l2_single_cta_runtime_m_n_body(
    acts,
    acts_sf,
    weights,
    weights_sf,
    out,
    marker,
    M,
    N_C: tl.constexpr,
    K_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    n_offset,
):
    offs_m = tl.arange(0, JIT_BLOCK_M)
    offs_n = n_offset + tl.arange(0, JIT_BLOCK_N)

    acc = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_C, JIT_BLOCK_K):
        offs_k = k0 + tl.arange(0, JIT_BLOCK_K)
        a = tl.load(
            acts + offs_m[:, None] * K_C + offs_k[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        b = tl.load(
            weights + offs_n[None, :] * K_C + offs_k[:, None],
            mask=offs_n[None, :] < N_C,
            other=0.0,
        )
        partial = tl.dot(a, b, out_dtype=tl.float32)
        act_scale = tl.load(
            acts_sf + (k0 // 64) * NUM_PADDED_M_C + offs_m,
            mask=offs_m < M,
            other=0.0,
        )
        weight_scale = tl.load(
            weights_sf + (offs_n // 128) * (K_C // 128) + (k0 // 128),
            mask=offs_n < N_C,
            other=0.0,
        )
        acc += partial * act_scale[:, None] * weight_scale[None, :]

    tl.store(
        out + offs_m[:, None] * N_C + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N_C),
    )
    tl.store(marker, 0x4C2208)


@triton.jit
def _single_expert_l1_body(
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    POOL_BASE_C: tl.constexpr,
    LOCAL_EXPERT_C: tl.constexpr,
):
    l1_acts_e = l1_acts + POOL_BASE_C * H_C
    l1_acts_sf_e = l1_acts_sf + POOL_BASE_C
    l1_topk_weights_e = l1_topk_weights + POOL_BASE_C
    l2_acts_e = l2_acts + POOL_BASE_C * I_C
    l2_acts_sf_e = l2_acts_sf + POOL_BASE_C
    l1_weights_e = l1_weights + LOCAL_EXPERT_C * (2 * I_C * H_C)
    l1_weights_sf_e = l1_weights_sf + LOCAL_EXPERT_C * ((2 * I_C // 128) * (H_C // 128))

    if JIT_L1_TILE_LOOP != 0:
        for i_offset in tl.range(0, JIT_L1_I_TILES * JIT_BLOCK_I, JIT_BLOCK_I):
            _l1_single_cta_runtime_i_body(
                l1_acts_e,
                l1_acts_sf_e,
                l1_topk_weights_e,
                l1_weights_e,
                l1_weights_sf_e,
                l2_acts_e,
                l2_acts_sf_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                i_offset,
            )
    else:
        if JIT_L1_I_TILES >= 1:
            _l1_single_cta_two_tile_body(
                l1_acts_e,
                l1_acts_sf_e,
                l1_topk_weights_e,
                l1_weights_e,
                l1_weights_sf_e,
                l2_acts_e,
                l2_acts_sf_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                0,
            )
        if JIT_L1_I_TILES >= 2:
            _l1_single_cta_two_tile_body(
                l1_acts_e,
                l1_acts_sf_e,
                l1_topk_weights_e,
                l1_weights_e,
                l1_weights_sf_e,
                l2_acts_e,
                l2_acts_sf_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                64,
            )


@triton.jit
def _single_expert_l2_body(
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    POOL_BASE_C: tl.constexpr,
    LOCAL_EXPERT_C: tl.constexpr,
):
    l2_acts_e = l2_acts + POOL_BASE_C * I_C
    l2_acts_sf_e = l2_acts_sf + POOL_BASE_C
    l2_out_e = l2_out + POOL_BASE_C * H_C
    l2_weights_e = l2_weights + LOCAL_EXPERT_C * (H_C * I_C)
    l2_weights_sf_e = l2_weights_sf + LOCAL_EXPERT_C * ((H_C // 128) * (I_C // 128))

    if JIT_L2_H_TILES >= 1:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                0,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                0,
            )
    if JIT_L2_H_TILES >= 2:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N,
            )
    if JIT_L2_H_TILES >= 3:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 2,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 2,
            )
    if JIT_L2_H_TILES >= 4:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 3,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 3,
            )
    if JIT_L2_H_TILES >= 5:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 4,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 4,
            )
    if JIT_L2_H_TILES >= 6:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 5,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 5,
            )
    if JIT_L2_H_TILES >= 7:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 6,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 6,
            )
    if JIT_L2_H_TILES >= 8:
        if JIT_L2_SCALAR != 0:
            _l2_single_cta_scalar_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 7,
            )
        else:
            _l2_single_cta_tile_body(
                l2_acts_e,
                l2_acts_sf_e,
                l2_weights_e,
                l2_weights_sf_e,
                l2_out_e,
                marker,
                M_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                JIT_BLOCK_N * 7,
            )


@triton.jit
def _single_expert_l1_l2_body(
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    POOL_BASE_C: tl.constexpr,
    LOCAL_EXPERT_C: tl.constexpr,
):
    _single_expert_l1_body(
        l1_acts,
        l1_acts_sf,
        l1_topk_weights,
        l1_weights,
        l1_weights_sf,
        l2_acts,
        l2_acts_sf,
        marker,
        M_C,
        H_C,
        I_C,
        NUM_PADDED_M_C,
        POOL_BASE_C,
        LOCAL_EXPERT_C,
    )
    tl.debug_barrier()
    _single_expert_l2_body(
        l2_acts,
        l2_acts_sf,
        l2_weights,
        l2_weights_sf,
        l2_out,
        marker,
        M_C,
        H_C,
        I_C,
        NUM_PADDED_M_C,
        POOL_BASE_C,
        LOCAL_EXPERT_C,
    )


@triton.jit
def _l1_single_kernel_worker(
    compute_reader,
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    wait_result = compute_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))

    _single_expert_l1_l2_body(
        l1_acts,
        l1_acts_sf,
        l1_topk_weights,
        l1_weights,
        l1_weights_sf,
        l2_acts,
        l2_acts_sf,
        l2_weights,
        l2_weights_sf,
        l2_out,
        marker,
        EXPERT0_COUNT_C,
        H_C,
        I_C,
        NUM_PADDED_M_C,
        0,
        0,
    )
    if NUM_EXPERTS_PER_RANK_C >= 2:
        tl.debug_barrier()
        _single_expert_l1_l2_body(
            l1_acts,
            l1_acts_sf,
            l1_topk_weights,
            l1_weights,
            l1_weights_sf,
            l2_acts,
            l2_acts_sf,
            l2_weights,
            l2_weights_sf,
            l2_out,
            marker,
            EXPERT1_COUNT_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            EXPERT1_POOL_BASE_C,
            1,
        )
    tl.debug_barrier()
    tle_raw.call(
        edsl_userhopper_ws_tldot_combine_write,
        [
            symm_buffer,
            l2_out,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
        ],
    )
    tle_raw.call(
        uh.edsl_userhopper_ws_combine_reduce,
        [
            symm_buffer,
            y,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            CLEANUP_WORKSPACE_C,
        ],
    )
    tl.store(marker, 0x4C3307)
    compute_reader.release(0)


@triton.jit
def _l1_l2_expert_wave_single_kernel_worker(
    compute_reader,
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    wait_result = compute_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))

    _single_expert_l1_body(
        l1_acts,
        l1_acts_sf,
        l1_topk_weights,
        l1_weights,
        l1_weights_sf,
        l2_acts,
        l2_acts_sf,
        marker,
        EXPERT0_COUNT_C,
        H_C,
        I_C,
        NUM_PADDED_M_C,
        0,
        0,
    )
    if NUM_EXPERTS_PER_RANK_C >= 2:
        tl.debug_barrier()
        _single_expert_l1_body(
            l1_acts,
            l1_acts_sf,
            l1_topk_weights,
            l1_weights,
            l1_weights_sf,
            l2_acts,
            l2_acts_sf,
            marker,
            EXPERT1_COUNT_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            EXPERT1_POOL_BASE_C,
            1,
        )

    tl.debug_barrier()
    _single_expert_l2_body(
        l2_acts,
        l2_acts_sf,
        l2_weights,
        l2_weights_sf,
        l2_out,
        marker,
        EXPERT0_COUNT_C,
        H_C,
        I_C,
        NUM_PADDED_M_C,
        0,
        0,
    )
    if NUM_EXPERTS_PER_RANK_C >= 2:
        tl.debug_barrier()
        _single_expert_l2_body(
            l2_acts,
            l2_acts_sf,
            l2_weights,
            l2_weights_sf,
            l2_out,
            marker,
            EXPERT1_COUNT_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            EXPERT1_POOL_BASE_C,
            1,
        )

    tl.debug_barrier()
    tle_raw.call(
        edsl_userhopper_ws_tldot_combine_write,
        [
            symm_buffer,
            l2_out,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
        ],
    )
    tle_raw.call(
        uh.edsl_userhopper_ws_combine_reduce,
        [
            symm_buffer,
            y,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            CLEANUP_WORKSPACE_C,
        ],
    )
    tl.store(marker, 0x4C3507)
    compute_reader.release(0)


@triton.jit
def _wait_l1_arrival_one_block(
    symm_buffer,
    M_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    POOL_BASE_C: tl.constexpr,
):
    if M_C > 0:
        arrival_offset: tl.constexpr = 32 + NUM_EXPERTS_C * 8 * 2 + NUM_EXPERTS_PER_RANK_C * 8
        arrival_u32 = (symm_buffer + arrival_offset).to(tl.pointer_type(tl.uint32))
        first_block: tl.constexpr = POOL_BASE_C // 64
        while tl.load(arrival_u32 + first_block, volatile=True) < M_C:
            pass


@triton.jit
def _wait_l1_arrival_one_block_dynamic(
    symm_buffer,
    M,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    pool_base,
):
    if M > 0:
        arrival_offset: tl.constexpr = 32 + NUM_EXPERTS_C * 8 * 2 + NUM_EXPERTS_PER_RANK_C * 8
        arrival_u32 = (symm_buffer + arrival_offset).to(tl.pointer_type(tl.uint32))
        first_block = pool_base // 64
        while tl.load(arrival_u32 + first_block, volatile=True) < M:
            pass


@triton.jit
def _l1_l2_multi_cta_expert_wave_worker(
    compute_reader,
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    cta_done,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    wait_result = compute_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    pid = tl.program_id(0)

    if pid == 1:
        _wait_l1_arrival_one_block(
            symm_buffer,
            EXPERT0_COUNT_C,
            NUM_EXPERTS_C,
            NUM_EXPERTS_PER_RANK_C,
            0,
        )
        _single_expert_l1_body(
            l1_acts,
            l1_acts_sf,
            l1_topk_weights,
            l1_weights,
            l1_weights_sf,
            l2_acts,
            l2_acts_sf,
            marker,
            EXPERT0_COUNT_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            0,
            0,
        )
        if JIT_MULTI_CTA_L1_ONLY == 0:
            tl.debug_barrier()
            _single_expert_l2_body(
                l2_acts,
                l2_acts_sf,
                l2_weights,
                l2_weights_sf,
                l2_out,
                marker,
                EXPERT0_COUNT_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                0,
                0,
            )
        tl.atomic_add(cta_done, 1, sem="release")
    if pid == 2 and NUM_EXPERTS_PER_RANK_C >= 2:
        _wait_l1_arrival_one_block(
            symm_buffer,
            EXPERT1_COUNT_C,
            NUM_EXPERTS_C,
            NUM_EXPERTS_PER_RANK_C,
            EXPERT1_POOL_BASE_C,
        )
        _single_expert_l1_body(
            l1_acts,
            l1_acts_sf,
            l1_topk_weights,
            l1_weights,
            l1_weights_sf,
            l2_acts,
            l2_acts_sf,
            marker,
            EXPERT1_COUNT_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            EXPERT1_POOL_BASE_C,
            1,
        )
        if JIT_MULTI_CTA_L1_ONLY == 0:
            tl.debug_barrier()
            _single_expert_l2_body(
                l2_acts,
                l2_acts_sf,
                l2_weights,
                l2_weights_sf,
                l2_out,
                marker,
                EXPERT1_COUNT_C,
                H_C,
                I_C,
                NUM_PADDED_M_C,
                EXPERT1_POOL_BASE_C,
                1,
            )
        tl.atomic_add(cta_done, 1, sem="release")

    if pid == 0:
        while tl.load(cta_done, volatile=True) < NUM_EXPERTS_PER_RANK_C:
            pass
        if JIT_MULTI_CTA_L1_ONLY == 0:
            tle_raw.call(
                edsl_userhopper_ws_tldot_combine_write,
                [
                    symm_buffer,
                    l2_out,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    H_C,
                    I_C,
                    NUM_PADDED_M_C,
                ],
            )
            tle_raw.call(
                uh.edsl_userhopper_ws_combine_reduce,
                [
                    symm_buffer,
                    y,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    H_C,
                    I_C,
                    NUM_PADDED_M_C,
                    CLEANUP_WORKSPACE_C,
                ],
            )
            tl.store(marker, 0x4C3707)
        else:
            tl.store(marker, 0x4C3711)
    compute_reader.release(0)


@triton.jit
def _l1_tile_split_multi_cta_worker(
    compute_reader,
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    marker,
    cta_done,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
):
    wait_result = compute_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    pid = tl.program_id(0)

    tile_cta = pid - 1
    active_tile_ctas: tl.constexpr = NUM_EXPERTS_PER_RANK_C * JIT_L1_I_TILES
    if tile_cta >= 0 and tile_cta < active_tile_ctas:
        local_expert = tile_cta // JIT_L1_I_TILES
        tile_idx = tile_cta - local_expert * JIT_L1_I_TILES
        count = tl.where(local_expert == 0, EXPERT0_COUNT_C, EXPERT1_COUNT_C)
        pool_base = tl.where(local_expert == 0, 0, EXPERT1_POOL_BASE_C)

        _wait_l1_arrival_one_block_dynamic(
            symm_buffer,
            count,
            NUM_EXPERTS_C,
            NUM_EXPERTS_PER_RANK_C,
            pool_base,
        )

        l1_acts_e = l1_acts + pool_base * H_C
        l1_acts_sf_e = l1_acts_sf + pool_base
        l1_topk_weights_e = l1_topk_weights + pool_base
        l2_acts_e = l2_acts + pool_base * I_C
        l2_acts_sf_e = l2_acts_sf + pool_base
        l1_weights_e = l1_weights + local_expert * (2 * I_C * H_C)
        l1_weights_sf_e = l1_weights_sf + local_expert * ((2 * I_C // 128) * (H_C // 128))
        i_offset = tile_idx * JIT_BLOCK_I

        _l1_single_cta_runtime_m_i_body(
            l1_acts_e,
            l1_acts_sf_e,
            l1_topk_weights_e,
            l1_weights_e,
            l1_weights_sf_e,
            l2_acts_e,
            l2_acts_sf_e,
            marker,
            count,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            i_offset,
        )
        tl.atomic_add(cta_done, 1, sem="release")

    if pid == 0:
        while tl.load(cta_done, volatile=True) < NUM_EXPERTS_PER_RANK_C * JIT_L1_I_TILES:
            pass
        tl.store(marker, 0x4C3712)
    compute_reader.release(0)


@triton.jit
def _l1_l2_tile_split_multi_cta_worker(
    compute_reader,
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    cta_done,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
    SKIP_COMBINE_C: tl.constexpr,
    SKIP_REDUCE_C: tl.constexpr,
):
    wait_result = compute_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    pid = tl.program_id(0)

    l1_tile_ctas: tl.constexpr = NUM_EXPERTS_PER_RANK_C * JIT_L1_I_TILES
    l2_tile_ctas: tl.constexpr = NUM_EXPERTS_PER_RANK_C * JIT_L2_H_TILES
    tile_cta = pid - 1

    if tile_cta >= 0 and tile_cta < l1_tile_ctas:
        local_expert = tile_cta // JIT_L1_I_TILES
        tile_idx = tile_cta - local_expert * JIT_L1_I_TILES
        count = tl.where(local_expert == 0, EXPERT0_COUNT_C, EXPERT1_COUNT_C)
        pool_base = tl.where(local_expert == 0, 0, EXPERT1_POOL_BASE_C)

        _wait_l1_arrival_one_block_dynamic(
            symm_buffer,
            count,
            NUM_EXPERTS_C,
            NUM_EXPERTS_PER_RANK_C,
            pool_base,
        )

        l1_acts_e = l1_acts + pool_base * H_C
        l1_acts_sf_e = l1_acts_sf + pool_base
        l1_topk_weights_e = l1_topk_weights + pool_base
        l2_acts_e = l2_acts + pool_base * I_C
        l2_acts_sf_e = l2_acts_sf + pool_base
        l1_weights_e = l1_weights + local_expert * (2 * I_C * H_C)
        l1_weights_sf_e = l1_weights_sf + local_expert * ((2 * I_C // 128) * (H_C // 128))
        i_offset = tile_idx * JIT_BLOCK_I

        _l1_single_cta_runtime_m_i_body(
            l1_acts_e,
            l1_acts_sf_e,
            l1_topk_weights_e,
            l1_weights_e,
            l1_weights_sf_e,
            l2_acts_e,
            l2_acts_sf_e,
            marker,
            count,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            i_offset,
        )
        tl.atomic_add(cta_done, 1, sem="release")

    l2_tile_cta = tile_cta - l1_tile_ctas
    if l2_tile_cta >= 0 and l2_tile_cta < l2_tile_ctas:
        while tl.load(cta_done, volatile=True) < NUM_EXPERTS_PER_RANK_C * JIT_L1_I_TILES:
            pass

        local_expert = l2_tile_cta // JIT_L2_H_TILES
        hidden_tile = l2_tile_cta - local_expert * JIT_L2_H_TILES
        count = tl.where(local_expert == 0, EXPERT0_COUNT_C, EXPERT1_COUNT_C)
        pool_base = tl.where(local_expert == 0, 0, EXPERT1_POOL_BASE_C)

        l2_acts_e = l2_acts + pool_base * I_C
        l2_acts_sf_e = l2_acts_sf + pool_base
        l2_out_e = l2_out + pool_base * H_C
        l2_weights_e = l2_weights + local_expert * (H_C * I_C)
        l2_weights_sf_e = l2_weights_sf + local_expert * ((H_C // 128) * (I_C // 128))
        n_offset = hidden_tile * JIT_BLOCK_N

        _l2_single_cta_runtime_m_n_body(
            l2_acts_e,
            l2_acts_sf_e,
            l2_weights_e,
            l2_weights_sf_e,
            l2_out_e,
            marker,
            count,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            n_offset,
        )
        tl.atomic_add(cta_done + 1, 1, sem="release")

    if pid == 0:
        while tl.load(cta_done + 1, volatile=True) < NUM_EXPERTS_PER_RANK_C * JIT_L2_H_TILES:
            pass
        if SKIP_COMBINE_C == 0:
            tle_raw.call(
                edsl_userhopper_ws_tldot_combine_write,
                [
                    symm_buffer,
                    l2_out,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    H_C,
                    I_C,
                    NUM_PADDED_M_C,
                ],
            )
            if SKIP_REDUCE_C == 0:
                tle_raw.call(
                    uh.edsl_userhopper_ws_combine_reduce,
                    [
                        symm_buffer,
                        y,
                        NUM_RANKS_C,
                        NUM_EXPERTS_C,
                        NUM_MAX_TOKENS_PER_RANK_C,
                        NUM_TOPK_C,
                        H_C,
                        I_C,
                        NUM_PADDED_M_C,
                        CLEANUP_WORKSPACE_C,
                    ],
                )
        tl.store(marker, 0x4C3713)
    compute_reader.release(0)


@triton.jit
def _l1_expert_wave_split_worker(
    compute_reader,
    l2_writer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
):
    wait_result = compute_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))

    _single_expert_l1_body(
        l1_acts,
        l1_acts_sf,
        l1_topk_weights,
        l1_weights,
        l1_weights_sf,
        l2_acts,
        l2_acts_sf,
        marker,
        EXPERT0_COUNT_C,
        H_C,
        I_C,
        NUM_PADDED_M_C,
        0,
        0,
    )
    if NUM_EXPERTS_PER_RANK_C >= 2:
        tl.debug_barrier()
        _single_expert_l1_body(
            l1_acts,
            l1_acts_sf,
            l1_topk_weights,
            l1_weights,
            l1_weights_sf,
            l2_acts,
            l2_acts_sf,
            marker,
            EXPERT1_COUNT_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            EXPERT1_POOL_BASE_C,
            1,
        )

    tl.debug_barrier()
    l2_slot = l2_writer.acquire(0)
    tl.store(tle.gpu.local_ptr(l2_slot.done, (0,)), 1)
    l2_writer.commit(0)
    tl.store(marker, 0x4C3601)
    compute_reader.release(0)


@triton.jit
def _l2_combine_expert_wave_split_worker(
    l2_reader,
    symm_buffer,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    M_C: tl.constexpr,
    H_C: tl.constexpr,
    I_C: tl.constexpr,
    NUM_PADDED_M_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    wait_result = l2_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))

    _single_expert_l2_body(
        l2_acts,
        l2_acts_sf,
        l2_weights,
        l2_weights_sf,
        l2_out,
        marker,
        EXPERT0_COUNT_C,
        H_C,
        I_C,
        NUM_PADDED_M_C,
        0,
        0,
    )
    if NUM_EXPERTS_PER_RANK_C >= 2:
        tl.debug_barrier()
        _single_expert_l2_body(
            l2_acts,
            l2_acts_sf,
            l2_weights,
            l2_weights_sf,
            l2_out,
            marker,
            EXPERT1_COUNT_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            EXPERT1_POOL_BASE_C,
            1,
        )

    tl.debug_barrier()
    tle_raw.call(
        edsl_userhopper_ws_tldot_combine_write,
        [
            symm_buffer,
            l2_out,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
        ],
    )
    tle_raw.call(
        uh.edsl_userhopper_ws_combine_reduce,
        [
            symm_buffer,
            y,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            H_C,
            I_C,
            NUM_PADDED_M_C,
            CLEANUP_WORKSPACE_C,
        ],
    )
    tl.store(marker, 0x4C3607)
    l2_reader.release(0)


@triton.jit
def _single_kernel_dispatch_receiver_l1_tldot_kernel(
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_dispatch_receiver_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_receiver_compute_sync",
        done=compute_done,
    )
    tle.gpu.warp_specialize(
        [
            (
                uh._dispatch_pipe_partition,
                (
                    dispatch_pipe.writer(),
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                uh._receiver_pipe_to_compute_partition,
                (
                    dispatch_pipe.reader(),
                    compute_pipe.writer(),
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _l1_single_kernel_worker,
                (
                    compute_pipe.reader(),
                    symm_buffer,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y,
                    marker,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                    CLEANUP_WORKSPACE_C,
                ),
            ),
        ],
        [1, 4],
        [80, 180],
    )


@triton.jit
def _single_kernel_dispatch_receiver_l1_l2_expert_wave_tldot_kernel(
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_expert_wave_dispatch_receiver_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_expert_wave_receiver_compute_sync",
        done=compute_done,
    )
    tle.gpu.warp_specialize(
        [
            (
                uh._dispatch_pipe_partition,
                (
                    dispatch_pipe.writer(),
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                uh._receiver_pipe_to_compute_partition,
                (
                    dispatch_pipe.reader(),
                    compute_pipe.writer(),
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _l1_l2_expert_wave_single_kernel_worker,
                (
                    compute_pipe.reader(),
                    symm_buffer,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y,
                    marker,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                    CLEANUP_WORKSPACE_C,
                ),
            ),
        ],
        [4, 4],
        [80, 180],
    )


@triton.jit
def _single_kernel_dispatch_receiver_l1_l2_split_workers_tldot_kernel(
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    l2_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_split_dispatch_receiver_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_split_receiver_l1_sync",
        done=compute_done,
    )
    l2_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_split_l1_l2_sync",
        done=l2_done,
    )
    tle.gpu.warp_specialize(
        [
            (
                uh._dispatch_pipe_partition,
                (
                    dispatch_pipe.writer(),
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                uh._receiver_pipe_to_compute_partition,
                (
                    dispatch_pipe.reader(),
                    compute_pipe.writer(),
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _l1_expert_wave_split_worker,
                (
                    compute_pipe.reader(),
                    l2_pipe.writer(),
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    marker,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                ),
            ),
            (
                _l2_combine_expert_wave_split_worker,
                (
                    l2_pipe.reader(),
                    symm_buffer,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y,
                    marker,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                    CLEANUP_WORKSPACE_C,
                ),
            ),
        ],
        [1, 4, 4],
        [80, 180, 180],
    )


@triton.jit
def _single_kernel_dispatch_receiver_l1_l2_multi_cta_expert_wave_tldot_kernel(
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    cta_done,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_multi_cta_dispatch_receiver_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_multi_cta_receiver_compute_sync",
        done=compute_done,
    )
    tle.gpu.warp_specialize(
        [
            (
                uh._dispatch_pipe_partition,
                (
                    dispatch_pipe.writer(),
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                uh._receiver_pipe_to_compute_partition,
                (
                    dispatch_pipe.reader(),
                    compute_pipe.writer(),
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _l1_l2_multi_cta_expert_wave_worker,
                (
                    compute_pipe.reader(),
                    symm_buffer,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y,
                    marker,
                    cta_done,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                    CLEANUP_WORKSPACE_C,
                ),
            ),
        ],
        [1, 4],
        [80, 180],
    )


@triton.jit
def _single_kernel_dispatch_receiver_l1_tile_split_multi_cta_tldot_kernel(
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    marker,
    cta_done,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_l1_tile_split_dispatch_receiver_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_l1_tile_split_receiver_compute_sync",
        done=compute_done,
    )
    tle.gpu.warp_specialize(
        [
            (
                uh._dispatch_pipe_partition,
                (
                    dispatch_pipe.writer(),
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                uh._receiver_pipe_to_compute_partition,
                (
                    dispatch_pipe.reader(),
                    compute_pipe.writer(),
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _l1_tile_split_multi_cta_worker,
                (
                    compute_pipe.reader(),
                    symm_buffer,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    marker,
                    cta_done,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_EXPERTS_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                ),
            ),
        ],
        [1, 4],
        [80, 180],
    )


@triton.jit
def _single_kernel_dispatch_receiver_l1_l2_tile_split_multi_cta_tldot_kernel(
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    cta_done,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
    SKIP_COMBINE_C: tl.constexpr,
    SKIP_REDUCE_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_l1_l2_tile_split_dispatch_receiver_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_l1_l2_tile_split_receiver_compute_sync",
        done=compute_done,
    )
    tle.gpu.warp_specialize(
        [
            (
                uh._dispatch_pipe_partition,
                (
                    dispatch_pipe.writer(),
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                uh._receiver_pipe_to_compute_partition,
                (
                    dispatch_pipe.reader(),
                    compute_pipe.writer(),
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _l1_l2_tile_split_multi_cta_worker,
                (
                    compute_pipe.reader(),
                    symm_buffer,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y,
                    marker,
                    cta_done,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                    CLEANUP_WORKSPACE_C,
                    SKIP_COMBINE_C,
                    SKIP_REDUCE_C,
                ),
            ),
        ],
        [1, 4],
        [80, 180],
    )


@triton.jit
def _single_kernel_dispatch_receiver_l1_l2_expert_wave_tldot_w2_kernel(
    symm_buffer,
    l1_acts,
    l1_acts_sf,
    l1_topk_weights,
    l1_weights,
    l1_weights_sf,
    l2_acts,
    l2_acts_sf,
    l2_weights,
    l2_weights_sf,
    l2_out,
    y,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    NUM_EXPERTS_PER_RANK_C: tl.constexpr,
    EXPERT0_COUNT_C: tl.constexpr,
    EXPERT1_COUNT_C: tl.constexpr,
    EXPERT1_POOL_BASE_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_expert_wave_w2_dispatch_receiver_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="single_kernel_expert_wave_w2_receiver_compute_sync",
        done=compute_done,
    )
    tle.gpu.warp_specialize(
        [
            (
                uh._dispatch_pipe_partition,
                (
                    dispatch_pipe.writer(),
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                uh._receiver_pipe_to_compute_partition,
                (
                    dispatch_pipe.reader(),
                    compute_pipe.writer(),
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _l1_l2_expert_wave_single_kernel_worker,
                (
                    compute_pipe.reader(),
                    symm_buffer,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y,
                    marker,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    NUM_EXPERTS_PER_RANK_C,
                    EXPERT0_COUNT_C,
                    EXPERT1_COUNT_C,
                    EXPERT1_POOL_BASE_C,
                    CLEANUP_WORKSPACE_C,
                ),
            ),
        ],
        [1, 2],
        [80, 180],
    )


def _check_supported_shape() -> None:
    if uh.NUM_EXPERTS_PER_RANK <= 0 or uh.NUM_EXPERTS_PER_RANK > 2:
        raise SystemExit("this smoke currently supports 1 or 2 local experts per rank")
    if uh.NUM_TOPK <= 0 or uh.NUM_TOPK > uh.NUM_RANKS:
        raise SystemExit("this smoke requires 0 < NUM_TOPK <= NUM_RANKS")
    if uh.HIDDEN not in (128, 256) or uh.INTERMEDIATE_HIDDEN != 128:
        raise SystemExit("this smoke currently requires H in {128, 256} and I=128")
    if uh.HIDDEN == 256 and os.environ.get("USERHOPPER_WS_ALLOW_EXPERIMENTAL_H256_TLDOT") != "1":
        raise SystemExit(
            "H=256 tl.dot path is experimental and currently expected to hit "
            "OutOfResources in the full single-kernel L1+L2 worker; set "
            "USERHOPPER_WS_ALLOW_EXPERIMENTAL_H256_TLDOT=1 to reproduce"
        )
    if PY_BLOCK_M <= 0 or PY_BLOCK_M > DEFAULT_PY_BLOCK_M:
        raise SystemExit(
            f"USERHOPPER_WS_BLOCK_M must be in [1, {DEFAULT_PY_BLOCK_M}], got {PY_BLOCK_M}"
        )
    if PY_BLOCK_N not in (32, 64):
        raise SystemExit(f"USERHOPPER_WS_L2_BLOCK_N must be 32 or 64, got {PY_BLOCK_N}")
    if uh.HIDDEN % PY_BLOCK_N != 0:
        raise SystemExit(f"USERHOPPER_WS_L2_BLOCK_N={PY_BLOCK_N} must divide HIDDEN={uh.HIDDEN}")
    if PY_L1_I_TILES <= 0 or PY_L1_I_TILES > DEFAULT_PY_L1_I_TILES:
        raise SystemExit(
            f"USERHOPPER_WS_L1_I_TILES must be 0 or in [1, {DEFAULT_PY_L1_I_TILES}], "
            f"got {os.environ['USERHOPPER_WS_L1_I_TILES']}"
        )
    if PY_L2_H_TILES <= 0 or PY_L2_H_TILES > DEFAULT_PY_L2_H_TILES:
        raise SystemExit(
            f"USERHOPPER_WS_L2_H_TILES must be 0 or in [1, {DEFAULT_PY_L2_H_TILES}], "
            f"got {os.environ['USERHOPPER_WS_L2_H_TILES']}"
        )
    counts = uh._expected_counts_for_rank(0)
    if any(count > PY_BLOCK_M for count in counts):
        raise SystemExit(f"this smoke requires received token-topk entries per expert <= {PY_BLOCK_M}")
    if any(uh._expected_counts_for_rank(rank) != counts for rank in range(uh.NUM_RANKS)):
        raise SystemExit("this smoke currently requires rank-invariant per-local-expert receive counts")


def _expected_l2_out(acts, acts_sf, weights, weights_sf, count: int):
    acts_f32 = acts[:count].to(torch.float32)
    weights_f32 = weights.to(torch.float32)
    out = torch.zeros((count, uh.HIDDEN), device=acts.device, dtype=torch.float32)
    for k0 in range(0, uh.INTERMEDIATE_HIDDEN, 64):
        a = acts_f32[:, k0:k0 + 64] * acts_sf[:count, k0 // 64][:, None]
        b = weights_f32[:, k0:k0 + 64]
        b = b * weights_sf[:, k0 // 128].repeat_interleave(128)[: uh.HIDDEN][:, None]
        out += a @ b.T
    return out


def main() -> None:
    _check_supported_shape()
    lib = uh._setup_lib()
    lib.userhopper_ws_nvshmem_init_wrapper()
    rank = lib.userhopper_ws_nvshmem_team_mype_wrapper()
    npes = lib.userhopper_ws_nvshmem_n_pes_wrapper()
    if npes != uh.NUM_RANKS:
        raise RuntimeError(f"this smoke expects exactly {uh.NUM_RANKS} PEs, got {npes}")
    per_expert_counts = uh._expected_counts_for_rank(rank)
    expert1_count = per_expert_counts[1] if uh.NUM_EXPERTS_PER_RANK > 1 else 0
    expert1_pool_base = uh._align(per_expert_counts[0], 64)

    torch.cuda.set_device(rank)
    device = triton.runtime.driver.active.get_active_torch_device()
    stream = torch.cuda.Stream(device=device)
    ptr = ctypes.c_void_p(lib.userhopper_ws_nvshmem_alloc_bytes_wrapper(uh.LAYOUT["total_bytes"]))

    whole = uh._view(ptr, 0, (uh.LAYOUT["total_bytes"],), torch.uint8, device)
    x_fp8 = uh._view(
        ptr,
        uh.LAYOUT["input_token"],
        (uh.NUM_MAX_TOKENS_PER_RANK, uh.HIDDEN),
        torch.float8_e4m3fn,
        device,
    )
    x_sf = uh._view(
        ptr,
        uh.LAYOUT["input_sf"],
        (uh.NUM_MAX_TOKENS_PER_RANK, uh.HIDDEN // 128),
        torch.float32,
        device,
    )
    topk_idx = uh._view(
        ptr,
        uh.LAYOUT["input_topk_idx"],
        (uh.NUM_MAX_TOKENS_PER_RANK, uh.NUM_TOPK),
        torch.int64,
        device,
    )
    topk_weights = uh._view(
        ptr,
        uh.LAYOUT["input_topk_weight"],
        (uh.NUM_MAX_TOKENS_PER_RANK, uh.NUM_TOPK),
        torch.float32,
        device,
    )
    l1_acts_u8 = uh._view(
        ptr,
        uh.LAYOUT["l1_token"],
        (uh.LAYOUT["num_max_pool_tokens"], uh.HIDDEN),
        torch.uint8,
        device,
    )
    l1_acts = uh._view(
        ptr,
        uh.LAYOUT["l1_token"],
        (uh.LAYOUT["num_max_pool_tokens"], uh.HIDDEN),
        torch.float8_e4m3fn,
        device,
    )
    l1_acts_sf = uh._view_strided(
        ptr,
        uh.LAYOUT["l1_sf"],
        (uh.LAYOUT["num_max_padded_sf_pool_tokens"], uh.HIDDEN // 128),
        (1, uh.LAYOUT["num_max_padded_sf_pool_tokens"]),
        torch.float32,
        device,
    )
    l1_topk_weights = uh._view(
        ptr,
        uh.LAYOUT["l1_topk_weight"],
        (uh.LAYOUT["num_max_pool_tokens"],),
        torch.float32,
        device,
    )
    l2_acts_u8 = uh._view(
        ptr,
        uh.LAYOUT["l2_token"],
        (uh.LAYOUT["num_max_pool_tokens"], uh.INTERMEDIATE_HIDDEN),
        torch.uint8,
        device,
    )
    l2_acts = uh._view(
        ptr,
        uh.LAYOUT["l2_token"],
        (uh.LAYOUT["num_max_pool_tokens"], uh.INTERMEDIATE_HIDDEN),
        torch.float8_e4m3fn,
        device,
    )
    l2_acts_sf = uh._view_strided(
        ptr,
        uh.LAYOUT["l2_sf"],
        (uh.LAYOUT["num_max_padded_sf_pool_tokens"], uh.INTERMEDIATE_HIDDEN // 64),
        (1, uh.LAYOUT["num_max_padded_sf_pool_tokens"]),
        torch.float32,
        device,
    )
    debug_words = uh._view(ptr, 0, (8,), torch.uint32, device)
    expected_count_words = uh._view(ptr, 0, (uh.NUM_EXPERTS_PER_RANK,), torch.uint32, device)
    send_count = uh._view(ptr, 32, (uh.NUM_EXPERTS * 2,), torch.uint64, device)
    recv_sum = uh._view(
        ptr,
        32 + uh.NUM_EXPERTS * 8 * 2,
        (uh.NUM_EXPERTS_PER_RANK,),
        torch.uint64,
        device,
    )
    arrival_offset = 32 + uh.NUM_EXPERTS * 8 * 2 + uh.NUM_EXPERTS_PER_RANK * 8
    arrival = uh._view(
        ptr,
        arrival_offset,
        (uh._align(uh.LAYOUT["num_max_pool_blocks"], 2),),
        torch.uint32,
        device,
    )
    l2_arrival_mask = uh._view(
        ptr,
        arrival_offset + uh._align(uh.LAYOUT["num_max_pool_blocks"], 2) * 4,
        (uh.LAYOUT["num_max_pool_blocks"],),
        torch.uint64,
        device,
    )
    marker = torch.empty((1,), dtype=torch.int32, device=device)
    cta_done = torch.empty((2,), dtype=torch.int32, device=device)
    l1_weights_u8 = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, 2 * uh.INTERMEDIATE_HIDDEN, uh.HIDDEN),
        dtype=torch.uint8,
        device=device,
    )
    l1_weights = l1_weights_u8.view(torch.float8_e4m3fn)
    l1_weights_sf = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, 2 * uh.INTERMEDIATE_HIDDEN // 128, uh.HIDDEN // 128),
        dtype=torch.float32,
        device=device,
    )
    l2_weights_u8 = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, uh.HIDDEN, uh.INTERMEDIATE_HIDDEN),
        dtype=torch.uint8,
        device=device,
    )
    l2_weights = l2_weights_u8.view(torch.float8_e4m3fn)
    l2_weights_sf = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, uh.HIDDEN // 128, uh.INTERMEDIATE_HIDDEN // 128),
        dtype=torch.float32,
        device=device,
    )
    l2_out = torch.empty(
        (uh.LAYOUT["num_max_pool_tokens"], uh.HIDDEN),
        dtype=torch.float32,
        device=device,
    )
    combine = uh._view(
        ptr,
        uh.LAYOUT["combine_token"],
        (uh.NUM_TOPK, uh.NUM_MAX_TOKENS_PER_RANK, uh.HIDDEN),
        torch.uint16,
        device,
    )
    y = torch.empty((uh.NUM_MAX_TOKENS_PER_RANK, uh.HIDDEN), dtype=torch.bfloat16, device=device)
    y_u8 = y.view(torch.uint8)

    try:
        with torch.cuda.stream(stream):
            whole.zero_()
            marker.zero_()
            l1_weight_values = (
                0.03125
                * (
                    1.0
                    + (
                        torch.arange(l1_weights_u8.numel(), dtype=torch.float32, device=device)
                        % 7.0
                    )
                )
            ).reshape_as(l1_weights)
            l1_weights.copy_(l1_weight_values)
            l1_weights_sf.copy_(
                (
                    1.0
                    + 0.125
                    * torch.arange(l1_weights_sf.numel(), dtype=torch.float32, device=device)
                ).reshape_as(l1_weights_sf)
            )
            l2_weight_values = (
                0.015625
                * (
                    1.0
                    + (
                        torch.arange(l2_weights_u8.numel(), dtype=torch.float32, device=device)
                        % 11.0
                    )
                )
            ).reshape_as(l2_weights)
            l2_weights.copy_(l2_weight_values)
            l2_weights_sf.copy_(
                (
                    0.5
                    + 0.0625
                    * torch.arange(l2_weights_sf.numel(), dtype=torch.float32, device=device)
                ).reshape_as(l2_weights_sf)
            )
            l2_out.zero_()
            y.zero_()
            expected_count_words.copy_(
                torch.tensor(per_expert_counts, dtype=torch.uint32, device=device)
            )
            debug_words[7] = 0x45585043
            for token in range(uh.NUM_TOKENS):
                x_fp8[token].fill_(uh._input_fp8_value(rank, token))
                for sf_idx in range(uh.HIDDEN // 128):
                    x_sf[token, sf_idx] = rank + 0.125 * (token + 1) + 0.01 * sf_idx
                for topk in range(uh.NUM_TOPK):
                    topk_idx[token, topk] = uh._route_expert(rank, topk)
                    topk_weights[token, topk] = rank + 0.25 * (token + 1) + 0.03125 * topk
        stream.synchronize()
        lib.userhopper_ws_nvshmem_barrier_all_wrapper()

        uh._install_cumodule_hook(lib)
        if MULTI_CTA_L2_TILE_SPLIT != 0:
            kernel = _single_kernel_dispatch_receiver_l1_l2_tile_split_multi_cta_tldot_kernel
        elif MULTI_CTA_L1_TILE_SPLIT != 0:
            kernel = _single_kernel_dispatch_receiver_l1_tile_split_multi_cta_tldot_kernel
        elif MULTI_CTA_EXPERT_WAVE != 0:
            kernel = _single_kernel_dispatch_receiver_l1_l2_multi_cta_expert_wave_tldot_kernel
        elif SPLIT_L1_L2_WORKERS != 0:
            kernel = _single_kernel_dispatch_receiver_l1_l2_split_workers_tldot_kernel
        elif EXPERT_WAVE_SINGLE_KERNEL != 0 and EXPERT_WAVE_COMPUTE_WARPS == 2:
            kernel = _single_kernel_dispatch_receiver_l1_l2_expert_wave_tldot_w2_kernel
        elif EXPERT_WAVE_SINGLE_KERNEL != 0:
            kernel = _single_kernel_dispatch_receiver_l1_l2_expert_wave_tldot_kernel
        else:
            kernel = _single_kernel_dispatch_receiver_l1_tldot_kernel
        expected_marker = (
            (
                0x4C3713
                if MULTI_CTA_L2_TILE_SPLIT != 0
                else (
                    0x4C3712
                    if MULTI_CTA_L1_TILE_SPLIT != 0
                    else (0x4C3711 if MULTI_CTA_L1_ONLY != 0 else 0x4C3707)
                )
            )
            if MULTI_CTA_EXPERT_WAVE != 0
            else (
                0x4C3607
                if SPLIT_L1_L2_WORKERS != 0
                else (0x4C3507 if EXPERT_WAVE_SINGLE_KERNEL != 0 else 0x4C3307)
            )
        )
        compute_order = (
            (
                "expert_wave_multi_cta_l1_l2_tile_split"
                if MULTI_CTA_L2_TILE_SPLIT != 0
                else (
                    "expert_wave_multi_cta_l1_tile_split"
                    if MULTI_CTA_L1_TILE_SPLIT != 0
                    else ("expert_wave_multi_cta_l1_only" if MULTI_CTA_L1_ONLY != 0 else "expert_wave_multi_cta")
                )
            )
            if MULTI_CTA_EXPERT_WAVE != 0
            else (
                "expert_wave_split_workers"
                if SPLIT_L1_L2_WORKERS != 0
                else ("expert_wave" if EXPERT_WAVE_SINGLE_KERNEL != 0 else "sequential")
            )
        )
        grid = (
            (1 + uh.NUM_EXPERTS_PER_RANK * PY_L1_I_TILES + uh.NUM_EXPERTS_PER_RANK * PY_L2_H_TILES,)
            if MULTI_CTA_L2_TILE_SPLIT != 0
            else (
                (1 + uh.NUM_EXPERTS_PER_RANK * PY_L1_I_TILES,)
                if MULTI_CTA_L1_TILE_SPLIT != 0
                else ((1 + uh.NUM_EXPERTS_PER_RANK,) if MULTI_CTA_EXPERT_WAVE != 0 else (1,))
            )
        )
        launch_times_us = []
        launch_start = torch.cuda.Event(enable_timing=True)
        launch_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            cta_done.zero_()
            marker.zero_()
            launch_start.record(stream)
            if MULTI_CTA_L2_TILE_SPLIT != 0:
                compiled = kernel[grid](
                    whole,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y_u8,
                    marker,
                    cta_done,
                    uh.NUM_TOKENS,
                    uh.EXPECTED_LOCAL_RECV_TOKENS,
                    uh.NUM_RANKS,
                    uh.NUM_EXPERTS,
                    uh.NUM_MAX_TOKENS_PER_RANK,
                    uh.NUM_TOPK,
                    uh.HIDDEN,
                    uh.INTERMEDIATE_HIDDEN,
                    uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                    uh.NUM_DISPATCH_WARPS,
                    uh.NUM_EXPERTS_PER_RANK,
                    per_expert_counts[0],
                    expert1_count,
                    expert1_pool_base,
                    CLEANUP_WORKSPACE,
                    SKIP_COMBINE,
                    SKIP_REDUCE,
                    num_warps=uh.NUM_WARPS,
                    maxnreg=uh.MAXNREG,
                )
            elif MULTI_CTA_L1_TILE_SPLIT != 0:
                compiled = kernel[grid](
                    whole,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    marker,
                    cta_done,
                    uh.NUM_TOKENS,
                    uh.EXPECTED_LOCAL_RECV_TOKENS,
                    uh.NUM_RANKS,
                    uh.NUM_EXPERTS,
                    uh.NUM_MAX_TOKENS_PER_RANK,
                    uh.NUM_TOPK,
                    uh.HIDDEN,
                    uh.INTERMEDIATE_HIDDEN,
                    uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                    uh.NUM_DISPATCH_WARPS,
                    uh.NUM_EXPERTS_PER_RANK,
                    per_expert_counts[0],
                    expert1_count,
                    expert1_pool_base,
                    num_warps=uh.NUM_WARPS,
                    maxnreg=uh.MAXNREG,
                )
            elif MULTI_CTA_EXPERT_WAVE != 0:
                compiled = kernel[grid](
                    whole,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y_u8,
                    marker,
                    cta_done,
                    uh.NUM_TOKENS,
                    uh.EXPECTED_LOCAL_RECV_TOKENS,
                    uh.NUM_RANKS,
                    uh.NUM_EXPERTS,
                    uh.NUM_MAX_TOKENS_PER_RANK,
                    uh.NUM_TOPK,
                    uh.HIDDEN,
                    uh.INTERMEDIATE_HIDDEN,
                    uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                    uh.NUM_DISPATCH_WARPS,
                    uh.NUM_EXPERTS_PER_RANK,
                    per_expert_counts[0],
                    expert1_count,
                    expert1_pool_base,
                    CLEANUP_WORKSPACE,
                    num_warps=uh.NUM_WARPS,
                    maxnreg=uh.MAXNREG,
                )
            else:
                compiled = kernel[grid](
                whole,
                l1_acts,
                l1_acts_sf,
                l1_topk_weights,
                l1_weights,
                l1_weights_sf,
                l2_acts,
                l2_acts_sf,
                l2_weights,
                l2_weights_sf,
                l2_out,
                y_u8,
                marker,
                uh.NUM_TOKENS,
                uh.EXPECTED_LOCAL_RECV_TOKENS,
                uh.NUM_RANKS,
                uh.NUM_EXPERTS,
                uh.NUM_MAX_TOKENS_PER_RANK,
                uh.NUM_TOPK,
                uh.HIDDEN,
                uh.INTERMEDIATE_HIDDEN,
                uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                uh.NUM_DISPATCH_WARPS,
                uh.NUM_EXPERTS_PER_RANK,
                per_expert_counts[0],
                expert1_count,
                expert1_pool_base,
                CLEANUP_WORKSPACE,
                num_warps=uh.NUM_WARPS,
                maxnreg=uh.MAXNREG,
                )
            launch_end.record(stream)
        stream.synchronize()
        launch_times_us.append(launch_start.elapsed_time(launch_end) * 1000.0)
        lib.userhopper_ws_nvshmem_barrier_all_wrapper()

        if int(marker.cpu()[0]) != expected_marker:
            raise SystemExit(f"single-kernel L1 marker mismatch: {int(marker.cpu()[0])}")
        ttgir = compiled.asm.get("ttgir", "")
        ptx = compiled.asm.get("ptx", "")
        if "ttg.warp_specialize" not in ttgir:
            raise SystemExit("single-kernel TTGIR missing ttg.warp_specialize")
        if "wgmma.mma_async" not in ptx and "mma.sync" not in ptx:
            raise SystemExit("single-kernel PTX missing Tensor Core mma instruction")

        expected_indices, expected_rows, expected_sf, expected_weight, _, _ = uh._expected_receive(rank)
        got_l1 = l1_acts_u8.detach().cpu()[expected_indices]
        got_l1_sf = l1_acts_sf.detach().cpu()[expected_indices, :]
        got_l1_weight = l1_topk_weights.detach().cpu()[expected_indices]
        if not torch.equal(got_l1, expected_rows):
            raise SystemExit("single-kernel receiver L1 token mismatch")
        if not torch.allclose(got_l1_sf, expected_sf):
            raise SystemExit("single-kernel receiver L1 scale mismatch")
        if not torch.allclose(got_l1_weight, expected_weight):
            raise SystemExit("single-kernel receiver L1 topk weight mismatch")

        expected_l2_float = uh._expected_l2_token_floats(
            rank,
            l1_weights_u8.detach().cpu(),
            l1_weights_sf.detach().cpu(),
        )
        expected_l2_sf, expected_l2_scaled = uh._expected_l2_sf_and_scaled_floats(expected_l2_float)
        expected_l2_tokens = uh._float_to_cuda_satfinite_e4m3_bytes(expected_l2_scaled, device)
        got_l2_tokens = l2_acts_u8.detach().cpu()[expected_indices]
        got_l2_sf = l2_acts_sf.detach().cpu()[expected_indices, :]
        computed_intermediate = PY_L1_I_TILES * 64
        full_l1_validation = computed_intermediate >= uh.INTERMEDIATE_HIDDEN
        if not torch.equal(
            got_l2_tokens[:, :computed_intermediate],
            expected_l2_tokens[:, :computed_intermediate],
        ):
            raise SystemExit(
                "single-kernel L1 Tensor Core l2_acts mismatch: "
                f"checked_intermediate={computed_intermediate} "
                f"got={got_l2_tokens[:, :16].tolist()} expected={expected_l2_tokens[:, :16].tolist()}"
            )
        expected_l2_sf_checked = expected_l2_sf[:, :PY_L1_I_TILES]
        got_l2_sf_checked = got_l2_sf[:, :PY_L1_I_TILES]
        l2_sf_tol = max(1e-5, 1e-4 * float(expected_l2_sf_checked.abs().max().item()))
        if not torch.allclose(got_l2_sf_checked, expected_l2_sf_checked, atol=l2_sf_tol, rtol=1e-4):
            max_abs = float((got_l2_sf_checked - expected_l2_sf_checked).abs().max().item())
            raise SystemExit(
                "single-kernel L1 Tensor Core l2_acts_sf mismatch: "
                f"checked_i_tiles={PY_L1_I_TILES} max_abs={max_abs} tol={l2_sf_tol}"
            )

        pool_base = 0
        max_l2_abs = 0.0
        l2_out_tol = 0.0
        computed_hidden = PY_L2_H_TILES * PY_BLOCK_N
        l2_is_computed = MULTI_CTA_L2_TILE_SPLIT != 0 or MULTI_CTA_L1_ONLY == 0
        full_l2_validation = l2_is_computed and computed_hidden >= uh.HIDDEN
        full_output_validation = full_l1_validation and full_l2_validation
        combine_validation = full_output_validation and SKIP_COMBINE == 0
        y_validation = combine_validation and SKIP_REDUCE == 0
        if l2_is_computed:
            for local_expert, count in enumerate(per_expert_counts):
                if count > 0:
                    expected_out = _expected_l2_out(
                        l2_acts[pool_base:pool_base + count],
                        l2_acts_sf[pool_base:],
                        l2_weights[local_expert],
                        l2_weights_sf[local_expert],
                        count,
                    )
                    got_out = l2_out[pool_base:pool_base + count]
                    expected_checked = expected_out[:, :computed_hidden]
                    got_checked = got_out[:, :computed_hidden]
                    expert_max_abs = float((got_checked - expected_checked).abs().max().item())
                    expert_max_ref = float(expected_checked.abs().max().item())
                    expert_tol = max(1e-2, 5e-3 * expert_max_ref)
                    max_l2_abs = max(max_l2_abs, expert_max_abs)
                    l2_out_tol = max(l2_out_tol, expert_tol)
                    if not torch.allclose(got_checked, expected_checked, atol=expert_tol, rtol=5e-3):
                        raise SystemExit(
                            "single-kernel L2 Tensor Core output mismatch: "
                            f"local_expert={local_expert} checked_hidden={computed_hidden} "
                            f"max_abs={expert_max_abs} tol={expert_tol}"
                        )
                pool_base += uh._align(count, 64)

        combine_expected_mask = None
        expected_combine_valid = None
        expected_y = None
        combine_tol = 0.0
        y_tol = 0.0
        if combine_validation:
            expected_combine = uh._expected_combine_float(
                rank,
                l1_weights_u8.detach().cpu(),
                l1_weights_sf.detach().cpu(),
                l2_weights_u8.detach().cpu(),
                l2_weights_sf.detach().cpu(),
                device,
            ).to(torch.bfloat16).float()
            got_combine = combine.detach().cpu().contiguous().view(torch.bfloat16).float()
            combine_expected_mask = torch.zeros(
                (uh.NUM_TOPK, uh.NUM_MAX_TOKENS_PER_RANK), dtype=torch.bool
            )
            for token in range(uh.NUM_TOKENS):
                for topk in range(uh.NUM_TOPK):
                    if uh._route_expert(rank, topk) >= 0:
                        combine_expected_mask[topk, token] = True
            got_combine_valid = got_combine[combine_expected_mask]
            expected_combine_valid = expected_combine[combine_expected_mask]
            combine_tol = max(
                1e-1,
                5e-3 * float(expected_combine_valid.abs().max().item() if expected_combine_valid.numel() else 1.0),
            )
            if not torch.allclose(got_combine_valid, expected_combine_valid, atol=combine_tol, rtol=5e-3):
                raise SystemExit(f"single-kernel combine mismatch: tol={combine_tol}")
            expected_y = got_combine.sum(dim=0).to(torch.bfloat16)
            if y_validation:
                got_y = y.detach().cpu()
                y_tol = max(1e-1, 5e-3 * float(expected_y[: uh.NUM_TOKENS].float().abs().max().item()))
                if not torch.allclose(got_y[: uh.NUM_TOKENS].float(), expected_y[: uh.NUM_TOKENS].float(), atol=y_tol, rtol=5e-3):
                    raise SystemExit(f"single-kernel y mismatch: tol={y_tol}")
        if CLEANUP_WORKSPACE != 0:
            uh._validate_workspace_cleanup(
                send_count.detach().cpu(),
                recv_sum.detach().cpu(),
                arrival.detach().cpu(),
                l2_arrival_mask.detach().cpu(),
            )

        for repeat_idx in range(1, REPEAT_LAUNCHES):
            lib.userhopper_ws_nvshmem_barrier_all_wrapper()
            launch_start = torch.cuda.Event(enable_timing=True)
            launch_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                marker.zero_()
                cta_done.zero_()
                launch_start.record(stream)
                if MULTI_CTA_L2_TILE_SPLIT != 0:
                    compiled = kernel[grid](
                        whole,
                        l1_acts,
                        l1_acts_sf,
                        l1_topk_weights,
                        l1_weights,
                        l1_weights_sf,
                        l2_acts,
                        l2_acts_sf,
                        l2_weights,
                        l2_weights_sf,
                        l2_out,
                        y_u8,
                        marker,
                        cta_done,
                        uh.NUM_TOKENS,
                        uh.EXPECTED_LOCAL_RECV_TOKENS,
                        uh.NUM_RANKS,
                        uh.NUM_EXPERTS,
                        uh.NUM_MAX_TOKENS_PER_RANK,
                        uh.NUM_TOPK,
                        uh.HIDDEN,
                        uh.INTERMEDIATE_HIDDEN,
                        uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                        uh.NUM_DISPATCH_WARPS,
                        uh.NUM_EXPERTS_PER_RANK,
                        per_expert_counts[0],
                        expert1_count,
                        expert1_pool_base,
                        CLEANUP_WORKSPACE,
                        SKIP_COMBINE,
                        SKIP_REDUCE,
                        num_warps=uh.NUM_WARPS,
                        maxnreg=uh.MAXNREG,
                    )
                elif MULTI_CTA_L1_TILE_SPLIT != 0:
                    compiled = kernel[grid](
                        whole,
                        l1_acts,
                        l1_acts_sf,
                        l1_topk_weights,
                        l1_weights,
                        l1_weights_sf,
                        l2_acts,
                        l2_acts_sf,
                        marker,
                        cta_done,
                        uh.NUM_TOKENS,
                        uh.EXPECTED_LOCAL_RECV_TOKENS,
                        uh.NUM_RANKS,
                        uh.NUM_EXPERTS,
                        uh.NUM_MAX_TOKENS_PER_RANK,
                        uh.NUM_TOPK,
                        uh.HIDDEN,
                        uh.INTERMEDIATE_HIDDEN,
                        uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                        uh.NUM_DISPATCH_WARPS,
                        uh.NUM_EXPERTS_PER_RANK,
                        per_expert_counts[0],
                        expert1_count,
                        expert1_pool_base,
                        num_warps=uh.NUM_WARPS,
                        maxnreg=uh.MAXNREG,
                    )
                elif MULTI_CTA_EXPERT_WAVE != 0:
                    compiled = kernel[grid](
                        whole,
                        l1_acts,
                        l1_acts_sf,
                        l1_topk_weights,
                        l1_weights,
                        l1_weights_sf,
                        l2_acts,
                        l2_acts_sf,
                        l2_weights,
                        l2_weights_sf,
                        l2_out,
                        y_u8,
                        marker,
                        cta_done,
                        uh.NUM_TOKENS,
                        uh.EXPECTED_LOCAL_RECV_TOKENS,
                        uh.NUM_RANKS,
                        uh.NUM_EXPERTS,
                        uh.NUM_MAX_TOKENS_PER_RANK,
                        uh.NUM_TOPK,
                        uh.HIDDEN,
                        uh.INTERMEDIATE_HIDDEN,
                        uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                        uh.NUM_DISPATCH_WARPS,
                        uh.NUM_EXPERTS_PER_RANK,
                        per_expert_counts[0],
                        expert1_count,
                        expert1_pool_base,
                        CLEANUP_WORKSPACE,
                        num_warps=uh.NUM_WARPS,
                        maxnreg=uh.MAXNREG,
                    )
                else:
                    compiled = kernel[grid](
                    whole,
                    l1_acts,
                    l1_acts_sf,
                    l1_topk_weights,
                    l1_weights,
                    l1_weights_sf,
                    l2_acts,
                    l2_acts_sf,
                    l2_weights,
                    l2_weights_sf,
                    l2_out,
                    y_u8,
                    marker,
                    uh.NUM_TOKENS,
                    uh.EXPECTED_LOCAL_RECV_TOKENS,
                    uh.NUM_RANKS,
                    uh.NUM_EXPERTS,
                    uh.NUM_MAX_TOKENS_PER_RANK,
                    uh.NUM_TOPK,
                    uh.HIDDEN,
                    uh.INTERMEDIATE_HIDDEN,
                    uh.LAYOUT["num_max_padded_sf_pool_tokens"],
                    uh.NUM_DISPATCH_WARPS,
                    uh.NUM_EXPERTS_PER_RANK,
                    per_expert_counts[0],
                    expert1_count,
                    expert1_pool_base,
                    CLEANUP_WORKSPACE,
                    num_warps=uh.NUM_WARPS,
                    maxnreg=uh.MAXNREG,
                    )
                launch_end.record(stream)
            stream.synchronize()
            launch_times_us.append(launch_start.elapsed_time(launch_end) * 1000.0)
            lib.userhopper_ws_nvshmem_barrier_all_wrapper()
            if int(marker.cpu()[0]) != expected_marker:
                raise SystemExit(
                    f"repeat launch {repeat_idx} marker mismatch: {int(marker.cpu()[0])}"
                )
            uh._validate_workspace_cleanup(
                send_count.detach().cpu(),
                recv_sum.detach().cpu(),
                arrival.detach().cpu(),
                l2_arrival_mask.detach().cpu(),
            )
            if combine_validation:
                repeat_combine = combine.detach().cpu().contiguous().view(torch.bfloat16).float()
                repeat_combine_valid = repeat_combine[combine_expected_mask]
                if not torch.allclose(
                    repeat_combine_valid,
                    expected_combine_valid,
                    atol=combine_tol,
                    rtol=5e-3,
                ):
                    raise SystemExit(f"repeat launch {repeat_idx} combine mismatch: tol={combine_tol}")
                if y_validation:
                    repeat_y = y.detach().cpu()
                    if not torch.allclose(
                        repeat_y[: uh.NUM_TOKENS].float(),
                        expected_y[: uh.NUM_TOKENS].float(),
                        atol=y_tol,
                        rtol=5e-3,
                    ):
                        raise SystemExit(f"repeat launch {repeat_idx} y mismatch: tol={y_tol}")

        steady_launch_times_us = launch_times_us[1:] if len(launch_times_us) > 1 else launch_times_us
        print(
            "rank={} userhopper_single_kernel_l1_l2_combine_tldot_smoke=PASS checked={} "
            "counts={} ws=checked raw_nvshmem=checked tensor_core=checked "
            "combine={} y={} compute_order={} compute_warps={} l1_i_tiles={}/{} "
            "l2_h_tiles={}/{} l2_block_n={} l2_mode={} cleanup={} repeats={} "
            "launch_avg_us={:.3f} launch_min_us={:.3f} launch_max_us={:.3f} "
            "launch_steady_avg_us={:.3f} launch_steady_min_us={:.3f} launch_steady_max_us={:.3f} "
            "l2_max_abs={:.6g} l2_tol={:.6g}".format(
                rank,
                len(expected_indices),
                uh._expected_counts_for_rank(rank),
                "skipped" if SKIP_COMBINE != 0 else ("checked" if full_output_validation else "partial-skip"),
                "skipped" if (SKIP_COMBINE != 0 or SKIP_REDUCE != 0) else ("checked" if full_output_validation else "partial-skip"),
                compute_order,
                EXPERT_WAVE_COMPUTE_WARPS if EXPERT_WAVE_SINGLE_KERNEL != 0 else 4,
                PY_L1_I_TILES,
                DEFAULT_PY_L1_I_TILES,
                PY_L2_H_TILES,
                DEFAULT_PY_L2_H_TILES,
                PY_BLOCK_N,
                "scalar" if L2_SCALAR != 0 else "tensorcore",
                CLEANUP_WORKSPACE,
                REPEAT_LAUNCHES,
                sum(launch_times_us) / len(launch_times_us),
                min(launch_times_us),
                max(launch_times_us),
                sum(steady_launch_times_us) / len(steady_launch_times_us),
                min(steady_launch_times_us),
                max(steady_launch_times_us),
                max_l2_abs,
                l2_out_tol,
            ),
            flush=True,
        )
    finally:
        lib.userhopper_ws_nvshmem_barrier_all_wrapper()
        lib.userhopper_ws_nvshmem_finalize_wrapper(ptr)


if __name__ == "__main__":
    main()
