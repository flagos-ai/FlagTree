# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""17 fused_moe (MoE fused expert GEMM) — TLE port v1.0 (MetaX C550).

  - plain-half (fp16/bf16), no quantization, silu activation
  - PAIR_GATE_UP_DOT fused GEMM1 (port of _metax_pair_fused_moe_kernel)
  - plain GEMM2 (port of fused_moe_kernel non-quant branch)
  - real moe_align_block_size: 4 triton stages (stage1 atomic counts,
    stage2_vec cumsum, stage3 aligned cumsum, stage4 scatter)
  - fused moe_sum kernel applies the router weights in-kernel

Invariant: gemm1 and gemm2 MUST use the same BLOCK_SIZE_M — both stages
read the same token sort, so a mismatch goes out of bounds on
sorted_token_ids. Enforced by an assertion in fused_experts_impl.

PAIR tile note: the PAIR GEMM1 kernel doubles BLOCK_SIZE_N internally, so
the widest pair tile that fits shared memory is about half the width of the
equivalent plain tile.  That is why gemm1 keeps BN=64 while gemm2 may use
BN=128 or 256.

Retuning method: every tier decision below came from interleaved
measurement (all candidates rotated once per round, median across 5-7
rounds), not from single-shot do_bench.  Single-shot numbers on this part
move by up to 15% between runs, which is larger than any difference being
tuned for, and two early conclusions in this file were artifacts of it.

Accuracy: 20/20 shapes vs fp32 reference at bf16 (worst max_abs 0.00098).
test_correctness sweeps every tier boundary in _qwen_tp2_config and its +1
(M = 1, 8, 9, 64, 384, 385, 896, 897, 1152, 1153, 1664, 1665, 2048, 2049,
3072, 3073, 4096, 4097, 8192, 8193), because the tier table is the thing
being tuned and an unlisted shape is untested.  The reference is an
expert-grouped fp32 chain, checked against the slow token-loop oracle by
test_reference_equivalence (agrees to fp32 rounding).

Two baselines, both driven through their own kernels (bench_ab prints two
speedup columns, gems/TLE and mcoplib/TLE):
  flaggems  FlagGems-vllm/src/flaggems_vllm/runtime/backend/_metax/ops/
            fused_moe.py — its own 5-kernel pipeline and MC550 tier tables.
  mcoplib   /home/flagos-ly/mcoplib/mcoplib/triton_fused_moe.py — the SGL
            GEMM kernel plus moe_sum_reduce.  mcoplib ships no orchestration
            layer, so the pipeline around it is assembled the way SGLang
            drives those kernels; see the _McoplibMoE docstring for exactly
            what was supplied and why it cannot flatter the TLE numbers.

Run:
  python3 17-fused_moe.py --accuracy        # reference cross-check + tier sweep
  python3 17-fused_moe.py --perf            # TLE latency only
  python3 17-fused_moe.py --ab              # flaggems and mcoplib speedups
  python3 17-fused_moe.py                   # all of the above
"""
import sys
import os
from typing import Any

import torch
import triton
import triton.language as tl

# =============================================================================
# Scope: plain-half (fp16/bf16), no quantization, silu, PAIR GEMM1 path.
# Semantics are a faithful port of _metax_pair_fused_moe_kernel +
# fused_moe_kernel (flaggems-vllm _metax backend). Correctness reference:
# fp32 torch GEMM chain (token-loop, one slot at a time).
# =============================================================================


@triton.jit
def _tle_pair_fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_K: tl.constexpr,
    N_DIVISIBLE_BY_BLOCK_N: tl.constexpr,
):
    """GEMM1 PAIR path: one CTA computes gate and up tiles together, silu
    fused in-register. Faithful port of _metax_pair_fused_moe_kernel."""
    pid = tl.program_id(axis=0)
    N_out = N // 2
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_experts == -1:
        # EP: expert not on this rank -> zeros
        offs_zero = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_zero_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_zero[None, :]
        c_zero_mask = token_mask[:, None]
        if not N_DIVISIBLE_BY_BLOCK_N:
            c_zero_mask = c_zero_mask & (offs_zero[None, :] < N_out)
        tl.store(c_zero_ptrs,
                 tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type),
                 mask=c_zero_mask)
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
    offs_pair_bn = tl.where(
        offs_pair < BLOCK_SIZE_N,
        pid_n * BLOCK_SIZE_N + offs_pair,
        N_out + pid_n * BLOCK_SIZE_N + offs_pair - BLOCK_SIZE_N,
    )
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am) + offs_k[None, :] * stride_ak
    b_pair_ptrs = (
        b_ptr + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_pair_bn[None, :] * stride_bn
    )
    pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if K_DIVISIBLE_BY_BLOCK_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            if N_DIVISIBLE_BY_BLOCK_N:
                b_pair = tl.load(b_pair_ptrs)
            else:
                b_pair = tl.load(b_pair_ptrs, mask=offs_pair_bn[None, :] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                        other=0.0)
            b_pair = tl.load(
                b_pair_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_pair_bn[None, :] < N),
                other=0.0,
            )
        pair_acc += tl.dot(a, b_pair)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_pair_ptrs += BLOCK_SIZE_K * stride_bk

    # FlagTree Triton requires permutation dimensions as positional args.
    gate_up = tl.trans(
        tl.reshape(pair_acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N)), 0, 2, 1
    )
    gate_acc, up_acc = tl.split(gate_up)
    gate_sig = tl.sigmoid(gate_acc)
    accumulator = (
        gate_acc.to(compute_type) * gate_sig.to(compute_type) * up_acc.to(compute_type)
    )

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None]
    if not N_DIVISIBLE_BY_BLOCK_N:
        c_mask = c_mask & (offs_cn[None, :] < N_out)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _tle_fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_K: tl.constexpr,
    N_DIVISIBLE_BY_BLOCK_N: tl.constexpr,
):
    """GEMM2 plain path: no pair, no silu (activation already applied)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_experts == -1:
        offs_zero = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_zero_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_zero[None, :]
        c_zero_mask = token_mask[:, None]
        if not N_DIVISIBLE_BY_BLOCK_N:
            c_zero_mask = c_zero_mask & (offs_zero[None, :] < N)
        tl.store(c_zero_ptrs,
                 tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type),
                 mask=c_zero_mask)
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am) + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if K_DIVISIBLE_BY_BLOCK_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            if N_DIVISIBLE_BY_BLOCK_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                        other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None]
    if not N_DIVISIBLE_BY_BLOCK_N:
        c_mask = c_mask & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# =============================================================================
# Host side
# =============================================================================

def _qwen_tp2_config(M: int, gemm_stage: str) -> dict[str, Any]:
    """MC550 plain-half tier table (E256topk8 Qwen3.6 TP2 class).

    Two constraints shape this table and both are enforced elsewhere:
      * gemm1 and gemm2 must share BLOCK_SIZE_M -- they read one token sort
        (asserted in fused_experts_impl);
      * the PAIR GEMM1 tile doubles BLOCK_SIZE_N internally, so a pair tile
        that fits is roughly half as wide as the equivalent plain tile.

    Tiers were measured interleaved (median of 5-7 rounds), not from a single
    do_bench.  Single-shot numbers on this part swing up to 15% run to run,
    which is larger than every difference the tuning is chasing.

    Two boundaries sit where the measurement put them rather than on round
    numbers: the 128x32 gemm2 tile fades between M=3072 and 3328, and the
    BM32 -> BM64 step for small M is monotone over 385..896.
    """
    is_gemm1 = gemm_stage == "gemm1"
    if M <= 8:
        block_m, block_n, block_k = (16, 64, 64) if is_gemm1 else (16, 64, 64)
        group_m, num_warps, num_stages = (1, 4, 4) if is_gemm1 else (1, 4, 3)
    elif M <= 384:
        # gemm2 128x32 wins across 8/32/64/96/128/384: at M=64 it is
        # 0.307 ms vs 0.331 ms for the 64x64 tile.  This tier is the widest
        # one below 385, so the change carries most of the small-M range.
        block_m, block_n, block_k = (16, 64, 64) if is_gemm1 else (16, 128, 32)
        group_m, num_warps, num_stages = (1, 4, 4) if is_gemm1 else (1, 4, 3)
    elif M <= 896:
        # BM32 -> BM64 with gemm2 128x32.  Monotone win across 385/512/640/
        # 768/896; at M=896 it is 0.446 ms vs 0.524 ms for the BM32 tier.
        # The upstream stages=4 is dropped to 3 for the PAIR tile.
        block_m, block_n, block_k = (64, 64, 64) if is_gemm1 else (64, 128, 32)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (8, 4, 3)
    elif M <= 1152:
        block_m, block_n, block_k = (64, 64, 64) if is_gemm1 else (64, 128, 32)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (8, 4, 3)
    elif M <= 1664:
        # PAIR deviation: BN 128 -> 64 keeps BK=64 within the smem limit.
        # BLOCK_SIZE_M pinned to 64 (gemm2's value) for the shared sort.
        block_m, block_n, block_k = (64, 64, 64) if is_gemm1 else (64, 128, 32)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (1, 8, 3)
    elif M <= 2048:
        # gemm2 128x32 at stages=3 beats stages=1 here (0.619 vs 0.650).
        block_m, block_n, block_k = (128, 64, 64) if is_gemm1 else (128, 128, 32)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (2, 4, 3)
    elif M <= 3072:
        block_m, block_n, block_k = (128, 64, 64) if is_gemm1 else (128, 128, 32)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (8, 4, 3)
    elif M <= 4096:
        # gemm2 widens to BN=256 with stages=3: 1.042 ms at M=4096 vs
        # 1.196 ms for the 128x32/stages=1 tile.  BM stays 128 so the shared
        # token sort is unchanged.
        block_m, block_n, block_k = (128, 64, 64) if is_gemm1 else (128, 256, 32)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (1, 8, 3)
    elif M <= 8192:
        # Same PAIR smem deviation as above: BN 128 -> 64 keeps BK=64.
        # Measured 1.600 ms (stages=3) vs 1.720 ms (128,128,32).
        block_m, block_n, block_k = (128, 64, 64) if is_gemm1 else (128, 128, 64)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (8, 4, 1)
    else:
        block_m, block_n, block_k = (128, 64, 64) if is_gemm1 else (128, 128, 64)
        group_m, num_warps, num_stages = (1, 8, 3) if is_gemm1 else (4, 8, 1)
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


# ---- 4-stage moe_align_block_size (port of the non-TLE orchestration) ----

@triton.jit
def _align_stage1(
    topk_ids_ptr, tokens_cnts_ptr,
    num_experts: tl.constexpr, numel,
    tokens_per_thread: tl.constexpr,
    sorted_token_ids_ptr, expert_ids_ptr,
    numel_sorted: tl.constexpr, numel_expert: tl.constexpr,
    block_size_sorted: tl.constexpr, block_size_expert: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_s = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    tl.store(sorted_token_ids_ptr + offs_s, numel, mask=offs_s < numel_sorted)
    offs_e = pid * block_size_expert + tl.arange(0, block_size_expert)
    tl.store(expert_ids_ptr + offs_e, -1, mask=offs_e < numel_expert)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1, mask=mask)


@triton.jit
def _align_stage2_vec(tokens_cnts_ptr, num_experts: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, tl.cumsum(token_cnt, 0))


@triton.jit
def _align_stage2(tokens_cnts_ptr, num_experts: tl.constexpr):
    pid = tl.program_id(0)
    last = 0
    for i in range(1, num_experts + 1):
        last = last + tl.load(tokens_cnts_ptr + i * num_experts + pid)
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last)


@triton.jit
def _align_stage3(
    total_ptr, tokens_cnts_ptr, cumsum_ptr,
    num_experts: tl.constexpr, num_experts_pow2: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts
    eo = tl.arange(0, num_experts_pow2)
    mask = eo < num_experts
    cnts = tl.load(tokens_cnts_ptr + off_cnt + eo, mask=mask, other=0)
    aligned = tl.cdiv(cnts, block_size) * block_size
    tl.store(cumsum_ptr + 1 + eo, tl.cumsum(aligned, 0), mask=mask)
    tl.store(total_ptr, tl.sum(aligned, 0))


@triton.jit(do_not_specialize=["numel"])
def _align_stage4(
    topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    tokens_cnts_ptr, cumsum_ptr,
    num_experts: tl.constexpr, block_size: tl.constexpr,
    numel, tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)
    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)
    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    offset = tl.arange(0, tokens_per_thread) + start_idx
    mask = offset < numel
    expert_id = tl.load(topk_ids_ptr + offset, mask=mask)
    rank = tl.atomic_add(tokens_cnts_ptr + off_t + expert_id, 1, mask=mask)
    rank_post_pad = rank + tl.load(cumsum_ptr + expert_id, mask=mask)
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset, mask=mask)


def _ceil_div(a, b):
    return (a + b - 1) // b


def moe_align_block_size(topk_ids, num_experts, block_size):
    """4-stage sorted-token alignment. Returns (sorted_ids, expert_ids,
    num_tokens_post_pad) matching the flaggems entry point."""
    numel = topk_ids.numel()
    max_padded = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_padded = min(numel * block_size, max_padded)
    sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=topk_ids.device)
    numel_expert = triton.cdiv(max_padded, block_size)
    expert_ids = torch.empty((numel_expert,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    tokens_per_thread = triton.next_power_of_2(max(1, _ceil_div(numel, num_experts)))
    block_sorted = triton.next_power_of_2(max(1, _ceil_div(max_padded, num_experts)))
    block_expert = triton.next_power_of_2(max(1, _ceil_div(numel_expert, num_experts)))
    E_pow2 = triton.next_power_of_2(num_experts)
    grid = (num_experts,)

    _align_stage1[grid](
        topk_ids, tokens_cnts, num_experts, numel, tokens_per_thread,
        sorted_ids, expert_ids, max_padded, numel_expert,
        block_sorted, block_expert,
    )
    if num_experts == E_pow2:
        _align_stage2_vec[grid](tokens_cnts, num_experts)
    else:
        _align_stage2[grid](tokens_cnts, num_experts)
    _align_stage3[(1,)](num_tokens_post_pad, tokens_cnts, cumsum,
                        num_experts, E_pow2, block_size)
    _align_stage4[grid](
        topk_ids, sorted_ids, expert_ids, tokens_cnts, cumsum,
        num_experts, block_size, numel, tokens_per_thread,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


# ---- moe_sum: fused top-k reduce (router weight applied in-kernel) ----

@triton.jit
def _moe_sum_kernel(
    input_ptr, output_ptr, router_weights_ptr,
    rw_stride_token, rw_stride_topk,
    num_tokens, hidden_size: tl.constexpr,
    TOPK: tl.constexpr, APPLY_ROUTER_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    input_base = input_ptr + token_idx * TOPK * hidden_size + hidden_offsets
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for expert_idx in tl.static_range(0, TOPK):
        values = tl.load(input_base + expert_idx * hidden_size, mask=hidden_mask, other=0.0)
        if APPLY_ROUTER_WEIGHT:
            rw = tl.load(router_weights_ptr + token_idx * rw_stride_token
                         + expert_idx * rw_stride_topk)
            values = values.to(tl.float32) * rw.to(tl.float32)
        acc += values.to(tl.float32)
    tl.store(output_ptr + token_idx * hidden_size + hidden_offsets,
             acc.to(output_ptr.dtype.element_ty), mask=hidden_mask)


def moe_sum(input, output, router_weights=None):
    """Top-k reduce, router weight applied in-kernel."""
    num_tokens, topk, hidden_size = input.shape
    if hidden_size <= 256:
        block = 256
    elif hidden_size <= 512:
        block = 512
    else:
        block = 1024
    grid = (num_tokens, triton.cdiv(hidden_size, block))
    rw = input if router_weights is None else router_weights
    _moe_sum_kernel[grid](
        input, output, rw,
        router_weights.stride(0) if router_weights is not None else 0,
        router_weights.stride(1) if router_weights is not None else 0,
        num_tokens, hidden_size,
        TOPK=topk, APPLY_ROUTER_WEIGHT=router_weights is not None,
        BLOCK_SIZE=block, num_warps=8,
    )


def invoke_tle_pair_moe_kernel(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids, expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
):
    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    N = B.size(1)
    K = B.size(2)
    actual_N = N // 2
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(actual_N, META["BLOCK_SIZE_N"]),
    )
    _tle_pair_fused_moe_kernel[grid](
        A, B, C,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        K_DIVISIBLE_BY_BLOCK_K=(K % config["BLOCK_SIZE_K"] == 0),
        N_DIVISIBLE_BY_BLOCK_N=(actual_N % config["BLOCK_SIZE_N"] == 0),
        **config,
    )


def invoke_tle_moe_kernel(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids, expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
):
    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    N = B.size(1)
    K = B.size(2)
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _tle_fused_moe_kernel[grid](
        A, B, C,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        K_DIVISIBLE_BY_BLOCK_K=(K % config["BLOCK_SIZE_K"] == 0),
        N_DIVISIBLE_BY_BLOCK_N=(N % config["BLOCK_SIZE_N"] == 0),
        **config,
    )


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """Plain-half silu MoE: align -> PAIR GEMM1 (silu fused) -> GEMM2 ->
    fused weighted top-k reduce.  No quantization; the naive block
    assignment path is not used here (always sorted token ids)."""
    assert hidden_states.is_contiguous()
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1
    num_tokens = hidden_states.size(0)
    E, N, K = w1.size()
    top_k_num = topk_ids.size(1)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    else:
        raise ValueError(f"Unsupported dtype {hidden_states.dtype}")

    out = hidden_states if inplace else torch.empty_like(hidden_states)
    # PAIR GEMM1 output is the silu-activated intermediate, width N//2.
    cache1 = torch.empty(
        num_tokens * top_k_num * (N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    ).view(num_tokens, top_k_num, N // 2)
    cache3 = torch.empty(  # GEMM2 output, pre-reduce
        num_tokens * top_k_num * K, device=hidden_states.device, dtype=hidden_states.dtype
    ).view(num_tokens, top_k_num, K)

    cfg1 = _qwen_tp2_config(num_tokens, "gemm1")
    cfg2 = _qwen_tp2_config(num_tokens, "gemm2")
    # Both stages read the SAME token sort, so a differing BLOCK_SIZE_M
    # silently misreads sorted_token_ids (flaggems raises on this too).
    if cfg1["BLOCK_SIZE_M"] != cfg2["BLOCK_SIZE_M"]:
        raise ValueError(
            f"gemm1 BLOCK_SIZE_M {cfg1['BLOCK_SIZE_M']} must match the "
            f"base config's {cfg2['BLOCK_SIZE_M']}: both read the same token sort"
        )
    block_m = cfg1["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, E, block_m
    )

    # GEMM1: PAIR fused path. A = raw hidden_states (one row per (token,slot)
    # via offs_token // top_k), silu applied in-register, no routed weight.
    invoke_tle_pair_moe_kernel(
        hidden_states, w1, cache1,
        None,  # no routed weight on input
        sorted_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k_num,
        config=cfg1,
        compute_type=compute_type,
    )

    # GEMM2: plain path. A = silu-activated cache1 (one row per (token,slot),
    # so top_k=1 for the A-row mapping), no routed weight in-kernel.
    invoke_tle_moe_kernel(
        cache1.view(num_tokens * top_k_num, N // 2), w2, cache3,
        None,
        sorted_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=1,
        config=cfg2,
        compute_type=compute_type,
    )
    # top-k reduce with router weight, fused kernel
    moe_sum(cache3, out, router_weights=topk_weights)
    return out


# =============================================================================
# Correctness: fp32 torch reference
# =============================================================================

def reference_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids):
    """fp32 torch chain, token-loop: GEMM1 -> silu -> GEMM2 -> weighted top-k
    reduce. One slot at a time so the top-k reduction axis is explicit.

    This is the slow, obviously-correct version.  It is kept as the oracle
    that reference_fused_experts_fast is validated against.
    """
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    topk = topk_ids.size(1)
    hs = hidden_states.to(torch.float32)
    out = torch.zeros(M, K, device=hidden_states.device, dtype=torch.float32)
    for t in range(M):
        for s in range(topk):
            e = topk_ids[t, s].item()
            gu = hs[t] @ w1[e].float().t()          # [N]
            g, u = gu[: N // 2], gu[N // 2:]
            act = g * torch.sigmoid(g) * u          # [N/2]
            h = act @ w2[e].float().t()             # [K]
            out[t] += h * topk_weights[t, s].float()
    return out.to(hidden_states.dtype)


def reference_fused_experts_fast(hidden_states, w1, w2, topk_weights, topk_ids):
    """Same maths as reference_fused_experts, grouped by expert.

    The token-loop oracle is O(M*topk) Python iterations (65536 at M=8192) and
    too slow to be run at every tier boundary.  This version groups the
    (token, slot) pairs by expert and does one fp32 GEMM per expert, so it
    costs ~E GEMMs regardless of M and needs no [M, N, K] expansion.

    It is NOT the oracle: batched fp32 GEMMs accumulate in a different order
    than the per-token loop, so it agrees only to fp32 rounding.  That is what
    test_reference_equivalence checks before this is used out to M=8192.
    """
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    topk = topk_ids.size(1)
    hs = hidden_states.to(torch.float32)
    w1f = w1.float()
    w2f = w2.float()
    out = torch.zeros(M, K, device=hidden_states.device, dtype=torch.float32)

    flat_ids = topk_ids.reshape(-1).long()          # [M*topk]
    token_of = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
    weight_of = topk_weights.reshape(-1).float()    # [M*topk]

    # One pass per expert; every (token, slot) pair assigned to it is handled
    # as a batch, so peak memory is (pairs_of_one_expert, N).
    for e in range(E):
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        toks = token_of[sel]                        # [n_e]
        a = hs.index_select(0, toks)                # [n_e, K]
        gu = a @ w1f[e].t()                         # [n_e, N]
        g, u = gu[:, : N // 2], gu[:, N // 2:]
        act = g * torch.sigmoid(g) * u              # [n_e, N/2]
        h = act @ w2f[e].t()                        # [n_e, K]
        h = h * weight_of[sel][:, None]
        out.index_add_(0, toks, h)
    return out.to(hidden_states.dtype)


def test_reference_equivalence():
    """The fast reference must agree with the token-loop oracle bit for bit
    (within fp32 rounding) before it is trusted at large M."""
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    for M, E, N, K, topk in [(1, 16, 64, 128, 2), (7, 16, 64, 128, 4),
                             (33, 32, 128, 256, 3)]:
        hs = torch.randn(M, K, device=device, dtype=dtype)
        w1 = torch.randn(E, N, K, device=device, dtype=dtype) * 0.01
        w2 = torch.randn(E, K, N // 2, device=device, dtype=dtype) * 0.01
        tw = torch.rand(M, topk, device=device, dtype=dtype)
        ti = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
        a = reference_fused_experts(hs, w1, w2, tw, ti).float()
        b = reference_fused_experts_fast(hs, w1, w2, tw, ti).float()
        d = (a - b).abs().max().item()
        print(f"  ref-equivalence M={M:4d} topk={topk}: max_abs={d:.8f}")
        assert d < 2e-3, f"fast reference diverges from the oracle at M={M}: {d}"
    print("test_reference_equivalence: passed")


def test_correctness():
    """Every tier boundary in _qwen_tp2_config, plus the M=1 and large-M
    ends.  The tier table is what this file tunes, so a shape that is not
    listed here is untested."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    shapes = [
        # (M, E, N, K, topk) - Qwen TP2 MoE shapes (flaggems PAIR domain)
        # each tier boundary and its +1, so both sides of every step run
        (1, 256, 256, 2048, 8),      # M<=8 tier
        (8, 256, 256, 2048, 8),      # M<=8 / M<=384 edge
        (9, 256, 256, 2048, 8),      # just past M<=8
        (64, 256, 256, 2048, 8),
        (384, 256, 256, 2048, 8),    # M<=384 edge
        (385, 256, 256, 2048, 8),    # just past M<=384
        (896, 256, 256, 2048, 8),    # M<=896 edge
        (897, 256, 256, 2048, 8),    # just past M<=896
        (1152, 256, 256, 2048, 8),   # M<=1152 edge
        (1153, 256, 256, 2048, 8),
        (1664, 256, 256, 2048, 8),   # M<=1664 edge
        (1665, 256, 256, 2048, 8),
        (2048, 256, 256, 2048, 8),   # M<=2048 edge
        (2049, 256, 256, 2048, 8),
        (3072, 256, 256, 2048, 8),   # M<=3072 edge
        (3073, 256, 256, 2048, 8),
        (4096, 256, 256, 2048, 8),   # M<=4096 edge, gemm2 BN=256 tier
        (4097, 256, 256, 2048, 8),
        (8192, 256, 256, 2048, 8),   # M<=8192 edge
        (8193, 256, 256, 2048, 8),   # else tier
    ]
    worst = 0.0
    for M, E, N, K, topk in shapes:
        hidden_states = torch.randn(M, K, device=device, dtype=dtype)
        w1 = torch.randn(E, N, K, device=device, dtype=dtype) * 0.01
        w2 = torch.randn(E, K, N // 2, device=device, dtype=dtype) * 0.01
        topk_weights = torch.rand(M, topk, device=device, dtype=dtype)
        topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
        out = fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids, inplace=False
        )
        ref = reference_fused_experts_fast(
            hidden_states, w1, w2, topk_weights, topk_ids
        )
        diff = (out.float() - ref.float()).abs()
        max_abs = diff.max().item()
        worst = max(worst, max_abs)
        print(f"M={M:5d} E={E} N={N} K={K} topk={topk} "
              f"max_abs={max_abs:.6f}")
        assert max_abs < 0.05, f"shape M={M}: max_abs {max_abs} too large"
    print(f"test_correctness: all {len(shapes)} passed (worst max_abs={worst:.6f})")


def bench(only_m=None):
    import triton.testing as testing
    shapes = [
        (8, 256, 256, 2048, 8),
        (64, 256, 256, 2048, 8),
        (512, 256, 256, 2048, 8),
        (2048, 256, 256, 2048, 8),
        (4096, 256, 256, 2048, 8),
        (8192, 256, 256, 2048, 8),
    ]
    if only_m:
        shapes = [s for s in shapes if s[0] in only_m]
    for M, E, N, K, topk in shapes:
        hidden_states = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.01
        w2 = torch.randn(E, K, N // 2, device="cuda", dtype=torch.bfloat16) * 0.01
        topk_weights = torch.rand(M, topk, device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.randint(0, E, (M, topk), device="cuda", dtype=torch.int32)

        def run():
            fused_experts_impl(
                hidden_states, w1, w2, topk_weights, topk_ids, inplace=False
            )
        ms = testing.do_bench(run, warmup=10, rep=50, return_mode="median")
        print(f"M={M:5d} fused_moe bf16: {ms:.4f} ms")


# =============================================================================
# Two baselines, both driven through their own kernels
#
#   flaggems: FlagGems-vllm _metax backend fused_experts_impl, called as-is:
#             its own 5-kernel pipeline plus the MC550-tuned per-M tiers.
#   mcoplib : /home/flagos-ly/mcoplib/mcoplib/triton_fused_moe.py — the SGL
#             GEMM kernel (sgl_fused_moe_kernel) and moe_sum_reduce_triton.
#             mcoplib ships no orchestration layer (no moe_align_block_size,
#             no silu_and_mul, no fused_experts_impl), so the pipeline below
#             assembles it the way SGLang drives those kernels:
#                 align -> GEMM1 -> silu*mul -> GEMM2 -> moe_sum_reduce
#             Two choices keep this baseline charitable, i.e. they can only
#             shrink the TLE speedup:
#               * alignment reuses the TLE 4-stage align kernel rather than a
#                 slower stand-in, and
#               * each stage's tile config is swept over MCOPLIB_CANDIDATES.
#             BLOCK_SIZE_M is shared by both stages (same token sort), so the
#             sweep is per-BM with the stages optimised jointly.
# =============================================================================

FLAGGEMS_SRC = "/home/flagos-ly/FlagGems-vllm/src"
MCOPLIB_ROOT = "/home/flagos-ly/mcoplib"

# (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages)
MCOPLIB_CANDIDATES = [
    (16, 64, 64, 1, 4, 3),
    (16, 128, 64, 1, 4, 3),
    (32, 64, 64, 8, 4, 3),
    (32, 128, 64, 8, 8, 3),
    (64, 64, 64, 8, 8, 3),
    (64, 128, 64, 1, 8, 3),
    (64, 128, 32, 8, 4, 3),
    (128, 64, 64, 8, 8, 3),
    (128, 128, 64, 8, 8, 3),
    (128, 128, 32, 8, 4, 4),
]

# No static shared-memory pre-filter here.  Estimating
# (BM*BK + BK*BN) * dtype * stages rejects configurations the compiler
# actually fits, and rejecting them silently biases every speedup in the
# TLE's favour.  Candidates are tried for real and pruned on failure
# instead -- the sweep is the source of truth.


def _load_path(p, name):
    if not os.path.isdir(p):
        print(f"[load] {name}: missing {p} -> column n/a")
        return False
    if p not in sys.path:
        sys.path.insert(0, p)
    return True


def _load_flaggems():
    """Soft-import the flaggems fused_moe module; None on any failure."""
    if not _load_path(FLAGGEMS_SRC, "flaggems"):
        return None
    try:
        from flaggems_vllm.runtime.backend._metax.ops import fused_moe as vfm
        return vfm
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _load_mcoplib():
    """Soft-import the mcoplib GEMM + reduce kernels."""
    if not _load_path(MCOPLIB_ROOT, "mcoplib"):
        return None
    try:
        from mcoplib.triton_fused_moe import (
            sgl_invoke_fused_moe_kernel,
            _moe_sum_reduce_kernel,
        )
        return sgl_invoke_fused_moe_kernel, _moe_sum_reduce_kernel
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _load_silu():
    """The fused silu*mul that SGLang feeds between the two GEMMs.

    Falls back to a torch chain only if flaggems is unavailable; the fallback
    is the slower (3-pass) form, so it never flatters the mcoplib baseline.
    """
    if not _load_path(FLAGGEMS_SRC, "flaggems"):
        return None
    try:
        from flaggems_vllm.ops import silu_and_mul_out
        return silu_and_mul_out
    except Exception:
        return None


# mcoplib's moe_sum_reduce_triton wrapper launches num_warps=16, i.e. 1024
# threads with the 64-wide waves this backend uses.  C550 caps a CTA at 512
# threads, so that wrapper cannot run here at all.  Launch the same kernel
# with every tuned parameter unchanged except the warp count, clamped to the
# hardware maximum -- otherwise this baseline has no reduce stage.
MC_MAX_WARPS = 8


def _mcoplib_sum_reduce(input_3d, output, kernel):
    token_num, topk_num, hidden_dim = input_3d.shape
    block_dim = 2048
    grid = (token_num, triton.cdiv(hidden_dim, block_dim))
    kernel[grid](
        input_3d,
        *input_3d.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=1.0,
        BLOCK_M=1,
        BLOCK_DIM=block_dim,
        NUM_STAGE=1,
        num_warps=MC_MAX_WARPS,
    )


class _McoplibMoE:
    """mcoplib GEMM kernels driven the way SGLang drives them.

    What mcoplib actually ships is the GEMM kernel (sgl_fused_moe_kernel) and
    the reduce kernel.  It has no moe_align_block_size, no silu_and_mul and no
    fused_experts_impl, and vllm is not installed here, so a benchmark of "the
    mcoplib baseline" has to supply the rest.  What this class supplies, and
    why none of it can flatter the TLE numbers:

      align    reuses the TLE 4-stage alignment kernel.  Using the TLE's own
               align can only make the baseline faster.
      silu     flaggems silu_and_mul_out, the fused kernel SGLang feeds
               between the two GEMMs.  Measured faster than a torch chain at
               every M tested (1.6x at M=1 through 2.3x at M=8192), so the
               fallback is strictly slower.
      reduce   mcoplib's own _moe_sum_reduce_kernel.  Its wrapper hardcodes
               num_warps=16, which is 1024 threads at this backend's 64-wide
               waves and cannot launch on C550 (CTA cap 512).  Only the warp
               count is clamped, to 8; BLOCK_DIM/ NUM_STAGE / BLOCK_M and the
               launch maths are mcoplib's.

    Config search: candidates are tried for real and pruned on failure.  An
    earlier version pre-filtered on an estimated (BM*BK + BK*BN)*2*stages
    shared-memory figure; that rejected BM=128 tiles the compiler actually
    fits, which silently weakened the baseline.  Do not reinstate it.
    """

    def __init__(self, kernels, silu_out):
        self._invoke, self._reduce_kernel = kernels
        self._sum_reduce = _mcoplib_sum_reduce
        self._silu_out = silu_out
        self._cfg_cache = {}

    def _silu_mul(self, cache1, cache2, rows, act_w):
        """silu(gate) * up, gate/up being the two halves of the GEMM1 output."""
        g = cache1[..., :act_w].reshape(rows, act_w)
        u = cache1[..., act_w:].reshape(rows, act_w)
        if self._silu_out is not None:
            self._silu_out(g, u, cache2)
        else:
            cache2.copy_(torch.nn.functional.silu(g).mul(u))
        return cache2

    def _gemm(self, A, B, C, tw, ti, sorted_ids, expert_ids, ntpp,
              mul_w, top_k, cfg, compute_type):
        self._invoke(
            A, B, None, C,
            None, None, None,                 # A_scale, B_scale, B_zp
            tw,
            ti,
            sorted_ids, expert_ids, ntpp,
            mul_w, top_k,
            config=cfg,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

    def _cands(self, bm, n_out):
        out = []
        for c in MCOPLIB_CANDIDATES:
            if c[0] != bm:
                continue
            if c[1] > n_out:
                continue
            out.append(c)
        return out

    def _sweep(self, run_fn, cands):
        import triton.testing as testing
        best = None
        for (_, bn, bk, gm, nw, ns) in cands:
            cfg = dict(BLOCK_SIZE_M=cands[0][0], BLOCK_SIZE_N=bn,
                       BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm,
                       num_warps=nw, num_stages=ns)
            try:
                ms = testing.do_bench(lambda: run_fn(cfg), warmup=10, rep=30,
                                      return_mode="median")
            except Exception:
                continue
            if best is None or ms < best[0]:
                best = (ms, cfg)
        return best

    def autotune(self, hs, w1, w2, tw, ti, compute_type):
        """Per-BM joint search over (gemm1 tile, gemm2 tile), scored by
        align + gemm1 + gemm2 wall time."""
        import triton.testing as testing
        M = hs.shape[0]
        E, N, K = w1.size()
        topk = ti.size(1)
        rows = M * topk
        act_w = N // 2
        best = None
        for bm in (16, 32, 64, 128):
            c1 = self._cands(bm, act_w)
            c2 = self._cands(bm, K)
            if not c1 or not c2:
                continue
            sorted_ids, expert_ids, ntpp = moe_align_block_size(ti, E, bm)
            cache1 = torch.empty(rows * N, device=hs.device, dtype=hs.dtype).view(M, topk, N)
            cache2 = torch.empty(rows * act_w, device=hs.device, dtype=hs.dtype).view(rows, act_w)
            cache3 = torch.empty(rows * K, device=hs.device, dtype=hs.dtype).view(M, topk, K)

            t_align = testing.do_bench(
                lambda: moe_align_block_size(ti, E, bm), warmup=10, rep=30,
                return_mode="median")

            b1 = self._sweep(
                lambda cfg: self._gemm(hs, w1, cache1, None, ti, sorted_ids,
                                       expert_ids, ntpp, False, topk, cfg, compute_type),
                c1)
            if b1 is None:
                continue
            self._silu_mul(cache1, cache2, rows, act_w)

            tw2 = tw.reshape(rows, 1)
            b2 = self._sweep(
                lambda cfg: self._gemm(cache2, w2, cache3, tw2, ti, sorted_ids,
                                       expert_ids, ntpp, True, 1, cfg, compute_type),
                c2)
            if b2 is None:
                continue
            total = t_align + b1[0] + b2[0]
            if best is None or total < best[0]:
                best = (total, dict(bm=bm, cfg1=b1[1], cfg2=b2[1],
                                    align_ms=t_align, g1_ms=b1[0], g2_ms=b2[0]))
        return best[1] if best else None

    def config_for(self, hs, w1, w2, tw, ti):
        key = (hs.size(0), tuple(w1.shape), tuple(w2.shape), ti.size(1))
        if key not in self._cfg_cache:
            self._cfg_cache[key] = self.autotune(
                hs, w1, w2, tw, ti, tl.bfloat16)
        return self._cfg_cache[key]

    def __call__(self, hs, w1, w2, tw, ti, cfg):
        M = hs.shape[0]
        E, N, K = w1.size()
        topk = ti.size(1)
        rows = M * topk
        act_w = N // 2

        sorted_ids, expert_ids, ntpp = moe_align_block_size(ti, E, cfg["bm"])
        cache1 = torch.empty(rows * N, device=hs.device, dtype=hs.dtype).view(M, topk, N)
        cache2 = torch.empty(rows * act_w, device=hs.device, dtype=hs.dtype).view(rows, act_w)
        cache3 = torch.empty(rows * K, device=hs.device, dtype=hs.dtype).view(M, topk, K)
        out = torch.empty_like(hs)

        self._gemm(hs, w1, cache1, None, ti, sorted_ids, expert_ids, ntpp,
                   False, topk, cfg["cfg1"], tl.bfloat16)
        self._silu_mul(cache1, cache2, rows, act_w)
        self._gemm(cache2, w2, cache3, tw.reshape(rows, 1), ti, sorted_ids,
                   expert_ids, ntpp, True, 1, cfg["cfg2"], tl.bfloat16)
        self._sum_reduce(cache3, out, self._reduce_kernel)
        return out


def _bench_interleaved(fns, rounds=3, warmup=10, rep=50):
    """Interleaved measurement: rotate through every closure once per round
    and take the per-closure median across rounds.  Beats sequential
    do_bench pairs, which pick up the clock ladder and ordering bias."""
    import triton.testing as testing
    acc = [[] for _ in fns]
    for _ in range(rounds):
        for i, fn in enumerate(fns):
            acc[i].append(
                testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))
    return [sorted(a)[len(a) // 2] for a in acc]


def bench_ab(only_m=None, rounds=3):
    gems = _load_flaggems()
    mc = _load_mcoplib()
    mc_impl = _McoplibMoE(mc, _load_silu()) if mc else None

    shapes = [
        (1, 256, 256, 2048, 8),
        (8, 256, 256, 2048, 8),
        (64, 256, 256, 2048, 8),
        (512, 256, 256, 2048, 8),
        (2048, 256, 256, 2048, 8),
        (4096, 256, 256, 2048, 8),
        (8192, 256, 256, 2048, 8),
    ]
    if only_m:
        shapes = [s for s in shapes if s[0] in only_m]

    print(f"{'M':>6} {'TLE ms':>9} "
          f"{'flaggems ms':>11} {'gems/TLE':>8} "
          f"{'mcop ms':>9} {'mcp/TLE':>8} {'match':>6}  mcoplib config")
    print("-" * 97)

    for M, E, N, K, topk in shapes:
        torch.manual_seed(0)
        hs = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.01
        w2 = torch.randn(E, K, N // 2, device="cuda", dtype=torch.bfloat16) * 0.01
        gating = torch.randn(M, E, device="cuda", dtype=torch.float32)
        tw, ti = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.bfloat16).contiguous()
        ti = ti.to(torch.int32).contiguous()

        fn_t = lambda: fused_experts_impl(hs, w1, w2, tw, ti, inplace=False)
        out_t = fn_t()

        fns = [fn_t]
        out_g = out_m = None
        if gems is not None:
            def fn_g():
                return gems.fused_experts_impl(hs, w1, w2, tw, ti, inplace=False)
            out_g = fn_g()
            fns.append(fn_g)

        mc_cfg = None
        if mc_impl is not None:
            mc_cfg = mc_impl.config_for(hs, w1, w2, tw, ti)
            if mc_cfg is None:
                # no candidate compiled: fall back to a plain 64x64x64 tile
                plain = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                             GROUP_SIZE_M=8, num_warps=8, num_stages=3)
                mc_cfg = dict(bm=64, cfg1=plain, cfg2=plain)

            def fn_m(cfg=mc_cfg):
                return mc_impl(hs, w1, w2, tw, ti, cfg)
            out_m = fn_m()
            fns.append(fn_m)

        ms = _bench_interleaved(fns, rounds=rounds)
        t_tle = ms[0]
        i = 1
        s_g = s_m = float("nan")
        ms_g = ms_m = float("nan")
        if gems is not None:
            ms_g = ms[i]; s_g = ms_g / t_tle; i += 1
        if mc_impl is not None:
            ms_m = ms[i]; s_m = ms_m / t_tle

        match = "-"
        if out_g is not None or out_m is not None:
            marks = []
            if out_g is not None:
                marks.append("g" if torch.allclose(
                    out_t.float(), out_g.float(), atol=0.05, rtol=0.05) else "x")
            if out_m is not None:
                marks.append("m" if torch.allclose(
                    out_t.float(), out_m.float(), atol=0.05, rtol=0.05) else "x")
            match = "".join(marks)

        cfgstr = ""
        if mc_cfg is not None:
            c1, c2 = mc_cfg["cfg1"], mc_cfg["cfg2"]
            cfgstr = (f"BM{mc_cfg['bm']} "
                      f"g1={c1['BLOCK_SIZE_N']}x{c1['BLOCK_SIZE_K']} "
                      f"g2={c2['BLOCK_SIZE_N']}x{c2['BLOCK_SIZE_K']}")

        def fmt(x):
            return f"{x:8.3f}" if x == x else f"{'n/a':>8}"

        print(f"{M:>6} {t_tle:9.4f} {ms_g:11.4f} {fmt(s_g)} "
              f"{ms_m:9.4f} {fmt(s_m)} {match:>6}  {cfgstr}")

    print("\nmatch: 'g' = TLE==flaggems, 'm' = TLE==mcoplib, 'x' = mismatch")
    print("both speedup columns > 1 means the shape needs no further work")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="17 fused_moe: accuracy and performance tests",
        epilog=(
            "examples:\n"
            "  python3 17-fused_moe.py --accuracy       # accuracy only\n"
            "  python3 17-fused_moe.py --perf           # TLE latency only\n"
            "  python3 17-fused_moe.py --ab             # three-way speedup only\n"
            "  python3 17-fused_moe.py                  # everything\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--accuracy", action="store_true",
                    help="accuracy only: reference cross-check + all tier "
                         "boundaries vs fp32")
    ap.add_argument("--perf", action="store_true",
                    help="TLE latency only (no baselines)")
    ap.add_argument("--ab", action="store_true",
                    help="three-way speedup only (flaggems and mcoplib vs TLE)")
    ap.add_argument("--shapes", default="",
                    help="comma separated M values for --perf / --ab, "
                         "e.g. 64,512")
    ap.add_argument("--rounds", type=int, default=3,
                    help="interleaved A/B rounds (default 3)")
    a = ap.parse_args()

    only = {int(x) for x in a.shapes.split(",") if x.strip()} or None
    # No selector -> run everything.
    run_all = not (a.accuracy or a.perf or a.ab)

    if a.accuracy or run_all:
        test_reference_equivalence()
        test_correctness()
    if a.perf or run_all:
        bench(only_m=only)
    if a.ab or run_all:
        bench_ab(only_m=only, rounds=a.rounds)
