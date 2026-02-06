"""
MoE Align Block Size (TLE Tutorial)
=================================

This tutorial compares two variants of MoE align block size:
- triton: the opt path
- tle: the vllm-like path

It validates correctness and benchmarks performance on synthetic and optional
real data.
"""

# %%
# Setup
# -----

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


# %%
# Kernels (opt path)
# ------------------


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage1_opt(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * BLOCK
    off_c = (pid + 1) * num_experts

    offsets = start_idx + tl.arange(0, BLOCK)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1, mask=mask)


@triton.jit
def moe_align_block_size_stage2_vec(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    cnt = tl.cumsum(token_cnt, axis=0)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, cnt)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts

    expert_offsets = tl.arange(0, num_experts)
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values)

    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
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
    token_idx_in_expert = tl.atomic_add(tokens_cnts_ptr + off_t + expert_id, 1, mask=mask)
    rank_post_pad = token_idx_in_expert + tl.load(cumsum_ptr + expert_id, mask=mask)
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset, mask=mask)


# %%
# Kernels (vllm-like path)
# ------------------------


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_sort_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    cumsum_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0, cache_modifier=".cv")
    rank = tl.atomic_add(cumsum_ptr + expert_id, 1, mask=mask, sem="relaxed")
    tl.store(sorted_token_ids_ptr + rank, offsets, mask=mask)


@triton.jit(do_not_specialize=["numel", "total_elems"])
def moe_align_block_size_vllm_small_batch_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    numel,
    total_elems,
    BLOCK_INIT: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    num_sort_blocks = tl.cdiv(total_elems, BLOCK_INIT)
    num_blocks_out = tl.cdiv(total_elems, BLOCK_SIZE)

    i = 0
    while i < num_sort_blocks:
        base = i * BLOCK_INIT
        offsets = base + tl.arange(0, BLOCK_INIT)
        mask = offsets < total_elems
        tl.store(sorted_token_ids_ptr + offsets, numel, mask=mask)
        i += 1

    offsets = tl.arange(0, BLOCK_TOKENS)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    valid = mask & (expert_id < num_experts)
    expert_id = tl.where(valid, expert_id, 0)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)

    matches = expert_offsets[:, None] == expert_id[None, :]
    token_mask = valid[None, :]
    counts = tl.sum(tl.where(matches & token_mask, 1, 0), axis=1).to(tl.int32)
    aligned = tl.cdiv(counts, BLOCK_SIZE) * BLOCK_SIZE
    cumsum = tl.cumsum(aligned, axis=0)
    base_offsets = cumsum - aligned
    total = tl.sum(aligned, axis=0)
    tl.store(num_tokens_post_pad_ptr, total)

    base_block = 0
    while base_block < num_blocks_out:
        block_ids = base_block + tl.arange(0, BLOCK_OUT)
        block_valid = block_ids < num_blocks_out
        block_start = block_ids * BLOCK_SIZE
        block_valid_block = block_valid & (block_start < total)
        block_expert_id = tl.sum(block_start[:, None] >= cumsum[None, :], axis=1)
        block_expert_id = tl.where(block_valid_block, block_expert_id, 0)
        tl.store(expert_ids_ptr + block_ids, block_expert_id, mask=block_valid)
        base_block += BLOCK_OUT

    prefix = tl.cumsum(tl.where(matches & token_mask, 1, 0), axis=1).to(tl.int32)
    token_rank = tl.sum(tl.where(matches, prefix, 0), axis=0)
    base = tl.sum(tl.where(matches, base_offsets[:, None], 0), axis=0)
    rank = base + token_rank - 1
    tl.store(sorted_token_ids_ptr + rank, offsets, mask=valid)


@triton.jit(do_not_specialize=["numel", "total_elems"])
def moe_align_block_size_vllm_stage1_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    cumsum_ptr,
    num_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    numel,
    max_num_tokens_padded,
    THREADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(numel, THREADS)
    if pid == 1:
        num_sort_blocks = tl.cdiv(max_num_tokens_padded, THREADS)
        i = 0
        while i < num_sort_blocks:
            base = i * THREADS
            offsets = base + tl.arange(0, THREADS)
            init_mask = offsets < max_num_tokens_padded
            tl.store(sorted_token_ids_ptr + offsets, numel, mask=init_mask)
            i += 1
        return

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts

    smem_counts = tle.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    smem_ptrs = tle.local_ptr(smem_counts, (expert_offsets, ))
    tl.store(smem_ptrs, 0)
    tl.debug_barrier()

    ones = tl.full((THREADS, ), 1, tl.int32)
    i = 0
    while i + 1 < num_blocks:
        base = i * THREADS
        offsets = base + tl.arange(0, THREADS)
        expert_id = tl.load(topk_ids_ptr + offsets).to(tl.int32)
        count_ptrs = tle.local_ptr(smem_counts, (expert_id, ))
        tl.atomic_add(count_ptrs, ones, sem="relaxed", scope="cta")
        i += 1

    base = (num_blocks - 1) * THREADS
    offsets = base + tl.arange(0, THREADS)
    tail_mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=tail_mask, other=0).to(tl.int32)
    valid = tail_mask & (expert_id < num_experts)
    expert_id = tl.where(valid, expert_id, 0)
    count_ptrs = tle.local_ptr(smem_counts, (expert_id, ))
    tl.atomic_add(count_ptrs, ones, mask=valid, sem="relaxed", scope="cta")

    counts = tl.load(smem_ptrs)
    counts = tl.where(expert_mask, counts, 0)
    aligned = tl.cdiv(counts, BLOCK_SIZE) * BLOCK_SIZE
    cumsum = tl.cumsum(aligned, axis=0)
    tl.store(cumsum_ptr + 0, 0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum, mask=expert_mask)
    total = tl.sum(aligned, axis=0)
    tl.store(num_tokens_post_pad_ptr, total)


@triton.jit
def moe_align_block_size_vllm_stage2_kernel(
    cumsum_ptr,
    expert_ids_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_VEC: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = tl.load(cumsum_ptr + pid) // BLOCK_SIZE
    end_idx = tl.load(cumsum_ptr + pid + 1) // BLOCK_SIZE
    num_blocks = end_idx - start_idx

    pid_vec = tl.full((BLOCK_VEC, ), pid, tl.int32)
    num_full_blocks = (num_blocks // BLOCK_VEC) * BLOCK_VEC

    i = 0
    while i < num_full_blocks:
        offs = i + tl.arange(0, BLOCK_VEC)
        tl.store(expert_ids_ptr + start_idx + offs, pid_vec)
        i += BLOCK_VEC

    if i < num_blocks:
        offs = i + tl.arange(0, BLOCK_VEC)
        mask = offs < num_blocks
        tl.store(expert_ids_ptr + start_idx + offs, pid_vec, mask=mask)


# %%
# Python wrappers
# ---------------


def _allocate_outputs(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    pad_sorted_ids: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty((max_num_tokens_padded, ), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks, ), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1, ), dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, num_tokens_post_pad


def _launch_common_opt(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    numel_sorted_token_ids = sorted_token_ids.numel()
    numel_expert_ids = expert_ids.numel()

    grid = (num_experts, )
    tokens_cnts = torch.zeros((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.zeros((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))
    block_size_sorted = triton.next_power_of_2(ceil_div(numel_sorted_token_ids, num_experts))
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))

    moe_align_block_size_stage1_opt[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        sorted_token_ids,
        expert_ids,
        numel_sorted_token_ids,
        numel_expert_ids,
        block_size_sorted,
        block_size_expert,
        BLOCK=tokens_per_thread,
    )
    if num_experts == triton.next_power_of_2(num_experts):
        moe_align_block_size_stage2_vec[grid](
            tokens_cnts,
            num_experts,
        )
    else:
        moe_align_block_size_stage2[grid](
            tokens_cnts,
            num_experts,
        )
    moe_align_block_size_stage3[(1, )](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def moe_align_block_size_triton_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    _launch_common_opt(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )


def moe_align_block_size_tle_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    small_batch_expert_mode = (numel < 1024) and (num_experts <= 64)
    if small_batch_expert_mode:
        max_num_tokens_padded = sorted_token_ids.numel()
        block_expert = triton.cdiv(num_experts, 32) * 32
        block_init = 256
        block_tokens = triton.next_power_of_2(numel if numel > 0 else 1)
        moe_align_block_size_vllm_small_batch_kernel[(1, )](
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            num_experts,
            numel,
            max_num_tokens_padded,
            BLOCK_INIT=block_init,
            BLOCK_TOKENS=block_tokens,
            BLOCK_SIZE=block_size,
            BLOCK_OUT=128,
            BLOCK_EXPERT=block_expert,
        )
        return
    if num_experts > 1024:
        _launch_common_opt(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        return
    cumsum = torch.zeros((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)

    block_expert = triton.cdiv(num_experts, 32) * 32
    THREADS = 1024
    max_num_tokens_padded = sorted_token_ids.numel()

    moe_align_block_size_vllm_stage1_kernel[(2, )](
        topk_ids,
        sorted_token_ids,
        cumsum,
        num_tokens_post_pad,
        num_experts,
        numel,
        max_num_tokens_padded,
        THREADS=THREADS,
        BLOCK_SIZE=block_size,
        BLOCK_EXPERT=block_expert,
        num_warps=32,
        num_stages=1,
    )

    grid = (num_experts, )
    moe_align_block_size_vllm_stage2_kernel[grid](
        cumsum,
        expert_ids,
        BLOCK_SIZE=block_size,
        BLOCK_VEC=32,
        num_warps=1,
        num_stages=1,
    )

    block_sort = 256
    grid = (triton.cdiv(numel, block_sort), )
    moe_align_block_size_sort_kernel[grid](
        topk_ids,
        sorted_token_ids,
        cumsum,
        numel,
        BLOCK=block_sort,
        num_warps=block_sort // 32,
        num_stages=1,
    )


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, num_experts, block_size, pad_sorted_ids)
    moe_align_block_size_triton_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size_tle(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, num_experts, block_size, pad_sorted_ids)
    moe_align_block_size_tle_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


# %%
# Correctness
# -----------


def _rand_topk_ids(num_tokens: int, num_experts: int) -> torch.Tensor:
    return torch.randint(0, num_experts, (num_tokens, ), device=DEVICE, dtype=torch.int32)


def run_correctness(
    num_tokens: int,
    num_experts: int,
    block_size: int,
):
    torch.manual_seed(0)
    topk_ids = _rand_topk_ids(num_tokens, num_experts)

    triton_sorted, triton_expert, triton_num_post = moe_align_block_size_triton(topk_ids, block_size, num_experts)
    tle_sorted, tle_expert, tle_num_post = moe_align_block_size_tle(topk_ids, block_size, num_experts)

    torch.testing.assert_close(triton_num_post, tle_num_post)
    num_post = int(triton_num_post.item())
    num_blocks = ceil_div(num_post, block_size)

    torch.testing.assert_close(triton_expert[:num_blocks], tle_expert[:num_blocks])

    counts = torch.bincount(topk_ids, minlength=num_experts)
    aligned = torch.div(counts + (block_size - 1), block_size, rounding_mode="floor") * block_size
    cumsum = torch.cumsum(aligned, dim=0).to(torch.int32)
    torch.testing.assert_close(tle_num_post, cumsum[-1:])

    def _check_sorted(sorted_ids: torch.Tensor) -> None:
        start = 0
        for expert_id in range(num_experts):
            end = int(cumsum[expert_id].item())
            tokens = sorted_ids[start:end]
            valid_mask = tokens < num_tokens
            if counts[expert_id] > 0:
                torch.testing.assert_close(valid_mask.sum(), counts[expert_id])
                torch.testing.assert_close(
                    topk_ids[tokens[valid_mask]],
                    torch.full_like(tokens[valid_mask], expert_id),
                )
            start = end

    _check_sorted(triton_sorted)
    _check_sorted(tle_sorted)

    if num_post < triton_sorted.numel():
        pad_val = triton_sorted[num_post:]
        assert torch.all(pad_val >= num_tokens)

    print("Correctness check passed (triton vs tle).")


# %%
# Benchmark
# ---------


def _moe_shapes() -> List[Tuple[int, int]]:
    deepseek_v32 = [
        (16384, 256),
        (32768, 256),
        (65536, 256),
        (131072, 256),
    ]
    return [
        (128, 16),
        (256, 16),
        (512, 16),
        (512, 64),
        (4096, 64),
        (8192, 64),
        (16384, 128),
        (32768, 128),
        (65536, 256),
    ] + deepseek_v32


def _bench_one(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    provider: str,
) -> Tuple[float, float, float]:
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, num_experts, block_size, False)
    init_fn, kernel_fn = _make_bench_runner(topk_ids, block_size, num_experts, provider, sorted_ids, expert_ids,
                                            num_tokens_post_pad)
    return _bench_kernel_only(init_fn, kernel_fn)


def _make_bench_runner(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    provider: str,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    numel = topk_ids.numel()
    if provider == "triton":
        numel_sorted_token_ids = sorted_ids.numel()
        numel_expert_ids = expert_ids.numel()
        grid = (num_experts, )
        tokens_cnts = torch.empty((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
        cumsum = torch.empty((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)
        tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))
        block_size_sorted = triton.next_power_of_2(ceil_div(numel_sorted_token_ids, num_experts))
        block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))

        def init_fn():
            tokens_cnts.zero_()
            cumsum.zero_()

        def kernel_fn():
            moe_align_block_size_stage1_opt[grid](
                topk_ids,
                tokens_cnts,
                num_experts,
                numel,
                sorted_ids,
                expert_ids,
                numel_sorted_token_ids,
                numel_expert_ids,
                block_size_sorted,
                block_size_expert,
                BLOCK=tokens_per_thread,
            )
            if num_experts == triton.next_power_of_2(num_experts):
                moe_align_block_size_stage2_vec[grid](
                    tokens_cnts,
                    num_experts,
                )
            else:
                moe_align_block_size_stage2[grid](
                    tokens_cnts,
                    num_experts,
                )
            moe_align_block_size_stage3[(1, )](
                num_tokens_post_pad,
                tokens_cnts,
                cumsum,
                num_experts,
                block_size,
            )
            moe_align_block_size_stage4[grid](
                topk_ids,
                sorted_ids,
                expert_ids,
                tokens_cnts,
                cumsum,
                num_experts,
                block_size,
                numel,
                tokens_per_thread,
            )

        return init_fn, kernel_fn

    if provider == "tle":
        small_batch_expert_mode = (numel < 1024) and (num_experts <= 64)
        if small_batch_expert_mode:
            max_num_tokens_padded = sorted_ids.numel()
            block_expert = triton.cdiv(num_experts, 32) * 32
            block_init = 256
            block_tokens = triton.next_power_of_2(numel if numel > 0 else 1)

            def init_fn():
                return

            def kernel_fn():
                moe_align_block_size_vllm_small_batch_kernel[(1, )](
                    topk_ids,
                    sorted_ids,
                    expert_ids,
                    num_tokens_post_pad,
                    num_experts,
                    numel,
                    max_num_tokens_padded,
                    BLOCK_INIT=block_init,
                    BLOCK_TOKENS=block_tokens,
                    BLOCK_SIZE=block_size,
                    BLOCK_OUT=128,
                    BLOCK_EXPERT=block_expert,
                )

            return init_fn, kernel_fn

        if num_experts > 1024:
            return _make_bench_runner(topk_ids, block_size, num_experts, "triton", sorted_ids, expert_ids,
                                      num_tokens_post_pad)

        cumsum = torch.empty((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)
        block_expert = triton.cdiv(num_experts, 32) * 32
        threads = 1024
        max_num_tokens_padded = sorted_ids.numel()
        stage2_grid = (num_experts, )
        block_sort = 256
        sort_grid = (triton.cdiv(numel, block_sort), )

        def init_fn():
            return

        def kernel_fn():
            moe_align_block_size_vllm_stage1_kernel[(2, )](
                topk_ids,
                sorted_ids,
                cumsum,
                num_tokens_post_pad,
                num_experts,
                numel,
                max_num_tokens_padded,
                THREADS=threads,
                BLOCK_SIZE=block_size,
                BLOCK_EXPERT=block_expert,
                num_warps=32,
                num_stages=1,
            )
            moe_align_block_size_vllm_stage2_kernel[stage2_grid](
                cumsum,
                expert_ids,
                BLOCK_SIZE=block_size,
                BLOCK_VEC=32,
                num_warps=1,
                num_stages=1,
            )
            moe_align_block_size_sort_kernel[sort_grid](
                topk_ids,
                sorted_ids,
                cumsum,
                numel,
                BLOCK=block_sort,
                num_warps=block_sort // 32,
                num_stages=1,
            )

        return init_fn, kernel_fn

    raise ValueError(f"unknown provider: {provider}")


def _bench_kernel_only(init_fn, kernel_fn, warmup: int = 20, iters: int = 100) -> Tuple[float, float, float]:
    for _ in range(warmup):
        init_fn()
        kernel_fn()
    torch.cuda.synchronize()

    starts = []
    ends = []
    for _ in range(iters):
        init_fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_fn()
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize()

    times = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=torch.float32)
    return (
        float(torch.quantile(times, 0.5).item()),
        float(torch.quantile(times, 0.2).item()),
        float(torch.quantile(times, 0.8).item()),
    )


def run_benchmark(shapes: Iterable[Tuple[int, int]], block_size: int) -> None:
    providers = ["triton", "tle"]
    print(f"block_size={block_size}")
    header = "num_tokens,num_experts," + ",".join([f"{p}_ms" for p in providers])
    print(header)
    for num_tokens, num_experts in shapes:
        topk_ids = _rand_topk_ids(num_tokens, num_experts)
        sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, num_experts, block_size, False)
        moe_align_block_size_triton_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
        moe_align_block_size_tle_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)

        times_ms = []
        for p in providers:
            ms, _, _ = _bench_one(topk_ids, block_size, num_experts, p)
            times_ms.append(ms)
        row = f"{num_tokens},{num_experts}," + ",".join([f"{t:.4f}" for t in times_ms])
        print(row)


def _zipf_probs(num_experts: int, alpha: float) -> torch.Tensor:
    ranks = torch.arange(1, num_experts + 1, device=DEVICE, dtype=torch.float32)
    probs = 1.0 / (ranks**alpha)
    return probs / probs.sum()


def _sample_topk_ids(num_tokens: int, num_experts: int, probs: torch.Tensor) -> torch.Tensor:
    ids = torch.multinomial(probs, num_tokens, replacement=True)
    return ids.to(torch.int32)


def _moe_realistic_shapes() -> List[Tuple[int, int]]:
    return [
        (256, 512),
        (512, 512),
        (1024, 512),
        (2048, 512),
        (4096, 512),
        (8192, 512),
        (16384, 512),
        (32768, 512),
        (65536, 512),
        (163840, 512),
    ]


def run_realistic_benchmark(block_size: int) -> None:
    providers = ["triton", "tle"]
    print("num_tokens,num_experts,source," + ",".join([f"{p}_ms" for p in providers]))
    for num_tokens, num_experts in _moe_realistic_shapes():
        probs = _zipf_probs(num_experts, alpha=1.2)
        topk_ids = _sample_topk_ids(num_tokens, num_experts, probs)
        times_ms = []
        for p in providers:
            ms, _, _ = _bench_one(topk_ids, block_size, num_experts, p)
            times_ms.append(ms)
        row = f"{num_tokens},{num_experts},zipf," + ",".join([f"{t:.4f}" for t in times_ms])
        print(row)


def _load_real_topk_ids(path: str) -> torch.Tensor:
    path_obj = Path(path)
    if path_obj.is_dir():
        path_obj = path_obj / "topk_ids.pt"
    topk_ids = torch.load(path_obj, map_location=DEVICE)
    if topk_ids.device != DEVICE:
        topk_ids = topk_ids.to(DEVICE)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    return topk_ids.contiguous()


def run_real_data_benchmark(topk_ids_path: str, num_experts: int, block_size: int) -> None:
    providers = ["triton", "tle"]
    topk_ids = _load_real_topk_ids(topk_ids_path)
    num_tokens = topk_ids.numel()
    max_id = int(topk_ids.max().item()) if num_tokens > 0 else -1
    if max_id >= num_experts:
        print(f"warning: max topk_id {max_id} >= num_experts {num_experts}")
    print(f"num_tokens={num_tokens}, num_experts={num_experts}, block_size={block_size}, source=real")
    header = "provider,ms"
    print(header)
    for p in providers:
        ms, _, _ = _bench_one(topk_ids, block_size, num_experts, p)
        print(f"{p},{ms:.4f}")


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=16, help="MoE block size")
    parser.add_argument("--num_tokens", type=int, default=8192, help="num tokens")
    parser.add_argument("--num_experts", type=int, default=64, help="num experts")
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness checks")
    parser.add_argument("--real_data", type=str, default="",
                        help="path to topk_ids.pt (or directory containing it) for real-data benchmark")
    args = parser.parse_args(argv)

    if not args.skip_correctness:
        run_correctness(args.num_tokens, args.num_experts, args.block_size)

    if args.real_data:
        run_real_data_benchmark(args.real_data, args.num_experts, args.block_size)
    else:
        run_realistic_benchmark(args.block_size)


if __name__ == "__main__":
    main()
