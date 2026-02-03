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
    num_experts: tl.constexpr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    valid = mask & (expert_id < num_experts)
    rank = tl.atomic_add(cumsum_ptr + expert_id, 1, mask=valid)
    tl.store(sorted_token_ids_ptr + rank, offsets, mask=valid)


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
    NUM_SORT_BLOCKS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    NUM_BLOCKS_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    for i in range(NUM_SORT_BLOCKS):
        base = i * BLOCK_INIT
        offsets = base + tl.arange(0, BLOCK_INIT)
        mask = offsets < total_elems
        tl.store(sorted_token_ids_ptr + offsets, numel, mask=mask)

    offsets = tl.arange(0, BLOCK_TOKENS)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    valid = mask & (expert_id < num_experts)
    expert_id = tl.where(valid, expert_id, 0)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts

    matches = expert_offsets[:, None] == expert_id[None, :]
    token_mask = valid[None, :]
    counts = tl.sum(tl.where(matches & token_mask, 1, 0), axis=1).to(tl.int32)
    aligned = tl.cdiv(counts, BLOCK_SIZE) * BLOCK_SIZE
    cumsum = tl.cumsum(aligned, axis=0)
    base_offsets = cumsum - aligned
    total = tl.sum(aligned, axis=0)
    tl.store(num_tokens_post_pad_ptr, total)

    for base_block in range(0, NUM_BLOCKS_OUT, BLOCK_OUT):
        block_ids = base_block + tl.arange(0, BLOCK_OUT)
        block_valid = block_ids < NUM_BLOCKS_OUT
        block_start = block_ids * BLOCK_SIZE
        block_valid_block = block_valid & (block_start < total)
        block_expert_id = tl.sum(block_start[:, None] >= cumsum[None, :], axis=1)
        block_expert_id = tl.where(block_valid_block, block_expert_id, 0)
        tl.store(expert_ids_ptr + block_ids, block_expert_id, mask=block_valid)

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
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    numel,
    total_elems,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_SORT_BLOCKS: tl.constexpr,
    NUM_BLOCKS_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_EXPERT_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 1:
        for i in range(NUM_SORT_BLOCKS):
            base = i * BLOCK
            offsets = base + tl.arange(0, BLOCK)
            init_mask = offsets < total_elems
            tl.store(sorted_token_ids_ptr + offsets, numel, mask=init_mask)
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

    ones = tl.full((BLOCK, ), 1, tl.int32)
    if NUM_BLOCKS > 0:
        if NUM_BLOCKS > 1:
            for i in range(NUM_BLOCKS - 1):
                base = i * BLOCK
                offsets = base + tl.arange(0, BLOCK)
                expert_id = tl.load(topk_ids_ptr + offsets).to(tl.int32)
                valid = expert_id < num_experts
                expert_id = tl.where(valid, expert_id, 0)
                count_ptrs = tle.local_ptr(smem_counts, (expert_id, ))
                tl.atomic_add(count_ptrs, ones, mask=valid)

        base = (NUM_BLOCKS - 1) * BLOCK
        offsets = base + tl.arange(0, BLOCK)
        tail_mask = offsets < numel
        expert_id = tl.load(topk_ids_ptr + offsets, mask=tail_mask, other=0).to(tl.int32)
        valid = tail_mask & (expert_id < num_experts)
        expert_id = tl.where(valid, expert_id, 0)
        count_ptrs = tle.local_ptr(smem_counts, (expert_id, ))
        tl.atomic_add(count_ptrs, ones, mask=valid)

    tl.debug_barrier()
    counts = tl.load(smem_ptrs)
    counts = tl.where(expert_mask, counts, 0)
    aligned = tl.cdiv(counts, BLOCK_SIZE) * BLOCK_SIZE
    cumsum = tl.cumsum(aligned, axis=0)
    tl.store(cumsum_ptr + 0, 0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum, mask=expert_mask)
    total = tl.sum(aligned, axis=0)
    tl.store(num_tokens_post_pad_ptr, total)

    for expert_base in range(0, BLOCK_EXPERT, BLOCK_EXPERT_GROUP):
        group_offsets = expert_base + tl.arange(0, BLOCK_EXPERT_GROUP)
        group_mask = group_offsets < num_experts
        start = tl.load(cumsum_ptr + group_offsets, mask=group_mask, other=0)
        end = tl.load(cumsum_ptr + group_offsets + 1, mask=group_mask, other=0)
        start_block = start // BLOCK_SIZE
        end_block = end // BLOCK_SIZE
        for base_block in range(0, NUM_BLOCKS_OUT, BLOCK_OUT):
            block_ids = base_block + tl.arange(0, BLOCK_OUT)
            valid_block = block_ids < NUM_BLOCKS_OUT
            block_ids_2d = block_ids[None, :] + tl.zeros((BLOCK_EXPERT_GROUP, 1), tl.int32)
            expert_vals = group_offsets[:, None] + tl.zeros((1, BLOCK_OUT), tl.int32)
            expert_mask_2d = (valid_block[None, :]
                              & group_mask[:, None]
                              & (block_ids_2d >= start_block[:, None])
                              & (block_ids_2d < end_block[:, None]))
            tl.store(expert_ids_ptr + block_ids_2d, expert_vals, mask=expert_mask_2d)


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
        total_elems = sorted_token_ids.numel()
        block_expert = triton.cdiv(num_experts, 32) * 32
        num_blocks_out = triton.cdiv(total_elems, block_size)
        block_init = 256
        block_tokens = triton.next_power_of_2(numel if numel > 0 else 1)
        num_sort_blocks = triton.cdiv(total_elems, block_init)
        moe_align_block_size_vllm_small_batch_kernel[(1, )](
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            num_experts,
            numel,
            total_elems,
            BLOCK_INIT=block_init,
            NUM_SORT_BLOCKS=num_sort_blocks,
            BLOCK_TOKENS=block_tokens,
            NUM_BLOCKS_OUT=num_blocks_out,
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

    expert_ids.fill_(0)

    block_expert = triton.cdiv(num_experts, 32) * 32
    BLOCK_TOKENS = 1024
    total_elems = sorted_token_ids.numel()
    num_blocks = triton.cdiv(numel, BLOCK_TOKENS)
    num_sort_blocks = triton.cdiv(total_elems, BLOCK_TOKENS)
    num_blocks_out = triton.cdiv(total_elems, block_size)

    moe_align_block_size_vllm_stage1_kernel[(2, )](
        topk_ids,
        sorted_token_ids,
        cumsum,
        expert_ids,
        num_tokens_post_pad,
        num_experts,
        numel,
        total_elems,
        BLOCK=BLOCK_TOKENS,
        NUM_BLOCKS=num_blocks,
        NUM_SORT_BLOCKS=num_sort_blocks,
        NUM_BLOCKS_OUT=num_blocks_out,
        BLOCK_SIZE=block_size,
        BLOCK_OUT=128,
        BLOCK_EXPERT=block_expert,
        BLOCK_EXPERT_GROUP=128,
        num_warps=32,
        num_stages=1,
    )

    block_sort = 256
    grid = (triton.cdiv(numel, block_sort), )
    moe_align_block_size_sort_kernel[grid](
        topk_ids,
        sorted_token_ids,
        cumsum,
        num_experts,
        numel,
        BLOCK=block_sort,
        num_warps=8,
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
    if provider == "triton":
        fn = lambda: moe_align_block_size_triton_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids,
                                                      num_tokens_post_pad)
    elif provider == "tle":
        fn = lambda: moe_align_block_size_tle_impl(topk_ids, num_experts, block_size, sorted_ids, expert_ids,
                                                   num_tokens_post_pad)
    else:
        raise ValueError(f"unknown provider: {provider}")

    return triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])


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


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=128, help="MoE block size")
    parser.add_argument("--num_tokens", type=int, default=8192, help="num tokens")
    parser.add_argument("--num_experts", type=int, default=64, help="num experts")
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness checks")
    args = parser.parse_args(argv)

    if not args.skip_correctness:
        run_correctness(args.num_tokens, args.num_experts, args.block_size)

    run_realistic_benchmark(args.block_size)


if __name__ == "__main__":
    main()
