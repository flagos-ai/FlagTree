"""
MoE Align Block Size (TLE Tutorial)
=================================

This tutorial benchmarks multiple MoE align implementations:
- triton baseline (stage1 + stage2)
- triton atomic (stage1+stage2 fused)
- triton atomic fused (stage1+2 + stage3+4 fused)
- tle cluster fused
- sglang cuda
- yiakwy-xpu cuda (cooperative multi-block)
"""

import argparse
import hashlib
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import urllib.request

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()
BLOCK_CLUSTER_MESH_8 = tle.device_mesh({"block_cluster": [("cluster_x", 8)]})
TLE_CLUSTER_SIZE = 8
SGLANG_MOE_ALIGN_KERNEL_URL = (
    "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/sgl-kernel/csrc/moe/moe_align_kernel.cu")
YIAKWY_MOE_ALIGN_KERNEL_URL = (
    "https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/AMD-sglang-benchmark-fork/"
    "d9831e330bd312fc00557187834d0f5b12ea5c70/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu")


@lru_cache(maxsize=64)
def _block_mesh(num_blocks: int):
    return tle.device_mesh({"block": [("block_x", int(num_blocks))]})


_TRITON_ALLOCATOR_INSTALLED = False


def _install_triton_default_allocator() -> None:
    global _TRITON_ALLOCATOR_INSTALLED
    if _TRITON_ALLOCATOR_INSTALLED:
        return

    def _alloc(size: int, _alignment: int, _stream: Optional[int]):
        return torch.empty((size, ), dtype=torch.uint8, device=DEVICE)

    triton.set_allocator(_alloc)
    _TRITON_ALLOCATOR_INSTALLED = True


SGLANG_MOE_ALIGN_BINDING_CPP = r"""
#include <torch/extension.h>

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    bool pad_sorted_token_ids);

void moe_align_block_size_embedded(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad) {
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  // SGLang kernel uses expert_id = topk_id + 1 and reserves slot 0 as a dummy.
  // Pass (num_experts + 1) to cover all valid experts [0, num_experts).
  const int64_t num_experts_with_dummy = num_experts + 1;
  auto cumsum_buffer = torch::zeros({num_experts_with_dummy + 1}, opts);
  moe_align_block_size(
      topk_ids,
      num_experts_with_dummy,
      block_size,
      sorted_token_ids,
      experts_ids,
      num_tokens_post_pad,
      cumsum_buffer,
      true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_block_size", &moe_align_block_size_embedded, "Embedded sglang moe_align_block_size");
}
"""

YIAKWY_MOE_ALIGN_BINDING_CPP = r"""
#include <torch/extension.h>

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer);

void moe_align_block_size_embedded(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer) {
  moe_align_block_size(
      topk_ids,
      num_experts,
      block_size,
      sorted_token_ids,
      experts_ids,
      num_tokens_post_pad,
      token_cnts_buffer,
      cumsum_buffer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_block_size", &moe_align_block_size_embedded, "Embedded yiakwy-xpu moe_align_block_size");
}
"""


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage1_triton(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts
    counts = tl.histogram(expert_id, BLOCK_EXPERT, mask=mask)
    tl.store(tokens_cnts_ptr + off_c + expert_offsets, counts, mask=expert_mask)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage1_2_triton_atomic(
    topk_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts
    counts = tl.histogram(expert_id, BLOCK_EXPERT, mask=mask)
    prefix_before = tl.atomic_add(cumsum_ptr + expert_offsets, counts, mask=expert_mask)
    tl.store(tokens_cnts_ptr + off_t + expert_offsets, prefix_before, mask=expert_mask)


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
    num_experts_next_power_of_2: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts

    expert_offsets = tl.arange(0, num_experts_next_power_of_2)
    mask = expert_offsets < num_experts
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets, mask=mask, other=0)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values, mask=mask)

    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)


@triton.jit
def moe_align_block_size_stage3_from_cumsum(
    total_tokens_post_pad_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    num_experts_next_power_of_2: tl.constexpr,
    block_size: tl.constexpr,
):
    expert_offsets = tl.arange(0, num_experts_next_power_of_2)
    mask = expert_offsets < num_experts
    token_cnts = tl.load(cumsum_ptr + expert_offsets, mask=mask, other=0)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values, mask=mask)
    tl.store(cumsum_ptr, 0)

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


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage4_triton_atomic_smem(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    expert_offsets = tl.arange(0, BLOCK_EXPERT)

    local_counts = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    row_ptrs = tokens_cnts_ptr + off_t + expert_offsets
    tle.gpu.copy(row_ptrs, local_counts, [num_experts])

    offset = tl.arange(0, tokens_per_thread) + start_idx
    mask = offset < numel
    expert_id = tl.load(topk_ids_ptr + offset, mask=mask, other=0).to(tl.int32)
    count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id, ))
    token_idx_in_expert = tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")
    rank_post_pad = token_idx_in_expert + tl.load(cumsum_ptr + expert_id, mask=mask, other=0)
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset, mask=mask)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_triton_atomic_fused_coop(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    cumsum_ptr,
    mesh: tl.constexpr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    EXPERTS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts
    token_offsets = tl.arange(0, BLOCK_TOKENS)

    # Stage 0: init outputs + global scratch.
    for base in range(pid * BLOCK_TOKENS, numel_sorted_token_ids, NUM_BLOCKS * BLOCK_TOKENS):
        offs = base + token_offsets
        tl.store(sorted_token_ids_ptr + offs, numel, mask=offs < numel_sorted_token_ids)
    for base in range(pid * BLOCK_TOKENS, numel_expert_ids, NUM_BLOCKS * BLOCK_TOKENS):
        offs = base + token_offsets
        tl.store(expert_ids_ptr + offs, 0, mask=offs < numel_expert_ids)
    if pid == 0:
        tl.store(cumsum_ptr + expert_offsets, 0, mask=expert_mask)
    tle.distributed_barrier(mesh)

    # Stage 1: per-program local histogram + global atomic prefix.
    local_counts = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_counts_ptrs = tle.gpu.local_ptr(local_counts, (expert_offsets, ))
    tl.store(local_counts_ptrs, 0, mask=expert_mask)

    for base in range(pid * BLOCK_TOKENS, numel, NUM_BLOCKS * BLOCK_TOKENS):
        offs = base + token_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id, ))
        tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")

    local_counts_vals = tl.load(local_counts_ptrs, mask=expert_mask, other=0)
    prefix_before = tl.atomic_add(cumsum_ptr + expert_offsets, local_counts_vals, mask=expert_mask, sem="acq_rel",
                                  scope="gpu")
    tl.store(local_counts_ptrs, prefix_before, mask=expert_mask)
    tle.distributed_barrier(mesh)

    # Stage 2: rank0 aligns expert totals and builds starts in global cumsum.
    if pid == 0:
        total_counts = tl.load(cumsum_ptr + expert_offsets, mask=expert_mask, other=0)
        aligned_counts = tl.cdiv(total_counts, block_size) * block_size
        expert_starts = tl.cumsum(aligned_counts, axis=0) - aligned_counts
        tl.store(cumsum_ptr + expert_offsets, expert_starts, mask=expert_mask)
        total_tokens = tl.sum(aligned_counts, axis=0)
        tl.store(num_tokens_post_pad_ptr, total_tokens)
    tle.distributed_barrier(mesh)

    # Cache expert starts in shared once; Stage 4 reuses it for every token.
    expert_starts_local = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    expert_starts_ptrs = tle.gpu.local_ptr(expert_starts_local, (expert_offsets, ))
    expert_starts_vals = tl.load(cumsum_ptr + expert_offsets, mask=expert_mask, other=0)
    tl.store(expert_starts_ptrs, expert_starts_vals, mask=expert_mask)

    # Stage 3: fill expert ids.
    total_tokens = tl.load(num_tokens_post_pad_ptr)
    for local_expert_idx in range(EXPERTS_PER_PROG):
        expert_id = pid + local_expert_idx * NUM_BLOCKS
        valid_expert = expert_id < num_experts
        start_idx = tl.load(tle.gpu.local_ptr(expert_starts_local, (expert_id, )), mask=valid_expert, other=0)
        next_expert = expert_id + 1
        has_next = valid_expert & (next_expert < num_experts)
        end_idx = tl.load(tle.gpu.local_ptr(expert_starts_local, (next_expert, )), mask=has_next, other=total_tokens)
        end_idx = tl.where(has_next, end_idx, total_tokens)
        start_idx = tl.where(valid_expert, start_idx, 0)
        end_idx = tl.where(valid_expert, end_idx, 0)
        for i in range(start_idx, end_idx, block_size):
            tl.store(expert_ids_ptr + i // block_size, expert_id)

    # Stage 4: scatter tokens using local prefix + expert starts.
    for base in range(pid * BLOCK_TOKENS, numel, NUM_BLOCKS * BLOCK_TOKENS):
        offs = base + token_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id, ))
        rank_with_prefix = tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")
        rank_base = tl.load(tle.gpu.local_ptr(expert_starts_local, (expert_id, )), mask=mask, other=0)
        rank_post_pad = rank_with_prefix + rank_base
        tl.store(sorted_token_ids_ptr + rank_post_pad, offs, mask=mask)


def _supports_tle_cluster_remote() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9


def _pick_tle_fused_launch_params(numel: int, num_experts: int) -> Tuple[int, int]:
    # Empirical tuning on SM90+:
    # - For larger expert counts, bigger token tiles are helpful for
    #   cluster-fused path.
    # - For smaller expert counts, keep conservative settings.
    if num_experts >= 256:
        if numel >= 32768:
            return 4096, 4
        if numel >= 1024:
            return 1024, 4
        return 256, 8

    if numel <= 512:
        return 128, 8
    if num_experts <= 64 and numel <= 2048:
        return 128, 8
    return 256, 8


def _pick_triton_atomic_fused_launch_params(numel: int, num_experts: int) -> Tuple[int, int]:
    # Empirical tuning (RTX 50-class + SM90+) for cooperative-grid fused path:
    # - Small/medium numel: smaller token tiles avoid under-populated grids.
    # - Large numel: use bigger token tiles to reduce loop overhead.
    if num_experts >= 256:
        if numel <= 16384:
            return 256, 8
        if numel <= 32768:
            return 512, 4
        return 1024, 4
    return _pick_tle_fused_launch_params(numel, num_experts)


def _pick_triton_atomic_fused_block_cap_mult(numel: int, num_experts: int, block_tokens: int) -> int:
    if num_experts < 256:
        return 4
    if block_tokens <= 256 or numel <= 16384:
        return 16
    if numel <= 65536:
        return 16
    return 16


def _pick_triton_atomic_fused_num_blocks(numel: int, num_experts: int, block_tokens: int) -> int:
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(DEVICE)
    sm_count = int(getattr(props, "multi_processor_count", 1))
    token_programs = triton.cdiv(numel, block_tokens)
    cap_mult = _pick_triton_atomic_fused_block_cap_mult(numel, num_experts, block_tokens)
    block_cap = sm_count * cap_mult
    # Allow trying num_blocks > num_experts for large-token cooperative runs.
    # Cooperative-launch failures still fallback by halving in caller.
    return max(1, min(token_programs, block_cap))


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_tle_cluster_fused(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    mesh: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    EXPERTS_PER_SHARD: tl.constexpr,
):
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    is_rank0 = cluster_rank == 0
    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    expert_mask = expert_offsets < num_experts

    # Stage 0: initialize outputs (sentinel sorted ids + zero expert ids).
    init_offsets = tl.arange(0, BLOCK_TOKENS)
    for base in range(cluster_rank * BLOCK_TOKENS, numel_sorted_token_ids, CLUSTER_SIZE * BLOCK_TOKENS):
        offs = base + init_offsets
        mask = offs < numel_sorted_token_ids
        tl.store(sorted_token_ids_ptr + offs, numel, mask=mask)
    for base in range(cluster_rank * BLOCK_TOKENS, numel_expert_ids, CLUSTER_SIZE * BLOCK_TOKENS):
        offs = base + init_offsets
        mask = offs < numel_expert_ids
        tl.store(expert_ids_ptr + offs, 0, mask=mask)

    local_counts = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    cumsum_local = tle.gpu.alloc(
        [BLOCK_EXPERT],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    # Stage 1a: initialize rank0 DSMEM cumsum before counting.
    rank0_cumsum_ptrs = tle.gpu.local_ptr(cumsum_local, (expert_offsets, ))
    if is_rank0:
        tl.store(rank0_cumsum_ptrs, 0, mask=expert_mask)
    tle.distributed_barrier(mesh)

    # Stage 1b: per-shard local histogram in shared memory.
    local_counts_ptrs = tle.gpu.local_ptr(local_counts, (expert_offsets, ))
    tl.store(local_counts_ptrs, 0, mask=expert_mask)

    for base in range(cluster_rank * BLOCK_TOKENS, numel, CLUSTER_SIZE * BLOCK_TOKENS):
        offs = base + init_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id, ))
        tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")

    # Stage 1c: directly atomic-add shard histogram into rank0 cumsum DSMEM.
    # The atomic return value is this shard's prefix_before for scatter.
    local_counts_vals = tl.load(local_counts_ptrs, mask=expert_mask, other=0)
    rank0_cumsum_remote = tle.remote(cumsum_local, 0, scope=mesh)
    rank0_cumsum_remote_ptrs = tle.gpu.local_ptr(rank0_cumsum_remote, (expert_offsets, ))
    prefix_before = tl.atomic_add(rank0_cumsum_remote_ptrs, local_counts_vals, mask=expert_mask, sem="relaxed",
                                  scope="cta")
    # Reuse local_counts to hold per-shard prefix_before for Stage 4 scatter.
    tl.store(local_counts_ptrs, prefix_before, mask=expert_mask)

    tle.distributed_barrier(mesh)

    # Stage 2: rank0 aligns expert totals and materializes start offsets
    # (leading-zero semantics) in rank0 DSMEM.
    if is_rank0:
        total_counts = tl.load(rank0_cumsum_ptrs, mask=expert_mask, other=0)
        aligned_counts = tl.cdiv(total_counts, block_size) * block_size
        # One-shot vectorized prefix sum over BLOCK_EXPERT, then convert to
        # exclusive scan: start[e] = inclusive[e] - aligned_counts[e].
        expert_cumsum_inclusive = tl.cumsum(aligned_counts, axis=0)
        expert_start_offsets = expert_cumsum_inclusive - aligned_counts
        tl.store(rank0_cumsum_ptrs, expert_start_offsets, mask=expert_mask)

        total_tokens = tl.sum(aligned_counts, axis=0)
        tl.store(num_tokens_post_pad_ptr, total_tokens)

    tle.distributed_barrier(mesh)

    # Stage 4a: fill expert_ids (moved from Stage 3).
    # Each shard handles one contiguous expert segment.
    rank0_cumsum_remote = tle.remote(cumsum_local, 0, scope=mesh)
    rank0_cumsum_remote_ptrs = tle.gpu.local_ptr(rank0_cumsum_remote, (expert_offsets, ))
    cumsum_vals = tl.load(rank0_cumsum_remote_ptrs, mask=expert_mask, other=0)
    tl.store(tle.gpu.local_ptr(cumsum_local, (expert_offsets, )), cumsum_vals, mask=expert_mask)
    total_tokens = tl.load(num_tokens_post_pad_ptr)

    for local_expert_idx in range(EXPERTS_PER_SHARD):
        expert_idx = cluster_rank * EXPERTS_PER_SHARD + local_expert_idx
        expert_id = expert_idx
        valid_expert = expert_id < num_experts
        start_ptr = tle.gpu.local_ptr(cumsum_local, (expert_id, ))
        start_idx = tl.load(start_ptr, mask=valid_expert, other=0)
        next_expert_id = expert_id + 1
        has_next = valid_expert & (next_expert_id < num_experts)
        next_ptr = tle.gpu.local_ptr(cumsum_local, (next_expert_id, ))
        end_from_next = tl.load(next_ptr, mask=has_next, other=0)
        end_idx = tl.where(has_next, end_from_next, total_tokens)
        start_idx = tl.where(valid_expert, start_idx, 0)
        end_idx = tl.where(valid_expert, end_idx, 0)
        for i in range(start_idx, end_idx, block_size):
            tl.store(expert_ids_ptr + i // block_size, expert_idx)

    tle.distributed_barrier(mesh)

    # Stage 4b: second token pass, scatter with (prefix_before + rank_in_shard) + expert_base.
    # local_counts currently holds prefix_before from Stage 1c.
    for base in range(cluster_rank * BLOCK_TOKENS, numel, CLUSTER_SIZE * BLOCK_TOKENS):
        offs = base + init_offsets
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)

        count_ptrs = tle.gpu.local_ptr(local_counts, (expert_id, ))
        rank_with_prefix = tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")
        base_ptrs = tle.gpu.local_ptr(cumsum_local, (expert_id, ))
        rank_base = tl.load(base_ptrs, mask=mask, other=0)
        rank_post_pad = rank_with_prefix + rank_base
        tl.store(sorted_token_ids_ptr + rank_post_pad, offs, mask=mask)


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


def _make_bench_runner(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    provider: str = "triton",
    triton_stage12_atomic: bool = False,
):
    numel = topk_ids.numel()
    numel_sorted_token_ids = sorted_ids.numel()
    numel_expert_ids = expert_ids.numel()
    grid = (num_experts, )
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))
    num_experts_next_power_of_2 = triton.next_power_of_2(num_experts)
    block_size_sorted = triton.next_power_of_2(ceil_div(numel_sorted_token_ids, num_experts))
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))
    block_expert_hist = triton.next_power_of_2(num_experts)
    # TLE alloc currently requires power-of-two block shape.
    block_expert_cluster = triton.next_power_of_2(num_experts)
    experts_per_shard = ceil_div(num_experts, TLE_CLUSTER_SIZE)
    block_tokens_tle, num_warps_tle = _pick_tle_fused_launch_params(numel, num_experts)
    block_tokens_taf, num_warps_taf = _pick_triton_atomic_fused_launch_params(numel, num_experts)
    triton_atomic_fused_num_blocks = _pick_triton_atomic_fused_num_blocks(numel, num_experts, block_tokens_taf)
    triton_atomic_fused_experts_per_prog = ceil_div(num_experts, triton_atomic_fused_num_blocks)
    use_cluster_tle_fused = (provider == "tle_cluster_fused" and topk_ids.is_cuda and _supports_tle_cluster_remote()
                             and block_expert_cluster <= 1024)
    use_triton_atomic_fused = (provider == "triton_atomic_fused" and topk_ids.is_cuda and block_expert_cluster <= 1024)
    tokens_cnts = None
    cumsum = None
    if provider in ("triton", ):
        tokens_cnts = torch.empty((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
        cumsum = torch.empty((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)
    elif provider == "triton_atomic_fused":
        cumsum = torch.empty((num_experts, ), dtype=torch.int32, device=topk_ids.device)

    def init_fn():
        if tokens_cnts is not None:
            tokens_cnts.zero_()
        if cumsum is not None:
            cumsum.zero_()

    def kernel_fn():
        if provider == "triton":
            if triton_stage12_atomic:
                moe_align_block_size_stage1_2_triton_atomic[grid](
                    topk_ids,
                    tokens_cnts,
                    cumsum,
                    num_experts,
                    numel,
                    tokens_per_thread,
                    sorted_ids,
                    expert_ids,
                    numel_sorted_token_ids,
                    numel_expert_ids,
                    block_size_sorted,
                    block_size_expert,
                    BLOCK_EXPERT=block_expert_hist,
                )
                moe_align_block_size_stage3_from_cumsum[(1, )](
                    num_tokens_post_pad,
                    cumsum,
                    num_experts,
                    num_experts_next_power_of_2,
                    block_size,
                )
                moe_align_block_size_stage4_triton_atomic_smem[grid](
                    topk_ids,
                    sorted_ids,
                    expert_ids,
                    tokens_cnts,
                    cumsum,
                    num_experts,
                    block_size,
                    numel,
                    tokens_per_thread,
                    BLOCK_EXPERT=block_expert_hist,
                )
            else:
                moe_align_block_size_stage1_triton[grid](
                    topk_ids,
                    tokens_cnts,
                    num_experts,
                    numel,
                    tokens_per_thread,
                    sorted_ids,
                    expert_ids,
                    numel_sorted_token_ids,
                    numel_expert_ids,
                    block_size_sorted,
                    block_size_expert,
                    BLOCK_EXPERT=block_expert_hist,
                )
                if num_experts == triton.next_power_of_2(num_experts):
                    moe_align_block_size_stage2_vec[grid](tokens_cnts, num_experts)
                else:
                    moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
                moe_align_block_size_stage3[(1, )](
                    num_tokens_post_pad,
                    tokens_cnts,
                    cumsum,
                    num_experts,
                    num_experts_next_power_of_2,
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
        elif provider == "triton_atomic_fused":
            if not use_triton_atomic_fused:
                raise ValueError("triton_atomic_fused requires CUDA SM90+ and num_experts <= 1024")
            nonlocal triton_atomic_fused_num_blocks, triton_atomic_fused_experts_per_prog
            while True:
                try:
                    moe_align_block_size_triton_atomic_fused_coop[(triton_atomic_fused_num_blocks, )](
                        topk_ids,
                        sorted_ids,
                        expert_ids,
                        num_tokens_post_pad,
                        cumsum,
                        _block_mesh(triton_atomic_fused_num_blocks),
                        num_experts,
                        block_size,
                        numel,
                        numel_sorted_token_ids,
                        numel_expert_ids,
                        NUM_BLOCKS=triton_atomic_fused_num_blocks,
                        BLOCK_TOKENS=block_tokens_taf,
                        BLOCK_EXPERT=block_expert_cluster,
                        EXPERTS_PER_PROG=triton_atomic_fused_experts_per_prog,
                        num_ctas=1,
                        num_warps=num_warps_taf,
                        launch_cooperative_grid=True,
                    )
                    break
                except Exception as ex:
                    msg = str(ex).lower()
                    if "no allocator was set" in msg:
                        _install_triton_default_allocator()
                        continue
                    if triton_atomic_fused_num_blocks <= 1 or "cooperative" not in msg:
                        raise
                    triton_atomic_fused_num_blocks = max(1, triton_atomic_fused_num_blocks // 2)
                    triton_atomic_fused_experts_per_prog = ceil_div(num_experts, triton_atomic_fused_num_blocks)
        elif provider == "tle_cluster_fused":
            if not use_cluster_tle_fused:
                raise ValueError("tle_cluster_fused requires CUDA SM90+ and num_experts <= 1024")
            moe_align_block_size_tle_cluster_fused[(1, )](
                topk_ids,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                num_experts,
                block_size,
                numel,
                numel_sorted_token_ids,
                numel_expert_ids,
                mesh=BLOCK_CLUSTER_MESH_8,
                CLUSTER_SIZE=TLE_CLUSTER_SIZE,
                BLOCK_TOKENS=block_tokens_tle,
                BLOCK_EXPERT=block_expert_cluster,
                EXPERTS_PER_SHARD=experts_per_shard,
                num_ctas=1,
                num_warps=num_warps_tle,
            )
        else:
            raise ValueError(f"unknown provider: {provider}")

    return init_fn, kernel_fn


def moe_align_block_size_triton_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    triton_stage12_atomic: bool = False,
) -> None:
    init_fn, kernel_fn = _make_bench_runner(
        topk_ids,
        block_size,
        num_experts,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        provider="triton",
        triton_stage12_atomic=triton_stage12_atomic,
    )
    init_fn()
    kernel_fn()


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool = False,
    triton_stage12_atomic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids, expert_ids, num_tokens_post_pad = _allocate_outputs(topk_ids, num_experts, block_size, pad_sorted_ids)
    moe_align_block_size_triton_impl(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        triton_stage12_atomic=triton_stage12_atomic,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def _resolve_sglang_cuda_moe_align() -> Optional[Callable]:
    embedded_fn = _load_embedded_sglang_cuda_moe_align()
    if embedded_fn is not None:
        return embedded_fn

    candidates = [
        ("sgl_kernel.moe", "moe_align_block_size"),
        ("sgl_kernel", "moe_align_block_size"),
        ("sgl_kernel.ops", "moe_align_block_size"),
        ("sglang.srt.layers.moe.fused_moe_triton", "moe_align_block_size"),
    ]
    for module_name, fn_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, fn_name, None)
        if fn is not None:
            return fn
    return None


@lru_cache(maxsize=1)
def _load_embedded_sglang_cuda_moe_align() -> Optional[Callable]:
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as ex:
        print(f"warning: cannot import torch cpp_extension: {ex}")
        return None

    try:
        with urllib.request.urlopen(SGLANG_MOE_ALIGN_KERNEL_URL, timeout=20) as resp:
            kernel_src = resp.read().decode("utf-8")
    except Exception as ex:
        print(f"warning: failed to download sglang moe_align kernel source: {ex}")
        return None

    kernel_src = kernel_src.replace("#include <THC/THCAtomics.cuh>\n", "")
    kernel_src = kernel_src.replace(
        '#include "utils.h"\n',
        """
#include <algorithm>
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32
#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \\
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \\
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \\
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \\
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \\
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)
#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \\
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))
inline uint32_t next_pow2(uint32_t x) noexcept {
  if (x <= 1) return 1;
  return 1u << (32 - __builtin_clz(x - 1));
}
""",
    )
    kernel_src = kernel_src.replace(
        "const int32_t threads = max((int32_t)num_experts, WARP_SIZE);",
        "const int32_t threads = std::max((int32_t)num_experts, (int32_t)WARP_SIZE);",
    )
    if '#include "utils.h"' in kernel_src:
        print("warning: failed to patch sglang kernel source include utils.h")
        return None

    source_digest = hashlib.sha1((SGLANG_MOE_ALIGN_BINDING_CPP + kernel_src).encode("utf-8")).hexdigest()[:12]
    extension_name = f"flagtree_sglang_moe_align_{source_digest}"

    try:
        module = load_inline(
            name=extension_name,
            cpp_sources=[SGLANG_MOE_ALIGN_BINDING_CPP],
            cuda_sources=[kernel_src],
            functions=None,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as ex:
        print(f"warning: failed to compile embedded sglang moe_align kernel: {ex}")
        return None

    fn = getattr(module, "moe_align_block_size", None)
    if fn is None:
        print("warning: embedded sglang module has no moe_align_block_size symbol")
    return fn


@lru_cache(maxsize=1)
def _load_embedded_yiakwy_cuda_moe_align() -> Optional[Callable]:
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as ex:
        print(f"warning: cannot import torch cpp_extension: {ex}")
        return None

    try:
        with urllib.request.urlopen(YIAKWY_MOE_ALIGN_KERNEL_URL, timeout=20) as resp:
            kernel_src = resp.read().decode("utf-8")
    except Exception as ex:
        print(f"warning: failed to download yiakwy moe_align kernel source: {ex}")
        return None

    kernel_src = kernel_src.replace("#include <THC/THCAtomics.cuh>\n", "")
    kernel_src = kernel_src.replace(
        '#include "utils.h"\n',
        """
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define WARP_SIZE 32
#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \\
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \\
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \\
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \\
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \\
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)
#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \\
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))
""",
    )
    if '#include "utils.h"' in kernel_src:
        print("warning: failed to patch yiakwy kernel source include utils.h")
        return None
    # Ensure the cooperative kernel launches on PyTorch's current CUDA stream.
    kernel_src = kernel_src.replace(
        "cudaLaunchCooperativeKernel((void*)kernel, num_blocks, block_threads, kernelArgs);",
        "C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, dim3(num_blocks), dim3(block_threads), kernelArgs, 0, stream));",
    )

    source_digest = hashlib.sha1((YIAKWY_MOE_ALIGN_BINDING_CPP + kernel_src).encode("utf-8")).hexdigest()[:12]
    extension_name = f"flagtree_yiakwy_moe_align_{source_digest}"

    try:
        module = load_inline(
            name=extension_name,
            cpp_sources=[YIAKWY_MOE_ALIGN_BINDING_CPP],
            cuda_sources=[kernel_src],
            functions=None,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as ex:
        print(f"warning: failed to compile embedded yiakwy moe_align kernel: {ex}")
        return None

    fn = getattr(module, "moe_align_block_size", None)
    if fn is None:
        print("warning: embedded yiakwy module has no moe_align_block_size symbol")
    return fn


def _make_yiakwy_cuda_runner(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    # The published kernel hard-codes MAX_NUM_EXPERTS=256 and assumes num_experts <= block_threads (256).
    if num_experts > 256:
        return None
    if not topk_ids.is_cuda:
        return None

    moe_align_fn = _load_embedded_yiakwy_cuda_moe_align()
    if moe_align_fn is None:
        return None

    # Kernel expects 2D topk_ids: [num_tokens, K]. Use K=1 view.
    topk_ids_2d = topk_ids.view(-1, 1)

    tokens_cnts = torch.empty((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.empty((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)

    def init_fn():
        # Make padding deterministic for correctness checks.
        sorted_ids.fill_(topk_ids.numel())
        expert_ids.zero_()
        num_tokens_post_pad.zero_()
        tokens_cnts.zero_()
        cumsum.zero_()

    def kernel_fn():
        moe_align_fn(
            topk_ids_2d,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            tokens_cnts,
            cumsum,
        )

    return init_fn, kernel_fn


def _make_sglang_cuda_runner(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    moe_align_fn = _resolve_sglang_cuda_moe_align()
    if moe_align_fn is None:
        return None

    cumsum_buffer = torch.empty((num_experts + 1, ), dtype=torch.int32, device=topk_ids.device)

    def init_fn():
        num_tokens_post_pad.zero_()
        cumsum_buffer.zero_()

    def _call_with_variant(use_cumsum_buffer: bool, use_pad_sorted_token_ids: bool):
        args = [topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad]
        if use_cumsum_buffer:
            args.append(cumsum_buffer)
        if use_pad_sorted_token_ids:
            args.append(False)
        moe_align_fn(*args)

    selected_variant = None
    for variant in ((True, True), (True, False), (False, False)):
        try:
            init_fn()
            _call_with_variant(*variant)
            selected_variant = variant
            break
        except TypeError:
            continue
    if selected_variant is None:
        return None

    def kernel_fn():
        _call_with_variant(*selected_variant)

    return init_fn, kernel_fn


def _rand_topk_ids(num_tokens: int, num_experts: int) -> torch.Tensor:
    return torch.randint(0, num_experts, (num_tokens, ), device=DEVICE, dtype=torch.int32)


def run_correctness(
    num_tokens: int,
    num_experts: int,
    block_size: int,
):
    torch.manual_seed(0)
    topk_ids = _rand_topk_ids(num_tokens, num_experts)
    triton_sorted, triton_expert, triton_num_post = moe_align_block_size_triton(
        topk_ids,
        block_size,
        num_experts,
        triton_stage12_atomic=False,
    )
    triton_atomic_sorted, triton_atomic_expert, triton_atomic_num_post = moe_align_block_size_triton(
        topk_ids,
        block_size,
        num_experts,
        triton_stage12_atomic=True,
    )
    outputs = {
        "triton": (triton_sorted, triton_expert, triton_num_post),
        "triton_atomic": (triton_atomic_sorted, triton_atomic_expert, triton_atomic_num_post),
    }
    try:
        triton_atomic_fused_sorted, triton_atomic_fused_expert, triton_atomic_fused_num_post = _allocate_outputs(
            topk_ids, num_experts, block_size, False)
        init_fn_taf, kernel_fn_taf = _make_bench_runner(
            topk_ids,
            block_size,
            num_experts,
            triton_atomic_fused_sorted,
            triton_atomic_fused_expert,
            triton_atomic_fused_num_post,
            provider="triton_atomic_fused",
            triton_stage12_atomic=False,
        )
        init_fn_taf()
        kernel_fn_taf()
        outputs["triton_atomic_fused"] = (
            triton_atomic_fused_sorted,
            triton_atomic_fused_expert,
            triton_atomic_fused_num_post,
        )
    except Exception as ex:
        print(f"Correctness: skip triton_atomic_fused ({ex})")

    if topk_ids.is_cuda and _supports_tle_cluster_remote() and triton.next_power_of_2(num_experts) <= 1024:
        sorted_ids_cf, expert_ids_cf, num_tokens_post_pad_cf = _allocate_outputs(topk_ids, num_experts, block_size,
                                                                                 False)
        init_fn_cf, kernel_fn_cf = _make_bench_runner(
            topk_ids,
            block_size,
            num_experts,
            sorted_ids_cf,
            expert_ids_cf,
            num_tokens_post_pad_cf,
            provider="tle_cluster_fused",
            triton_stage12_atomic=False,
        )
        init_fn_cf()
        kernel_fn_cf()
        outputs["tle_cluster_fused"] = (sorted_ids_cf, expert_ids_cf, num_tokens_post_pad_cf)
    else:
        print("Correctness: skip tle_cluster_fused (requires CUDA SM90+ and BLOCK_EXPERT<=1024).")

    sorted_ids_s, expert_ids_s, num_tokens_post_pad_s = _allocate_outputs(topk_ids, num_experts, block_size, False)
    sglang_runner = _make_sglang_cuda_runner(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_s,
        expert_ids_s,
        num_tokens_post_pad_s,
    )
    if sglang_runner is None:
        print("Correctness: skip sglang_cuda (not available).")
    else:
        init_fn_s, kernel_fn_s = sglang_runner
        init_fn_s()
        kernel_fn_s()
        outputs["sglang_cuda"] = (sorted_ids_s, expert_ids_s, num_tokens_post_pad_s)

    try:
        sorted_ids_y, expert_ids_y, num_tokens_post_pad_y = _allocate_outputs(topk_ids, num_experts, block_size, False)
        yiakwy_runner = _make_yiakwy_cuda_runner(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids_y,
            expert_ids_y,
            num_tokens_post_pad_y,
        )
        if yiakwy_runner is None:
            print("Correctness: skip yiakwy_cuda (unsupported or unavailable).")
        else:
            init_fn_y, kernel_fn_y = yiakwy_runner
            init_fn_y()
            kernel_fn_y()
            outputs["yiakwy_cuda"] = (sorted_ids_y, expert_ids_y, num_tokens_post_pad_y)
    except Exception as ex:
        print(f"Correctness: skip yiakwy_cuda ({ex})")

    counts = torch.bincount(topk_ids, minlength=num_experts)
    aligned = torch.div(counts + (block_size - 1), block_size, rounding_mode="floor") * block_size
    cumsum = torch.cumsum(aligned, dim=0).to(torch.int32)
    for _name, (_sorted, _expert, num_post) in outputs.items():
        torch.testing.assert_close(num_post, cumsum[-1:])

    valid_length = int(cumsum[-1].item())
    num_blocks = valid_length // block_size
    expected_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=DEVICE, dtype=torch.int32),
        aligned.to(torch.int32) // block_size,
    )
    for _name, (_sorted, expert_ids, _num_post) in outputs.items():
        torch.testing.assert_close(expert_ids[:num_blocks], expected_expert_ids)

    for _name, (sorted_ids, _expert, _num_post) in outputs.items():
        start = 0
        for expert_id in range(num_experts):
            end = int(cumsum[expert_id].item())
            tokens = sorted_ids[start:end]
            valid_tokens = tokens[tokens < num_tokens]
            if counts[expert_id] > 0:
                assert valid_tokens.numel() == int(counts[expert_id].item())
                torch.testing.assert_close(
                    topk_ids[valid_tokens],
                    torch.full_like(valid_tokens, expert_id),
                )
            start = end

        if valid_length < sorted_ids.numel():
            assert torch.all(sorted_ids[valid_length:] >= num_tokens)

    print(f"Correctness check passed ({', '.join(outputs.keys())}).")


def _moe_realistic_shapes(num_experts: int) -> List[Tuple[int, int]]:
    return [
        (256, num_experts),
        (512, num_experts),
        (1024, num_experts),
        (2048, num_experts),
        (4096, num_experts),
        (8192, num_experts),
        (16384, num_experts),
        (32768, num_experts),
        (65536, num_experts),
        (163840, num_experts),
    ]


def _zipf_probs(num_experts: int, alpha: float) -> torch.Tensor:
    ranks = torch.arange(1, num_experts + 1, device=DEVICE, dtype=torch.float32)
    probs = 1.0 / (ranks**alpha)
    return probs / probs.sum()


def _sample_topk_ids(num_tokens: int, num_experts: int, probs: torch.Tensor) -> torch.Tensor:
    ids = torch.multinomial(probs, num_tokens, replacement=True)
    return ids.to(torch.int32)


def run_realistic_benchmark(block_size: int, num_experts: int) -> None:
    print("num_tokens,num_experts,source,triton_ms,triton_atomic_ms,triton_atomic_fused_ms,"
          "tle_cluster_fused_ms,sglang_cuda_ms,yiakwy_cuda_ms")
    sglang_available = _resolve_sglang_cuda_moe_align() is not None
    if not sglang_available:
        print("warning: sglang cuda moe_align_block_size not found, sglang_cuda_ms will be na")
    yiakwy_available = num_experts <= 256 and _load_embedded_yiakwy_cuda_moe_align() is not None
    if not yiakwy_available:
        print("warning: yiakwy cuda moe_align_block_size not available for this config, yiakwy_cuda_ms will be na")
    yiakwy_warned = False
    triton_atomic_fused_warned = False
    tle_cluster_fused_warned = False
    probs = _zipf_probs(num_experts, alpha=1.2)
    for num_tokens, _ in _moe_realistic_shapes(num_experts):
        topk_ids = _sample_topk_ids(num_tokens, num_experts, probs)
        sorted_ids_t, expert_ids_t, num_tokens_post_pad_t = _allocate_outputs(topk_ids, num_experts, block_size, False)
        init_fn_t, kernel_fn_t = _make_bench_runner(
            topk_ids,
            block_size,
            num_experts,
            sorted_ids_t,
            expert_ids_t,
            num_tokens_post_pad_t,
            provider="triton",
            triton_stage12_atomic=False,
        )
        sorted_ids_ta, expert_ids_ta, num_tokens_post_pad_ta = _allocate_outputs(topk_ids, num_experts, block_size,
                                                                                 False)
        init_fn_ta, kernel_fn_ta = _make_bench_runner(
            topk_ids,
            block_size,
            num_experts,
            sorted_ids_ta,
            expert_ids_ta,
            num_tokens_post_pad_ta,
            provider="triton",
            triton_stage12_atomic=True,
        )
        ms_t, _, _ = triton.testing.do_bench(
            lambda: (init_fn_t(), kernel_fn_t()),
            warmup=20,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
        ms_ta, _, _ = triton.testing.do_bench(
            lambda: (init_fn_ta(), kernel_fn_ta()),
            warmup=20,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
        ms_taf = None
        try:
            sorted_ids_taf, expert_ids_taf, num_tokens_post_pad_taf = _allocate_outputs(
                topk_ids, num_experts, block_size, False)
            init_fn_taf, kernel_fn_taf = _make_bench_runner(
                topk_ids,
                block_size,
                num_experts,
                sorted_ids_taf,
                expert_ids_taf,
                num_tokens_post_pad_taf,
                provider="triton_atomic_fused",
                triton_stage12_atomic=False,
            )
            ms_taf, _, _ = triton.testing.do_bench(
                lambda: (init_fn_taf(), kernel_fn_taf()),
                warmup=20,
                rep=100,
                quantiles=[0.5, 0.2, 0.8],
            )
        except Exception as ex:
            if not triton_atomic_fused_warned:
                print(f"warning: triton_atomic_fused unavailable, triton_atomic_fused_ms will be na ({ex})")
                triton_atomic_fused_warned = True
        ms_cf = None
        try:
            sorted_ids_cf, expert_ids_cf, num_tokens_post_pad_cf = _allocate_outputs(
                topk_ids, num_experts, block_size, False)
            init_fn_cf, kernel_fn_cf = _make_bench_runner(
                topk_ids,
                block_size,
                num_experts,
                sorted_ids_cf,
                expert_ids_cf,
                num_tokens_post_pad_cf,
                provider="tle_cluster_fused",
            )
            ms_cf, _, _ = triton.testing.do_bench(
                lambda: (init_fn_cf(), kernel_fn_cf()),
                warmup=20,
                rep=100,
                quantiles=[0.5, 0.2, 0.8],
            )
        except Exception as ex:
            if not tle_cluster_fused_warned:
                print(f"warning: tle_cluster_fused unavailable, tle_cluster_fused_ms will be na ({ex})")
                tle_cluster_fused_warned = True

        ms_s = None
        sglang_runner = None
        if sglang_available:
            sorted_ids_s, expert_ids_s, num_tokens_post_pad_s = _allocate_outputs(topk_ids, num_experts, block_size,
                                                                                  False)
            sglang_runner = _make_sglang_cuda_runner(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids_s,
                expert_ids_s,
                num_tokens_post_pad_s,
            )
        if sglang_runner is not None:
            init_fn_s, kernel_fn_s = sglang_runner
            ms_s, _, _ = triton.testing.do_bench(
                lambda: (init_fn_s(), kernel_fn_s()),
                warmup=20,
                rep=100,
                quantiles=[0.5, 0.2, 0.8],
            )
        ms_y = None
        if yiakwy_available:
            sorted_ids_y, expert_ids_y, num_tokens_post_pad_y = _allocate_outputs(topk_ids, num_experts, block_size,
                                                                                  False)
            yiakwy_runner = _make_yiakwy_cuda_runner(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids_y,
                expert_ids_y,
                num_tokens_post_pad_y,
            )
            if yiakwy_runner is not None:
                init_fn_y, kernel_fn_y = yiakwy_runner
                try:
                    ms_y, _, _ = triton.testing.do_bench(
                        lambda: (init_fn_y(), kernel_fn_y()),
                        warmup=20,
                        rep=100,
                        quantiles=[0.5, 0.2, 0.8],
                    )
                except Exception as ex:
                    if not yiakwy_warned:
                        print(f"warning: yiakwy_cuda launch failed, yiakwy_cuda_ms will be na ({ex})")
                        yiakwy_warned = True
        ms_cf_str = "na" if ms_cf is None else f"{float(ms_cf):.4f}"
        ms_taf_str = "na" if ms_taf is None else f"{float(ms_taf):.4f}"
        ms_s_str = "na" if ms_s is None else f"{float(ms_s):.4f}"
        ms_y_str = "na" if ms_y is None else f"{float(ms_y):.4f}"
        print(
            f"{num_tokens},{num_experts},zipf,{float(ms_t):.4f},{float(ms_ta):.4f},{ms_taf_str},{ms_cf_str},{ms_s_str},{ms_y_str}"
        )


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


def run_real_data_benchmark(
    topk_ids_path: str,
    num_experts: int,
    block_size: int,
) -> None:
    topk_ids = _load_real_topk_ids(topk_ids_path)
    num_tokens = topk_ids.numel()
    max_id = int(topk_ids.max().item()) if num_tokens > 0 else -1
    if max_id >= num_experts:
        print(f"warning: max topk_id {max_id} >= num_experts {num_experts}")

    sorted_ids_t, expert_ids_t, num_tokens_post_pad_t = _allocate_outputs(topk_ids, num_experts, block_size, False)
    init_fn_t, kernel_fn_t = _make_bench_runner(
        topk_ids,
        block_size,
        num_experts,
        sorted_ids_t,
        expert_ids_t,
        num_tokens_post_pad_t,
        provider="triton",
        triton_stage12_atomic=False,
    )
    sorted_ids_ta, expert_ids_ta, num_tokens_post_pad_ta = _allocate_outputs(topk_ids, num_experts, block_size, False)
    init_fn_ta, kernel_fn_ta = _make_bench_runner(
        topk_ids,
        block_size,
        num_experts,
        sorted_ids_ta,
        expert_ids_ta,
        num_tokens_post_pad_ta,
        provider="triton",
        triton_stage12_atomic=True,
    )
    ms_t, _, _ = triton.testing.do_bench(
        lambda: (init_fn_t(), kernel_fn_t()),
        warmup=20,
        rep=100,
        quantiles=[0.5, 0.2, 0.8],
    )
    ms_ta, _, _ = triton.testing.do_bench(
        lambda: (init_fn_ta(), kernel_fn_ta()),
        warmup=20,
        rep=100,
        quantiles=[0.5, 0.2, 0.8],
    )
    ms_taf = None
    try:
        sorted_ids_taf, expert_ids_taf, num_tokens_post_pad_taf = _allocate_outputs(topk_ids, num_experts, block_size,
                                                                                    False)
        init_fn_taf, kernel_fn_taf = _make_bench_runner(
            topk_ids,
            block_size,
            num_experts,
            sorted_ids_taf,
            expert_ids_taf,
            num_tokens_post_pad_taf,
            provider="triton_atomic_fused",
            triton_stage12_atomic=False,
        )
        ms_taf, _, _ = triton.testing.do_bench(
            lambda: (init_fn_taf(), kernel_fn_taf()),
            warmup=20,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
    except Exception as ex:
        print(f"warning: triton_atomic_fused unavailable, triton_atomic_fused=na ({ex})")
    ms_cf = None
    try:
        sorted_ids_cf, expert_ids_cf, num_tokens_post_pad_cf = _allocate_outputs(topk_ids, num_experts, block_size,
                                                                                 False)
        init_fn_cf, kernel_fn_cf = _make_bench_runner(
            topk_ids,
            block_size,
            num_experts,
            sorted_ids_cf,
            expert_ids_cf,
            num_tokens_post_pad_cf,
            provider="tle_cluster_fused",
        )
        ms_cf, _, _ = triton.testing.do_bench(
            lambda: (init_fn_cf(), kernel_fn_cf()),
            warmup=20,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
    except Exception as ex:
        print(f"warning: tle_cluster_fused unavailable, tle_cluster_fused=na ({ex})")

    ms_s = None
    sorted_ids_s, expert_ids_s, num_tokens_post_pad_s = _allocate_outputs(topk_ids, num_experts, block_size, False)
    sglang_runner = _make_sglang_cuda_runner(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_s,
        expert_ids_s,
        num_tokens_post_pad_s,
    )
    if sglang_runner is not None:
        init_fn_s, kernel_fn_s = sglang_runner
        ms_s, _, _ = triton.testing.do_bench(
            lambda: (init_fn_s(), kernel_fn_s()),
            warmup=20,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
    else:
        print("warning: sglang cuda moe_align_block_size not found, sglang_cuda=na")

    ms_y = None
    sorted_ids_y, expert_ids_y, num_tokens_post_pad_y = _allocate_outputs(topk_ids, num_experts, block_size, False)
    yiakwy_runner = _make_yiakwy_cuda_runner(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_y,
        expert_ids_y,
        num_tokens_post_pad_y,
    )
    if yiakwy_runner is not None:
        init_fn_y, kernel_fn_y = yiakwy_runner
        try:
            ms_y, _, _ = triton.testing.do_bench(
                lambda: (init_fn_y(), kernel_fn_y()),
                warmup=20,
                rep=100,
                quantiles=[0.5, 0.2, 0.8],
            )
        except Exception as ex:
            print(f"warning: yiakwy cuda moe_align_block_size failed, yiakwy_cuda=na ({ex})")
    else:
        print("warning: yiakwy cuda moe_align_block_size not available, yiakwy_cuda=na")

    print(f"num_tokens={num_tokens}, num_experts={num_experts}, block_size={block_size}, source=real")
    print("provider,ms")
    print(f"triton,{float(ms_t):.4f}")
    print(f"triton_atomic,{float(ms_ta):.4f}")
    if ms_taf is None:
        print("triton_atomic_fused,na")
    else:
        print(f"triton_atomic_fused,{float(ms_taf):.4f}")
    if ms_cf is None:
        print("tle_cluster_fused,na")
    else:
        print(f"tle_cluster_fused,{float(ms_cf):.4f}")
    if ms_s is None:
        print("sglang_cuda,na")
    else:
        print(f"sglang_cuda,{float(ms_s):.4f}")
    if ms_y is None:
        print("yiakwy_cuda,na")
    else:
        print(f"yiakwy_cuda,{float(ms_y):.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=16, help="MoE block size")
    parser.add_argument("--num_tokens", type=int, default=8192, help="num tokens")
    parser.add_argument("--num_experts", type=int, default=64, help="num experts")
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness checks")
    parser.add_argument("--real_data", type=str, default="", help="path to topk_ids.pt")
    args = parser.parse_args(argv)

    if not args.skip_correctness:
        run_correctness(args.num_tokens, args.num_experts, args.block_size)

    if args.real_data:
        run_real_data_benchmark(
            args.real_data,
            args.num_experts,
            args.block_size,
        )
    else:
        run_realistic_benchmark(
            args.block_size,
            args.num_experts,
        )


if __name__ == "__main__":
    main()
