"""Route sorting + padding for fused MoE.

Produces sorted_token_ids, expert_ids, num_tokens_post_padded — the
three arrays that turn a sparse (token, expert) routing into a dense
block-aligned GEMM layout.

Implementation: custom Gluon/Triton kernels (histogram + scatter),
which is more efficient than a host-side Python loop for large M.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# ---------------------------------------------------------------------------
# Step 1: histogram kernel — count how many (token, topk) pairs each expert
# receives.  Each CTA processes a chunk of topk_ids.
# ---------------------------------------------------------------------------

@gluon.jit
def _moe_hist_kernel(
    topk_ids_ptr,
    histogram_ptr,
    num_tokens_topk,  # M * topk
    num_experts,
):
    """One element per CTA: atomically increment histogram[expert_id]."""
    pid = gl.program_id(axis=0)
    if pid < num_tokens_topk:
        eid = gl.load(topk_ids_ptr + pid).to(gl.int64)
        gl.atomic_add(histogram_ptr + eid, 1)


# ---------------------------------------------------------------------------
# Step 2: scatter kernel — write sorted_token_ids using padded offsets
# ---------------------------------------------------------------------------

@gluon.jit
def _moe_scatter_kernel(
    topk_ids_ptr,
    cumsum_counter_ptr,  # mutable counter (atomic_add claims slots)
    padded_cumsum_ptr,  # prefix-sum after padding (read-only offsets)
    sorted_token_ids_ptr,
    num_tokens_topk,
):
    """One element per CTA: scatter one token into sorted_token_ids."""
    pid = gl.program_id(axis=0)
    if pid < num_tokens_topk:
        eid = gl.load(topk_ids_ptr + pid).to(gl.int64)
        # Claim a slot in this expert's region via atomic increment
        pos = gl.atomic_add(cumsum_counter_ptr + eid, 1)
        # Write offset in the padded layout
        base = gl.load(padded_cumsum_ptr + eid)
        sorted_idx = base + pos
        # sorted_token_ids stores the flat index token*topk + k
        gl.store(sorted_token_ids_ptr + sorted_idx, pid)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sentinel_idx: int | None = None,
) -> tuple:
    """Align MoE routing to block-aligned GEMM layout.

    Args:
        topk_ids: [M, topk] int tensor of expert assignments per token.
        block_size: BLOCK_M used by the GEMM kernel (must divide the
                    padded expert counts).
        num_experts: total number of experts E.
        sentinel_idx: index to use for padding rows in sorted_token_ids.
            Plan B (zero-sentinel): set to M * topk so padding rows point
            to the zero-sentinel row appended to the input tensor. This
            makes all K-loop loads uniform (no token_mask needed), which
            is required for cpasync's C500 swizzled shared memory layout.
            If None, defaults to M_topk (original behavior, same value but
            explicitly named for clarity).

    Returns:
        sorted_token_ids: [EM] int tensor — sorted token*topk+k indices.
        expert_ids: [num_m_blocks] int tensor — expert id per M-block.
        num_tokens_post_padded: [1] int tensor — padding boundary.
    """
    M_topk = topk_ids.numel()
    topk_ids_flat = topk_ids.flatten().to(torch.int32)

    # Default sentinel: original behavior (out-of-bounds → token_mask=False)
    if sentinel_idx is None:
        sentinel_idx = M_topk

    # Step 1: histogram — count tokens per expert
    histogram = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    _moe_hist_kernel[(M_topk,)](
        topk_ids_flat,
        histogram, 
        M_topk, 
        num_experts,
    )

    # Step 2: host-side prefix sum + padding
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    cumsum[1:] = torch.cumsum(histogram, dim=0)

    # Pad each expert count to a multiple of block_size
    padded_counts = ((histogram + block_size - 1) // block_size) * block_size
    padded_cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    padded_cumsum[1:] = torch.cumsum(padded_counts, dim=0)
    total_padded = padded_cumsum[-1].item()

    # Step 3: scatter kernel — write sorted_token_ids
    # Plan B: padding fill = sentinel_idx (points to zero-sentinel row in
    # the padded input tensor). The scatter kernel only writes valid entries;
    # the fill value here covers padding slots that no valid entry maps to.
    sorted_token_ids = torch.full(
        (total_padded,), sentinel_idx, dtype=torch.int32, device=topk_ids.device,
    )

    # cumsum_counter starts at zero — atomic_add will claim slots 0,1,2,...
    cumsum_counter = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)

    _moe_scatter_kernel[(M_topk,)](
        topk_ids_flat, 
        cumsum_counter, 
        padded_cumsum,
        sorted_token_ids, 
        M_topk,
    )

    # Step 4: build expert_ids on host (simple, reliable)
    num_m_blocks = total_padded // block_size
    expert_ids = torch.full((num_m_blocks, ), -1, dtype=torch.int32, device=topk_ids.device)
    expert_ids_host = expert_ids.cpu()
    block_idx = 0
    for e in range(num_experts):
        start = padded_cumsum[e].item()
        end = padded_cumsum[e + 1].item()
        num_blocks = (end - start) // block_size
        for b in range(num_blocks):
            if block_idx < num_m_blocks:
                expert_ids_host[block_idx] = e
                block_idx += 1
    expert_ids.copy_(expert_ids_host.to(topk_ids.device))

    num_tokens_post_padded = torch.tensor(
        [total_padded], 
        dtype=torch.int32, 
        device=topk_ids.device,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded