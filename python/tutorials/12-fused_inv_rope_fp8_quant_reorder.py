"""
Fused Inverse RoPE + FP8 Quantization + Reorder
================================================

In this tutorial, you will learn how to use **reorder** operations in Triton kernels
and compare the reorder approach with the pipeline approach.

In doing so, you will learn about:

* When to use reorder vs. pipeline for different workload patterns.

* Trade-offs between memory access patterns and computational efficiency.

* Performance comparison between reorder and pipeline versions.

"""

# %%
# Motivations
# -----------
#
# This tutorial focuses on **reordering** as an alternative optimization strategy to pipelining.
#
# **Reorder approach:**
# - Transforms tensor layout in memory (e.g., [tokens, heads, dims] → [groups, tokens, heads_per_group, dims])
# - Simpler control flow with explicit memory layout transformations
# - Better for scenarios where memory access patterns dominate performance
#
# **Pipeline approach:**
# - Uses software pipelining to overlap memory loads and computation
# - Better for compute-bound kernels with predictable memory access patterns
# - Requires more registers and shared memory for staging buffers
#
# By comparing these two approaches on the same fused operation (inverse RoPE + FP8 quantization),
# you will understand when to choose reorder over pipeline based on your workload characteristics.

import logging
import sys
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

import triton.experimental.tle.language as tle


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_fp8():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


if supports_fp8():
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32

DEVICE = triton.runtime.driver.active.get_active_torch_device()

logger = logging.getLogger(__name__)

# %%
# Helper Functions
# ----------------
#
# TMA (Tensor Memory Accelerator) requires alignment for optimal performance on modern GPUs.


def _get_tma_aligned_size(size: int, align: int) -> int:
    """Align size to TMA boundary for optimal memory access."""
    return ((size + align - 1) // align) * align


# %%
# Compute Kernel
# --------------
#
# The fused kernel performs three operations in sequence:
# 1. Inverse RoPE: Undo rotary position embeddings using precomputed cos/sin cache
# 2. FP8 Quantization: Convert to FP8 with per-group dynamic scaling
# 3. Reorder: Transpose the output layout from [tokens, heads, dims] to [groups, tokens, heads_per_group, dims]
#
# Key optimizations:
# - Load cos/sin cache once per token (shared across all heads)
# - Compute quantization scales on-the-fly
# - Zero-fill padding tokens efficiently
# - Use TMA-aligned memory accesses when possible


@triton.jit
def _fused_inv_rope_fp8_quant_per_head_reorder(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh_base = tl.program_id(1).to(tl.int64) * HEADS_PER_PROGRAM

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE

    if pid_token >= num_tokens:
        # Zero-fill scales for padding tokens across all heads in this program.
        for i in tl.static_range(0, HEADS_PER_PROGRAM):
            global_head = pid_gh_base + i
            g = global_head // heads_per_group
            head_in_group = global_head % heads_per_group
            qb_start = head_in_group * CHUNKS_PER_HEAD
            if TMA_ALIGNED_SCALES:
                scale_addr = (scale_ptr + g * scale_stride_group + pid_token + head_in_group * scale_stride_k)
                tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
            else:
                block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
                qb_indices = qb_start + block_offsets
                scale_addrs = (scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k)
                tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD, ), dtype=tl.float32))
        return

    offsets = tl.arange(0, HEAD_DIM)
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start
    cs_idx = tl.maximum(rope_local >> 1, 0)
    is_even = (rope_local & 1) == 0

    # Load cos/sin once — shared across all heads for the same token.
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)

    for i in tle.range(0, HEADS_PER_PROGRAM, loop_unroll_factor=HEADS_PER_PROGRAM, reorder=True):
        global_head = pid_gh_base + i
        g = global_head // heads_per_group
        head_in_group = global_head % heads_per_group
        qb_start = head_in_group * CHUNKS_PER_HEAD

        input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

        x = tl.load(input_base + offsets).to(tl.float32)

        x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32)
        x_add = x * cos_v + x_partner * sin_v
        x_sub = x * cos_v - x_partner * sin_v
        rotated = tl.where(is_even, x_add, x_sub)
        x = tl.where(is_rope, rotated, x)

        x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
        block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
        scales = block_absmax * (1.0 / fp8_max)
        if TMA_ALIGNED_SCALES:
            scales = tl.math.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10))))

        scales_exp = tl.reshape(
            tl.broadcast_to(
                tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
                (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
            ),
            (HEAD_DIM, ),
        )
        x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

        fp8_base = (fp8_ptr + g * fp8_stride_group + pid_token * fp8_stride_token + qb_start * QUANT_GROUP_SIZE)
        tl.store(fp8_base + offsets, x_quant)

        block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
        qb_indices = qb_start + block_offsets
        if TMA_ALIGNED_SCALES:
            scale_bits = scales.to(tl.int32, bitcast=True)
            ue8m0_bytes = (scale_bits >> 23) & 0xFF
            packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
            scale_addr = (scale_ptr + g * scale_stride_group + pid_token + head_in_group * scale_stride_k)
            tl.store(scale_addr, packed_val)
        else:
            scale_addrs = (scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k)
            tl.store(scale_addrs, scales)


def fused_inv_rope_fp8_quant_reorder(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    tma_aligned_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS FUSED INV ROPE FP8 QUANT REORDER")
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert fp8_dtype == torch.float8_e4m3fn, "only torch.float8_e4m3fn is supported"
    assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
    assert positions.ndim == 1, "`positions` must be 1D"
    assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
    assert o.stride(-1) == 1, "head_dim must be contiguous"
    assert positions.shape[0] == o.shape[0], "positions and o token count mismatch"

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    chunks_per_head = head_dim // quant_group_size
    if tma_aligned_scales:
        assert (chunks_per_head <= 4), "packed UE8M0 path currently expects at most 4 scale blocks per head"

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    tma_aligned_t = _get_tma_aligned_size(num_tokens, 4)

    if tma_aligned_scales:
        scale_inner = (num_scale_blocks + 3) // 4
        scale_dtype = torch.int32
    else:
        scale_inner = num_scale_blocks
        scale_dtype = torch.float32

    finfo = torch.finfo(fp8_dtype)
    fp8_q = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
    scale = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )

    # Determine how many heads each program handles.
    # Use smaller batches (2 or 4) to control register pressure, as each head
    # holds HEAD_DIM × multiple arrays simultaneously.
    total_heads = n_groups * heads_per_group
    heads_per_program = 1
    for candidate in [4, 2]:
        if total_heads % candidate == 0:
            heads_per_program = candidate
            break

    grid = (tma_aligned_t, total_heads // heads_per_program)
    _fused_inv_rope_fp8_quant_per_head_reorder[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_q,
        scale,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_q.stride(0),
        fp8_stride_token=fp8_q.stride(1),
        scale_stride_group=scale.stride(0),
        scale_stride_k=scale.stride(2),
        fp8_max=finfo.max,
        eps=eps,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        HEADS_PER_PROGRAM=heads_per_program,
    )

    return fp8_q.transpose(0, 1), scale.transpose(0, 1)


@triton.jit
def _fused_inv_rope_fp8_quant_per_head_pipeline(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh_base = tl.program_id(1).to(tl.int64) * HEADS_PER_PROGRAM

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE

    if pid_token >= num_tokens:
        # Zero-fill scales for padding tokens across all heads in this program.
        for i in tl.static_range(0, HEADS_PER_PROGRAM):
            global_head = pid_gh_base + i
            g = global_head // heads_per_group
            head_in_group = global_head % heads_per_group
            qb_start = head_in_group * CHUNKS_PER_HEAD
            if TMA_ALIGNED_SCALES:
                scale_addr = (scale_ptr + g * scale_stride_group + pid_token + head_in_group * scale_stride_k)
                tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
            else:
                block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
                qb_indices = qb_start + block_offsets
                scale_addrs = (scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k)
                tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD, ), dtype=tl.float32))
        return

    offsets = tl.arange(0, HEAD_DIM)
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start
    cs_idx = tl.maximum(rope_local >> 1, 0)
    is_even = (rope_local & 1) == 0

    # Load cos/sin once — shared across all heads for the same token.
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)

    for i in tl.range(0, HEADS_PER_PROGRAM, loop_unroll_factor=HEADS_PER_PROGRAM, num_stages=2):
        global_head = pid_gh_base + i
        g = global_head // heads_per_group
        head_in_group = global_head % heads_per_group
        qb_start = head_in_group * CHUNKS_PER_HEAD

        input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

        x = tl.load(input_base + offsets).to(tl.float32)

        x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32)
        x_add = x * cos_v + x_partner * sin_v
        x_sub = x * cos_v - x_partner * sin_v
        rotated = tl.where(is_even, x_add, x_sub)
        x = tl.where(is_rope, rotated, x)

        x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
        block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
        scales = block_absmax * (1.0 / fp8_max)
        if TMA_ALIGNED_SCALES:
            scales = tl.math.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10))))

        scales_exp = tl.reshape(
            tl.broadcast_to(
                tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
                (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
            ),
            (HEAD_DIM, ),
        )
        x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

        fp8_base = (fp8_ptr + g * fp8_stride_group + pid_token * fp8_stride_token + qb_start * QUANT_GROUP_SIZE)
        tl.store(fp8_base + offsets, x_quant)

        block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
        qb_indices = qb_start + block_offsets
        if TMA_ALIGNED_SCALES:
            scale_bits = scales.to(tl.int32, bitcast=True)
            ue8m0_bytes = (scale_bits >> 23) & 0xFF
            packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
            scale_addr = (scale_ptr + g * scale_stride_group + pid_token + head_in_group * scale_stride_k)
            tl.store(scale_addr, packed_val)
        else:
            scale_addrs = (scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k)
            tl.store(scale_addrs, scales)


def reference_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    tma_aligned_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS FUSED INV ROPE FP8 QUANT REORDER")
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert fp8_dtype == torch.float8_e4m3fn, "only torch.float8_e4m3fn is supported"
    assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
    assert positions.ndim == 1, "`positions` must be 1D"
    assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
    assert o.stride(-1) == 1, "head_dim must be contiguous"
    assert positions.shape[0] == o.shape[0], "positions and o token count mismatch"

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    chunks_per_head = head_dim // quant_group_size
    if tma_aligned_scales:
        assert (chunks_per_head <= 4), "packed UE8M0 path currently expects at most 4 scale blocks per head"

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    tma_aligned_t = _get_tma_aligned_size(num_tokens, 4)

    if tma_aligned_scales:
        scale_inner = (num_scale_blocks + 3) // 4
        scale_dtype = torch.int32
    else:
        scale_inner = num_scale_blocks
        scale_dtype = torch.float32

    finfo = torch.finfo(fp8_dtype)
    fp8_q = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
    scale = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )

    # Determine how many heads each program handles.
    # Use smaller batches (2 or 4) to control register pressure, as each head
    # holds HEAD_DIM × multiple arrays simultaneously.
    total_heads = n_groups * heads_per_group
    heads_per_program = 1
    for candidate in [4, 2]:
        if total_heads % candidate == 0:
            heads_per_program = candidate
            break

    grid = (tma_aligned_t, total_heads // heads_per_program)
    _fused_inv_rope_fp8_quant_per_head_pipeline[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_q,
        scale,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_q.stride(0),
        fp8_stride_token=fp8_q.stride(1),
        scale_stride_group=scale.stride(0),
        scale_stride_k=scale.stride(2),
        fp8_max=finfo.max,
        eps=eps,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        HEADS_PER_PROGRAM=heads_per_program,
    )

    return fp8_q.transpose(0, 1), scale.transpose(0, 1)


# %%
# Unit Test
# ---------
#
# We verify correctness by comparing our fused kernel against the reference implementation.


def test_fused_inv_rope_fp8_quant(
    num_tokens: int = 32,
    n_groups: int = 2,
    heads_per_group: int = 4,
    head_dim: int = 512,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
):
    """Test correctness of the fused kernel."""

    num_heads = n_groups * heads_per_group
    print(f"\nTesting with num_tokens={num_tokens}, num_heads={num_heads} "
          f"(n_groups={n_groups}, heads_per_group={heads_per_group}), "
          f"head_dim={head_dim}, nope_dim={nope_dim}, rope_dim={rope_dim}, "
          f"quant_group_size={quant_group_size}")

    # Setup
    max_pos = 2048

    # Create test data
    o = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, max_pos, (num_tokens, ), device=DEVICE, dtype=torch.int64)
    cos_sin_cache = torch.randn(max_pos, rope_dim, device=DEVICE, dtype=torch.float32)

    # Run fused kernel
    fp8_fused, scale_fused = fused_inv_rope_fp8_quant_reorder(o, positions, cos_sin_cache, n_groups, heads_per_group,
                                                              nope_dim, rope_dim, quant_group_size)

    # Run reference
    fp8_ref, scale_ref = reference_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim,
                                                      rope_dim, quant_group_size)

    # The fused kernel returns [num_tokens, n_groups, heads_per_group * head_dim]
    # We need to reshape for comparison with reference which is [num_tokens, num_heads, head_dim]
    fp8_fused_reshaped = fp8_fused.reshape(num_tokens, num_heads, head_dim)
    scale_fused_reshaped = scale_fused.reshape(num_tokens, num_heads, -1)

    fp8_ref_reshaped = fp8_ref.reshape(num_tokens, num_heads, head_dim)
    scale_ref_reshaped = scale_ref.reshape(num_tokens, num_heads, -1)
    # Compare results
    if SUPPORTED_FP8_DTYPE != torch.float32:
        # FP8 has limited precision, so we use looser tolerances
        fp8_fused_f32 = fp8_fused_reshaped.to(torch.float32)
        fp8_ref_f32 = fp8_ref_reshaped.to(torch.float32)

        max_diff = torch.max(torch.abs(fp8_fused_f32 - fp8_ref_f32))
        relative_diff = max_diff / (torch.max(torch.abs(fp8_ref_f32)) + 1e-6)

        print(f"FP8 max absolute difference: {max_diff.item():.6f}")
        print(f"FP8 max relative difference: {relative_diff.item():.6f}")

        # Check scales
        scale_max_diff = torch.max(torch.abs(scale_fused_reshaped - scale_ref_reshaped))
        scale_relative_diff = scale_max_diff / (torch.max(torch.abs(scale_ref_reshaped)) + 1e-6)

        print(f"Scale max absolute difference: {scale_max_diff.item():.6f}")
        print(f"Scale max relative difference: {scale_relative_diff.item():.6f}")

        # Assert correctness with tolerances appropriate for FP8
        assert relative_diff < 0.1, f"FP8 values differ too much: {relative_diff}"
        assert scale_relative_diff < 0.01, f"Scales differ too much: {scale_relative_diff}"

        print("Correctness test passed!")
    else:
        print("FP8 not supported on this device, using float32 (no quantization)")
        torch.testing.assert_close(fp8_fused_reshaped, fp8_ref_reshaped, rtol=0.0, atol=0.0)
        torch.testing.assert_close(scale_fused_reshaped, scale_ref_reshaped, rtol=0.0, atol=0.0)
        print("Correctness test passed (float32 mode)!")


# Run correctness tests with appropriate parameters
# Default config: head_dim=512 (nope_dim=448 + rope_dim=64), quant_group_size=128
test_fused_inv_rope_fp8_quant(num_tokens=32, n_groups=2, heads_per_group=4)
test_fused_inv_rope_fp8_quant(num_tokens=64, n_groups=4, heads_per_group=2)

if '--only_unit_test' in sys.argv:
    sys.exit(0)

# %%
# Benchmark
# ---------
#
# We benchmark the fused kernel against a naive implementation that runs the operations separately.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_tokens'],
        x_vals=[2**i for i in range(8, 13)],
        line_arg='provider',
        line_vals=['triton-reorder', 'triton-pipeline'],
        line_names=["TritonReorder", "TritonPipeline"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="fused-inv-rope-fp8-quant-performance",
        args={
            'n_groups': 8,
            'heads_per_group': 16,
            'head_dim': 512,
            'nope_dim': 448,
            'rope_dim': 64,
            'quant_group_size': 128,
        },
    ))
def benchmark(num_tokens, n_groups, heads_per_group, head_dim, nope_dim, rope_dim, quant_group_size, provider):
    """Benchmark the fused kernel against separate operations."""

    num_heads = n_groups * heads_per_group
    max_pos = 2048

    # Create test data
    o = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, max_pos, (num_tokens, ), device=DEVICE, dtype=torch.int64)
    cos_sin_cache = torch.randn(max_pos, rope_dim, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton-reorder':
        fn = lambda: fused_inv_rope_fp8_quant_reorder(o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim,
                                                      rope_dim, quant_group_size)
    else:
        fn = lambda: reference_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim,
                                                  rope_dim, quant_group_size)

    ms, ms_min, ms_max = triton.testing.do_bench_cudagraph(
        fn,
        quantiles=quantiles,
    )

    # Calculate effective bandwidth
    # Read: o (input) + positions + cos_sin_cache
    # Write: fp8_q (output) + scale
    bytes_read = (o.numel() * o.element_size() + positions.numel() * positions.element_size() +
                  cos_sin_cache.numel() * cos_sin_cache.element_size())

    num_groups_quant = head_dim // quant_group_size
    bytes_write = (
        o.numel() * 1 +  # FP8 is 1 byte
        num_tokens * num_heads * num_groups_quant * 4)  # float32 scales

    total_bytes = bytes_read + bytes_write
    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(ms_min), gbps(ms_max)


benchmark.run(show_plots=True, print_data=True)
