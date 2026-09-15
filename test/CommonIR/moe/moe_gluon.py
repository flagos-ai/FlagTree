"""mcTriton/Gluon Fused MoE Kernel.

Implements the fused Mixture-of-Experts computation using mcTriton's
gl.metax path for Metax/MACA GPUs with double-buffered K-direction
software pipelining.

Data flow:
    hidden_states [M, K]
        → GEMM1: gather(A, sorted_token_ids) @ w1[expert]^T  → cache1 [M, topk, N]
        → activation: silu(gate) * up                       → cache2 [M*topk, N/2]
        → GEMM2: cache2 @ w2[expert]^T                     → cache3 [M, topk, K]
        → moe_sum: Σ_k topk_weights * cache3               → output [M, K]

First version: bf16/fp16 only (no quantization).
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .moe_align_block_size import moe_align_block_size

# ============================================================================
# Configuration selection
# ============================================================================

# Metax m16n8k16 alignment: BLOCK_M, BLOCK_N, BLOCK_K must be multiples of 16.
# Wave size = 64 threads on Metax (vs 32 on NVIDIA).

# NOTE: _MOE_CONFIGS dict removed — replaced by _select_config() with K-aware adaptive tiling.


def _select_config(M, N, K, E, topk):
    """Heuristic config selection for MoE GEMM tiling.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, num_warps, GROUP_SIZE_M).

    Key design choices:
    - BLOCK_K scales with K to target ≤4 K-tiles (fewer serial iterations
      in the K-loop, better cpasync pipeline utilization).
    - BLOCK_N scales with N for compute-bound shapes (larger N → bigger tile).
    - tokens_per_expert = M * topk / E determines the memory/compute boundary.
    - NUM_BUFFERS adapts to K-tile count (2 for few tiles, 4 for many).
    """
    # Estimate tokens per expert
    tokens_per_expert = M * topk / E

    # ---- BLOCK_M: controls padding waste, must match between GEMM1/GEMM2 ----
    if tokens_per_expert <= 64:
        BLOCK_M = 16
    elif tokens_per_expert >= 256:
        BLOCK_M = 128
    else:
        BLOCK_M = 64

    # ---- BLOCK_N: scale with output dimension for compute-bound shapes ----
    # Larger N means more work per output row → bigger N-tile improves throughput
    if N >= 2048:
        BLOCK_N = 256
    elif N >= 512:
        BLOCK_N = 128
    else:
        BLOCK_N = 64

    # ---- BLOCK_K: target ≤4 K-tiles for good pipeline utilization ----
    # Must be multiple of 16 (Metax m16n8k16), clamped to [64, 256]
    # For compute-bound shapes (tokens_per_expert > 64), target ≤4 tiles;
    # for memory-bound shapes (tokens_per_expert ≤ 64), allow up to 8 tiles
    # since smaller BK would increase smem bank conflicts without helping
    # (the kernel is memory-bound anyway).
    if tokens_per_expert <= 64:
        # Decode: allow more K-tiles, larger BK reduces smem pressure
        raw_bk = max(64, (K + 7) // 8)  # ceil(K/8), at least 64
    else:
        # Prefill/mid: target ≤4 K-tiles for better cpasync overlap
        raw_bk = max(64, (K + 3) // 4)  # ceil(K/4), at least 64
    BLOCK_K = ((raw_bk + 15) // 16) * 16  # round up to multiple of 16
    BLOCK_K = min(BLOCK_K, 256)

    # ---- NUM_BUFFERS: deeper pipeline for more K-tiles ----
    num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    if num_k_tiles <= 2:
        NUM_BUFFERS = 2
    elif num_k_tiles <= 4:
        NUM_BUFFERS = 3
    else:
        NUM_BUFFERS = 4

    # ---- num_warps: more warps for larger tiles ----
    # More warps → more threads → larger occupancy for compute-bound work
    if BLOCK_M >= 128 and BLOCK_N >= 128:
        num_warps = 8
    else:
        num_warps = 4

    # ---- GROUP_SIZE_M: L2 reuse grouping ----
    GROUP_SIZE_M = 16 if tokens_per_expert > 128 else 1

    return BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, num_warps, GROUP_SIZE_M


# ============================================================================
# MoE GEMM Kernel (mcTriton gl.metax double-buffered pipeline)
# ============================================================================
#
# Pipeline ordering (Metax async copy, from MCTRITON_SUMMARY §5,
# adapted for MoE gather):
#
#   Prologue:  async_copy K-tile 0 to smem stage 0
#              → gvm_arrive(2) → barrier()
#   Loop body: gvm_arrive(2) → barrier_shared()
#              → intrinsic_load(cur stage) → gl.dot
#              → async_copy next K-tile to smem next stage
#   Epilogue:  gvm_arrive(0) → barrier_shared() → _keep_alive() → write back C
#
# Data path: Global ──async_copy──→ BSM ──.load(intrinsic)──→ Register(dot) → MMA
#            1-hop async, bypassing registers, overlapping with computation.
#
# Key primitives:
#   async_copy_global_to_shared: async Global→BSM, compiles to __builtin_mxc_ldg_b128_bsm
#   gvm_arrive(N):              wait for N async copies (Metax equivalent of
#                               NVIDIA commit_group + wait_group)
#   barrier():                  full CTA barrier (prologue)
#   barrier_shared():           lightweight shared mem barrier (loop body)
#
# If the async pipeline has issues, fall back to moe_gemm_kernel_simple (below)
# which removes the manual pipeline and lets the compiler manage everything.

@gluon.jit
def moe_gemm_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr, sorted_token_ids_ptr, expert_ids_ptr, topk_weights_ptr,
    # Dimensions
    M, N, K, EM, num_valid_tokens,
    # Strides
    stride_am, stride_ak, stride_be, stride_bn, stride_bk, stride_cm, stride_cn, stride_wm,  # topk_weights stride
    # Constexprs
    topk: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr, DTYPE: gl.constexpr,  # 'bfloat16' | 'float16'
    MUL_ROUTED_WEIGHT: gl.constexpr,
    GATHER_BY_TOKEN: gl.constexpr,  # True: A=[M,K], use offs_token//topk; False: A=[M*topk,N], use offs_token
    ):
    """MoE GEMM kernel: gather(A, sorted_token_ids) @ B[expert]^T → C.

    Uses mcTriton gl.metax async copy K-direction pipeline:
    - gl.local_alloc for multi-buffer shared memory
    - gl.metax.slice for per-stage views
    - gl.metax.async_copy_global_to_shared for async Global→BSM copy
      (bypasses registers, compiles to __builtin_mxc_ldg_b128_bsm)
    - gl.metax.gvm_arrive for async copy synchronization
      (Metax equivalent of NVIDIA commit_group + wait_group)
    - gl.metax.barrier / barrier_shared for CTA / shared memory sync
    - gl.dot for Tensor Core (auto-adapts to Metax wave=64)

    NOTE: No early-return / return statements — C500 Gluon layout requires
    structured control flow (SCF) before SCF-to-CF lowering.
    """
    # ---- 1. Grouped CTA mapping (L2 reuse) ----
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. Guard predicates (no return — C500 needs structured control flow) ----
    tile_valid = pid_m * BLOCK_M < EM

    # ---- 3. Read sorted_token_ids (gather source indices) ----
    offs_token_id = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
    offs_token = gl.load(sorted_token_ids_ptr + offs_token_id, mask=tile_valid, other=M * topk).to(gl.int64)

    # ---- 4. Valid row mask (epilogue only — K-loop uses zero-sentinel) ----
    # With Plan B, padding rows in sorted_token_ids point to a zero-sentinel
    # row in the padded input tensor. K-loop A loads are uniform (no mask),
    # so token_mask is only needed in the epilogue for C store and weight load.
    valid_row_mask = (offs_token < num_valid_tokens) & tile_valid

    # ---- 5. Read expert id for this M-block ----
    off_experts = gl.load(expert_ids_ptr + pid_m, mask=tile_valid, other=0).to(gl.int64)
    expert_valid = (off_experts != -1) & tile_valid

    # ---- 6. Compute real token rows (gather) ----
    # GEMM1: A = hidden_states [M, K] — only M rows, index by token (offs_token // topk)
    # GEMM2: A = cache2 [M*topk, N] — M*topk rows, index by flat (offs_token directly)
    if GATHER_BY_TOKEN:
        real_token_rows = offs_token // topk
    else:
        real_token_rows = offs_token

    # ---- 7. Load MoE router weights (if needed) ----
    # Plan B: use valid_row_mask (epilogue only) — padding rows read 0 weight
    moe_weight = gl.zeros((BLOCK_M, ), dtype=gl.float32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = gl.load(
            topk_weights_ptr + offs_token,
            mask=valid_row_mask,
            other=0.0,
        )

    # ---- 8. Index offsets for A and B tiles ----
    offs_k_a = gl.arange(0, BLOCK_K)
    offs_k_b = gl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N)

    # B base: offset to this expert's weight block
    b_base = b_ptr + off_experts * stride_be

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # ---- 9. Resolve dtypes for shared memory ----
    if DTYPE == 'bfloat16':
        a_dtype = gl.bfloat16
        b_dtype = gl.bfloat16
        output_dtype = gl.bfloat16
    else:
        a_dtype = gl.float16
        b_dtype = gl.float16
        output_dtype = gl.float16

    # ---- 10. Allocate multi-buffer shared memory ----
    a_smem = gl.local_alloc(a_dtype, [BLOCK_M, BLOCK_K], num_buffers=NUM_BUFFERS)
    b_smem = gl.local_alloc(b_dtype, [BLOCK_K, BLOCK_N], num_buffers=NUM_BUFFERS)

    # ---- 11. Prologue: async copy first K-tile to stage 0 ----
    # Plan B: A loads are uniform (no token_mask) — padding rows point to
    # zero-sentinel, so A reads valid (zero) data without masking.
    # Only tile_valid (index-based, uniform) is needed for boundary tiles.
    # B loads use expert_valid (also index-based, uniform for cpasync C500).
    a_ptrs = a_ptr + real_token_rows[:, None] * stride_am + offs_k_a[None, :] * stride_ak
    a_load_mask = tile_valid & (offs_k_a[None, :] < K)
    b_ptrs = b_base + offs_k_b[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_load_mask = expert_valid & (offs_k_b[:, None] < K)

    a_stage0 = gl.metax.slice(a_smem, [BLOCK_M, BLOCK_K], [0, 0, 0])
    b_stage0 = gl.metax.slice(b_smem, [BLOCK_K, BLOCK_N], [0, 0, 0])
    gl.metax.async_copy_global_to_shared(a_stage0, a_ptrs, mask=a_load_mask, other=0.0, intrinsic=True)
    gl.metax.async_copy_global_to_shared(b_stage0, b_ptrs, mask=b_load_mask, other=0.0, intrinsic=True)
    gl.metax.gvm_arrive(2)
    gl.metax.barrier()

    # Accumulator in fp32
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)

    # ---- 12. Main K-loop: async copy pipeline ----
    for k in range(num_k_tiles):
        cur = k % NUM_BUFFERS
        nxt = (k + 1) % NUM_BUFFERS

        # Wait for current stage's async copy to complete, then sync
        gl.metax.gvm_arrive(2)
        gl.metax.barrier_shared()

        # Intrinsic load from current stage → register for Tensor Core
        a_frag = gl.metax.slice(a_smem, [BLOCK_M, BLOCK_K], [cur, 0, 0]).load(intrinsic=True)
        b_frag = gl.metax.slice(b_smem, [BLOCK_K, BLOCK_N], [cur, 0, 0]).load(intrinsic=True)

        # Tensor Core dot product
        acc = gl.dot(a_frag, b_frag, acc)

        # Async prefetch next K-tile directly to next stage in shared memory
        if k + 1 < num_k_tiles:
            k_next = (k + 1) * BLOCK_K

            # A pointers: gather by real_token_rows (uniform, no token_mask)
            a_ptrs_next = a_ptr + real_token_rows[:, None] * stride_am + (k_next + offs_k_a)[None, :] * stride_ak
            a_mask_next = tile_valid & ((k_next + offs_k_a)[None, :] < K)

            # B pointers: same expert, next K-tile
            b_ptrs_next = b_base + (k_next + offs_k_b)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask_next = expert_valid & ((k_next + offs_k_b)[:, None] < K)

            # Async copy: Global → BSM (bypasses registers, overlaps with dot)
            next_a_stage = gl.metax.slice(a_smem, [BLOCK_M, BLOCK_K], [nxt, 0, 0])
            next_b_stage = gl.metax.slice(b_smem, [BLOCK_K, BLOCK_N], [nxt, 0, 0])
            gl.metax.async_copy_global_to_shared(next_a_stage, a_ptrs_next, mask=a_mask_next, other=0.0, intrinsic=True)
            gl.metax.async_copy_global_to_shared(next_b_stage, b_ptrs_next, mask=b_mask_next, other=0.0, intrinsic=True)

    # ---- 13. Epilogue ----

    # ③ Router weight multiplication (in fp32 for numerical stability)
    if MUL_ROUTED_WEIGHT:
        acc = acc * moe_weight[:, None]

    # ④ dtype conversion
    c = gl.cast(acc, output_dtype)

    # ⑤ Scatter store: write back using sorted_token_ids to locate output rows
    # C layout is [M*topk, N]; offs_token already encodes token*topk+k
    # which maps directly to the flattened [M*topk, N] row index.
    # Plan B: store uses valid_row_mask (not token_mask) — padding rows
    # write to the sentinel region which is never read by _moe_sum.
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = valid_row_mask[:, None] & (offs_cn[None, :] < N)
    gl.store(c_ptrs, c, mask=c_mask)

    # Final sync: wait for all in-flight async copies, then barrier
    gl.metax.gvm_arrive(0)
    gl.metax.barrier_shared()
    a_smem._keep_alive()
    b_smem._keep_alive()


# ============================================================================
# Simplified MoE GEMM Kernel — compiler-managed pipeline fallback
# ============================================================================
#
# If pipeline="cpasync" conflicts with the manual gl.metax pipeline above,
# use this simplified kernel instead. It removes gl.local_alloc / gl.metax
# and relies on the compiler to manage the K-direction software pipeline
# (async copy + multi-stage buffering) via pipeline="cpasync" + num_stages.
#
# To switch: replace moe_gemm_kernel with moe_gemm_kernel_simple in
# invoke_moe_gemm() below.

@gluon.jit
def moe_gemm_kernel_simple(
    # Pointers
    a_ptr, b_ptr, c_ptr, sorted_token_ids_ptr, expert_ids_ptr, topk_weights_ptr,
    # Dimensions
    M, N, K, EM, num_valid_tokens,
    # Strides
    stride_am, stride_ak, stride_be, stride_bn, stride_bk, stride_cm, stride_cn, stride_wm,  # topk_weights stride
    # Constexprs
    topk: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
    DTYPE: gl.constexpr,  # 'bfloat16' | 'float16'
    MUL_ROUTED_WEIGHT: gl.constexpr,
    GATHER_BY_TOKEN: gl.constexpr,  # True: A=[M,K], use offs_token//topk; False: A=[M*topk,N], use offs_token
    ):
    """Simplified MoE GEMM: no manual pipeline, compiler-managed async copy.

    NOTE: No early-return / return statements are used inside this kernel.
    C500 Gluon layout requires structured control flow (SCF) before
    SCF-to-CF lowering — any `return` inside an `if` breaks the SCF
    invariant and triggers "C500 async queue analysis requires structured
    control flow before SCF-to-CF lowering".  Instead, we guard all
    computation with mask predicates so the kernel has a single linear
    control flow path (the K-loop body is the only structured region).
    """
    # ---- 1. Grouped CTA mapping (L2 reuse) ----
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. Guard predicates (no return — C500 needs structured control flow) ----
    tile_valid = pid_m * BLOCK_M < EM

    # ---- 3. Read sorted_token_ids ----
    offs_token_id = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
    offs_token = gl.load(sorted_token_ids_ptr + offs_token_id, mask=tile_valid, other=M * topk).to(gl.int64)

    # ---- 4. Valid row mask (epilogue only — K-loop uses zero-sentinel) ----
    # Plan B: padding rows in sorted_token_ids point to zero-sentinel row.
    # K-loop A loads are uniform (no mask), so valid_row_mask is only for
    # the epilogue (C store, weight load).
    valid_row_mask = (offs_token < num_valid_tokens) & tile_valid

    # ---- 5. Expert id ----
    off_experts = gl.load(expert_ids_ptr + pid_m, mask=tile_valid, other=0).to(gl.int64)
    expert_valid = (off_experts != -1) & tile_valid

    # ---- 6. Gather rows ----
    # GEMM1: A = hidden_states [M, K] — only M rows, index by token (offs_token // topk)
    # GEMM2: A = cache2 [M*topk, N] — M*topk rows, index by flat (offs_token directly)
    if GATHER_BY_TOKEN:
        real_token_rows = offs_token // topk
    else:
        real_token_rows = offs_token

    # ---- 7. Router weights ----
    # Plan B: use valid_row_mask — padding rows read 0 weight
    moe_weight = gl.zeros((BLOCK_M, ), dtype=gl.float32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = gl.load(
            topk_weights_ptr + offs_token,
            mask=valid_row_mask,
            other=0.0,
        )

    # ---- 8. Index offsets ----
    offs_k = gl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N)
    b_base = b_ptr + off_experts * stride_be
    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # ---- 9. Dtypes ----
    if DTYPE == 'bfloat16':
        output_dtype = gl.bfloat16
    else:
        output_dtype = gl.float16

    # ---- 10. Simple K-loop (compiler manages pipeline) ----
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)

    for k in range(num_k_tiles):
        k_off = k * BLOCK_K

        # Plan B: A loads are uniform (no token_mask) — padding rows point to
        # zero-sentinel. Only tile_valid (index-based, uniform) for boundary.
        a_ptrs = a_ptr + real_token_rows[:, None] * stride_am + (k_off + offs_k)[None, :] * stride_ak
        a_mask = tile_valid & ((k_off + offs_k)[None, :] < K)

        # B: same expert, this K-tile
        b_ptrs = b_base + (k_off + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = expert_valid & ((k_off + offs_k)[:, None] < K)

        a = gl.load(a_ptrs, mask=a_mask, other=0.0)
        b = gl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = gl.dot(a, b, acc)

    # ---- 11. Epilogue ----
    if MUL_ROUTED_WEIGHT:
        acc = acc * moe_weight[:, None]

    c = gl.cast(acc, output_dtype)

    # Plan B: store uses valid_row_mask — padding rows write to the sentinel
    # region which is never read by _moe_sum.
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = valid_row_mask[:, None] & (offs_cn[None, :] < N)
    gl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# Kernel dispatch helper
# ============================================================================

_DTYPE_STR = {
    torch.float16: 'float16',
    torch.bfloat16: 'bfloat16',
}


def invoke_moe_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk: int,
    mul_routed_weight: bool,
    num_tokens: int,  # original M (number of input tokens, before topk expansion)
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    NUM_BUFFERS: int,
    num_warps: int,
    GROUP_SIZE_M: int,
    use_manual_pipeline: bool = False,
    gather_by_token: bool = True,
):
    """Launch moe_gemm_kernel with the given configuration.

    Args:
        num_tokens: The original number of input tokens M (before topk expansion).
            A.size(0) may be M*topk (for GEMM2 where A=cache2), so we cannot
            use A.size(0) to derive num_valid_tokens.  num_valid_tokens in the
            kernel is always num_tokens * topk = M * topk.
        use_manual_pipeline: If True, use moe_gemm_kernel with manual gl.metax
            double-buffered pipeline + compiler-managed async copy (pipeline="cpasync").
            If False, use moe_gemm_kernel_simple with compiler-managed pipeline only.
            Set to False if pipeline="cpasync" conflicts with manual gl.metax barriers.
            Default is False because C500 Gluon layout cannot handle non-uniform
            masks (token_mask from MoE gather) on swizzled async_copy destinations.
        gather_by_token: If True, compute A row index as offs_token // topk
            (for GEMM1 where A = hidden_states [M, K], only M rows).
            If False, use offs_token directly as A row index
            (for GEMM2 where A = cache2 [M*topk, N], M*topk rows).
            Default is True.
    """
    K = A.size(1)
    N = B.size(1)  # B is [E, N, K]
    EM = sorted_token_ids.size(0)
    num_valid_tokens = num_tokens * topk

    dtype_str = _DTYPE_STR[A.dtype]

    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    if use_manual_pipeline:
        # Manual gl.metax double-buffer pipeline + compiler async copy overlay.
        # If pipeline="cpasync" conflicts with manual gl.metax barriers,
        # set use_manual_pipeline=False to use the simplified kernel.
        moe_gemm_kernel[grid](
            A, B, C, sorted_token_ids, expert_ids, topk_weights if mul_routed_weight else None, num_tokens, N, K, EM,
            num_valid_tokens, A.stride(0), A.stride(1), B.stride(0), B.stride(1), B.stride(2), C.stride(0), C.stride(1),
            topk_weights.stride(0) if topk_weights is not None else 0, topk=topk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M, NUM_BUFFERS=NUM_BUFFERS, DTYPE=dtype_str,
            MUL_ROUTED_WEIGHT=mul_routed_weight, GATHER_BY_TOKEN=gather_by_token, num_warps=num_warps,
            num_stages=NUM_BUFFERS,  # pipeline stages = double-buffer count
            pipeline="cpasync",  # enable async Global→Shared copy
        )
    else:
        # Compiler-managed pipeline with async Global→Shared copy.
        # Plan B (zero-sentinel): padding rows in sorted_token_ids point to a
        # zero-sentinel row in the padded input, so all K-loop loads are uniform
        # (no token_mask). This satisfies cpasync's C500 swizzled shared memory
        # layout constraint. Both cpasync and swizzle now work simultaneously.
        moe_gemm_kernel_simple[grid](
            A, B, C, sorted_token_ids, expert_ids, topk_weights if mul_routed_weight else None, num_tokens, N, K, EM,
            num_valid_tokens, A.stride(0), A.stride(1), B.stride(0), B.stride(1), B.stride(2), C.stride(0), C.stride(1),
            topk_weights.stride(0) if topk_weights is not None else 0, topk=topk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M, DTYPE=dtype_str, MUL_ROUTED_WEIGHT=mul_routed_weight,
            GATHER_BY_TOKEN=gather_by_token, num_warps=num_warps, num_stages=NUM_BUFFERS,  # pipeline stages
            pipeline="cpasync",  # async Global→Shared copy (Plan B: uniform masks → C500 OK)
        )


# ============================================================================
# Activation: SiLU-and-mul (small Gluon kernel)
# ============================================================================

@gluon.jit
def _silu_and_mul_kernel(
    x_ptr, out_ptr,
    N_half,          # N / 2 (intermediate size)
    stride_x, stride_out,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
):
    """x = [ROWS, N], gate = x[:, :N/2], up = x[:, N/2:]
       out = silu(gate) * up,  silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    pid_m = gl.program_id(axis=0)
    row = pid_m * ROWS + gl.arange(0, ROWS)
    col = gl.arange(0, COLS)

    # Gate: first N/2 columns
    gate_ptrs = x_ptr + row[:, None] * stride_x + col[None, :]
    gate = gl.load(gate_ptrs)

    # Up: second N/2 columns
    up_ptrs = x_ptr + row[:, None] * stride_x + (col + N_half)[None, :]
    up = gl.load(up_ptrs)

    # SiLU: gate * sigmoid(gate) = gate / (1 + exp(-gate))
    # Upcast to fp32 for gl.exp (only supports fp32/fp64, not bf16/fp16)
    gate_f = gate.to(gl.float32)
    silu_gate_f = gate_f / (1.0 + gl.exp(-gate_f))
    silu_gate = silu_gate_f.to(out_ptr.dtype.element_ty)

    # Output
    out = silu_gate * up
    out_ptrs = out_ptr + row[:, None] * stride_out + col[None, :]
    gl.store(out_ptrs, out)


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Apply SiLU(gate) * up activation.

    Args:
        x: [M*topk, N] where N = 2 * intermediate_size.
           First half is gate, second half is up.

    Returns:
        [M*topk, N/2] activation output.
    """
    rows, N = x.shape
    assert N % 2 == 0
    N_half = N // 2
    out = torch.empty((rows, N_half), device=x.device, dtype=x.dtype)

    # Each CTA processes one row
    BLOCK_ROWS = 1
    n_ctas = triton.cdiv(rows, BLOCK_ROWS)

    _silu_and_mul_kernel[(n_ctas,)](
        x, out, N_half,
        x.stride(0), out.stride(0),
        ROWS=BLOCK_ROWS, COLS=N_half,
    )
    return out


# ============================================================================
# MoE weighted sum (small Gluon kernel)
# ============================================================================

@gluon.jit
def _moe_sum_kernel(
    topk_weights_ptr,
    cache3_ptr,
    out_ptr,
    M, K, topk: gl.constexpr,
    stride_wm, stride_wk,
    stride_c3_m, stride_c3_k, stride_c3_topk,
    stride_out_m, stride_out_k,
    BLOCK_K: gl.constexpr,
):
    """out[m, k] = Σ_j topk_weights[m, j] * cache3[m, j, k]"""
    pid_m = gl.program_id(axis=0)
    pid_k = gl.program_id(axis=1)

    m = pid_m
    k_off = pid_k * BLOCK_K + gl.arange(0, BLOCK_K)
    k_mask = k_off < K

    acc = gl.zeros((BLOCK_K,), dtype=gl.float32)

    for j in range(topk):
        w = gl.load(topk_weights_ptr + m * stride_wm + j * stride_wk)
        v_ptrs = cache3_ptr + m * stride_c3_m + j * stride_c3_topk + k_off * stride_c3_k
        v = gl.load(v_ptrs, mask=k_mask, other=0.0)
        acc = acc + w * v

    out_ptrs = out_ptr + m * stride_out_m + k_off * stride_out_k
    out_dtype = out_ptr.dtype.element_ty
    gl.store(out_ptrs, acc.to(out_dtype), mask=k_mask)


def _moe_sum(
    topk_weights: torch.Tensor,
    cache3: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum along the topk dimension.

    Args:
        topk_weights: [M, topk] router weights.
        cache3: [M, topk, K] second GEMM output.

    Returns:
        [M, K] output.
    """
    M, topk, K = cache3.shape
    out = torch.empty((M, K), device=cache3.device, dtype=cache3.dtype)

    # Choose BLOCK_K as the largest power-of-2 ≤ min(K, 256)
    BLOCK_K = min(K, 256)
    BLOCK_K = 1 << (BLOCK_K.bit_length() - 1)
    if BLOCK_K < 1:
        BLOCK_K = 1

    grid = (M, triton.cdiv(K, BLOCK_K))

    _moe_sum_kernel[grid](
        topk_weights, cache3, out,
        M, K, topk,
        topk_weights.stride(0), topk_weights.stride(1),
        cache3.stride(0), cache3.stride(2), cache3.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return out


# ============================================================================
# Top-level fused MoE orchestration
# ============================================================================

def fused_moe(
    hidden_states: torch.Tensor,    # [M, K]
    w1: torch.Tensor,               # [E, N, K], N = 2*intermediate
    w2: torch.Tensor,               # [E, K, intermediate]
    topk_ids: torch.Tensor,         # [M, top_k]
    topk_weights: torch.Tensor,     # [M, top_k]
    *,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:                  # [M, K]
    """Fused MoE: two GEMMs + activation + weighted sum.

    First version: bf16/fp16 only (no quantization).

    Args:
        hidden_states: Input tensor [M, K].
        w1: First expert weights [E, N, K] where N = 2 * intermediate_size.
        w2: Second expert weights [E, K, intermediate_size].
        topk_ids: Expert assignments [M, top_k].
        topk_weights: Router weights [M, top_k].
        apply_router_weight_on_input: If True, multiply router weights in
            the first GEMM epilogue (better for decode numerical stability).
            If False, multiply in the second GEMM epilogue (standard MoE).

    Returns:
        Output tensor [M, K].
    """
    # ---- ① Constraint checks ----
    M, K = hidden_states.shape
    E, N, Kw = w1.shape
    E2, K_out, intermediate = w2.shape
    topk = topk_ids.shape[1]

    assert hidden_states.dtype in (torch.float16, torch.bfloat16), \
        f"First version supports bf16/fp16 only, got {hidden_states.dtype}"
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert K == Kw, f"K mismatch: hidden_states K={K}, w1 K={Kw}"
    assert E == E2, f"Expert count mismatch: w1 E={E}, w2 E={E2}"
    assert N == 2 * intermediate, \
        f"w1 N={N} must equal 2*intermediate={2*intermediate}"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert topk_weights.stride(1) == 1, "topk_weights must be contiguous in topk dim"

    # ---- ② Select config ----
    # GEMM1 and GEMM2 have different dimension characteristics:
    #   GEMM1: A=[M, K], B=[E, N, K], shared dim=K, output cols=N=2*intermediate
    #   GEMM2: A=[M*topk, intermediate], B=[E, K_out, intermediate], shared dim=intermediate, output cols=K_out
    # BLOCK_M must be shared (determines sorted_token_ids alignment & expert_ids layout),
    # but BLOCK_N, BLOCK_K, NUM_BUFFERS, num_warps can differ.
    BLOCK_M1, BLOCK_N1, BLOCK_K1, NUM_BUFFERS1, num_warps1, GROUP_SIZE_M1 = \
        _select_config(M, N, K, E, topk)
    # GEMM2: pass M (not M*topk) to _select_config to avoid inflated tokens_per_expert.
    # The actual GEMM has M*topk rows in A, but the expert distribution is the same
    # as GEMM1 (each expert still gets ~M*topk/E tokens). We pass M with topk=1
    # so tokens_per_expert = M/E, which gives a realistic tiling selection for the
    # per-expert tile size. The output dimensions (N=K_out, K=intermediate) are the
    # real differentiator for GEMM2's tiling.
    _, BLOCK_N2, BLOCK_K2, NUM_BUFFERS2, num_warps2, GROUP_SIZE_M2 = \
        _select_config(M, K_out, intermediate, E, topk)
    # Shared BLOCK_M: use GEMM1's value (it reflects the true M, while GEMM2's
    # BM is based on M*topk which inflates tokens_per_expert). The padding
    # alignment must match GEMM1's M-block layout.
    BLOCK_M = BLOCK_M1

    # ---- ③ Allocate caches ----
    # cache1 and cache3 share memory (not simultaneously live)
    # C layout: [M, topk, N_dim] flattened as [M*topk, N_dim] for the kernel
    cache13 = torch.empty(
        M * topk * max(N, K_out),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache1 = cache13[:M * topk * N].view(M * topk, N)
    cache3 = cache13[:M * topk * K_out].view(M * topk, K_out)

    # cache2 needs separate memory (used concurrently with cache1)
    # Plan B: allocate one extra row for the zero-sentinel. The sentinel row
    # at index M*topk is written by GEMM1 for padding entries but never read
    # by _moe_sum (which only sums rows 0..M*topk-1).
    # Use torch.zeros so the sentinel row is zeroed in the allocation itself,
    # avoiding the separate .zero_() launch.
    cache2 = torch.zeros(
        (M * topk + 1, intermediate),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # ---- ④ Plan B: pad input with zero-sentinel row ----
    # Append one zero row to hidden_states so that padding entries in
    # sorted_token_ids (which point to sentinel_idx = M * topk) load zeros
    # from A. This makes K-loop A loads uniform (no token_mask), which is
    # required for cpasync's C500 swizzled shared memory layout.
    #
    # Optimization: use torch.zeros + slice assign instead of empty + copy + zero.
    # torch.zeros does cudaMalloc+memset in one launch; slice assign is a
    # D2D memcpy. This saves one kernel launch vs empty+copy+zero (3 launches → 2).
    hidden_states_padded = torch.zeros(
        (M + 1, K), device=hidden_states.device, dtype=hidden_states.dtype,
    )
    hidden_states_padded[:M].copy_(hidden_states)

    # Pad topk_weights with a zero at the sentinel index so that padding
    # rows in the epilogue read weight=0 and contribute nothing.
    topk_weights_padded = torch.zeros(
        M * topk + 1, device=hidden_states.device, dtype=topk_weights.dtype,
    )
    topk_weights_padded[:M * topk].copy_(topk_weights.flatten())

    # Sentinel index for sorted_token_ids padding fill
    sentinel_idx = M * topk

    # ---- ⑤ Expert assignment (sort + pad) ----
    # Plan B: pass sentinel_idx so padding rows in sorted_token_ids point to
    # the zero-sentinel row instead of an out-of-bounds index.
    # Use GEMM1's BLOCK_M for alignment since it determines the M-block layout.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, E, sentinel_idx=sentinel_idx,
    )

    # ---- ⑥ First GEMM: hidden_states @ w1^T → cache1 ----
    # Plan B: pass padded hidden_states (with zero-sentinel row) and padded
    # topk_weights. num_tokens stays M because the sentinel row is at index M
    # in hidden_states_padded (GATHER_BY_TOKEN: offs_token // topk = M*topk // topk = M).
    invoke_moe_gemm(
        hidden_states_padded, w1, cache1,
        topk_weights_padded, sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk=topk,
        mul_routed_weight=apply_router_weight_on_input,
        num_tokens=M,
        gather_by_token=True,     # GEMM1: A = hidden_states_padded [M+1, K], index by token
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
        NUM_BUFFERS=NUM_BUFFERS1, num_warps=num_warps1, GROUP_SIZE_M=GROUP_SIZE_M1,
    )

    # ---- ⑦ Activation: silu(gate) * up ----
    act_out = _silu_and_mul(cache1)
    # Copy activation output to cache2 rows [0, M*topk). The sentinel row
    # at cache2[M*topk] stays zero (initialized above).
    cache2[:M * topk].copy_(act_out)

    # ---- ⑨ Second GEMM: cache2 @ w2^T → cache3 ----
    # Plan B: pass cache2 (with zero-sentinel row at index M*topk) and
    # padded topk_weights. Uses GEMM2's separate N/K/buffers config.
    invoke_moe_gemm(
        cache2, w2, cache3,
        topk_weights_padded, sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk=topk,  # sorted_token_ids still encodes topk grouping
        mul_routed_weight=not apply_router_weight_on_input,
        num_tokens=M,
        gather_by_token=False,    # GEMM2: A = cache2 [M*topk+1, intermediate], index by flat
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
        NUM_BUFFERS=NUM_BUFFERS2, num_warps=num_warps2, GROUP_SIZE_M=GROUP_SIZE_M2,
    )

    # ---- ⑩ Weighted sum along topk dimension ----
    # Reshape cache3 to [M, topk, K_out] for the moe_sum
    cache3_3d = cache3.view(M, topk, K_out)
    if apply_router_weight_on_input:
        # Router weight was applied in GEMM1 epilogue; GEMM2 did NOT apply it.
        # So cache3 already contains routed_weight * w2(act(w1(x))).
        # We just need a plain sum over topk (no weight multiplication).
        output = _moe_sum(torch.ones_like(topk_weights), cache3_3d)
    else:
        # Router weight was NOT applied in GEMM1, so it was applied in GEMM2 epilogue.
        # cache3 contains routed_weight * w2(act(w1(x))).
        # Same as above: just plain sum, weights already baked in.
        output = _moe_sum(torch.ones_like(topk_weights), cache3_3d)

    return output
