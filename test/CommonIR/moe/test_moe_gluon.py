"""Tests for mcTriton/Gluon Fused MoE kernel.

Correctness tests compare the fused_moe output against a per-expert
PyTorch reference implementation.
"""

import pytest
import torch

from .moe_gluon import fused_moe, _silu_and_mul, _moe_sum
from .moe_align_block_size import moe_align_block_size


# ============================================================================
# Reference implementation (per-expert PyTorch loop)
# ============================================================================

def fused_moe_ref(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Reference MoE: per-expert PyTorch matmul loop.

    When apply_router_weight_on_input=False (standard MoE):
        output[m] = Σ_k weight[m,k] · W2_k(silu_and_mul(W1_k(x_m)))

    When apply_router_weight_on_input=True (decode-optimized path):
        output[m] = Σ_k silu_and_mul(weight[m,k] · W1_k(x_m)) @ W2_k^T
        The weight is applied BEFORE the non-linear activation, which is
        mathematically different from the standard formula but reduces
        intermediate data movement during decode.
    """
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
    intermediate = N // 2

    output = torch.zeros_like(hidden_states)

    for m in range(M):
        for k_idx in range(topk):
            expert = topk_ids[m, k_idx].item()
            weight = topk_weights[m, k_idx].item()

            # GEMM1: x @ w1[expert]^T
            x = hidden_states[m]  # [K]
            gate_up = x @ w1[expert].t()  # [N]

            if apply_router_weight_on_input:
                # Weight applied BEFORE activation (decode-optimized path)
                gate_up = weight * gate_up

            # Activation: silu(gate) * up
            gate = gate_up[:intermediate]
            up = gate_up[intermediate:]
            act = torch.nn.functional.silu(gate) * up  # [intermediate]

            # GEMM2: act @ w2[expert]^T
            y = act @ w2[expert].t()  # [K]

            if not apply_router_weight_on_input:
                # Weight applied AFTER both GEMMs (standard path)
                y = weight * y

            output[m] += y

    return output


# ============================================================================
# Test shapes: (M, E, topk, K, intermediate)
# ============================================================================

_TEST_SHAPES = [(16, 8, 2, 1024, 512),  # decode small batch
                (128, 8, 2, 1024, 512),  # medium batch
                (64, 4, 2, 512, 256),  # small, tile-friendly
                (48, 4, 2, 512, 256),  # non-tile-aligned
                ]

_DTYPES = [torch.bfloat16, torch.float16]


# ============================================================================
# MoE align block size tests
# ============================================================================

class TestMoeAlignBlockSize:
    """Tests for the route sorting + padding utility."""

    @pytest.mark.parametrize("M, E, topk", [(16, 8, 2), (64, 4, 4), (128, 8, 2)])
    def test_basic_properties(self, M, E, topk):
        """sorted_token_ids and expert_ids have the right shapes and values."""
        topk_ids = torch.randint(0, E, (M, topk), device="cuda", dtype=torch.int32)
        block_size = 16

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size,
            E,
        )

        # num_tokens_post_padded is a multiple of block_size
        total_padded = num_tokens_post_padded.item()
        assert total_padded % block_size == 0

        # All valid entries in sorted_token_ids are < M * topk
        valid_mask = sorted_token_ids < M * topk
        assert valid_mask.sum().item() == M * topk

        # expert_ids has no -1 entries (all blocks assigned)
        assert (expert_ids >= 0).all().item()

    @pytest.mark.parametrize("M, E, topk", [(16, 8, 2), (32, 4, 2)])
    def test_coverage(self, M, E, topk):
        """Every (token, topk) pair appears exactly once in sorted_token_ids."""
        topk_ids = torch.randint(0, E, (M, topk), device="cuda", dtype=torch.int32)
        block_size = 16

        sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size, E)

        # Extract valid entries
        valid = sorted_token_ids[sorted_token_ids < M * topk].sort()[0]
        expected = torch.arange(M * topk, dtype=torch.int32, device="cuda")
        assert torch.equal(valid, expected)


# ============================================================================
# Fused MoE correctness tests
# ============================================================================

class TestFusedMoe:
    """End-to-end correctness tests for fused_moe."""

    @pytest.mark.parametrize("shape", _TEST_SHAPES)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_correctness(self, shape, dtype):
        """Fused MoE output matches per-expert PyTorch reference."""
        M, E, topk, K, intermediate = shape
        N = 2 * intermediate

        device = "cuda"
        torch.manual_seed(42)

        hidden_states = torch.randn(M, K, device=device, dtype=dtype)
        w1 = torch.randn(E, N, K, device=device, dtype=dtype) * 0.1
        w2 = torch.randn(E, K, intermediate, device=device, dtype=dtype) * 0.1
        topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
        topk_weights = torch.rand(M, topk, device=device, dtype=dtype)
        # Ensure contiguous
        topk_weights = topk_weights.contiguous()

        # Reference
        ref_output = fused_moe_ref(
            hidden_states,
            w1,
            w2,
            topk_ids,
            topk_weights,
        )

        # Fused
        fused_output = fused_moe(
            hidden_states,
            w1,
            w2,
            topk_ids,
            topk_weights,
        )

        # Tolerance: bf16 has ~1e-2 precision, fp16 ~1e-3
        # if dtype == torch.bfloat16:
        #     atol, rtol = 5e-1, 1e-1,
        # else:
        #     atol, rtol = 5e-2, 5e-2
        if dtype == torch.bfloat16:
            atol, rtol = 5e-1, 1e-1
        else:
            atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(fused_output, ref_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("shape", _TEST_SHAPES)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_router_weight_on_input(self, shape, dtype):
        """每条路径 vs 对应 reference（两条路径数学不等价，不能互比）"""
        M, E, topk, K, intermediate = shape
        N = 2 * intermediate

        device = "cuda"
        torch.manual_seed(123)

        hidden_states = torch.randn(M, K, device=device, dtype=dtype)
        w1 = torch.randn(E, N, K, device=device, dtype=dtype) * 0.1
        w2 = torch.randn(E, K, intermediate, device=device, dtype=dtype) * 0.1
        topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
        topk_weights = torch.rand(M, topk, device=device, dtype=dtype).contiguous()

        if dtype == torch.bfloat16:
            atol, rtol = 5e-1, 1e-1
        else:
            atol, rtol = 5e-2, 5e-2

        for apply_rw in (False, True):
            ref = fused_moe_ref(
                hidden_states, 
                w1, 
                w2, 
                topk_ids,
                topk_weights,
                apply_router_weight_on_input=apply_rw,
            )
            fused = fused_moe(
                hidden_states, 
                w1,
                w2, 
                topk_ids,
                topk_weights,
                apply_router_weight_on_input=apply_rw,
            )
            torch.testing.assert_close(fused, ref, atol=atol, rtol=rtol)

    def test_output_shape(self):
        """Output shape matches input shape."""
        M, E, topk, K, intermediate = 16, 4, 2, 512, 256
        N = 2 * intermediate

        hidden_states = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.1
        w2 = torch.randn(E, K, intermediate, device="cuda", dtype=torch.bfloat16) * 0.1
        topk_ids = torch.randint(0, E, (M, topk), device="cuda", dtype=torch.int32)
        topk_weights = torch.rand(M, topk, device="cuda", dtype=torch.bfloat16).contiguous()

        output = fused_moe(hidden_states, w1, w2, topk_ids, topk_weights)
        assert output.shape == (M, K)
        assert output.dtype == hidden_states.dtype


# ============================================================================
# Silu-and-mul unit test
# ============================================================================

class TestSiluAndMul:
    """Tests for the _silu_and_mul activation kernel."""

    def test_correctness(self):
        M, N_half = 32, 128
        N = 2 * N_half

        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        gate = x[:, :N_half]
        up = x[:, N_half:]
        expected = torch.nn.functional.silu(gate) * up

        result = _silu_and_mul(x)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


# ============================================================================
# MoE sum unit test
# ============================================================================

class TestMoeSum:
    """Tests for the _moe_sum weighted reduction kernel."""

    def test_correctness(self):
        M, topk, K = 16, 4, 256

        topk_weights = torch.rand(M, topk, device="cuda", dtype=torch.bfloat16).contiguous()
        cache3 = torch.randn(M, topk, K, device="cuda", dtype=torch.bfloat16)

        # Reference: einsum
        expected = torch.einsum("mk,mkd->md", topk_weights.float(), cache3.float()).to(torch.bfloat16)

        result = _moe_sum(topk_weights, cache3)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
