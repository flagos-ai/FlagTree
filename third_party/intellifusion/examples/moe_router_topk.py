"""E10 MoE routing: softmax/sigmoid, group-limited top-k and weight normalization."""

import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _route(Logits, Bias, Indices, Weights, E: tl.constexpr, G: tl.constexpr, TOP_G: tl.constexpr,
           GROUP_SCORE_N: tl.constexpr, K: tl.constexpr, USE_BIAS: tl.constexpr, SOFTMAX: tl.constexpr,
           NORMALIZE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, BLOCK)
    valid = e < E
    logits = tl.load(Logits + row * E + e, valid, other=0).to(tl.float32)
    if SOFTMAX:
        max_logit = tl.max(tl.where(valid, logits, -float("inf")), 0)
        exp_logits = tl.exp(logits - max_logit)
        probabilities = exp_logits / tl.sum(tl.where(valid, exp_logits, 0.0), 0)
    else:
        probabilities = 1.0 / (1.0 + tl.exp(-logits))
    if USE_BIAS:
        bias = tl.load(Bias + e, valid, other=0).to(tl.float32)
        score = probabilities + bias
    else:
        score = probabilities

    group_size: tl.constexpr = E // G
    groups = e // group_size
    allowed = tl.full((BLOCK, ), False, tl.int1)
    remaining_groups = tl.full((BLOCK, ), True, tl.int1)
    for _ in range(TOP_G):
        # Group score is the sum of its best one or two biased expert scores.
        best = tl.full((BLOCK, ), -float("inf"), tl.float32)
        second = tl.full((BLOCK, ), -float("inf"), tl.float32)
        for slot in range(group_size):
            candidate = tl.gather(score, groups * group_size + slot, 0)
            best_is_lower = candidate > best
            second = tl.where(best_is_lower, best, tl.maximum(second, candidate))
            best = tl.maximum(best, candidate)
        group_score = best + tl.where(GROUP_SCORE_N == 2, second, 0.0)
        representative = valid & (e % group_size == 0) & remaining_groups
        selected_score = tl.max(tl.where(representative, group_score, -float("inf")), 0)
        selected_group = tl.min(tl.where(representative & (group_score == selected_score), groups, 2147483647), 0)
        allowed |= valid & (groups == selected_group)
        remaining_groups &= groups != selected_group

    scores_left = tl.where(allowed, score, -float("inf"))
    selected_sum = 0.0
    for k in range(K):
        peak = tl.max(scores_left, 0)
        index = tl.min(tl.where((scores_left == peak) & valid, e, 2147483647), 0)
        probability = tl.sum(tl.where(e == index, probabilities, 0.0), 0)
        tl.store(Indices + row * K + k, index)
        tl.store(Weights + row * K + k, probability)
        selected_sum += probability
        scores_left = tl.where(e == index, -float("inf"), scores_left)
    if NORMALIZE:
        for k in range(K):
            probability = tl.load(Weights + row * K + k)
            # Underflow can leave every selected sigmoid probability at zero.
            tl.store(Weights + row * K + k,
                     tl.where(selected_sum > 0, probability / tl.maximum(selected_sum, 1.0e-38), 1.0 / K))


def route(logits, k, *, groups=1, top_groups=1, group_score_n=1, activation="softmax", bias=None, normalize=True):
    """Return (expert_indices, weights); bias affects selection, not weights.

    Each row has E experts split into contiguous equal-size groups. Pick the
    top_groups by the sum of their best group_score_n biased expert scores,
    then choose k biased scores within those groups. Ties choose lower index.
    Normalized weights sum to one; input and bias are never mutated.
    If all selected probabilities underflow to zero, weights are uniform.
    """
    if logits.device.type != "ifwa" or logits.ndim != 2 or logits.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("logits must be IFWA FP32/BF16 [tokens, experts]")
    tokens, experts = logits.shape
    if not logits.is_contiguous() or experts < 1 or experts > 256 or experts & (experts - 1):
        raise ValueError("contiguous power-of-two experts in [1, 256] required")
    if groups < 1 or experts % groups or groups & (groups - 1):
        raise ValueError("groups must be a power-of-two divisor of experts")
    if not 1 <= top_groups <= groups or group_score_n not in (1, 2) or group_score_n > experts // groups:
        raise ValueError("invalid group selection")
    if not 1 <= k <= (experts // groups) * top_groups:
        raise ValueError("invalid k")
    if activation not in ("softmax", "sigmoid"):
        raise ValueError("activation must be softmax or sigmoid")
    if bias is not None and (bias.shape != (experts, ) or bias.device != logits.device or bias.dtype != logits.dtype
                             or not bias.is_contiguous()):
        raise ValueError("bias must be contiguous [experts] on the same device/dtype")
    if logits.requires_grad or (bias is not None and bias.requires_grad):
        raise ValueError("this inference example does not implement backward")
    indices = torch.empty((tokens, k), device=logits.device, dtype=torch.int32)
    weights = torch.empty((tokens, k), device=logits.device, dtype=torch.float32)
    if tokens:
        _route[(tokens, )](logits, bias if bias is not None else logits, indices, weights, experts, groups, top_groups,
                           group_score_n, k, bias is not None, activation == "softmax", normalize, experts, num_warps=4,
                           enable_fp_fusion=False)
    return indices, weights


def reference(logits, k, groups, top_groups, group_score_n, activation, bias, normalize):
    probabilities = torch.softmax(logits.float(), -1) if activation == "softmax" else torch.sigmoid(logits.float())
    scores = probabilities + (bias.float() if bias is not None else 0)
    result_indices = []
    result_weights = []
    for p, s in zip(probabilities, scores):
        group_size = len(s) // groups
        group_scores = [
            sum(sorted(s[g * group_size:(g + 1) * group_size].tolist(), reverse=True)[:group_score_n])
            for g in range(groups)
        ]
        selected_groups = sorted(range(groups), key=lambda g: (-group_scores[g], g))[:top_groups]
        candidates = [i for g in selected_groups for i in range(g * group_size, (g + 1) * group_size)]
        chosen = sorted(candidates, key=lambda i: (-s[i].item(), i))[:k]
        values = p[chosen]
        if normalize:
            values = values / values.sum() if values.sum() > 0 else torch.full_like(values, 1.0 / k)
        result_indices.append(chosen)
        result_weights.append(values)
    return torch.tensor(result_indices, dtype=torch.int32), torch.stack(result_weights)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("experts,groups,top_groups,group_score_n,k,bias_on", [(8, 1, 1, 1, 2, False),
                                                                               (16, 4, 2, 2, 3, True),
                                                                               (32, 4, 1, 1, 4, True)])
@pytest.mark.parametrize("normalize", [True, False])
def test_routing(activation, dtype, experts, groups, top_groups, group_score_n, k, bias_on, normalize):
    generator = torch.Generator().manual_seed(201)
    cpu_logits = torch.randn((2, experts), generator=generator).to(dtype)
    cpu_bias = (torch.randn(experts, generator=generator) / 5).to(dtype) if bias_on else None
    expected_i, expected_w = reference(cpu_logits, k, groups, top_groups, group_score_n, activation, cpu_bias,
                                       normalize)
    logits = cpu_logits.to("ifwa:0")
    bias = cpu_bias.to("ifwa:0") if cpu_bias is not None else None
    indices, weights = route(logits, k, groups=groups, top_groups=top_groups, group_score_n=group_score_n,
                             activation=activation, bias=bias, normalize=normalize)
    torch.testing.assert_close(indices.cpu(), expected_i, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_w, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(logits.cpu(), cpu_logits, rtol=0, atol=0)
    if bias is not None:
        torch.testing.assert_close(bias.cpu(), cpu_bias, rtol=0, atol=0)


def test_zero_probability_normalization():
    cpu_logits = torch.full((2, 8), -1000.0)
    indices, weights = route(cpu_logits.to("ifwa:0"), 2, activation="sigmoid")
    torch.testing.assert_close(indices.cpu(), torch.tensor([[0, 1], [0, 1]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), torch.full((2, 2), 0.5), rtol=0, atol=0)

    cpu_logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]])
    bias = torch.tensor([0.0, 2.0, 2.0, 0.0])
    indices, weights = route(cpu_logits.to("ifwa:0"), 2, activation="softmax", bias=bias.to("ifwa:0"))
    torch.testing.assert_close(indices.cpu(), torch.tensor([[1, 2]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), torch.full((1, 2), 0.5), rtol=0, atol=0)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E10 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
