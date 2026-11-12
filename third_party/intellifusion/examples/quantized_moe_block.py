"""E23: MoE route -> stable dispatch -> NVFP4 expert MLP -> weighted combine."""

import sys

import pytest
import torch
import torch.nn.functional as F
import triton

from moe_permute_combine import stable_permute, weighted_combine
from moe_router_topk import reference as route_reference
from moe_router_topk import route
from nvfp4_block_scaled_linear import nvfp4_block_scaled_linear, reference_quantize
from swiglu_geglu import swiglu_geglu


def quantized_moe_block(x, router_logits, gate_weight, up_weight, down_weight, *, k=2):
    """Return [T,64] FP32; expert matrices [E,64,64], FP32 on one IFWA device.

    Route by softmax then normalize selected weights. Stable dispatch duplicates
    tokens once per selected expert; empty experts receive no GEMM launch. Each
    expert computes a separately quantized NVFP4 gate/up/down path. The
    activation is FP32 SiLU*up before down-input quantization.
    """
    if x.ndim != 2 or x.shape[1] != 64 or x.shape[0] < 1:
        raise ValueError("x must have nonempty [T,64] shape")
    experts = gate_weight.shape[0] if gate_weight.ndim == 3 else 0
    if experts < 1 or experts > 8 or experts & (experts - 1) or not 1 <= k <= experts:
        raise ValueError("expert count must be power of two <=8 and 1<=k<=E")
    if router_logits.shape != (x.shape[0], experts) or any(t.shape != (experts, 64, 64)
                                                           for t in (gate_weight, up_weight, down_weight)):
        raise ValueError("router or expert weight shape mismatch")
    tensors = (x, router_logits, gate_weight, up_weight, down_weight)
    if x.device.type != "ifwa" or any(
            t.device != x.device or t.dtype != torch.float32 or not t.is_contiguous() or t.requires_grad
            for t in tensors):
        raise ValueError("all inputs must be contiguous FP32 inference IFWA tensors")
    ids, route_weights = route(router_logits, k, activation="softmax", normalize=True)
    packed_x, inverse, counts = stable_permute(x, ids, experts)
    counts_cpu = counts.cpu().tolist()
    expert_output = torch.empty_like(packed_x)
    offset = 0
    for expert, count in enumerate(counts_cpu):
        if count:
            rows = packed_x[offset:offset + count]
            gate, _ = nvfp4_block_scaled_linear(rows, gate_weight[expert])
            up, _ = nvfp4_block_scaled_linear(rows, up_weight[expert])
            activation = swiglu_geglu(gate, up)
            down, _ = nvfp4_block_scaled_linear(activation, down_weight[expert])
            expert_output[offset:offset + count].copy_(down)
        offset += count
    return weighted_combine(expert_output, inverse, route_weights)


def _quantized_gemm(x, weight):
    _, _, a = reference_quantize(x, 1.0)
    _, _, b = reference_quantize(weight, 1.0)
    return a @ b.T


def reference(x, logits, gate_weight, up_weight, down_weight, k):
    experts = gate_weight.shape[0]
    ids, weights = route_reference(logits, k, 1, 1, 1, "softmax", None, True)
    result = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(k):
            expert = int(ids[token, slot])
            rows = x[token:token + 1]
            gate = _quantized_gemm(rows, gate_weight[expert])
            up = _quantized_gemm(rows, up_weight[expert])
            activated = F.silu(gate) * up
            down = _quantized_gemm(activated, down_weight[expert])
            result[token] += down[0] * weights[token, slot]
    return result


@pytest.mark.parametrize("tokens,experts,k", [(2, 2, 1), (3, 4, 2)])
def test_quantized_moe(tokens, experts, k):
    generator = torch.Generator().manual_seed(2301)
    x = torch.randn((tokens, 64), generator=generator) * 0.3
    logits = torch.randn((tokens, experts), generator=generator)
    if experts == 2:
        logits[:, 0] += 10.0  # expert 1 must remain empty
    weights = [torch.randn((experts, 64, 64), generator=generator) * 0.15 for _ in range(3)]
    expected = reference(x, logits, *weights, k)
    actual = quantized_moe_block(x.to("ifwa:0"), logits.to("ifwa:0"), *(w.to("ifwa:0") for w in weights), k=k).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E23 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
