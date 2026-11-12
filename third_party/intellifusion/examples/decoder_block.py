"""E22: a small causal decoder block composed from E01/E03/E04/E08 kernels."""

import sys

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from dense_linear_persistent import dense_linear_persistent
from flash_attention_prefill import flash_attention_prefill
from residual_rmsnorm import residual_rmsnorm
from swiglu_geglu import swiglu_geglu


@triton.jit
def _residual_add(A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, other=0).to(tl.float32)
    b = tl.load(B + i, i < N, other=0).to(tl.float32)
    tl.store(Y + i, a + b, i < N)


def _add(a, b):
    out = torch.empty_like(a)
    _residual_add[(triton.cdiv(a.numel(), 256), )](a, b, out, a.numel(), 256, enable_fp_fusion=False)
    return out


def decoder_block(x, norm1_weight, qkv_weight, output_weight, norm2_weight, gate_weight, up_weight, down_weight, *,
                  heads, eps=1e-6):
    """Return [B,S,H] BF16 IFWA output; H=heads*D, D in [16,128].

    All weights are contiguous BF16 on the same IFWA device. QKV weight is
    [3H,H]; O [H,H]; gate/up [I,H]; down [H,I]. No bias, dropout, cache,
    training gradients, or parallel residual path.
    """
    if x.ndim != 3 or x.dtype != torch.bfloat16 or x.device.type != "ifwa" or not x.is_contiguous():
        raise ValueError("x must be contiguous BF16 IFWA [B,S,H]")
    batch, seq, hidden = x.shape
    if min(batch, seq, hidden) < 1 or heads < 1 or hidden % heads or not 16 <= hidden // heads <= 128:
        raise ValueError("invalid batch/sequence/hidden/head dimensions")
    intermediate = gate_weight.shape[0] if gate_weight.ndim == 2 else 0
    shapes = ((hidden, ), (3 * hidden, hidden), (hidden, hidden), (hidden, ), (intermediate, hidden),
              (intermediate, hidden), (hidden, intermediate))
    weights = (norm1_weight, qkv_weight, output_weight, norm2_weight, gate_weight, up_weight, down_weight)
    if intermediate < 1 or any(
            w.shape != shape or w.dtype != x.dtype or w.device != x.device or not w.is_contiguous() or w.requires_grad
            for w, shape in zip(weights, shapes)) or x.requires_grad:
        raise ValueError("weight shape/device/dtype/contiguity mismatch or autograd requested")
    normed = residual_rmsnorm(x.view(batch * seq, hidden), norm1_weight, eps=eps)
    projected = dense_linear_persistent(normed, qkv_weight).to(x.dtype).view(batch, seq, 3, heads, hidden // heads)
    q, k, v = (projected[:, :, i].permute(0, 2, 1, 3).contiguous() for i in range(3))
    attended = flash_attention_prefill(q, k, v, causal=True)
    attended = attended.permute(0, 2, 1, 3).contiguous().view(batch * seq, hidden)
    attention_out = dense_linear_persistent(attended, output_weight)
    first_residual = _add(x.view(batch * seq, hidden), attention_out)
    normed2 = residual_rmsnorm(first_residual, norm2_weight, eps=eps)
    gate = dense_linear_persistent(normed2, gate_weight)
    up = dense_linear_persistent(normed2, up_weight)
    activated = swiglu_geglu(gate, up).to(x.dtype)
    down = dense_linear_persistent(activated, down_weight)
    return _add(first_residual, down).view(batch, seq, hidden)


def _cpu_norm(x, weight, eps):
    fp = x.float()
    return (fp * torch.rsqrt(fp.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def reference(x, weights, heads, eps):
    nw1, qkv_w, o_w, nw2, gate_w, up_w, down_w = weights
    batch, seq, hidden = x.shape
    normed = _cpu_norm(x, nw1, eps)
    projected = (normed.float().reshape(-1, hidden) @ qkv_w.float().T).to(x.dtype)
    projected = projected.view(batch, seq, 3, heads, hidden // heads)
    q, k, v = (projected[:, :, i].permute(0, 2, 1, 3).float() for i in range(3))
    scores = (q @ k.transpose(-1, -2)) / (hidden // heads)**0.5
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, -float("inf"))
    attended = (torch.softmax(scores, -1) @ v).to(x.dtype).permute(0, 2, 1, 3).reshape(-1, hidden)
    attention_out = attended.float() @ o_w.float().T
    first = (x.reshape(-1, hidden).float() + attention_out).to(x.dtype)
    normed2 = _cpu_norm(first, nw2, eps)
    gate = normed2.float() @ gate_w.float().T
    up = normed2.float() @ up_w.float().T
    activated = (F.silu(gate) * up).to(x.dtype)
    down = activated.float() @ down_w.float().T
    return (first.float() + down).to(x.dtype).view(batch, seq, hidden)


@pytest.mark.parametrize("batch,seq,hidden,heads,intermediate", [(1, 17, 32, 2, 64), (2, 5, 64, 2, 64)])
def test_decoder_block(batch, seq, hidden, heads, intermediate):
    generator = torch.Generator().manual_seed(2201)
    x = (torch.randn((batch, seq, hidden), generator=generator) * 0.15).to(torch.bfloat16)
    weight_shapes = ((hidden, ), (3 * hidden, hidden), (hidden, hidden), (hidden, ), (intermediate, hidden),
                     (intermediate, hidden), (hidden, intermediate))
    weights = [(torch.randn(s, generator=generator) * 0.12).to(torch.bfloat16) for s in weight_shapes]
    expected = reference(x, weights, heads, 1e-6)
    actual = decoder_block(x.to("ifwa:0"), *(w.to("ifwa:0") for w in weights), heads=heads).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E22 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
