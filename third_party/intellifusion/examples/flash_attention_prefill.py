"""E08: tiled online-softmax prefill attention, including GQA and padding."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _prefill(Q, K, V, O, Lengths, S: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
             SCALE: tl.constexpr, CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    batch_head = tl.program_id(1)
    batch = batch_head // HQ
    head = batch_head % HQ
    kv_head = head // (HQ // HKV)
    length = tl.load(Lengths + batch)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, BD)
    q_base = (batch * HQ + head) * S * D
    kv_base = (batch * HKV + kv_head) * S * D
    q = tl.load(Q + q_base + rows[:, None] * D + dims[None, :], (rows[:, None] < length) & (dims[None, :] < D), other=0)
    running_max = tl.full((BM, ), -float("inf"), tl.float32)
    running_sum = tl.zeros((BM, ), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, tl.cdiv(length, BN)):
        keys = start * BN + cols
        k = tl.load(K + kv_base + dims[:, None] + keys[None, :] * D, (dims[:, None] < D) & (keys[None, :] < length),
                    other=0)
        scores = tl.dot(q, k, input_precision="ieee") * SCALE
        valid = keys[None, :] < length
        if CAUSAL:
            valid = valid & (keys[None, :] <= rows[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        v = tl.load(V + kv_base + keys[:, None] * D + dims[None, :], (keys[:, None] < length) & (dims[None, :] < D),
                    other=0)
        acc = acc * alpha[:, None]
        acc = tl.dot(probs.to(q.dtype), v, acc)
        running_sum = running_sum * alpha + tl.sum(probs, 1)
        running_max = new_max
    result = tl.where(rows[:, None] < length, acc / running_sum[:, None], 0)
    tl.store(O + q_base + rows[:, None] * D + dims[None, :], result, (rows[:, None] < S) & (dims[None, :] < D))


def flash_attention_prefill(q, k, v, *, causal=False, lengths=None, scale=None):
    """Return [B,Hq,S,D], computing softmax(QK^T * scale)V per GQA group.

    Inputs: contiguous IFWA FP16/BF16 tensors; K and V have [B,Hkv,S,D].
    D is 16..128. Hq must be divisible by Hkv. Positive CPU lengths <= S
    describe valid prefixes; padded query outputs are zero. Causal masking is
    top-left aligned (self-attention only). Intermediate softmax probabilities
    are cast to the input dtype for the PV dot; running statistics are FP32.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected Q[B,Hq,S,D], K/V[B,Hkv,S,D]")
    b, hq, s, d = q.shape
    if min(b, hq, s) <= 0 or not 16 <= d <= 128 or k.shape[1] <= 0:
        raise ValueError("nonempty dimensions and 16 <= head dimension <= 128 are required")
    if k.shape[0] != b or k.shape[2:] != (s, d) or hq % k.shape[1]:
        raise ValueError("K/V dimensions must match Q, with Hq divisible by Hkv")
    if q.device.type != "ifwa" or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q must be an IFWA FP16/BF16 tensor")
    if any(t.device != q.device or t.dtype != q.dtype or not t.is_contiguous() or t.requires_grad for t in (q, k, v)):
        raise ValueError("inputs must be contiguous inference tensors of the same dtype/device")
    if lengths is None:
        lengths = [s] * b
    lengths_cpu = torch.as_tensor(lengths)
    if lengths_cpu.device.type != "cpu" or lengths_cpu.shape != (b, ) or lengths_cpu.dtype not in (torch.int32,
                                                                                                   torch.int64):
        raise ValueError("lengths must contain one CPU integer per batch item")
    if not bool(((lengths_cpu >= 1) & (lengths_cpu <= s)).all()):
        raise ValueError("lengths must be in [1, S]")
    scale = d**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    device_lengths = lengths_cpu.to(dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    _prefill[(triton.cdiv(s, 16), b * hq)](q, k, v, out, device_lengths, s, hq, k.shape[1], d, scale, causal, 16, 32,
                                           triton.next_power_of_2(d), num_warps=4, num_stages=1)
    return out


def reference(q, k, v, causal, lengths, scale):
    out = torch.zeros_like(q)
    group = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths):
        for head in range(q.shape[1]):
            scores = q[batch, head, :length].float() @ k[batch, head // group, :length].float().T * scale
            if causal:
                scores.masked_fill_(torch.triu(torch.ones(length, length, dtype=torch.bool), 1), -float("inf"))
            out[batch, head, :length] = torch.softmax(scores, -1) @ v[batch, head // group, :length].float()
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape,lengths", [((1, 2, 2, 17, 16), [17]), ((2, 4, 2, 35, 32), [35, 19]),
                                           ((1, 2, 1, 67, 64), [67])])
def test_prefill(dtype, causal, shape, lengths):
    b, hq, hkv, s, d = shape
    generator = torch.Generator().manual_seed(808)
    q = torch.randn(b, hq, s, d, generator=generator).to(dtype)
    k = torch.randn(b, hkv, s, d, generator=generator).to(dtype)
    v = torch.randn(b, hkv, s, d, generator=generator).to(dtype)
    scale = 0.7 / math.sqrt(d)
    expected = reference(q, k, v, causal, lengths, scale)
    device_inputs = [t.to("ifwa:0") for t in (q, k, v)]
    actual = flash_attention_prefill(*device_inputs, causal=causal, lengths=lengths, scale=scale).cpu()
    assert torch.isfinite(actual).all()
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for batch, length in enumerate(lengths):
        assert torch.count_nonzero(actual[batch, :, length:]) == 0
    for original, device in zip((q, k, v), device_inputs):
        torch.testing.assert_close(device.cpu(), original, rtol=0, atol=0)


if __name__ == "__main__":
    target = triton.runtime.driver.active.get_current_target()
    print(f"target={target}", flush=True)
    if target.backend != "ifwa":
        raise RuntimeError("this example requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
