"""E19: small dense/GQA causal or noncausal attention backward in Triton."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _backward(Q, K, V, DO, DQ, DK, DV, S: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
              SCALE: tl.constexpr, CAUSAL: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    row = tl.arange(0, BS)
    dim = tl.arange(0, BD)
    kbase = (batch * HKV + kv_head) * S * D
    k = tl.load(K + kbase + row[:, None] * D + dim[None, :], (row[:, None] < S) & (dim[None, :] < D), other=0)
    v = tl.load(V + kbase + row[:, None] * D + dim[None, :], (row[:, None] < S) & (dim[None, :] < D), other=0)
    grad_k = tl.zeros((BS, BD), tl.float32)
    grad_v = tl.zeros((BS, BD), tl.float32)
    for group_head in range(HQ // HKV):
        q_head = kv_head * (HQ // HKV) + group_head
        qbase = (batch * HQ + q_head) * S * D
        q = tl.load(Q + qbase + row[:, None] * D + dim[None, :], (row[:, None] < S) & (dim[None, :] < D), other=0)
        grad_out = tl.load(DO + qbase + row[:, None] * D + dim[None, :], (row[:, None] < S) & (dim[None, :] < D),
                           other=0)
        score = tl.dot(q, tl.trans(k)) * SCALE
        allowed = (row[:, None] < S) & (row[None, :] < S)
        if CAUSAL:
            allowed = allowed & (row[None, :] <= row[:, None])
        # Padded query rows have zero dO; finite mask keeps their softmax defined.
        score = tl.where(allowed, score, -1.0e6)
        probs = tl.exp(score - tl.max(score, 1)[:, None])
        probs = probs / tl.sum(probs, 1)[:, None]
        grad_p = tl.dot(grad_out, tl.trans(v))
        row_dot = tl.sum(grad_p * probs, 1)
        grad_score = probs * (grad_p - row_dot[:, None]) * SCALE
        grad_q = tl.dot(grad_score.to(q.dtype), k)
        grad_k += tl.dot(tl.trans(grad_score.to(q.dtype)), q)
        grad_v += tl.dot(tl.trans(probs.to(q.dtype)), grad_out)
        tl.store(DQ + qbase + row[:, None] * D + dim[None, :], grad_q, (row[:, None] < S) & (dim[None, :] < D))
    tl.store(DK + kbase + row[:, None] * D + dim[None, :], grad_k, (row[:, None] < S) & (dim[None, :] < D))
    tl.store(DV + kbase + row[:, None] * D + dim[None, :], grad_v, (row[:, None] < S) & (dim[None, :] < D))


def attention_backward(q, k, v, grad_output, *, causal=False, scale=None):
    """Return dQ/dK/dV for Q[B,Hq,S,D], K/V[B,Hkv,S,D], dO like Q.

    FP16/BF16 inputs, FP32 outputs. All dimensions contiguous and D>=16,
    1<=S<=64. This differentiates FP32 softmax of QK^T*scale and output PV.
    GQA repeats each KV head Hq/Hkv times; dK/dV sum over query heads.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or grad_output.shape != q.shape:
        raise ValueError('Q/dO must match and K/V must match rank-four shapes')
    b, hq, s, d = q.shape
    if min(b, hq, s, d) <= 0 or s > 64 or d < 16 or d > 128 or k.shape[0] != b or k.shape[2:] != (
            s, d) or k.shape[1] < 1 or hq % k.shape[1]:
        raise ValueError('requires 1<=S<=64, 16<=D<=128, compatible GQA heads')
    tensors = (q, k, v, grad_output)
    if q.device.type != 'ifwa' or q.dtype not in (torch.float16, torch.bfloat16) or any(
            t.device != q.device or t.dtype != q.dtype or not t.is_contiguous() or t.requires_grad for t in tensors):
        raise ValueError('requires contiguous inference FP16/BF16 IFWA tensors')
    scale = d**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    _backward[(b, k.shape[1])](q, k, v, grad_output, dq, dk, dv, s, d, hq, k.shape[1], scale, causal,
                               max(16, triton.next_power_of_2(s)), triton.next_power_of_2(d), num_warps=4,
                               enable_fp_fusion=False)
    return dq, dk, dv


def reference(q, k, v, grad_output, causal, scale):
    qr, kr, vr = (x.detach().float().requires_grad_() for x in (q, k, v))
    b, hq, s, _ = q.shape
    group = hq // k.shape[1]
    outputs = []
    for batch in range(b):
        heads = []
        for head in range(hq):
            scores = qr[batch, head] @ kr[batch, head // group].T * scale
            if causal:
                scores = scores.masked_fill(torch.triu(torch.ones(s, s, dtype=torch.bool), 1), -float('inf'))
            heads.append(torch.softmax(scores, -1) @ vr[batch, head // group])
        outputs.append(torch.stack(heads))
    torch.stack(outputs).backward(grad_output.float())
    return qr.grad, kr.grad, vr.grad


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('b,hq,hkv,s,d', [(1, 2, 2, 16, 16), (1, 4, 2, 20, 32)])
def test_attention_backward(dtype, causal, b, hq, hkv, s, d):
    gen = torch.Generator().manual_seed(190)
    cpu = [
        torch.randn(shape, generator=gen).to(dtype)
        for shape in ((b, hq, s, d), (b, hkv, s, d), (b, hkv, s, d), (b, hq, s, d))
    ]
    scale = 0.75 / math.sqrt(d)
    expected = reference(*cpu, causal, scale)
    dev = [x.to('ifwa:0') for x in cpu]
    actual = attention_backward(*dev, causal=causal, scale=scale)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), target, rtol=3e-2, atol=3e-2)
    for moved, original in zip(dev, cpu):
        torch.testing.assert_close(moved.cpu(), original, rtol=0, atol=0)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
