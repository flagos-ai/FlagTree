"""E18: causal depthwise convolution and SiLU gate, plus incremental state."""
import sys

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_full(X, W, G, Y, C: tl.constexpr, T: tl.constexpr, K: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < TOTAL
    channel = (idx // T) % C
    pos = idx % T
    row = idx // T
    acc = tl.full((BLOCK, ), 0.0, tl.float32)
    for tap in range(K):
        source_pos = pos - (K - 1 - tap)
        x = tl.load(X + row * T + source_pos, valid & (source_pos >= 0), other=0).to(tl.float32)
        weight = tl.load(W + channel * K + tap, valid, other=0).to(tl.float32)
        acc += x * weight
    gate = tl.load(G + idx, valid, other=0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(Y + idx, acc * gate * sig, valid)


@triton.jit
def _conv_step(X, State, W, G, Y, Next, C: tl.constexpr, K: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < TOTAL
    channel = idx % C
    acc = tl.full((BLOCK, ), 0.0, tl.float32)
    for tap in range(K - 1):
        old = tl.load(State + idx * (K - 1) + tap, valid, other=0).to(tl.float32)
        weight = tl.load(W + channel * K + tap, valid, other=0).to(tl.float32)
        acc += old * weight
        if tap < K - 2:
            newer = tl.load(State + idx * (K - 1) + tap + 1, valid, other=0)
            tl.store(Next + idx * (K - 1) + tap, newer, valid)
    x = tl.load(X + idx, valid, other=0).to(tl.float32)
    weight = tl.load(W + channel * K + K - 1, valid, other=0).to(tl.float32)
    acc += x * weight
    tl.store(Next + idx * (K - 1) + K - 2, x, valid)
    gate = tl.load(G + idx, valid, other=0).to(tl.float32)
    tl.store(Y + idx, acc * gate / (1.0 + tl.exp(-gate)), valid)


def _check(x, weight, gate):
    if x.device.type != 'ifwa' or x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('expected IFWA FP32/BF16')
    if x.shape != gate.shape or x.device != gate.device or x.dtype != gate.dtype:
        raise ValueError('gate must match input')
    if any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous() or t.requires_grad
           for t in (x, weight, gate)):
        raise ValueError('contiguous same-device inference tensors required')


def causal_conv1d_gate(x, weight, gate):
    """x/gate [B,C,T], weight[C,K], 2<=K<=8; FP32 accumulation."""
    _check(x, weight, gate)
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[1] or not 2 <= weight.shape[1] <= 8:
        raise ValueError('expected x[B,C,T], weight[C,K] with 2<=K<=8')
    if min(x.shape) <= 0:
        raise ValueError('nonempty dimensions required')
    b, c, t = x.shape
    k = weight.shape[1]
    out = torch.empty_like(x)
    _conv_full[(triton.cdiv(x.numel(), 128), )](x, weight, gate, out, c, t, k, x.numel(), 128, enable_fp_fusion=False)
    return out


def causal_conv1d_gate_step(x, state, weight, gate):
    """Incremental x/gate[B,C], state[B,C,K-1] oldest-first -> (y,next_state)."""
    _check(x, weight, gate)
    if x.ndim != 2 or state.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[
            1] or not 2 <= weight.shape[1] <= 8:
        raise ValueError('expected x[B,C], state[B,C,K-1], weight[C,K]')
    if state.shape != (
            *x.shape, weight.shape[1] - 1
    ) or state.device != x.device or state.dtype != x.dtype or not state.is_contiguous() or state.requires_grad:
        raise ValueError('state must be contiguous [B,C,K-1] on same device/dtype')
    if min(x.shape) <= 0:
        raise ValueError('nonempty dimensions required')
    y, next_state = torch.empty_like(x), torch.empty_like(state)
    _conv_step[(triton.cdiv(x.numel(), 128), )](x, state, weight, gate, y, next_state, x.shape[1], weight.shape[1],
                                                x.numel(), 128, enable_fp_fusion=False)
    return y, next_state


def reference_full(x, w, gate):
    b, c, t = x.shape
    k = w.shape[1]
    result = torch.zeros((b, c, t), dtype=torch.float32)
    for tap in range(k):
        delay = k - 1 - tap
        if delay < t:
            result[..., delay:] += x.float()[..., :t - delay] * w.float()[:, tap][None, :, None]
    return (result * F.silu(gate.float())).to(x.dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shape,k', [((1, 2, 3), 3), ((2, 3, 9), 4), ((1, 4, 17), 5)])
def test_causal_conv(dtype, shape, k):
    gen = torch.Generator().manual_seed(180)
    x = torch.randn(shape, generator=gen).to(dtype)
    w = torch.randn((shape[1], k), generator=gen).to(dtype)
    gate = torch.randn(shape, generator=gen).to(dtype)
    expected = reference_full(x, w, gate)
    out = causal_conv1d_gate(x.to('ifwa:0'), w.to('ifwa:0'), gate.to('ifwa:0'))
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.cpu(), expected, rtol=tol, atol=tol)
    state = x[..., -k + 1:].clone().contiguous()
    next_x = torch.randn((shape[0], shape[1]), generator=gen).to(dtype)
    next_gate = torch.randn_like(next_x).to(dtype)
    step_y, next_state = causal_conv1d_gate_step(next_x.to('ifwa:0'), state.to('ifwa:0'), w.to('ifwa:0'),
                                                 next_gate.to('ifwa:0'))
    expected_step = (torch.sum(torch.cat(
        (state.float(), next_x.float()[..., None]), -1) * w.float()[None], -1) * F.silu(next_gate.float())).to(dtype)
    torch.testing.assert_close(step_y.cpu(), expected_step, rtol=tol, atol=tol)
    torch.testing.assert_close(next_state.cpu(), torch.cat((state[..., 1:], next_x[..., None]), -1), rtol=0, atol=0)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
