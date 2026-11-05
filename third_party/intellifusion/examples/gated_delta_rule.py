"""E17: recurrent gated delta rule with varlen and a chunk-64 boundary."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta(Q, K, V, G, Beta, H0, Lengths, O, HT, T: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                 BK: tl.constexpr, BV: tl.constexpr, CHUNK: tl.constexpr):
    batch = tl.program_id(0)
    key_idx = tl.arange(0, BK)
    value_idx = tl.arange(0, BV)
    mask_state = (key_idx[:, None] < DK) & (value_idx[None, :] < DV)
    state = tl.load(H0 + batch * DK * DV + key_idx[:, None] * DV + value_idx[None, :], mask_state,
                    other=0).to(tl.float32)
    length = tl.load(Lengths + batch)
    for block in range(tl.cdiv(T, CHUNK)):
        for offset in range(CHUNK):
            t = block * CHUNK + offset
            if t < length:
                q = tl.load(Q + (batch * T + t) * DK + key_idx, key_idx < DK, other=0).to(tl.float32)
                k = tl.load(K + (batch * T + t) * DK + key_idx, key_idx < DK, other=0).to(tl.float32)
                v = tl.load(V + (batch * T + t) * DV + value_idx, value_idx < DV, other=0).to(tl.float32)
                gate = tl.load(G + batch * T + t)
                beta = tl.load(Beta + batch * T + t)
                state *= tl.exp(gate)
                predicted = tl.sum(state * k[:, None], 0)
                delta = beta * (v - predicted)
                state += k[:, None] * delta[None, :]
                out = tl.sum(state * (q * (DK**-.5))[:, None], 0)
                tl.store(O + (batch * T + t) * DV + value_idx, out, value_idx < DV)
    tl.store(HT + batch * DK * DV + key_idx[:, None] * DV + value_idx[None, :], state, mask_state)


def gated_delta_rule(q, k, v, gate, beta, initial_state, lengths):
    """Return (o[B,T,Dv], final_state[B,Dk,Dv]) in FP32.

    q/k[B,T,Dk], v[B,T,Dv] FP32/BF16; gate[B,T] log decay <=0,
    beta[B,T] in [0,1], state FP32. lengths is CPU integer[B], 0<=length<=T.
    h=exp(g)*h; delta=beta*(v-k@h); h+=outer(k,delta);
    o=(q/sqrt(Dk))@h. Output for padded tokens is zero; state unchanged for
    zero-length sequences. One program owns each sequence, avoiding write races.
    """
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3 or v.shape[:2] != q.shape[:2]:
        raise ValueError("q/k[B,T,Dk] and v[B,T,Dv] required")
    batch, t, dk = q.shape
    dv = v.shape[2]
    if min(batch, t, dk, dv) < 1 or dk > 64 or dv > 64:
        raise ValueError("nonempty shapes with Dk,Dv<=64 required")
    if gate.shape != (batch, t) or beta.shape != gate.shape or initial_state.shape != (batch, dk, dv):
        raise ValueError("gate, beta, or initial_state shape mismatch")
    if q.device.type != "ifwa" or q.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("q must be IFWA FP32/BF16")
    if any(tensor.device != q.device or tensor.dtype != q.dtype or not tensor.is_contiguous() or tensor.requires_grad
           for tensor in (q, k, v)):
        raise ValueError("q/k/v must share device and dtype, be contiguous, inference only")
    if any(tensor.device != q.device or tensor.dtype != torch.float32 or not tensor.is_contiguous()
           or tensor.requires_grad for tensor in (gate, beta, initial_state)):
        raise ValueError("gate, beta, and state must be contiguous FP32 on the same device")
    lengths_cpu = torch.as_tensor(lengths)
    if lengths_cpu.device.type != "cpu" or lengths_cpu.shape != (batch, ) or lengths_cpu.dtype not in (torch.int32,
                                                                                                       torch.int64):
        raise ValueError("lengths must be CPU integer [B]")
    if not bool(((lengths_cpu >= 0) & (lengths_cpu <= t)).all()):
        raise ValueError("lengths must lie in [0,T]")
    # Padding is explicitly zero-initialized because the kernel writes only valid tokens.
    out = torch.zeros((batch, t, dv), dtype=torch.float32, device=q.device)
    final = torch.empty_like(initial_state)
    lengths_dev = lengths_cpu.to(device=q.device, dtype=torch.int32)
    _gated_delta[(batch, )](q, k, v, gate, beta, initial_state, lengths_dev, out, final, t, dk, dv,
                            triton.next_power_of_2(dk), triton.next_power_of_2(dv), 64, num_warps=4,
                            enable_fp_fusion=False)
    return out, final


def reference(q, k, v, gate, beta, initial_state, lengths):
    output = torch.zeros((*v.shape[:2], v.shape[-1]), dtype=torch.float32)
    final = torch.empty_like(initial_state)
    for batch, length in enumerate(lengths):
        state = initial_state[batch].clone()
        for t in range(length):
            state = state * torch.exp(gate[batch, t])
            delta = beta[batch, t] * (v[batch, t].float() - k[batch, t].float() @ state)
            state += k[batch, t].float()[:, None] * delta[None, :]
            output[batch, t] = (q[batch, t].float() * q.shape[-1]**-.5) @ state
        final[batch] = state
    return output, final


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("t,lengths", [(1, [1, 0]), (7, [7, 3]), (70, [70, 43])])
def test_gated_delta_rule(dtype, t, lengths):
    g = torch.Generator().manual_seed(217)
    batch, dk, dv = 2, 8, 8
    q = (torch.randn(batch, t, dk, generator=g) * .15).to(dtype)
    k = (torch.randn(batch, t, dk, generator=g) * .15).to(dtype)
    v = (torch.randn(batch, t, dv, generator=g) * .15).to(dtype)
    gate = -torch.rand(batch, t, generator=g) * .02
    beta = torch.rand(batch, t, generator=g) * .5
    state = torch.randn(batch, dk, dv, generator=g) * .02
    expected_o, expected_h = reference(q, k, v, gate, beta, state, lengths)
    inputs = (q, k, v, gate, beta, state)
    dev = [value.to("ifwa:0") for value in inputs]
    actual_o, actual_h = gated_delta_rule(*dev, lengths)
    torch.testing.assert_close(actual_o.cpu(), expected_o, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_h.cpu(), expected_h, rtol=2e-4, atol=2e-5)
    for actual, original in zip(dev, inputs):
        torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E17 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
