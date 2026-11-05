"""E16: Mamba-style selective state update for one decode token on IFWA."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _selective_update(State, X, Dt, A, B, C, D, Z, Out, NewState, CHANNELS: tl.constexpr, N: tl.constexpr,
                      HAS_Z: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    channel = tl.program_id(1)
    state_idx = tl.arange(0, BLOCK)
    valid = state_idx < N
    x = tl.load(X + batch * CHANNELS + channel).to(tl.float32)
    dt = tl.load(Dt + batch * CHANNELS + channel)
    a = tl.load(A + channel * N + state_idx, valid, other=0)
    b = tl.load(B + batch * N + state_idx, valid, other=0)
    c = tl.load(C + batch * N + state_idx, valid, other=0)
    old = tl.load(State + (batch * CHANNELS + channel) * N + state_idx, valid, other=0)
    updated = tl.exp(dt * a) * old + dt * b * x
    updated = tl.where(valid, updated, 0.0)
    tl.store(NewState + (batch * CHANNELS + channel) * N + state_idx, updated, valid)
    y = tl.sum(updated * c, 0) + tl.load(D + channel) * x
    if HAS_Z:
        z = tl.load(Z + batch * CHANNELS + channel).to(tl.float32)
        y = y * (z / (1.0 + tl.exp(-z)))
    tl.store(Out + batch * CHANNELS + channel, y)


def mamba_selective_state_update(state, x, dt, a, b, c, d, z=None):
    """Return (output, new_state) for one token.

    state[batch,channels,state_dim] FP32; x[batch,channels] FP32/BF16;
    dt[batch,channels], A[channels,state_dim], B/C[batch,state_dim],
    D[channels] are FP32. A should be negative and dt positive for a decaying
    state. Equation: h'=exp(dt*A)*h + dt*B*x; y=sum(h'*C)+D*x;
    optionally y*=SiLU(z), with z of the same dtype/shape as x.
    """
    if x.ndim != 2 or min(x.shape) < 1 or state.ndim != 3 or state.shape[:2] != x.shape:
        raise ValueError("x[batch,channels] and state[batch,channels,state_dim] required")
    batch, channels = x.shape
    n = state.shape[2]
    if not 1 <= n <= 8192:
        raise ValueError("state_dim must be in [1,8192]")
    if dt.shape != x.shape or a.shape != (channels, n) or b.shape != (batch, n) or c.shape != b.shape or d.shape != (
            channels, ) or (z is not None and z.shape != x.shape):
        raise ValueError("dt, A, B, C, D, or z shape mismatch")
    if x.device.type != "ifwa" or x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("x must be FP32/BF16 on IFWA")
    parameters = (state, dt, a, b, c, d)
    if any(t.dtype != torch.float32 for t in parameters) or (z is not None and z.dtype != x.dtype):
        raise ValueError("state and parameters must be FP32; optional z matches x dtype")
    if any(t.device != x.device or not t.is_contiguous() or t.requires_grad
           for t in (x, *parameters, *((z, ) if z is not None else ()))):
        raise ValueError("inputs must be contiguous, same device, inference only")
    out = torch.empty_like(x)
    updated = torch.empty_like(state)
    _selective_update[(batch, channels)](state, x, dt, a, b, c, d, x if z is None else z, out, updated, channels, n, z
                                         is not None, triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    return out, updated


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 3, 7), (2, 4, 33)])
@pytest.mark.parametrize("has_z", [False, True])
def test_selective_update(dtype, shape, has_z):
    batch, channels, n = shape
    g = torch.Generator().manual_seed(216)
    state = torch.randn(shape, generator=g) * .2
    x = torch.randn(batch, channels, generator=g).to(dtype)
    dt = torch.rand(batch, channels, generator=g) * .2 + .01
    a = -torch.rand(channels, n, generator=g)
    b = torch.randn(batch, n, generator=g) * .1
    c = torch.randn(batch, n, generator=g) * .1
    d = torch.randn(channels, generator=g) * .1
    z = torch.randn(batch, channels, generator=g).to(dtype) if has_z else None
    reference_state = torch.exp(dt[..., None] * a[None, :, :]) * state + dt[..., None] * b[:, None, :] * x.float()[...,
                                                                                                                   None]
    reference_out = (reference_state * c[:, None, :]).sum(-1) + d[None, :] * x.float()
    if z is not None:
        reference_out *= torch.nn.functional.silu(z.float())
    original = [state, x, dt, a, b, c, d] + ([] if z is None else [z])
    dev = [t.to("ifwa:0") for t in original]
    out, updated = mamba_selective_state_update(*dev[:7], z=dev[7] if z is not None else None)
    torch.testing.assert_close(updated.cpu(), reference_state, rtol=2e-5, atol=2e-5)
    tol = 2e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.cpu(), reference_out.to(dtype), rtol=tol, atol=tol)
    for tensor, input_cpu in zip(dev, original):
        torch.testing.assert_close(tensor.cpu(), input_cpu, rtol=0, atol=0)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E16 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
