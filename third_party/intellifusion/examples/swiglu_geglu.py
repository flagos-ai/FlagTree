"""E03: SiLU(gate)*up and tanh-GELU(gate)*up, explicit forward/backward."""
import sys

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _activation_and_derivative(x, MODE: tl.constexpr):
    if MODE == "swiglu":
        sig = 1. / (1. + tl.exp(-x))
        activation = x * sig
        derivative = sig * (1. + x * (1. - sig))
    else:
        c: tl.constexpr = 0.7978845608028654
        u = c * (x + 0.044715 * x * x * x)
        e = tl.exp(-2. * tl.abs(u))
        t = tl.where(u < 0., -1., 1.) * ((1. - e) / (1. + e))
        activation = 0.5 * x * (1. + t)
        derivative = 0.5 * (1. + t) + 0.5 * x * (1. - t * t) * c * (1. + 3. * 0.044715 * x * x)
    return activation, derivative


@triton.jit
def _gated_activation(G, U, DY, Y, DG, DU, N: tl.constexpr, MODE: tl.constexpr, BACKWARD: tl.constexpr,
                      BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    g = tl.load(G + offset, mask, other=0).to(tl.float32)
    u = tl.load(U + offset, mask, other=0).to(tl.float32)
    activation, derivative = _activation_and_derivative(g, MODE)
    if BACKWARD:
        dy = tl.load(DY + offset, mask, other=0).to(tl.float32)
        tl.store(DG + offset, dy * u * derivative, mask)
        tl.store(DU + offset, dy * activation, mask)
    else:
        tl.store(Y + offset, activation * u, mask)


def _validate(gate, up, mode, grad=None):
    if mode not in ("swiglu", "geglu"):
        raise ValueError("mode must be swiglu or geglu (tanh approximation)")
    tensors = (gate, up) if grad is None else (gate, up, grad)
    if gate.device.type != "ifwa" or gate.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("inputs must be FP32 or BF16 on IFWA")
    if any(t.shape != gate.shape or t.device != gate.device or t.dtype != gate.dtype or not t.is_contiguous()
           or t.requires_grad for t in tensors):
        raise ValueError("inputs must share shape/device/dtype, be contiguous, and not require autograd")


def swiglu_geglu(gate, up, mode="swiglu"):
    """Return activation(gate)*up, same shape/dtype; no mutation or autograd."""
    _validate(gate, up, mode)
    out = torch.empty_like(gate)
    if gate.numel():
        _gated_activation[(triton.cdiv(gate.numel(), 256), )](gate, up, gate, out, gate, up, gate.numel(), mode, False,
                                                              256, enable_fp_fusion=False)
    return out


def swiglu_geglu_backward(gate, up, grad_output, mode="swiglu"):
    """Return explicit (d_gate, d_up) for an upstream gradient; inputs unchanged."""
    _validate(gate, up, mode, grad_output)
    dg, du = torch.empty_like(gate), torch.empty_like(up)
    if gate.numel():
        _gated_activation[(triton.cdiv(gate.numel(), 256), )](gate, up, grad_output, gate, dg, du, gate.numel(), mode,
                                                              True, 256, enable_fp_fusion=False)
    return dg, du


@pytest.mark.parametrize("mode", ["swiglu", "geglu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,wide", [((0, 33), False), ((3, 171), False), ((2, 256), True)])
def test_swiglu_geglu(mode, dtype, shape, wide):
    generator = torch.Generator().manual_seed(31)
    gate = torch.randn(shape, generator=generator).to(dtype)
    up = torch.randn(shape, generator=generator).to(dtype)
    dy = torch.randn(shape, generator=generator).to(dtype)
    if wide:
        gate = torch.linspace(-12, 12, gate.numel()).reshape(shape).to(dtype)
    g_ref, u_ref = gate.detach().float().requires_grad_(), up.detach().float().requires_grad_()
    activation = F.silu(g_ref) if mode == "swiglu" else F.gelu(g_ref, approximate="tanh")
    y_ref = activation * u_ref
    y_ref.backward(dy.float())
    dev = [t.to("ifwa:0") for t in (gate, up, dy)]
    actual = swiglu_geglu(*dev[:2], mode=mode)
    dg, du = swiglu_geglu_backward(*dev, mode=mode)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-2
    for output, expected in ((actual, y_ref.detach()), (dg, g_ref.grad), (du, u_ref.grad)):
        cpu = output.cpu()
        assert torch.isfinite(cpu).all()
        torch.testing.assert_close(cpu, expected.to(dtype), rtol=tolerance, atol=tolerance)
    for actual_input, original in zip(dev, (gate, up, dy)):
        torch.testing.assert_close(actual_input.cpu(), original, rtol=0, atol=0)


if __name__ == "__main__":
    target = triton.runtime.driver.active.get_current_target()
    print(f"E03 target={target}", flush=True)
    if target.backend != "ifwa":
        raise RuntimeError("this example requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
