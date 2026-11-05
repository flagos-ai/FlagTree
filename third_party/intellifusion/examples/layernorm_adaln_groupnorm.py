"""E15: independent LayerNorm, AdaLN modulation, and GroupNorm on IFWA."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm(X, W, B, SHIFT, SCALE, Y, H: tl.constexpr, EPS: tl.constexpr, ADA: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < H
    x = tl.load(X + row * H + col, valid, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / H
    centered = tl.where(valid, x - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / H
    normalized = centered * tl.rsqrt(variance + EPS)
    w = tl.load(W + col, valid, other=0).to(tl.float32)
    bias = tl.load(B + col, valid, other=0).to(tl.float32)
    out = normalized * w + bias
    if ADA:
        mod_scale = tl.load(SCALE + row * H + col, valid, other=0).to(tl.float32)
        mod_shift = tl.load(SHIFT + row * H + col, valid, other=0).to(tl.float32)
        out = out * (1.0 + mod_scale) + mod_shift
    tl.store(Y + row * H + col, out, valid)


@triton.jit
def _groupnorm(X, W, B, Y, C: tl.constexpr, L: tl.constexpr, GROUPS: tl.constexpr, EPS: tl.constexpr,
               BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    group = tl.program_id(1)
    group_channels: tl.constexpr = C // GROUPS
    count: tl.constexpr = group_channels * L
    idx = tl.arange(0, BLOCK)
    valid = idx < count
    channel = group * group_channels + idx // L
    address = (batch * C + channel) * L + idx % L
    x = tl.load(X + address, valid, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / count
    centered = tl.where(valid, x - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / count
    w = tl.load(W + channel, valid, other=0).to(tl.float32)
    bias = tl.load(B + channel, valid, other=0).to(tl.float32)
    out = centered * tl.rsqrt(variance + EPS) * w + bias
    tl.store(Y + address, out, valid)


def layernorm_adaln(x, weight, bias, *, shift=None, scale=None, eps=1e-5):
    """LayerNorm(x) affine; AdaLN adds per-token shift/scale after affine.

    x[T,H], weight/bias[H], shift/scale[T,H]. All share FP32/BF16 dtype.
    Both modulation tensors must be supplied together or both omitted.
    """
    if x.ndim != 2 or x.shape[0] < 1 or not 1 <= x.shape[1] <= 8192:
        raise ValueError("x must be nonempty [T,H] with H<=8192")
    if (shift is None) != (scale is None):
        raise ValueError("AdaLN requires both shift and scale")
    if weight.shape != (x.shape[1], ) or bias.shape != weight.shape:
        raise ValueError("weight and bias must have shape [H]")
    if shift is not None and (shift.shape != x.shape or scale.shape != x.shape):
        raise ValueError("AdaLN shift/scale must match x")
    inputs = (x, weight, bias) if shift is None else (x, weight, bias, shift, scale)
    _validate(inputs, eps)
    out = torch.empty_like(x)
    _layernorm[(x.shape[0], )](x, weight, bias, x if shift is None else shift,
                               x if scale is None else scale, out, x.shape[1], eps, shift is not None,
                               triton.next_power_of_2(x.shape[1]), num_warps=4, enable_fp_fusion=False)
    return out


def groupnorm(x, weight, bias, groups, *, eps=1e-5):
    """GroupNorm over contiguous [N,C,L], per-group C/groups * L elements."""
    if x.ndim != 3 or min(x.shape) < 1 or groups < 1 or x.shape[1] % groups:
        raise ValueError("x must be nonempty [N,C,L], with groups dividing C")
    n, c, length = x.shape
    if c // groups * length > 8192 or weight.shape != (c, ) or bias.shape != (c, ):
        raise ValueError("group size <=8192 and affine tensors [C] required")
    _validate((x, weight, bias), eps)
    out = torch.empty_like(x)
    _groupnorm[(n, groups)](x, weight, bias, out, c, length, groups, eps, triton.next_power_of_2(c // groups * length),
                            num_warps=4, enable_fp_fusion=False)
    return out


def _validate(inputs, eps):
    x = inputs[0]
    if x.device.type != "ifwa" or x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("inputs must be FP32/BF16 on IFWA")
    if any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous() or t.requires_grad for t in inputs):
        raise ValueError("all inputs must be contiguous, same device/dtype, inference only")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 129), (3, 512)])
@pytest.mark.parametrize("ada", [False, True])
def test_layernorm_adaln(dtype, shape, ada):
    g = torch.Generator().manual_seed(215)
    x = torch.randn(shape, generator=g).to(dtype)
    weight = torch.randn(shape[1], generator=g).to(dtype)
    bias = torch.randn(shape[1], generator=g).to(dtype)
    shift = (torch.randn(shape, generator=g) * .2).to(dtype)
    scale = (torch.randn(shape, generator=g) * .2).to(dtype)
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    variance = ((xf - mean)**2).mean(-1, keepdim=True)
    expected = (xf - mean) * torch.rsqrt(variance + 1e-5) * weight.float() + bias.float()
    if ada:
        expected = expected * (1 + scale.float()) + shift.float()
    dev = [t.to("ifwa:0") for t in (x, weight, bias, shift, scale)]
    actual = layernorm_adaln(*dev[:3], shift=dev[3] if ada else None, scale=dev[4] if ada else None).cpu()
    tol = 2e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected.to(dtype), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,groups", [((2, 6, 17), 3), ((1, 8, 32), 4)])
def test_groupnorm(dtype, shape, groups):
    g = torch.Generator().manual_seed(216)
    x = torch.randn(shape, generator=g).to(dtype)
    weight = torch.randn(shape[1], generator=g).to(dtype)
    bias = torch.randn(shape[1], generator=g).to(dtype)
    expected = torch.nn.functional.group_norm(x.float(), groups, weight.float(), bias.float(), eps=1e-5).to(dtype)
    dev = [t.to("ifwa:0") for t in (x, weight, bias)]
    actual = groupnorm(*dev, groups).cpu()
    tol = 2e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E15 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
