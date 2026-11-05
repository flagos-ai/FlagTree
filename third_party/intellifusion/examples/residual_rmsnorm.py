"""E01: inference RMSNorm, add -> norm, and norm -> add on IFWA."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl

MODES = ("rmsnorm", "add_norm", "norm_add")


@triton.jit
def _residual_rmsnorm(X, R, W, Y, H: tl.constexpr, EPS: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < H
    x = tl.load(X + row * H + col, mask, other=0).to(tl.float32)
    if MODE == "add_norm":
        r = tl.load(R + row * H + col, mask, other=0).to(tl.float32)
        # Match a materialized input-dtype residual sum before normalization.
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
    variance = tl.sum(x * x, 0) / H
    inv_rms = tl.rsqrt(variance + EPS)
    w = tl.load(W + col, mask, other=0).to(tl.float32)
    normalized = (x * inv_rms) * w
    if MODE == "norm_add":
        # Match a materialized input-dtype norm before adding the residual.
        normalized = normalized.to(X.dtype.element_ty).to(tl.float32)
        r = tl.load(R + row * H + col, mask, other=0).to(tl.float32)
        normalized = normalized + r
    tl.store(Y + row * H + col, normalized, mask)


def residual_rmsnorm(x, weight, residual=None, *, mode="rmsnorm", eps=1e-6):
    """Return a new tensor; all inputs are contiguous and on the same IFWA device.

    x: [tokens, hidden], weight: [hidden], FP32 or BF16 of the same dtype.
    residual: same shape/dtype as x, required only for the two residual modes.
    RMS(z) = cast_dtype((float(z) * rsqrt(mean(float(z)**2) + eps)) * float(weight)).
    rmsnorm: RMS(x); add_norm: RMS(cast_dtype(float(x) + float(residual)));
    norm_add: cast_dtype(float(RMS(x)) + float(residual)).
    FP32 accumulation; no input mutation, gradients, or returned residual sum.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if x.device.type != "ifwa":
        raise ValueError("x must be on IFWA")
    if x.ndim != 2 or not 1 <= x.shape[1] <= 8192:
        raise ValueError("x must have shape [tokens, hidden], with 1 <= hidden <= 8192")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("only FP32 and BF16 are supported")
    if weight.shape != (x.shape[1], ):
        raise ValueError("weight must have shape [hidden]")
    tensors = [x, weight]
    if mode == "rmsnorm":
        if residual is not None:
            raise ValueError("rmsnorm does not accept a residual")
    else:
        if residual is None or residual.shape != x.shape:
            raise ValueError("residual must have the same shape as x")
        tensors.append(residual)
    if any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous, with the same dtype and device")
    if any(t.requires_grad for t in tensors):
        raise ValueError("this is an inference-only example")
    out = torch.empty_like(x)
    if x.shape[0]:
        _residual_rmsnorm[(x.shape[0], )](x, residual if residual is not None else x, weight, out, x.shape[1], eps,
                                          mode, triton.next_power_of_2(x.shape[1]), num_warps=4, enable_fp_fusion=False)
    return out


def reference(x, weight, residual, mode, eps):
    """CPU PyTorch oracle with the same explicitly documented rounding points."""
    source = x if mode != "add_norm" else (x.float() + residual.float()).to(x.dtype)
    source = source.float()
    out = (source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + eps)) * weight.float()
    out = out.to(x.dtype)
    if mode == "norm_add":
        out = (out.float() + residual.float()).to(x.dtype)
    return out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,pattern", [((1, 1), "random"), ((3, 129), "random"), ((2, 2048), "random"),
                                           ((1, 7168), "random"), ((1, 8192), "random"), ((2, 127), "zero"),
                                           ((2, 127), "cancellation"), ((0, 128), "random")])
def test_correctness(mode, dtype, shape, pattern):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(shape, generator=generator).to(dtype)
    r = torch.randn(shape, generator=generator).to(dtype)
    w = torch.randn(shape[1], generator=generator).to(dtype)
    if pattern == "zero":
        x.zero_()
        r.zero_()
    elif pattern == "cancellation":
        r = -x
    eps = 1e-6
    expected = reference(x, w, r, mode, eps)
    x_dev, r_dev, w_dev = (t.to("ifwa:0") for t in (x, r, w))
    out = residual_rmsnorm(x_dev, w_dev, None if mode == "rmsnorm" else r_dev, mode=mode, eps=eps)
    actual = out.cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x_dev.cpu(), x, rtol=0, atol=0)
    torch.testing.assert_close(r_dev.cpu(), r, rtol=0, atol=0)
    torch.testing.assert_close(w_dev.cpu(), w, rtol=0, atol=0)


if __name__ == "__main__":
    import importlib.metadata

    target = triton.runtime.driver.active.get_current_target()
    print(f"triton={importlib.metadata.version('triton')}, torch={torch.__version__}, target={target}", flush=True)
    if target.backend != "ifwa":
        raise RuntimeError(f"expected IFWA backend, got {target}")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
