"""E14: scaled masked softmax with optional tanh softcap on IFWA."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax(X, MASK, Y, N: tl.constexpr, SCALE: tl.constexpr, CAP: tl.constexpr, HAS_CAP: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = tl.load(MASK + row * N + col, col < N, other=0)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32) * SCALE
    if HAS_CAP:
        x = CAP * (2.0 / (1.0 + tl.exp(-2.0 * x / CAP)) - 1.0)
    score = tl.where((col < N) & valid, x, -float("inf"))
    any_valid = tl.sum(valid.to(tl.int32), 0) > 0
    row_max = tl.where(any_valid, tl.max(score, 0), 0.0)
    weights = tl.exp(score - row_max)
    denominator = tl.sum(weights, 0)
    out = weights / tl.where(any_valid, denominator, 1.0)
    tl.store(Y + row * N + col, out, col < N)


def fused_softmax(x, mask, *, scale=1.0, softcap=None):
    """FP32 output for contiguous IFWA FP32/BF16 x[rows,cols], bool mask.

    Computes softmax(softcap*tanh(scale*x/softcap)) over valid columns.
    Without softcap the tanh stage is omitted. All-masked rows are zero.
    """
    if x.ndim != 2 or x.shape[0] < 1 or not 1 <= x.shape[1] <= 8192:
        raise ValueError("x must have shape [rows,cols], rows>0, 1<=cols<=8192")
    if x.device.type != "ifwa" or x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("x must be FP32/BF16 on IFWA")
    if mask.shape != x.shape or mask.dtype != torch.bool or mask.device != x.device:
        raise ValueError("mask must be an IFWA bool tensor matching x")
    if any(not t.is_contiguous() or t.requires_grad for t in (x, mask)):
        raise ValueError("inputs must be contiguous inference tensors")
    if not math.isfinite(scale) or scale <= 0 or (softcap is not None and (not math.isfinite(softcap) or softcap <= 0)):
        raise ValueError("scale and optional softcap must be finite and positive")
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    _softmax[(x.shape[0], )](x, mask, out, x.shape[1], scale, softcap or 1.0, softcap is not None,
                             triton.next_power_of_2(x.shape[1]), num_warps=4, enable_fp_fusion=False)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 19), (2, 127)])
@pytest.mark.parametrize("softcap", [None, 2.5])
def test_fused_softmax(dtype, shape, softcap):
    g = torch.Generator().manual_seed(214)
    x = torch.randn(shape, generator=g).to(dtype) * 4
    mask = torch.rand(shape, generator=g) > .2
    mask[0, -1] = True
    mask[-1, :] = False
    score = x.float() * .7
    if softcap is not None:
        score = softcap * torch.tanh(score / softcap)
    score = score.masked_fill(~mask, -float("inf"))
    expected = torch.softmax(score[:-1], -1)
    expected = torch.cat((expected, torch.zeros_like(score[-1:])), 0)
    x_dev, mask_dev = x.to("ifwa:0"), mask.to("ifwa:0")
    actual = fused_softmax(x_dev, mask_dev, scale=.7, softcap=softcap).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_dev.cpu(), x, rtol=0, atol=0)
    torch.testing.assert_close(mask_dev.cpu(), mask, rtol=0, atol=0)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E14 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
