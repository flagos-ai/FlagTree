"""E21: chunk-free, forward linear cross entropy without a logits tensor."""

import sys

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _linear_ce(X, W, Target, Loss, H: tl.constexpr, V: tl.constexpr, BH: tl.constexpr, BV: tl.constexpr):
    token = tl.program_id(0)
    hidden = tl.arange(0, BH)
    vocab = tl.arange(0, BV)
    x = tl.load(X + token * H + hidden, hidden < H, other=0).to(tl.float32)
    w = tl.load(W + vocab[:, None] * H + hidden[None, :], (vocab[:, None] < V) & (hidden[None, :] < H),
                other=0).to(tl.float32)
    logits = tl.sum(w * x[None, :], 1)
    logits = tl.where(vocab < V, logits, -float("inf"))
    maximum = tl.max(logits, 0)
    logsumexp = maximum + tl.log(tl.sum(tl.exp(logits - maximum), 0))
    label = tl.load(Target + token)
    selected = tl.sum(tl.where(vocab == label, logits, 0.0), 0)
    tl.store(Loss + token, logsumexp - selected)


def linear_cross_entropy(x, weight, target):
    """Return per-token FP32 CE; x[T,H], weight[V,H], CPU integer target[T].

    Input/weight are contiguous BF16/FP16 IFWA inference tensors; reduction is
    FP32. Target indices are checked on CPU then copied to IFWA. No [T,V] logits
    tensor is materialized. Forward only: no gradient or vocabulary chunking.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected x[T,H], weight[V,H]")
    tokens, hidden = x.shape
    vocab = weight.shape[0]
    if min(tokens, hidden, vocab) < 1 or hidden > 256 or vocab > 512:
        raise ValueError("requires positive T,H,V; H<=256 and V<=512")
    if x.device.type != "ifwa" or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be IFWA FP16/BF16")
    if weight.device != x.device or weight.dtype != x.dtype or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x/weight must be contiguous, same dtype/device")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("forward-only example does not implement autograd")
    if not isinstance(target, torch.Tensor) or target.device.type != "cpu" or target.shape != (
            tokens, ) or target.dtype not in (torch.int32, torch.int64):
        raise ValueError("target must be CPU int32/int64 [T]")
    if bool(((target < 0) | (target >= vocab)).any()):
        raise ValueError("target indices out of vocabulary range")
    device_target = target.to(dtype=torch.int32, device=x.device)
    out = torch.empty(tokens, device=x.device, dtype=torch.float32)
    _linear_ce[(tokens, )](x, weight, device_target, out, hidden, vocab, triton.next_power_of_2(hidden),
                           triton.next_power_of_2(vocab), num_warps=4, enable_fp_fusion=False)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 32, 17), (3, 64, 127), (2, 128, 255)])
def test_linear_ce(dtype, shape):
    tokens, hidden, vocab = shape
    generator = torch.Generator().manual_seed(2101)
    x = (torch.randn((tokens, hidden), generator=generator) * 0.3).to(dtype)
    w = (torch.randn((vocab, hidden), generator=generator) * 0.3).to(dtype)
    targets = torch.tensor([i * 71 % vocab for i in range(tokens)], dtype=torch.int64)
    expected = F.cross_entropy(x.float() @ w.float().T, targets, reduction="none")
    actual = linear_cross_entropy(x.to("ifwa:0"), w.to("ifwa:0"), targets).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E21 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
