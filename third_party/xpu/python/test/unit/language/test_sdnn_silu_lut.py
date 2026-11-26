import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _silu_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offsets)
    tl.store(dst + offsets, x / (1.0 + tl.exp(-x)))


def test_sdnn_silu_lut():
    if not torch.cuda.is_available():
        pytest.skip("requires a P800")
    target = triton.runtime.driver.active.get_current_target()
    if (target.arch, target.warp_size) != (3, 1):
        pytest.skip("requires a KL3 target")

    x = torch.linspace(-3.0, 3.0, 65536, dtype=torch.float32, device="cuda")
    out = torch.full_like(x, float("nan"))
    _silu_kernel[(8,)](x, out, BLOCK=8192, num_warps=8, num_stages=1, is_sdnn=True)
    torch.cuda.synchronize()
    max_abs_error = (out - torch.nn.functional.silu(x)).abs().max().item()
    assert max_abs_error < 3e-4, max_abs_error
