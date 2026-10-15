"""Native WGMMA descriptor users must not consume TileIR buffer handles."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._internal_testing import is_hopper_or_newer

pytestmark = pytest.mark.skipif(not is_hopper_or_newer(), reason="requires NVIDIA Hopper or newer")


@triton.jit
def _matmul(a, b, out, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    lhs = tle.gpu.alloc([64, 64], tl.float16, init_value=tl.load(a + offsets))
    rhs = tle.gpu.alloc([64, 64], tl.float16, init_value=tl.load(b + offsets))
    acc = tle.gpu.wgmma(lhs, rhs, trans_a=TRANS_A, trans_b=TRANS_B)
    acc = tle.gpu.wgmma_wait(0, acc)
    tl.store(out + offsets, acc)


@pytest.mark.parametrize("trans_a, trans_b", [(False, False), (True, False), (False, True), (True, True)])
def test_wgmma_buffer_bridge(trans_a, trans_b):
    torch.manual_seed(42)
    a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
    kernel = _matmul[(1, )](a, b, out, trans_a, trans_b, num_warps=4)
    lhs = a.T if trans_a else a
    rhs = b.T if trans_b else b
    torch.testing.assert_close(out, lhs.float() @ rhs.float(), atol=2e-2, rtol=2e-2)
    assert "!tile.buf" not in kernel.asm["ttgir"]
    assert "unrealized_conversion_cast" not in kernel.asm["ttgir"]
