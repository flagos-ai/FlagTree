"""Opt-in device checks. CPU compiler jobs must not enable these tests."""
import os
import pytest
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_MUSA_RUN_NUMERICAL_DEVICE_TESTS") != "1", reason="requires explicit MUSA device-test opt-in")


@triton.jit
def _convert_boundary(X, Y, N: tl.constexpr, EXP: tl.constexpr):
    i = tl.arange(0, 1024)
    x = tl.load(X + i, i < N, 0)
    y = tl.exp2(x) if EXP else x
    tl.store(Y + i, y, i < N)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_corrected_exp2_boundaries(width):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_musa")
    values = torch.tensor(
        [float("nan"),
         float("inf"), -float("inf"), 0., -0., -150., -149., -148., -127., -126., -1., 1., 127.], dtype=torch.float32)
    x = values.to("musa")
    y = torch.empty_like(x)
    _convert_boundary[(1, )](x, y, x.numel(), True, num_warps=4, exp2_vector_width=width, exp2_native=False)
    expected = torch.exp2(values.double()).float()
    torch.testing.assert_close(y.cpu(), expected, rtol=2e-6, atol=0, equal_nan=True)


def test_fp8_rne_boundaries():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_musa")
    # E4M3 midpoints, signed zero and representable subnormal boundaries.
    values = torch.tensor([0., -0., 2.**-10, 3 * 2.**-10, 2.**-9, 1.0625, 1.1875, -1.0625, -1.1875, 448.],
                          dtype=torch.float32)
    x = values.to("musa")
    y = torch.empty(x.shape, device="musa", dtype=torch.float8_e4m3fn)
    _convert_boundary[(1, )](x, y, x.numel(), False, num_warps=4, enable_fp8_burst2=True, fp8_downcast_burst4=True)
    assert torch.equal(y.cpu().view(torch.uint8), values.to(torch.float8_e4m3fn).view(torch.uint8))
