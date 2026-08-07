"""Test matmul + bucketize compiled via torch.compile (inductor/Triton)."""
import pytest
import torch


def _fn(x: torch.Tensor, y: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
    z = torch.mm(x, y)
    return torch.bucketize(z, buckets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bucketize_after_matmul_compiled():
    """Compiled matmul + bucketize should match eager result."""
    buckets = torch.arange(-100, 100, 10, device="cuda")
    x = torch.randn(64, 64, device="cuda").clamp(-99, 99)
    y = torch.randn(64, 64, device="cuda").clamp(-99, 99)

    expected = _fn(x, y, buckets)
    opt_fn = torch.compile(_fn, mode="max-autotune")
    actual = opt_fn(x, y, buckets)

    assert torch.equal(actual, expected)
