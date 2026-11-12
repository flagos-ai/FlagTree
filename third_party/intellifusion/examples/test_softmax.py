import pytest
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def softmax_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-block forward softmax kernel.

    Each program instance handles one row. Numerically stable via max subtraction.
    """
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # Load row with -inf padding for masked positions
    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-float("inf"))

    # Numerically stable softmax: subtract max before exp
    m = tl.max(x, axis=0)
    x = x - m
    y = tl.exp(x)
    s = tl.sum(y, axis=0)
    y = y / s
    # y = tl.softmax(x, axis=0)
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask)


def softmax(x: torch.Tensor, num_warps: int) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    grid = (n_rows, )
    softmax_kernel[grid](
        y,
        y.stride(0),
        x,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


def softmax_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float64:
        return 1e-6, 1e-6
    if dtype == torch.float32:
        return 1e-4, 1e-4
    raise TypeError(f"Unsupported softmax dtype: {dtype}")


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 680),
        (10, 1024),
    ],
)
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("dtype", [
    torch.float64,
    torch.float32,
    torch.bfloat16,
    torch.float16,
])
def test_softmax(shape: tuple[int, int], num_warps: int, dtype: torch.dtype):
    torch.manual_seed(0)
    n_rows, n_cols = shape

    x_cpu = torch.randn(n_rows, n_cols, device="cpu", dtype=dtype)
    x = x_cpu.to(DEVICE)
    y_triton = softmax(x, num_warps=num_warps)
    y_torch = torch.softmax(x_cpu, dim=1).to(DEVICE)
    torch.testing.assert_close(y_triton, y_torch, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
