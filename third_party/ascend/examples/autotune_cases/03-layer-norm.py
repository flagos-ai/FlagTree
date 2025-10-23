"""
Layer Normalization
=============
"""

import os

import torch
import torch_npu
import triton
import triton.language as tl
from triton.testing import do_bench_npu


@triton.autotune(
    configs=[],
    key={"x": "M", "y": "N"},
    split_params={"x": "XBLOCK_SIZE"},
    tiling_params={"y": "RBLOCK_SIZE"},
    low_dims=["y"],
    persistent_reduction=False,
    dual_reduction=False,
)
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,
    M,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    XBLOCK_SIZE: tl.constexpr,
    RBLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row_begin = tl.program_id(0) * XBLOCK_SIZE
    row_idx = row_begin + tl.arange(0, XBLOCK_SIZE)
    row_mask = row_idx < M
    row_offsets = row_idx[:, None] * stride
    # Compute mean
    _mean = tl.zeros((XBLOCK_SIZE, RBLOCK_SIZE), dtype=tl.float32)
    for off in range(0, N, RBLOCK_SIZE):
        col_idx = off + tl.arange(0, RBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:, None] & col_mask[None, :]
        a = tl.load(X + row_offsets + col_idx[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        _mean += a
    mean = tl.sum(_mean, axis=1, keep_dims=True) / N
    # Compute variance
    _var = tl.zeros((XBLOCK_SIZE, RBLOCK_SIZE), dtype=tl.float32)
    for off in range(0, N, RBLOCK_SIZE):
        col_idx = off + tl.arange(0, RBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(X + row_offsets + col_idx[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row_idx[:, None], mean, mask=row_mask[:, None])
    tl.store(Rstd + row_idx[:, None], rstd, mask=row_mask[:, None])
    # Normalize and apply linear transformation
    for off in range(0, N, RBLOCK_SIZE):
        col_idx = off + tl.arange(0, RBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:, None] & col_mask[None, :]
        w = tl.load(W + col_idx, mask=col_mask).reshape((1, RBLOCK_SIZE))
        b = tl.load(B + col_idx, mask=col_mask).reshape((1, RBLOCK_SIZE))
        x = tl.load(X + row_offsets + col_idx[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + row_offsets + col_idx[None, :], y, mask=mask)


def layer_norm_torch(args):
    x, w_shape, weight, bias, eps, dtype = args
    return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)


def layer_norm_autotune(args):
    x, weight, bias, eps = args
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta["XBLOCK_SIZE"]), 1, 1)
    # enqueue kernel
    _layer_norm_fwd_fused[grid](  #
        x_arg, y, weight, bias, mean, rstd, x_arg.stride(0), N, M, eps  #
    )
    return y


def test_layer_norm(shape, dtype, eps=1e-5):
    os.environ["TRITON_BENCH_METHOD"] = (
        "npu"  # use torch_npu.profiler to get calculating time
    )
    M, N = shape
    device = "npu"
    x_shape = shape
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device)
    bias = torch.rand(w_shape, dtype=dtype, device=device)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    y_torch = layer_norm_torch((x, w_shape, weight, bias, eps, dtype))
    y_triton = layer_norm_autotune((x, weight, bias, eps))
    assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=0)
    print(f"Layer Normalization {M},{N} {dtype} PASSED!")


if __name__ == "__main__":
    test_layer_norm((128, 128), torch.float16)
