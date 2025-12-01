# This is the Layer Norm forward pass from the Triton tutorial found here:
# https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import torch

import triton
import triton.language as tl
import pytest
import benchmark

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, device):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        x_arg_dev = x_arg.to(DEVICE)
        y_dev = y.to(DEVICE)
        weight_dev = weight.to(DEVICE)
        bias_dev = bias.to(DEVICE)
        mean_dev = mean.to(DEVICE)
        rstd_dev = rstd.to(DEVICE)
        # enqueue kernel
        # _layer_norm_fwd_fused[(M, )](  #
        #     x_arg, y, weight, bias, mean, rstd,  #
        #     x_arg.stride(0), N, eps,  #
        #     BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)

        _layer_norm_fwd_fused[(M, )](  #
            x_arg_dev, y_dev, weight_dev, bias_dev, mean_dev, rstd_dev,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        x = x_arg_dev.to("cpu")
        y = y_dev.to("cpu")

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y


@pytest.mark.parametrize("M, N, dtype, eps", [  #
    (M, N, dtype, eps) for M in [1151] for N in [8192] for dtype in [torch.float16] for eps in [1e-5]
])
def test_layer_norm(M, N, dtype, eps, device):
    layer_norm = LayerNorm.apply
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(False)

    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps, device)
    # TODO We can't compare against Torch layer_norm since it doesn't support float16 on CPU
    #y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)

    print(y_tri)
    #print(y_ref)

    # compare
    #assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


@benchmark.measure()
def bench_layernorm(size, provider):
    layer_norm = LayerNorm.apply
    device = 'cpu'
    eps = 1e-5
    # dtype = torch.float16
    dtype = torch.float32
    x_shape = (size, size)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(False)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps, device)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)

    print(y_tri)
    print(y_ref)

    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    # benchmark.select_cpu_backend()
    for X in [2**i for i in range(10, 13, 1)]:
        for provider in ['triton']:
            bench_layernorm(X, provider)
