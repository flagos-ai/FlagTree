# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports
import torch
import triton
import triton.language as tl

from tilegym.backend import mark_perf_ready
from tilegym.backend import register_impl

from .tanh import tanh_forward

# Kernel Helpers


@triton.jit
def _standard_normal_cdf(x):
    # inverse_sqrt_2 = 1.0 / tl.sqrt(2.0)
    inverse_sqrt_2 = 0.7071067811865475
    cdf = 0.5 * (1 + tl.math.erf(x * inverse_sqrt_2))
    return cdf


@triton.jit
def _standard_normal_pdf(x):
    # inverse_sqrt_2_pi = 1.0 / (tl.sqrt(2 * PI))
    inverse_sqrt_2_pi = 0.3989422804014327
    pdf = inverse_sqrt_2_pi * tl.exp((-0.5 * x * x).to(tl.float32))
    return pdf


@triton.jit
def gelu_tanh_forward(x):
    """
    Compute the approximate GELU activation function using tanh approximation.

    This is the fast approximation version of GELU that uses the hyperbolic tangent
    function to approximate the cumulative distribution function. The mathematical formula is:
    f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    This approximation is computationally faster than the exact version while maintaining
    good accuracy for most practical applications.

    Args:
        x: Input tensor values

    Returns:
        Approximate GELU activation output using tanh approximation
    """
    # sqrt_2_div_pi = tl.sqrt(2 / PI)
    sqrt_2_div_pi = 0.7978845608028654
    val = 0.5 * x * (1 + tanh_forward(sqrt_2_div_pi * (x + 0.044715 * x * x * x)))
    return val


@triton.jit
def gelu_forward(x):
    """
    Compute the exact GELU (Gaussian Error Linear Unit) activation function.

    This is the precise version of GELU using the cumulative distribution function (CDF)
    of the standard normal distribution. The mathematical formula is:
    f(x) = x * Φ(x)
    where Φ(x) is the CDF of the standard normal distribution.

    Args:
        x: Input tensor values

    Returns:
        GELU activation output: x * Φ(x)
    """
    return x * _standard_normal_cdf(x)


@triton.jit
def gelu_backward(x, dy):
    return dy * (_standard_normal_cdf(x) + x * _standard_normal_pdf(x))


# Device Kernels


@triton.jit
def _gelu_kernel(y_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, approximate):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute gelu
    if approximate:
        gelu_output = gelu_tanh_forward(x)
    else:
        gelu_output = gelu_forward(x)

    # Write back output to DRAM
    tl.store(y_ptr + offsets, gelu_output, mask=mask)


@triton.jit
def _gelu_kernel_backward(
    dx_ptr,
    dy_ptr,
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    gelu_grad_output = gelu_backward(x, dy)

    tl.store(dx_ptr + offsets, gelu_grad_output, mask=mask)


# Host Launchers & Public API


class _GeLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, approximate):
        assert approximate == "none" or approximate == "tanh", "Only `none` or `tanh` activations are supported"

        # Allocate output
        y = torch.empty_like(x)
        n_elements = y.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        BLOCK_SIZE = 1024
        _gelu_kernel[grid](y, x, n_elements, BLOCK_SIZE, approximate == "tanh")

        ctx.x = x
        return y

    @staticmethod
    def backward(ctx, dy):
        x = ctx.x
        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

        dx = torch.empty_like(dy)

        BLOCK_SIZE = 1024
        _gelu_kernel_backward[grid](dx, dy, x, n_elements, BLOCK_SIZE)

        return dx, None


@register_impl("gelu", backend="triton")
def gelu(input: torch.Tensor, approximate="none"):
    r"""
    Returns GeLU activation of input.
    If approximate is ``'tanh'`` then
    $f(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))$
    Else if approximate is ``'none'`` then
    $f(x) = x * \Phi(x)$
    Where $Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution.

    Args:
        input: Tensor
        approximate: ``'none'`` or ``'tanh'``
    """
    return _GeLU.apply(input.view(-1), approximate).view(input.shape)


# Backend Registration & Perf Markers

mark_perf_ready("gelu", "nvt")
