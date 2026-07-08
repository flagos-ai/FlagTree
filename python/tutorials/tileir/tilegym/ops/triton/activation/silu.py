# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports
import torch
import triton
import triton.language as tl

from tilegym.backend import register_impl

from .utils import sigmoid_forward

# Kernel Helpers


@triton.jit
def silu_forward(x):
    return x * sigmoid_forward(x)


@triton.jit
def silu_backward(x, dy):
    return dy * (sigmoid_forward(x) * (1.0 + x * (1.0 - sigmoid_forward(x))))


# Device Kernels


@triton.jit
def _silu_kernel(
    y_ptr,
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute silu
    silu_output = silu_forward(x)

    # Write back output to DRAM
    tl.store(y_ptr + offsets, silu_output, mask=mask)


@triton.jit
def _silu_kernel_backward(
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

    silu_grad_output = silu_backward(x, dy)

    tl.store(dx_ptr + offsets, silu_grad_output, mask=mask)


# Host Launchers & Public API


class _SiLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        # Allocate output
        y = torch.empty_like(x)
        n_elements = y.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

        _silu_kernel[grid](y, x, n_elements, BLOCK_SIZE=1024)
        ctx.x = x
        return y

    @staticmethod
    def backward(ctx, dy):
        x = ctx.x
        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

        dx = torch.empty_like(dy)

        _silu_kernel_backward[grid](dx, dy, x, n_elements=n_elements, BLOCK_SIZE=1024)

        return dx


@register_impl("silu", backend="triton")
def silu(input: torch.Tensor, ):
    r"""
    Returns SiLU activation of input.

    .. math::
        f(x) = x * \sigma (x)

    Args:
        input: Tensor
    """
    return _SiLU.apply(input.view(-1)).view(input.shape)
