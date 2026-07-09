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
def tanh_forward(x):
    return 2 * sigmoid_forward(2 * x) - 1


# Device Kernels


@triton.jit
def _tanh_kernel(
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

    # Compute tanh
    tanh_output = tanh_forward(x)

    # Write back output to DRAM
    tl.store(y_ptr + offsets, tanh_output, mask=mask)


@triton.jit
def _tanh_kernel_backward(
    dx_ptr,
    dy_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tanh_grad_output = dy * (1 - y * y)

    tl.store(dx_ptr + offsets, tanh_grad_output, mask=mask)


# Host Launchers & Public API


class _Tanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        # Allocate output
        y = torch.empty_like(x)
        n_elements = y.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

        _tanh_kernel[grid](y, x, n_elements, BLOCK_SIZE=1024)

        ctx.y = y
        return y

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

        dx = torch.empty_like(dy)

        _tanh_kernel_backward[grid](dx, dy, y, n_elements=n_elements, BLOCK_SIZE=1024)

        return dx


@register_impl("tanh", backend="triton")
def tanh(input: torch.Tensor, ):
    r"""
    Returns Tanh actication of input.

    .. math::
        f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    Args:
        input: Tensor
    """
    return _Tanh.apply(input.view(-1)).view(input.shape)
