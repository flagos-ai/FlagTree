# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
TODO: in place ops not implemented
The ReLU family (ReLU, ELU, LeaklyReLU, PReLU, RReLU, SELU, CELU).
ReLU: x | 0
ELU: x | alpha_constant * (exp(x) - 1)
LeakyReLU: x | alpha_constant * x
PReLU: x | alpha_learnable * x
RReLU: x | alpha_random * x
SELU: 1.67 * x | negative: 1.67*1.05 * (exp(x) - 1)
CELU: x | alpha_constant * (exp(x / alpha_constant) - 1)
"""

import torch
import triton
import triton.language as tl

from tilegym.backend import get_available_triton_backend
from tilegym.backend import mark_perf_ready
from tilegym.backend import register_impl

from .utils import exp_forward
"""
=======================================================
                    ReLU
=======================================================
"""

# Feature: relu

# Kernel Helpers


@triton.jit
def relu_forward(x):
    return tl.where(x > 0.0, x, 0.0)


@triton.jit
def relu_backward(x, dy):
    dydx = tl.where(x > 0.0, 1.0, 0.0)
    return dydx * dy


# Device Kernels


@triton.jit
def _relu_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # write-back
    y = relu_forward(x)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_bwd_kernel(
    dy_ptr,
    dx_ptr,
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    dx = relu_backward(x, dy)
    tl.store(dx_ptr + offsets, dx, mask=mask)


# Host Launchers & Public API


class _ReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        y = torch.empty_like(x)

        assert x.is_contiguous()

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _relu_fwd_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty_like(dy)
        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _relu_bwd_kernel[grid](dy, dx, ctx.x, n_elements, BLOCK_SIZE=1024)
        return dx


@register_impl("relu", backend="triton")
def relu(x):
    r"""
    Returns ReLU activation of x.

    Args:
        x: Tensor
    """
    return _ReLU.apply(x)


"""
=======================================================
                    ELU
=======================================================
"""

# Feature: elu

# Device Kernels


@triton.jit
def _elu_fwd_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # write-back
    y = tl.where(x > 0.0, x, alpha * (exp_forward(x) - 1))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _elu_bwd_kernel(
    dy_ptr,
    dx_ptr,
    x_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    dydx = tl.where(x > 0.0, 1.0, alpha * exp_forward(x))
    dx = dydx * dy
    tl.store(dx_ptr + offsets, dx, mask=mask)


# Host Launchers & Public API


class _ELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.x = x
        ctx.ALPHA = alpha

        y = torch.empty_like(x)

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _elu_fwd_kernel[grid](x, y, alpha, n_elements, BLOCK_SIZE=1024)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty_like(dy)

        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _elu_bwd_kernel[grid](dy, dx, ctx.x, ctx.ALPHA, n_elements, BLOCK_SIZE=1024)

        return dx, None


@register_impl("elu", backend="triton")
def elu(x, alpha=1.0):
    r"""
    Returns ELU activation of x.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    Args:
        x: Tensor
        alpha: Float, default = 1.0
    """
    return _ELU.apply(x, alpha)


"""
=======================================================
                    LeakyReLU
=======================================================
"""

# Feature: leaky_relu

# Device Kernels


@triton.jit
def _leaky_relu_fwd_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # write-back
    y = tl.where(x > 0.0, x, alpha * x)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _leaky_relu_bwd_kernel(
    dy_ptr,
    dx_ptr,
    x_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    dydx = tl.where(x > 0.0, 1.0, alpha)
    dx = dydx * dy
    tl.store(dx_ptr + offsets, dx, mask=mask)


# Host Launchers & Public API


class _LeakyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.x = x
        ctx.ALPHA = alpha

        y = torch.empty_like(x)

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _leaky_relu_fwd_kernel[grid](x, y, alpha, n_elements, BLOCK_SIZE=1024)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty_like(dy)

        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _leaky_relu_bwd_kernel[grid](dy, dx, ctx.x, ctx.ALPHA, n_elements, BLOCK_SIZE=1024)

        return dx, None


@register_impl("leaky_relu", backend="triton")
def leaky_relu(x, negative_slope=0.01):
    r"""
    Returns Leaky ReLU activation of x.

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        x: Tensor
        negative_slope: Float, default = 0.01
    """
    return _LeakyReLU.apply(x, negative_slope)


"""
=======================================================
                    PReLU
=======================================================
"""

# Feature: prelu

# Device Kernels


@triton.jit
def _scalar_prelu_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This is called when using scalar weight regardless of input shape.
    Grids (BLOCK_SIZE). Treats input as vector.
    Each block gets (BLOCK_SIZE,) input data, matching (1,) scalar weight.
    """
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr)
    # write-back
    y = tl.where(x > 0.0, x, w * x)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _vector_prelu_nc_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This is called when using vector weights and input is 2 dimensional. (N, C).
    Grids (N, C // BLOCK_SIZE).
    Each block gets (1, BLOCK_SIZE) input data, matching (BLOCK_SIZE,) weights
    """
    row = tl.program_id(0)
    x_ptr += row * stride  # move x to assigned row
    y_ptr += row * stride  # move y to assigned row

    col_start = tl.program_id(1) * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)  # column offsets for x and w
    mask = offsets < stride  # mask out offsets going over a row

    # load in x and w
    x = tl.load(x_ptr + offsets, mask=mask)  # (BLOCK_SIZE,)
    w = tl.load(w_ptr + offsets, mask=mask)  # (BLOCK_SIZE,)

    # compute PReLU
    y = tl.where(x > 0.0, x, w * x)  # (BLOCK_SIZE,)

    # store in y
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _vector_prelu_nchw_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    stride_n,
    stride_c,
    stride_w,
    C,
    W,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    This is called when using vector weights and input is >2 dimensional. e.g. (N, C, H, W)
    Grids(N, C // BLOCK_SIZE_C, W // BLOCK_SIZE_W)
    Each block gets (1, BLOCK_SIZE_C, BLOCK_SIZE_W) input data, matchine (BLOCK_SIZE,) weights
    """
    row = tl.program_id(0)
    x_ptr += row * stride_n
    y_ptr += row * stride_n

    col_start = tl.program_id(1) * BLOCK_SIZE_C
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_C)
    mask_C = col_offsets < C

    tub_start = tl.program_id(2) * BLOCK_SIZE_W
    tub_offsets = tub_start + tl.arange(0, BLOCK_SIZE_W)
    mask_W = tub_offsets < W

    offsets = col_offsets[:, None] * stride_c + tub_offsets[None, :] * stride_w
    mask = mask_C[:, None] & mask_W[None, :]

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)  # (BLOCK_SIZE_C, BLOCK_SIZE_W)
    w = tl.load(w_ptr + col_offsets, mask=mask_C)[:, None]  # (BLOCK_SIZE_C, 1)

    y = tl.where(x > 0.0, x, w * x)

    tl.store(y_ptr + offsets, y, mask=mask)


if get_available_triton_backend() == "nvt":

    @triton.jit
    def _scalar_prelu_bwd_stage1_kernel(dy_ptr, dx_ptr, dw_ptr, x_ptr, w_ptr, lock_ptr, n_elements, n_weights,
                                        BLOCK_SIZE: tl.constexpr,
                                        GROUP_SIZE_M: tl.constexpr,  # this should be the number of groups
                                        ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        lock_id = pid % GROUP_SIZE_M
        group_offset = lock_id * n_weights

        mask = offsets < n_elements
        dy = tl.load(dy_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)
        w = tl.load(w_ptr)

        dydx = tl.where(x > 0.0, 1.0, w)
        dx = dydx * dy
        tl.store(dx_ptr + offsets, dx, mask=mask)

        dydw = tl.where(x > 0.0, 0.0, x)
        dw = tl.sum(dydw * dy)

        r, token = tl.ext.atomic_cas(
            lock_ptr + lock_id,
            0,
            1,
            semantics="acquire",
            scope="device",
            has_result_token=True,
        )
        while r == 1:
            r, token = tl.ext.atomic_cas(
                lock_ptr + lock_id,
                0,
                1,
                semantics="acquire",
                scope="device",
                has_result_token=True,
            )

        rw, token = tl.ext.load(dw_ptr + group_offset, memToken=token, has_result_token=True)
        dw += rw
        token = tl.ext.store(dw_ptr + group_offset, dw, memToken=token, has_result_token=True)
        tl.ext.atomic_xchg(
            lock_ptr + lock_id,
            0,
            memToken=token,
            semantics="release",
            scope="device",
        )


@triton.jit
def _vector_prelu_nc_bwd_stage1_kernel(
    dy_ptr,
    dx_ptr,
    dw_ptr,
    x_ptr,
    w_ptr,
    lock_ptr,
    stride,
    lock_stride,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    This is called when using vector weights and input is 2 dimensional. (N, C).
    Grids (N, C // BLOCK_SIZE).
    Each block gets (1, BLOCK_SIZE) input data, matching (BLOCK_SIZE,) weights.
    dw is a (GROUP_SIZE, C) matrix for parallel reduction.
    Reduces (N, C) to (GROUP_SIZE, C)
    The reduction requires grabbing one of (GROUP_SIZE, C // BLOCK_SIZE) locks.
    """
    row = tl.program_id(0)
    # move to assigned row
    dx_ptr += row * stride
    dy_ptr += row * stride
    x_ptr += row * stride

    col_start = tl.program_id(1) * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)  # column offsets for dy and w
    mask = offsets < stride  # mask out offsets going over a row

    lock_row = row % GROUP_SIZE  # stage 1 reduction over rows
    lock_col = tl.program_id(1)  # every blocks in the same row gets a different lock.
    lock_ptr += lock_row * lock_stride + lock_col
    dw_ptr += lock_row * stride

    # load in dy, x, w
    dy = tl.load(dy_ptr + offsets, mask=mask)  # (BLOCK_SIZE,)
    x = tl.load(x_ptr + offsets, mask=mask)  # (BLOCK_SIZE,)
    w = tl.load(w_ptr + offsets, mask=mask)  # (BLOCK_SIZE,)

    # store in dx
    dydx = tl.where(x > 0.0, 1.0, w)  # (BLOCK_SIZE,)
    dx = dydx * dy
    tl.store(dx_ptr + offsets, dx, mask=mask)

    dydw = tl.where(x > 0.0, 0.0, x)  # (BLOCK_SIZE,)
    dw = dydw * dy  # (BLOCK_SIZE,) we don't sum over the C axis as in scalar case

    # spin for lock -> load -> sum -> store -> release lock
    r, token = tl.ext.atomic_cas(
        lock_ptr,
        0,
        1,
        semantics="acquire",
        scope="device",
        has_result_token=True,
    )
    while r == 1:
        r, token = tl.ext.atomic_cas(
            lock_ptr,
            0,
            1,
            semantics="acquire",
            scope="device",
            has_result_token=True,
        )

    rw, token = tl.ext.load(dw_ptr + offsets, mask=mask, memToken=token, has_result_token=True)
    dw += rw
    token = tl.ext.store(dw_ptr + offsets, dw, mask=mask, memToken=token, has_result_token=True)
    tl.ext.atomic_xchg(lock_ptr, 0, memToken=token, semantics="release", scope="device")


@triton.jit
def _vector_prelu_nchw_bwd_stage1_kernel(
    dy_ptr,
    dx_ptr,
    dw_ptr,
    x_ptr,
    w_ptr,
    lock_ptr,
    stride_n,
    stride_c,
    stride_w,
    C,
    W,
    lock_stride,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    This is called when using vector weights and input is >2 dimensional. (N, C, H, W).
    Grids (N, C // BLOCK_SIZE_C, W // BLOCK_SIZE_W).
    Each block gets (1, BLOCK_SIZE_C, BLOCK_SIZE_W) input data, matching (BLOCK_SIZE,) weights.
    dw is a (GROUP_SIZE, C) matrix for parallel reduction.
    Reduces (N, C) to (GROUP_SIZE, C)
    The reduction requires grabbing one of (GROUP_SIZE, C // BLOCK_SIZE) locks.
    """
    row = tl.program_id(0)
    # move to assigned row
    dx_ptr += row * stride_n
    dy_ptr += row * stride_n
    x_ptr += row * stride_n

    col_start = tl.program_id(1) * BLOCK_SIZE_C
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_C)  # column offsets for dy and w
    mask_C = col_offsets < C  # mask out offsets going over a row

    tub_start = tl.program_id(2) * BLOCK_SIZE_W
    tub_offsets = tub_start + tl.arange(0, BLOCK_SIZE_W)
    mask_W = tub_offsets < W

    offsets = col_offsets[:, None] * stride_c + tub_offsets[None, :] * stride_w
    mask = mask_C[:, None] & mask_W[None, :]

    lock_row = row % GROUP_SIZE  # stage 1 reduction over rows
    lock_col = tl.program_id(1)  # every blocks in the same row gets a different lock.
    lock_ptr += lock_row * lock_stride + lock_col
    dw_ptr += lock_row * C

    # load in dy, x, w
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)  # (BLOCK_SIZE_C, BLOCK_SIZE_W)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)  # (BLOCK_SIZE_C, BLOCK_SIZE_W)
    w = tl.load(w_ptr + col_offsets, mask=mask_C).to(tl.float32)[:, None]  # (BLOCK_SIZE_C, 1)

    # store in dx
    dydx = tl.where(x > 0.0, 1.0, w)  # (BLOCK_SIZE_C, BLOCK_SIZE_W)
    dx = dydx * dy
    tl.store(dx_ptr + offsets, dx, mask=mask)

    dydw = tl.where(x > 0.0, 0.0, x)  # (BLOCK_SIZE_C, BLOCK_SIZE_W)
    dw = tl.sum(dydw * dy, axis=1).to(w.dtype)  # (BLOCK_SIZE,) here we sum out the spatial axis.

    # spin for lock -> load -> sum -> store -> release lock
    r, token = tl.ext.atomic_cas(
        lock_ptr,
        0,
        1,
        semantics="acquire",
        scope="device",
        has_result_token=True,
    )
    while r == 1:
        r, token = tl.ext.atomic_cas(
            lock_ptr,
            0,
            1,
            semantics="acquire",
            scope="device",
            has_result_token=True,
        )

    rw, token = tl.ext.load(
        dw_ptr + col_offsets,
        mask=mask_C,
        memToken=token,
        has_result_token=True,
    )
    dw += rw
    token = tl.ext.store(
        dw_ptr + col_offsets,
        dw,
        mask=mask_C,
        memToken=token,
        has_result_token=True,
    )
    tl.ext.atomic_xchg(lock_ptr, 0, memToken=token, semantics="release", scope="device")


@triton.jit
def _vector_prelu_nc_bwd_stage2_kernel(
    _dw_ptr,
    dw_ptr,
    N,
    C,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    This is stage 2 of the previous kernel.
    Grids (C // BLOCK_SIZE_C)
    Each block gets (N, BLOCK_SIZE_C) _dw data from previous stage, sums out the first axis.
    Reduces (N, C) to (C,)
    Note that N here is equal to GROUP_SIZE from prvious stage.
    """
    col_start = tl.program_id(0) * BLOCK_SIZE_C
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_C)

    dw = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)  # TODO: generalize dtype here?

    for row_start in range(0, N, BLOCK_SIZE_N):
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE_N)
        mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < C)
        offsets = row_offsets[:, None] * C + col_offsets[None, :]
        dw += tl.load(_dw_ptr + offsets, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(dw_ptr + col_offsets, sum_dw, mask=col_offsets < C)


# Host Launchers & Public API


class _PReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w):
        # weight should either be a scalar or match the number of channels of input.
        n_weights = w.numel()
        assert n_weights == 1 or (x.dim() > 1 and n_weights == x.shape[1])

        ctx.x = x
        ctx.w = w

        y = torch.empty_like(x)

        if n_weights == 1:
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            _scalar_prelu_fwd_kernel[grid](x, y, w, n_elements, BLOCK_SIZE=1024)
        elif x.dim() == 2:
            N, C = x.shape
            grid = lambda meta: (N, triton.cdiv(C, meta["BLOCK_SIZE"]))
            _vector_prelu_nc_fwd_kernel[grid](x, y, w, x.stride(0), BLOCK_SIZE=1024)
        else:
            x_squashed = x.view(x.shape[0], x.shape[1], -1)
            N, C, W = x_squashed.shape
            stride_n, stride_c, stride_w = x_squashed.stride()
            grid = lambda meta: (
                N,
                triton.cdiv(C, meta["BLOCK_SIZE_C"]),
                triton.cdiv(W, meta["BLOCK_SIZE_W"]),
            )
            _vector_prelu_nchw_fwd_kernel[grid](
                x,
                y,
                w,
                stride_n,
                stride_c,
                stride_w,
                C,
                W,
                BLOCK_SIZE_C=32,
                BLOCK_SIZE_W=256,
            )

        return y

    @staticmethod
    def backward(ctx, dy):
        if get_available_triton_backend() == "nvt":
            GROUP_SIZE = 256
            BLOCK_SIZE = 1024
            n_weights = ctx.w.numel()

            dx = torch.empty_like(dy)
            dw = torch.empty_like(ctx.w)
            _dw = torch.zeros(
                (GROUP_SIZE, ctx.w.numel()),
                dtype=torch.float32,
                device=ctx.w.device,
            )

            if n_weights == 1:
                n_elements = dy.numel()
                locks = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=ctx.w.device)
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
                _scalar_prelu_bwd_stage1_kernel[grid](
                    dy,
                    dx,
                    _dw,
                    ctx.x,
                    ctx.w,
                    locks,
                    n_elements,
                    n_weights,
                    BLOCK_SIZE=BLOCK_SIZE,
                    GROUP_SIZE_M=GROUP_SIZE,
                )
                dw = _dw.sum(
                    dim=0, keepdim=True)  # TODO: am I allowed to do this? it's just summing a 64d vector into a scalar.
            elif dy.dim() == 2:
                N, C = dy.shape
                num_blocks_per_row = triton.cdiv(C, BLOCK_SIZE)
                locks = torch.zeros(
                    (GROUP_SIZE, num_blocks_per_row),
                    dtype=torch.int32,
                    device=ctx.w.device,
                )
                grid = lambda meta: (N, num_blocks_per_row)
                _vector_prelu_nc_bwd_stage1_kernel[grid](
                    dy,
                    dx,
                    _dw,
                    ctx.x,
                    ctx.w,
                    locks,
                    dy.stride(0),
                    locks.stride(0),
                    BLOCK_SIZE=BLOCK_SIZE,
                    GROUP_SIZE=GROUP_SIZE,
                )
                grid = lambda meta: (triton.cdiv(C, meta["BLOCK_SIZE_C"]), )
                _vector_prelu_nc_bwd_stage2_kernel[grid](_dw, dw, GROUP_SIZE, C, BLOCK_SIZE_N=32, BLOCK_SIZE_C=128)
            else:
                dy_squashed = dy.view(dy.shape[0], dy.shape[1], -1)
                N, C, W = dy_squashed.shape
                stride_n, stride_c, stride_w = dy_squashed.stride()
                BLOCK_SIZE_W = 32
                BLOCK_SIZE_C = BLOCK_SIZE // BLOCK_SIZE_W
                num_blocks_per_row = triton.cdiv(C, BLOCK_SIZE_C)
                locks = torch.zeros(
                    (GROUP_SIZE, num_blocks_per_row),
                    dtype=torch.int32,
                    device=ctx.w.device,
                )
                grid = lambda meta: (
                    N,
                    num_blocks_per_row,
                    triton.cdiv(W, BLOCK_SIZE_W),
                )
                _vector_prelu_nchw_bwd_stage1_kernel[grid](
                    dy,
                    dx,
                    _dw,
                    ctx.x,
                    ctx.w,
                    locks,
                    stride_n,
                    stride_c,
                    stride_w,
                    C,
                    W,
                    locks.stride(0),
                    BLOCK_SIZE_C=BLOCK_SIZE_C,
                    BLOCK_SIZE_W=BLOCK_SIZE_W,
                    GROUP_SIZE=GROUP_SIZE,
                )
                dw = _dw.sum(0)
                # grid = lambda meta: (triton.cdiv(C, meta['BLOCK_SIZE_C']),)
                # _vector_prelu_nc_bwd_stage2_kernel[grid](
                #     _dw, dw, GROUP_SIZE, C, BLOCK_SIZE_N=32, BLOCK_SIZE_C=128
                # )

            return dx, dw
        raise NotImplementedError("Backward is not supported")


@register_impl("prelu", backend="triton")
def prelu(x, weight):
    r"""
    Returns PReLU activation of x.

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here weight is a learnable parameter. It could be a vector or a scalar.
    When weight is a vector, its number of elements should match the number
    of channels of x. When weight is a scalar, the same weight is used across
    all input channels.

    Args:
        x: Tensor (N, C, \*)
        weight: Tensor (C,) or (1,)
    """
    return _PReLU.apply(x, weight)


"""
=======================================================
                    RReLU
=======================================================
"""

# Feature: rrelu

# Device Kernels


@triton.jit
def _rrelu_fwd_kernel(
    x_ptr,
    y_ptr,
    seed,
    lower,
    upper,
    training,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly generate alpha ~ U(lower, upper)
    if training:
        alpha = tl.rand(seed, offsets) * (upper - lower) + lower
        y = tl.where(x > 0.0, x, alpha * x)
    else:
        y = tl.where(x > 0.0, x, x * (upper + lower) / 2.0)
    # write-back
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _rrelu_bwd_kernel(
    dy_ptr,
    dx_ptr,
    x_ptr,
    seed,
    lower,
    upper,
    training,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    if training:
        alpha = tl.rand(seed, offsets) * (upper - lower) + lower
        dydx = tl.where(x > 0.0, 1.0, alpha)
    else:
        dydx = tl.where(x > 0.0, 1.0, (upper + lower) / 2.0)

    dx = dydx * dy
    tl.store(dx_ptr + offsets, dx, mask=mask)


# Host Launchers & Public API


class _RReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, seed, lower, upper, training):
        ctx.x = x
        ctx.seed = seed
        ctx.lower = lower
        ctx.upper = upper
        ctx.training = training

        y = torch.empty_like(x)

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _rrelu_fwd_kernel[grid](x, y, seed, lower, upper, training, n_elements, BLOCK_SIZE=1024)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty_like(dy)

        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _rrelu_bwd_kernel[grid](
            dy,
            dx,
            ctx.x,
            ctx.seed,
            ctx.lower,
            ctx.upper,
            ctx.training,
            n_elements,
            BLOCK_SIZE=1024,
        )

        return dx, None, None, None, None


@register_impl("rrelu", backend="triton")
def rrelu(x, seed, lower=1.0 / 8, upper=1.0 / 3, training=False):
    r"""
    Returns RReLU activation of x.

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    If training is true, then $a \sim U(\text{lower}, \text{upper})$ for each input element.
    Otherwise, $a = \frac{\text{lower} + \text{upper}}2$

    Args:
        x: Tensor
        seed: Float
        lower: Float, default = 0.125
        upper: Float, default = 1 / 3
        training: Boolean, default = False
    """
    return _RReLU.apply(x, seed, lower, upper, training)


"""
=======================================================
                    SELU
=======================================================
"""

# Feature: selu

# Device Kernels


@triton.jit
def _selu_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # params
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    # write-back
    y = scale * tl.where(x > 0.0, x, alpha * (exp_forward(x) - 1))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _selu_bwd_kernel(
    dy_ptr,
    dx_ptr,
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717

    dydx = scale * tl.where(x > 0.0, 1.0, alpha * exp_forward(x))
    dx = dydx * dy
    tl.store(dx_ptr + offsets, dx, mask=mask)


# Host Launchers & Public API


class _SELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.x = x

        y = torch.empty_like(x)

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _selu_fwd_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty_like(dy)

        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _selu_bwd_kernel[grid](dy, dx, ctx.x, n_elements, BLOCK_SIZE=1024)

        return dx


@register_impl("selu", backend="triton")
def selu(x):
    r"""
    Returns SeLU activation of x.

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))


    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    Args:
        x: Tensor
    """
    return _SELU.apply(x)


"""
=======================================================
                    CELU
=======================================================
"""

# Feature: celu

# Device Kernels


@triton.jit
def _celu_fwd_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # write-back
    y = tl.where(x > 0.0, x, alpha * (tl.exp(x / alpha) - 1))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _celu_bwd_kernel(
    dy_ptr,
    dx_ptr,
    x_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    dydx = tl.where(x > 0.0, 1.0, tl.exp(x / alpha))
    dx = dydx * dy
    tl.store(dx_ptr + offsets, dx, mask=mask)


# Host Launchers & Public API


class _CELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.x = x
        ctx.alpha = alpha

        y = torch.empty_like(x)

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _celu_fwd_kernel[grid](x, y, alpha, n_elements, BLOCK_SIZE=1024)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty_like(dy)

        n_elements = dy.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        _celu_bwd_kernel[grid](dy, dx, ctx.x, ctx.alpha, n_elements, BLOCK_SIZE=1024)

        return dx, None


@register_impl("celu", backend="triton")
def celu(x, alpha=1.0):
    r"""
    Returns CeLU actication of input.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    Args:
        input: Tensor
        alpha: Float, default = 1.0
    """
    return _CELU.apply(x, alpha)


mark_perf_ready("celu", "nvt")
mark_perf_ready("elu", "nvt")
mark_perf_ready("leaky_relu", "nvt")
mark_perf_ready("prelu", "nvt")
mark_perf_ready("relu", "nvt")
mark_perf_ready("rrelu", "nvt")
mark_perf_ready("selu", "nvt")
