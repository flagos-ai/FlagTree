# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import pytest
import triton
import triton.language as tl
import test_common

types_all = [
    (torch.float32, 'float32'),
]

case_2d = [
    # X, Y, XBLOCK, YBLOCK
    (512, 32, 4, 64),
]

case_3d = [
    # X, Y, Z, XBLOCK, YBLOCK, ZBLOCK
    (100, 40, 32, 10, 4, 4),
]

case_4d = [
    # X, Y, Z, W, XBLOCK, YBLOCK, ZBLOCK, WBLOCK
    (100, 80, 20, 16, 20, 4, 4, 4),
]


# ----------------------------------------------------------
# Triton kernel: addptr with implicit permute kernels
# ----------------------------------------------------------
@triton.jit
def addptr_implicit_perm_load_2d(
    ptr,
    out,
    ynumel,
    xnumel,
    stride_y,
    stride_x,
    out_stride_x,
    out_stride_y,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    mask = (x < xnumel) & (y < ynumel)

    offset = x * stride_x + y * stride_y
    val = tl.load(ptr + offset, mask=mask)

    out_offset = x * out_stride_x + y * out_stride_y
    tl.store(out + out_offset, val, mask=mask)


@triton.jit
def addptr_implicit_perm_store_2d(
    ptr,
    out,
    ynumel,
    xnumel,
    in_stride_x,
    in_stride_y,
    stride_y,
    stride_x,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    mask = (x < xnumel) & (y < ynumel)

    in_offset = x * in_stride_x + y * in_stride_y
    val = tl.load(ptr + in_offset, mask=mask)

    out_offset = x * stride_x + y * stride_y
    tl.store(out + out_offset, val, mask=mask)


@triton.jit
def addptr_implicit_perm_load_store_2d_static_stride(ptr, out, ynumel, xnumel, stride_y: tl.constexpr,
                                                     stride_x: tl.constexpr, YBLOCK: tl.constexpr,
                                                     XBLOCK: tl.constexpr):
    # logical indices (A^T view)
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK]
    mask = (x < xnumel) & (y < ynumel)

    # IMPORTANT:
    # ptr is a row-major A, but we interpret it as A^T via stride
    offset = x * stride_x + y * stride_y

    val = tl.load(ptr + offset, mask)
    tl.store(out + offset, val, mask)


@triton.jit
def addptr_implicit_perm_load_store_2d(
    ptr,
    out,
    ynumel,
    xnumel,
    stride_y,
    stride_x,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # logical indices (A^T view)
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK]

    mask = (x < xnumel) & (y < ynumel)

    # IMPORTANT:
    # ptr is a row-major A, but we interpret it as A^T via stride
    offset = x * stride_x + y * stride_y

    val = tl.load(ptr + offset, mask=mask)
    tl.store(out + offset, val, mask=mask)


@triton.jit
def addptr_implicit_perm_load_store_3d_static_stride(
    ptr,
    out,
    znumel,
    ynumel,
    xnumel,
    stride_z: tl.constexpr,
    stride_y: tl.constexpr,
    stride_x: tl.constexpr,
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    x = pid_x * XBLOCK + tl.arange(0, XBLOCK)[:, None, None]
    y = pid_y * YBLOCK + tl.arange(0, YBLOCK)[None, :, None]
    z = pid_z * ZBLOCK + tl.arange(0, ZBLOCK)[None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel)

    offset = x * stride_x + y * stride_y + z * stride_z
    val = tl.load(ptr + offset, mask=mask)
    tl.store(out + offset, val, mask=mask)


@triton.jit
def addptr_implicit_perm_load_store_3d(
    ptr,
    out,
    znumel,
    ynumel,
    xnumel,
    stride_z,
    stride_y,
    stride_x,
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    x = pid_x * XBLOCK + tl.arange(0, XBLOCK)[:, None, None]
    y = pid_y * YBLOCK + tl.arange(0, YBLOCK)[None, :, None]
    z = pid_z * ZBLOCK + tl.arange(0, ZBLOCK)[None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel)

    offset = x * stride_x + y * stride_y + z * stride_z
    val = tl.load(ptr + offset, mask=mask)
    tl.store(out + offset, val, mask=mask)


@triton.jit
def addptr_implicit_perm_load_store_4d_static_stride(
    ptr,
    out,
    wnumel,
    znumel,
    ynumel,
    xnumel,
    stride_w: tl.constexpr,
    stride_z: tl.constexpr,
    stride_y: tl.constexpr,
    stride_x: tl.constexpr,
    WBLOCK: tl.constexpr,
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)  # covers (w, x)
    pid1 = tl.program_id(1)  # y
    pid2 = tl.program_id(2)  # z

    xblocks_per_w = (xnumel + XBLOCK - 1) // XBLOCK

    w_pid = pid0 // xblocks_per_w
    x_pid = pid0 - w_pid * xblocks_per_w

    x0 = x_pid * XBLOCK
    y0 = pid1 * YBLOCK
    z0 = pid2 * ZBLOCK
    w0 = w_pid * WBLOCK

    x = x0 + tl.arange(0, XBLOCK)[:, None, None, None]
    y = y0 + tl.arange(0, YBLOCK)[None, :, None, None]
    z = z0 + tl.arange(0, ZBLOCK)[None, None, :, None]
    w = w0 + tl.arange(0, WBLOCK)[None, None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel) & (w < wnumel)

    offset = x * stride_x + y * stride_y + z * stride_z + w * stride_w
    val = tl.load(ptr + offset, mask=mask, other=0.0)
    tl.store(out + offset, val, mask=mask)


@triton.jit
def addptr_implicit_perm_load_store_4d(
    ptr,
    out,
    wnumel,
    znumel,
    ynumel,
    xnumel,
    stride_w,
    stride_z,
    stride_y,
    stride_x,
    WBLOCK: tl.constexpr,
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)  # covers (w, x)
    pid1 = tl.program_id(1)  # y
    pid2 = tl.program_id(2)  # z

    xblocks_per_w = (xnumel + XBLOCK - 1) // XBLOCK

    w_pid = pid0 // xblocks_per_w
    x_pid = pid0 - w_pid * xblocks_per_w

    x0 = x_pid * XBLOCK
    y0 = pid1 * YBLOCK
    z0 = pid2 * ZBLOCK
    w0 = w_pid * WBLOCK

    x = x0 + tl.arange(0, XBLOCK)[:, None, None, None]
    y = y0 + tl.arange(0, YBLOCK)[None, :, None, None]
    z = z0 + tl.arange(0, ZBLOCK)[None, None, :, None]
    w = w0 + tl.arange(0, WBLOCK)[None, None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel) & (w < wnumel)

    offset = x * stride_x + y * stride_y + z * stride_z + w * stride_w
    val = tl.load(ptr + offset, mask=mask, other=0.0)
    tl.store(out + offset, val, mask=mask)


# ----------------------------------------------------------
# Triton kernel: make_tensor_ptr with implicit permute kernels
# ----------------------------------------------------------
@triton.jit
def make_tensor_ptr_implicit_perm_load_2d(
    ptr,
    out,
    ynumel,
    xnumel,
    STRIDE_Y,
    STRIDE_X,
    OUT_STRIDE_X,
    OUT_STRIDE_Y,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y0 = tl.program_id(1) * YBLOCK
    x0 = tl.program_id(0) * XBLOCK
    y = y0 + tl.arange(0, YBLOCK)[None, :]
    x = x0 + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    ymask = y < ynumel
    mask = xmask & ymask

    tptr = tl.make_block_ptr(
        base=ptr,
        shape=(xnumel, ynumel),
        strides=(STRIDE_X, STRIDE_Y),
        offsets=(x0, y0),
        block_shape=(XBLOCK, YBLOCK),
        order=(0, 1),
    )

    val = tl.load(tptr)
    out_offset = x * OUT_STRIDE_X + y * OUT_STRIDE_Y
    tl.store(out + out_offset, val, mask=mask)


@triton.jit
def make_tensor_ptr_implicit_perm_store_2d(
    ptr,
    out,
    ynumel,
    xnumel,
    IN_STRIDE_X,
    IN_STRIDE_Y,
    STRIDE_Y,
    STRIDE_X,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y0 = tl.program_id(1) * YBLOCK
    x0 = tl.program_id(0) * XBLOCK
    y = y0 + tl.arange(0, YBLOCK)[None, :]
    x = x0 + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    ymask = y < ynumel
    mask = xmask & ymask

    in_offset = x * IN_STRIDE_X + y * IN_STRIDE_Y
    val = tl.load(ptr + in_offset, mask=mask)

    tptr = tl.make_block_ptr(
        base=out,
        shape=(xnumel, ynumel),
        strides=(STRIDE_X, STRIDE_Y),
        offsets=(x0, y0),
        block_shape=(XBLOCK, YBLOCK),
        order=(0, 1),
    )
    tl.store(tptr, val, boundary_check=(0, 1))


@triton.jit
def make_tensor_ptr_implicit_perm_load_store_2d_static_stride(ptr, out, ynumel, xnumel, STRIDE_Y: tl.constexpr,
                                                              STRIDE_X: tl.constexpr, YBLOCK: tl.constexpr,
                                                              XBLOCK: tl.constexpr):
    y0 = tl.program_id(1) * YBLOCK
    x0 = tl.program_id(0) * XBLOCK
    y = y0 + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK]
    x = x0 + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    xmask = x < xnumel
    ymask = y < ynumel
    mask = xmask & ymask

    tptr = tl.make_block_ptr(
        base=ptr,
        shape=(xnumel, ynumel),
        strides=(STRIDE_X, STRIDE_Y),
        offsets=(x0, y0),
        block_shape=(XBLOCK, YBLOCK),
        order=(0, 1),
    )

    val = tl.load(tptr)
    tl.store(out + (x * STRIDE_X + STRIDE_Y * y), val, mask=mask)


@triton.jit
def make_tensor_ptr_implicit_perm_load_store_3d_static_stride(
    ptr,
    out,
    znumel,
    ynumel,
    xnumel,
    STRIDE_Z: tl.constexpr,
    STRIDE_Y: tl.constexpr,
    STRIDE_X: tl.constexpr,
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    x0 = pid_x * XBLOCK
    y0 = pid_y * YBLOCK
    z0 = pid_z * ZBLOCK

    x = x0 + tl.arange(0, XBLOCK)[:, None, None]
    y = y0 + tl.arange(0, YBLOCK)[None, :, None]
    z = z0 + tl.arange(0, ZBLOCK)[None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel)

    tptr = tl.make_block_ptr(
        base=ptr,
        shape=(xnumel, ynumel, znumel),
        strides=(STRIDE_X, STRIDE_Y, STRIDE_Z),
        offsets=(x0, y0, z0),
        block_shape=(XBLOCK, YBLOCK, ZBLOCK),
        order=(0, 1, 2),
    )

    val = tl.load(tptr)
    tl.store(out + (x * STRIDE_X + y * STRIDE_Y + z * STRIDE_Z), val, mask=mask)


@triton.jit
def make_tensor_ptr_implicit_perm_load_store_3d(
    ptr,
    out,
    znumel,
    ynumel,
    xnumel,
    STRIDE_Z,
    STRIDE_Y,
    STRIDE_X,
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    x0 = pid_x * XBLOCK
    y0 = pid_y * YBLOCK
    z0 = pid_z * ZBLOCK

    x = x0 + tl.arange(0, XBLOCK)[:, None, None]
    y = y0 + tl.arange(0, YBLOCK)[None, :, None]
    z = z0 + tl.arange(0, ZBLOCK)[None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel)

    tptr = tl.make_block_ptr(
        base=ptr,
        shape=(xnumel, ynumel, znumel),
        strides=(STRIDE_X, STRIDE_Y, STRIDE_Z),
        offsets=(x0, y0, z0),
        block_shape=(XBLOCK, YBLOCK, ZBLOCK),
        order=(0, 1, 2),
    )

    val = tl.load(tptr)
    tl.store(out + (x * STRIDE_X + y * STRIDE_Y + z * STRIDE_Z), val, mask=mask)


@triton.jit
def make_tensor_ptr_implicit_perm_load_3d_static_stride(
    ptr,
    out,
    znumel,  # logical z (== X)
    ynumel,  # logical y (== Y)
    xnumel,  # logical x (== Z)
    STRIDE_Z: tl.constexpr,
    STRIDE_Y: tl.constexpr,
    STRIDE_X: tl.constexpr,
    # out is row-major with shape (xnumel, ynumel, znumel)
    OUT_STRIDE0: tl.constexpr,  # = ynumel*znumel
    OUT_STRIDE1: tl.constexpr,  # = znumel
    OUT_STRIDE2: tl.constexpr,  # = 1
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    x0 = pid_x * XBLOCK
    y0 = pid_y * YBLOCK
    z0 = pid_z * ZBLOCK

    x = x0 + tl.arange(0, XBLOCK)[:, None, None]
    y = y0 + tl.arange(0, YBLOCK)[None, :, None]
    z = z0 + tl.arange(0, ZBLOCK)[None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel)

    # load: implicit permute view
    tptr = tl.make_block_ptr(
        base=ptr,
        shape=(xnumel, ynumel, znumel),
        strides=(STRIDE_X, STRIDE_Y, STRIDE_Z),
        offsets=(x0, y0, z0),
        block_shape=(XBLOCK, YBLOCK, ZBLOCK),
        order=(0, 1, 2),
    )
    val = tl.load(tptr)

    # store: row-major output (no implicit permute)
    out_offset = x * OUT_STRIDE0 + y * OUT_STRIDE1 + z * OUT_STRIDE2
    tl.store(out + out_offset, val, mask=mask)


# ----------------------------------------------------------
# Triton kernel: advance with implicit permute kernels
# ----------------------------------------------------------
@triton.jit
def advance_implicit_perm_load_store_2d_static_stride(ptr, out, ynumel, xnumel, STRIDE_Y: tl.constexpr,
                                                      STRIDE_X: tl.constexpr, YBLOCK: tl.constexpr,
                                                      XBLOCK: tl.constexpr):
    y0 = tl.program_id(1) * YBLOCK
    x0 = tl.program_id(0) * XBLOCK
    y = y0 + tl.arange(0, YBLOCK)[None, :]
    x = x0 + tl.arange(0, XBLOCK)[:, None]
    mask = (x < xnumel) & (y < ynumel)

    tptr = tl.make_block_ptr(
        base=ptr,
        shape=(xnumel, ynumel),
        strides=(STRIDE_X, STRIDE_Y),
        offsets=(0, 0),
        block_shape=(XBLOCK, YBLOCK),
        order=(0, 1),
    )
    tptr2 = tl.advance(tptr, (x0, y0))
    val = tl.load(tptr2)
    tl.store(out + (x * STRIDE_X + y * STRIDE_Y), val, mask=mask)


@triton.jit
def advance_implicit_perm_load_store_3d_static_stride(
    ptr,
    out,
    znumel,
    ynumel,
    xnumel,
    STRIDE_Z: tl.constexpr,
    STRIDE_Y: tl.constexpr,
    STRIDE_X: tl.constexpr,
    ZBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    x0 = pid_x * XBLOCK
    y0 = pid_y * YBLOCK
    z0 = pid_z * ZBLOCK

    x = x0 + tl.arange(0, XBLOCK)[:, None, None]
    y = y0 + tl.arange(0, YBLOCK)[None, :, None]
    z = z0 + tl.arange(0, ZBLOCK)[None, None, :]

    mask = (x < xnumel) & (y < ynumel) & (z < znumel)

    tptr0 = tl.make_block_ptr(
        base=ptr,
        shape=(xnumel, ynumel, znumel),
        strides=(STRIDE_X, STRIDE_Y, STRIDE_Z),
        offsets=(0, 0, 0),
        block_shape=(XBLOCK, YBLOCK, ZBLOCK),
        order=(0, 1, 2),
    )
    tptr = tl.advance(tptr0, (x0, y0, z0))
    val = tl.load(tptr)
    tl.store(out + (x * STRIDE_X + y * STRIDE_Y + z * STRIDE_Z), val, mask=mask)


# ----------------------------------------------------------
# pytest case
# ----------------------------------------------------------
def ceil_div(a, b):
    return (a + b - 1) // b


def _assert_row_major_2d(A, X, Y):
    assert tuple(A.shape) == (X, Y)
    assert A.is_contiguous()
    assert A.stride() == (Y, 1)


def _assert_row_major_3d(A, X, Y, Z):
    # [X, Y, Z] contiguous -> stride = (Y*Z, Z, 1)
    assert tuple(A.shape) == (X, Y, Z)
    assert A.is_contiguous()
    assert A.stride() == (Y * Z, Z, 1)


def _assert_row_major_4d(A, X, Y, Z, W):
    # [X, Y, Z, W] contiguous -> stride = (Y*Z*W, Z*W, W, 1)
    assert tuple(A.shape) == (X, Y, Z, W)
    assert A.is_contiguous()
    assert A.stride() == (Y * Z * W, Z * W, W, 1)


# ----------------------------------------------------------
# pytest case: addptr kernels
# ----------------------------------------------------------
@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_load_2d(X, Y, XBLOCK, YBLOCK, dtype, sigtype):
    A = test_common.generate_tensor(shape=(X, Y), dtype=sigtype).npu()
    _assert_row_major_2d(A, X, Y)

    xnumel = Y
    ynumel = X

    STRIDE_X = 1
    STRIDE_Y = Y

    out = torch.empty((xnumel, ynumel), device="npu", dtype=A.dtype)
    assert out.is_contiguous()
    OUT_STRIDE_X = ynumel
    OUT_STRIDE_Y = 1

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    addptr_implicit_perm_load_2d[grid](
        A,
        out,
        ynumel,
        xnumel,
        STRIDE_Y,
        STRIDE_X,
        OUT_STRIDE_X,
        OUT_STRIDE_Y,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    ref = A.T.contiguous()
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_store_2d(X, Y, XBLOCK, YBLOCK, dtype, sigtype):
    """
    Test goal:
      - Real memory layout: A[X, Y], row-major (stride = (Y, 1))
      - Load uses row-major semantics only
      - Store uses implicit transpose semantics only
      - Output shape: [Y, X] row-major
      - Result: out == A.T
    """
    A = test_common.generate_tensor(shape=(X, Y), dtype=sigtype).npu()
    _assert_row_major_2d(A, X, Y)

    xnumel = X
    ynumel = Y

    IN_STRIDE_X = Y
    IN_STRIDE_Y = 1

    STRIDE_X = 1
    STRIDE_Y = X

    out = torch.empty((Y, X), device="npu", dtype=A.dtype)
    assert out.is_contiguous()

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    addptr_implicit_perm_store_2d[grid](
        A,
        out,
        ynumel,
        xnumel,
        IN_STRIDE_X,
        IN_STRIDE_Y,
        STRIDE_Y,
        STRIDE_X,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    ref = A.T.contiguous()
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_load_store_2d_static_stride(
    X,
    Y,
    XBLOCK,
    YBLOCK,
    dtype,
    sigtype,
):
    """
    Test goal:
      - Real memory layout: A[X, Y], row-major (stride = (Y, 1))
      - Kernel view:        A^T[Y, X], stride = (1, Y)
      - Kernel does load+store with identical offsets
      - Result must satisfy: out == in
    """
    A = test_common.generate_tensor(
        shape=(X, Y),
        dtype=sigtype,
    ).npu()

    _assert_row_major_2d(A, X, Y)

    out = torch.zeros_like(A)

    # A^T logical shape
    xnumel = Y  # cols of A
    ynumel = X  # rows of A

    # A^T logical stride
    stride_x = 1
    stride_y = Y

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        1,
    )

    addptr_implicit_perm_load_store_2d_static_stride[grid](
        A,
        out,
        ynumel,
        xnumel,
        stride_y,
        stride_x,
        XBLOCK=XBLOCK,
        YBLOCK=YBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_load_store_2d(
    X,
    Y,
    XBLOCK,
    YBLOCK,
    dtype,
    sigtype,
):
    """
    Same as static-stride version, but stride passed as runtime values.
    """
    A = test_common.generate_tensor(
        shape=(X, Y),
        dtype=sigtype,
    ).npu()

    _assert_row_major_2d(A, X, Y)

    out = torch.zeros_like(A)

    # A^T logical shape
    xnumel = Y  # cols of A
    ynumel = X  # rows of A

    # A^T logical stride
    stride_x = 1
    stride_y = Y

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        1,
    )

    addptr_implicit_perm_load_store_2d[grid](
        A,
        out,
        ynumel,
        xnumel,
        stride_y,
        stride_x,
        XBLOCK=XBLOCK,
        YBLOCK=YBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, XBLOCK, YBLOCK, ZBLOCK", case_3d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_load_store_3d_static_stride(X, Y, Z, XBLOCK, YBLOCK, ZBLOCK, dtype, sigtype):
    """
    Test goal:
      - Real memory layout: A[X, Y, Z], row-major (stride = (Y*Z, Z, 1))
      - Kernel view:        treat as permuted logical coords via stride:
                            offset = x*1 + y*Z + z*(Y*Z)
                            i.e. (x,y,z) mapped to base index (z, y, x)
      - Kernel does load+store with identical offsets
      - Result must satisfy: out == in
    """
    A = test_common.generate_tensor(shape=(X, Y, Z), dtype=sigtype).npu()
    _assert_row_major_3d(A, X, Y, Z)
    out = torch.zeros_like(A)

    # Logical shape for "A^(perm)" (x fastest)
    xnumel = Z
    ynumel = Y
    znumel = X

    # Logical strides (in elements): (1, Z, Y*Z)
    stride_x = 1
    stride_y = Z
    stride_z = Y * Z

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    addptr_implicit_perm_load_store_3d_static_stride[grid](
        A,
        out,
        znumel,
        ynumel,
        xnumel,
        stride_z,
        stride_y,
        stride_x,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, XBLOCK, YBLOCK, ZBLOCK", case_3d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_load_store_3d(X, Y, Z, XBLOCK, YBLOCK, ZBLOCK, dtype, sigtype):
    """
    Same as static-stride version, but stride passed as runtime values.
    """
    A = test_common.generate_tensor(shape=(X, Y, Z), dtype=sigtype).npu()
    _assert_row_major_3d(A, X, Y, Z)
    out = torch.zeros_like(A)

    xnumel = Z
    ynumel = Y
    znumel = X

    stride_x = 1
    stride_y = Z
    stride_z = Y * Z

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    addptr_implicit_perm_load_store_3d[grid](
        A,
        out,
        znumel,
        ynumel,
        xnumel,
        stride_z,
        stride_y,
        stride_x,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, W, XBLOCK, YBLOCK, ZBLOCK, WBLOCK", case_4d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_load_store_4d_static_stride(X, Y, Z, W, XBLOCK, YBLOCK, ZBLOCK, WBLOCK, dtype, sigtype):
    """
    Test goal:
      - Real memory layout: A[X, Y, Z, W], row-major (stride = (Y*Z*W, Z*W, W, 1))
      - Kernel view:        treat as permuted logical coords via stride:
                            offset = x*1 + y*W + z*(Z*W) + w*(Y*Z*W)
                            i.e. (x,y,z,w) mapped to base index (w, z, y, x)
      - Kernel does load+store with identical offsets
      - Result must satisfy: out == in
    """
    A = test_common.generate_tensor(shape=(X, Y, Z, W), dtype=sigtype).npu()
    _assert_row_major_4d(A, X, Y, Z, W)
    out = torch.zeros_like(A)

    # Logical shape (x fastest)
    xnumel = W
    ynumel = Z
    znumel = Y
    wnumel = X

    # Logical strides (in elements): (1, W, Z*W, Y*Z*W)
    stride_x = 1
    stride_y = W
    stride_z = Z * W
    stride_w = Y * Z * W

    # Kernel maps pid0 over (w, x). It uses xblocks_per_w computed from xnumel.
    xblocks_per_w = ceil_div(xnumel, XBLOCK)
    grid0 = wnumel * xblocks_per_w
    grid = (
        grid0,
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    addptr_implicit_perm_load_store_4d_static_stride[grid](
        A,
        out,
        wnumel,
        znumel,
        ynumel,
        xnumel,
        stride_w,
        stride_z,
        stride_y,
        stride_x,
        WBLOCK=WBLOCK,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, W, XBLOCK, YBLOCK, ZBLOCK, WBLOCK", case_4d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_addptr_implicit_perm_load_store_4d(X, Y, Z, W, XBLOCK, YBLOCK, ZBLOCK, WBLOCK, dtype, sigtype):
    """
    Same as static-stride version, but stride passed as runtime values.
    """
    A = test_common.generate_tensor(shape=(X, Y, Z, W), dtype=sigtype).npu()
    _assert_row_major_4d(A, X, Y, Z, W)
    out = torch.zeros_like(A)

    xnumel = W
    ynumel = Z
    znumel = Y
    wnumel = X

    stride_x = 1
    stride_y = W
    stride_z = Z * W
    stride_w = Y * Z * W

    xblocks_per_w = ceil_div(xnumel, XBLOCK)
    grid0 = wnumel * xblocks_per_w
    grid = (
        grid0,
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    addptr_implicit_perm_load_store_4d[grid](
        A,
        out,
        wnumel,
        znumel,
        ynumel,
        xnumel,
        stride_w,
        stride_z,
        stride_y,
        stride_x,
        WBLOCK=WBLOCK,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


# ----------------------------------------------------------
# pytest case: make_tensor_ptr kernels
# ----------------------------------------------------------
@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_make_tensor_ptr_implicit_perm_load_2d(X, Y, XBLOCK, YBLOCK, dtype, sigtype):
    """
    Test goal:
      - Real memory layout: A[X, Y], row-major (stride = (Y, 1))
      - Kernel view (logical): shape = (Y, X), stride = (1, Y) (implicit transpose)
      - Load via make_block_ptr with implicit permute stride
      - Store with row-major output stride
      - Output shape: [Y, X] row-major
      - Result: out == A.T
    """
    A = test_common.generate_tensor(shape=(X, Y), dtype=sigtype).npu()
    _assert_row_major_2d(A, X, Y)

    xnumel = Y
    ynumel = X

    STRIDE_X = 1
    STRIDE_Y = Y

    out = torch.empty((xnumel, ynumel), device="npu", dtype=A.dtype)
    assert out.is_contiguous()
    OUT_STRIDE_X = ynumel
    OUT_STRIDE_Y = 1

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    make_tensor_ptr_implicit_perm_load_2d[grid](
        A,
        out,
        ynumel,
        xnumel,
        STRIDE_Y,
        STRIDE_X,
        OUT_STRIDE_X,
        OUT_STRIDE_Y,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    ref = A.T.contiguous()
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_make_tensor_ptr_implicit_perm_store_2d(X, Y, XBLOCK, YBLOCK, dtype, sigtype):
    """
    Test goal:
      - Real memory layout: A[X, Y], row-major (stride = (Y, 1))
      - Load uses row-major semantics only
      - Store via make_block_ptr with implicit transpose semantics only
      - Output shape: [Y, X] row-major
      - Result: out == A.T
    """
    A = test_common.generate_tensor(shape=(X, Y), dtype=sigtype).npu()
    _assert_row_major_2d(A, X, Y)

    xnumel = X
    ynumel = Y

    IN_STRIDE_X = Y
    IN_STRIDE_Y = 1

    STRIDE_X = 1
    STRIDE_Y = X

    out = torch.empty((Y, X), device="npu", dtype=A.dtype)
    assert out.is_contiguous()

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    make_tensor_ptr_implicit_perm_store_2d[grid](
        A,
        out,
        ynumel,
        xnumel,
        IN_STRIDE_X,
        IN_STRIDE_Y,
        STRIDE_Y,
        STRIDE_X,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    ref = A.T.contiguous()
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_make_tensor_ptr_implicit_perm_load_store_2d_static_stride(X, Y, XBLOCK, YBLOCK, dtype, sigtype):
    """
    Test goal matches addptr_2d_static_stride, but uses tl.make_block_ptr + tl.load(tptr).
    Real layout: A[X,Y] row-major stride=(Y,1)
    Kernel view: A^T[Y,X] stride=(1,Y)
    Store is by explicit linear offset with same logical stride.
    """
    A = test_common.generate_tensor(shape=(X, Y), dtype=sigtype).npu()
    _assert_row_major_2d(A, X, Y)
    out = torch.zeros_like(A)

    xnumel = Y
    ynumel = X
    STRIDE_X = 1
    STRIDE_Y = Y

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    make_tensor_ptr_implicit_perm_load_store_2d_static_stride[grid](
        A,
        out,
        ynumel,
        xnumel,
        STRIDE_Y=STRIDE_Y,
        STRIDE_X=STRIDE_X,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, XBLOCK, YBLOCK, ZBLOCK", case_3d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_make_tensor_ptr_implicit_perm_load_store_3d_static_stride(X, Y, Z, XBLOCK, YBLOCK, ZBLOCK, dtype, sigtype):
    """
    Real layout: A[X,Y,Z] row-major stride=(Y*Z, Z, 1)
    Kernel view (logical): shape=(Z,Y,X), strides=(1, Z, Y*Z)
    """
    A = test_common.generate_tensor(shape=(X, Y, Z), dtype=sigtype).npu()
    _assert_row_major_3d(A, X, Y, Z)
    out = torch.zeros_like(A)

    xnumel = Z
    ynumel = Y
    znumel = X
    STRIDE_X = 1
    STRIDE_Y = Z
    STRIDE_Z = Y * Z

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    make_tensor_ptr_implicit_perm_load_store_3d_static_stride[grid](
        A,
        out,
        znumel,
        ynumel,
        xnumel,
        STRIDE_Z=STRIDE_Z,
        STRIDE_Y=STRIDE_Y,
        STRIDE_X=STRIDE_X,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, XBLOCK, YBLOCK, ZBLOCK", case_3d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_make_tensor_ptr_implicit_perm_load_store_3d(X, Y, Z, XBLOCK, YBLOCK, ZBLOCK, dtype, sigtype):
    """
    Same as static stride but STRIDE_* passed at runtime.
    """
    A = test_common.generate_tensor(shape=(X, Y, Z), dtype=sigtype).npu()
    _assert_row_major_3d(A, X, Y, Z)
    out = torch.zeros_like(A)

    xnumel = Z
    ynumel = Y
    znumel = X
    STRIDE_X = 1
    STRIDE_Y = Z
    STRIDE_Z = Y * Z

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    make_tensor_ptr_implicit_perm_load_store_3d[grid](
        A,
        out,
        znumel,
        ynumel,
        xnumel,
        STRIDE_Z,
        STRIDE_Y,
        STRIDE_X,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, XBLOCK, YBLOCK, ZBLOCK", case_3d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_make_tensor_ptr_implicit_perm_load_3d_static_stride(X, Y, Z, XBLOCK, YBLOCK, ZBLOCK, dtype, sigtype):
    A = test_common.generate_tensor(shape=(X, Y, Z), dtype=sigtype).npu()
    _assert_row_major_3d(A, X, Y, Z)

    # logical/permuted shape
    xnumel = Z
    ynumel = Y
    znumel = X

    # implicit-permute strides (elements)
    STRIDE_X = 1
    STRIDE_Y = Z
    STRIDE_Z = Y * Z

    # output is row-major of shape (Z, Y, X)
    out = torch.empty((xnumel, ynumel, znumel), device="npu", dtype=A.dtype)
    assert out.is_contiguous()
    OUT_STRIDE0 = ynumel * znumel  # Y*X
    OUT_STRIDE1 = znumel  # X
    OUT_STRIDE2 = 1

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    make_tensor_ptr_implicit_perm_load_3d_static_stride[grid](
        A,
        out,
        znumel,
        ynumel,
        xnumel,
        STRIDE_Z=STRIDE_Z,
        STRIDE_Y=STRIDE_Y,
        STRIDE_X=STRIDE_X,
        OUT_STRIDE0=OUT_STRIDE0,
        OUT_STRIDE1=OUT_STRIDE1,
        OUT_STRIDE2=OUT_STRIDE2,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    # expected: out[x,y,z] == A[z,y,x]  => out == A.permute(2,1,0)
    ref = A.permute(2, 1, 0).contiguous()
    torch.testing.assert_close(out, ref)


# ----------------------------------------------------------
# pytest case: advance kernels
# ----------------------------------------------------------
@pytest.mark.parametrize("X, Y, XBLOCK, YBLOCK", case_2d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_advance_implicit_perm_load_store_2d_static_stride(X, Y, XBLOCK, YBLOCK, dtype, sigtype):
    """
    Same goal as addptr_2d_static_stride, but uses tl.make_block_ptr + tl.advance.
    """
    A = test_common.generate_tensor(shape=(X, Y), dtype=sigtype).npu()
    _assert_row_major_2d(A, X, Y)
    out = torch.zeros_like(A)

    xnumel = Y
    ynumel = X
    STRIDE_X = 1
    STRIDE_Y = Y

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    advance_implicit_perm_load_store_2d_static_stride[grid](
        A,
        out,
        ynumel,
        xnumel,
        STRIDE_Y=STRIDE_Y,
        STRIDE_X=STRIDE_X,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


@pytest.mark.parametrize("X, Y, Z, XBLOCK, YBLOCK, ZBLOCK", case_3d)
@pytest.mark.parametrize("dtype, sigtype", types_all)
def test_advance_implicit_perm_load_store_3d_static_stride(X, Y, Z, XBLOCK, YBLOCK, ZBLOCK, dtype, sigtype):
    """
    Real layout: A[X,Y,Z] row-major stride=(Y*Z, Z, 1)
    Kernel view (logical): shape=(Z,Y,X), strides=(1, Z, Y*Z)
    """
    A = test_common.generate_tensor(shape=(X, Y, Z), dtype=sigtype).npu()
    _assert_row_major_3d(A, X, Y, Z)
    out = torch.zeros_like(A)

    xnumel = Z
    ynumel = Y
    znumel = X
    STRIDE_X = 1
    STRIDE_Y = Z
    STRIDE_Z = Y * Z

    grid = (
        ceil_div(xnumel, XBLOCK),
        ceil_div(ynumel, YBLOCK),
        ceil_div(znumel, ZBLOCK),
    )

    advance_implicit_perm_load_store_3d_static_stride[grid](
        A,
        out,
        znumel,
        ynumel,
        xnumel,
        STRIDE_Z=STRIDE_Z,
        STRIDE_Y=STRIDE_Y,
        STRIDE_X=STRIDE_X,
        ZBLOCK=ZBLOCK,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
    )

    torch.testing.assert_close(out, A)


if __name__ == "__main__":
    test_addptr_implicit_perm_load_2d(*case_2d[0], *types_all[0])
    test_addptr_implicit_perm_store_2d(*case_2d[0], *types_all[0])
    test_addptr_implicit_perm_load_store_2d_static_stride(*case_2d[0], *types_all[0])
    test_addptr_implicit_perm_load_store_2d(*case_2d[0], *types_all[0])
    test_addptr_implicit_perm_load_store_3d_static_stride(*case_3d[0], *types_all[0])
    test_addptr_implicit_perm_load_store_3d(*case_3d[0], *types_all[0])
    test_addptr_implicit_perm_load_store_4d_static_stride(*case_4d[0], *types_all[0])
    test_addptr_implicit_perm_load_store_4d(*case_4d[0], *types_all[0])

    test_make_tensor_ptr_implicit_perm_load_2d(*case_2d[0], *types_all[0])
    test_make_tensor_ptr_implicit_perm_store_2d(*case_2d[0], *types_all[0])
    test_make_tensor_ptr_implicit_perm_load_3d_static_stride(*case_3d[0], *types_all[0])
    test_make_tensor_ptr_implicit_perm_load_store_2d_static_stride(*case_2d[0], *types_all[0])
    test_make_tensor_ptr_implicit_perm_load_store_3d_static_stride(*case_3d[0], *types_all[0])
    test_make_tensor_ptr_implicit_perm_load_store_3d(*case_3d[0], *types_all[0])
    test_advance_implicit_perm_load_store_2d_static_stride(*case_2d[0], *types_all[0])
    test_advance_implicit_perm_load_store_3d_static_stride(*case_3d[0], *types_all[0])
