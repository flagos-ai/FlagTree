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

import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu


types_all = [
    (torch.float32, 'float32'),
]


shapes_common = [
    (8, 2048, 4),
]


def foo(a, b, shape):
    y = a.reshape(shape)
    y = y.permute(0,2,1) + b
    return y


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def triton_gpu(
    in_ptr0, in_ptr1, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :] # (1, YBLOCK)
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None] # (XBLOCK, 1)
    xmask = xindex < xnumel
    x2 = xindex # (XBLOCK, 1)
    y3 = yindex # (1, YBLOCK)
    y0 = yindex % SHAPE1  # (1, YBLOCK)
    y1 = (yindex // SHAPE1) # (1, YBLOCK)
    tmp0 = tl.load(in_ptr0 + (x2 + (SHAPE2*y3)), xmask) # (XBLOCK, YBLOCK)
    tmp1 = tl.load(in_ptr1 + (y0 + (SHAPE1*x2) + (SHAPE1*SHAPE2*y1)), xmask) # (XBLOCK, YBLOCK)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr + (x2 + (SHAPE2*y3)), tmp2, xmask)


@triton.jit
def k_load_perm_select(ptr, out, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    ymask = y < ynumel
    bad_mask = xmask | ymask
    val = tl.load(ptr + (x + 4 * y), bad_mask)
    tl.store(out + (x + 4 * y), val, xmask & ymask)


@triton.jit
def k_store_perm_select(in_ptr, out_ptr, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    ymask = y < ynumel
    bad_mask = xmask | ymask
    val = tl.load(in_ptr + (x + 4 * y), xmask & ymask)
    tl.store(out_ptr + (x + 4 * y), val, bad_mask)


@triton.jit
def k_load_moddiv_noperm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    # 2D tile：[YBLOCK, XBLOCK]
    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)

    mask = y < ynumel

    z = (yindex // SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_ = (yindex % SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_linear = z * SHAPE1 + y_                           # (YBLOCK, 1)
    offset_load = x + SHAPE2 * y_linear                  # (YBLOCK, XBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask)

    offset_store = x + SHAPE2 * y                        # (YBLOCK, XBLOCK)
    tl.store(out_ptr + offset_store, val, mask=mask)


@triton.jit
def k_store_moddiv_noperm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)

    mask = (y < ynumel) & (x < xnumel)

    offset_load = x + SHAPE2 * y                         # (YBLOCK, XBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask)

    z = (yindex // SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_ = (yindex % SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_linear = z * SHAPE1 + y_                           # (YBLOCK, 1)
    offset_store = x + SHAPE2 * y_linear                 # (YBLOCK, XBLOCK)
    tl.store(out_ptr + offset_store, val, mask=mask)


@triton.jit
def k_load_moddiv_perm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    X = xindex[:, None]                                 # (XBLOCK, 1)
    Y = yindex[None, :]                                 # (1, YBLOCK)
    mask_load = (X < xnumel) & (Y < ynumel)             # (XBLOCK, YBLOCK)

    z = (yindex // SHAPE1)[None, :]                     # (1, YBLOCK)
    y_ = (yindex % SHAPE1)[None, :]                     # (1, YBLOCK)
    offset_load = X + SHAPE2 * y_ + SHAPE2 * SHAPE1 * z # (XBLOCK, YBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask_load) # (XBLOCK, YBLOCK)

    y2 = yindex[:, None]                                # (YBLOCK, 1)
    x2 = xindex[None, :]                                # (1, XBLOCK)
    mask_store = (y2 < ynumel) & (x2 < xnumel)          # (YBLOCK, XBLOCK)
    offset_store = x2 + SHAPE2 * y2                     # (YBLOCK, XBLOCK)

    tl.store(out_ptr + offset_store, val.permute(1, 0), mask=mask_store)


@triton.jit
def k_store_moddiv_perm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)
    mask_load = (y < ynumel) & (x < xnumel)

    offset_load = x + SHAPE2 * y                         # (YBLOCK, XBLOCK)
    val = tl.load(in_ptr + offset_load, mask=mask_load)  # (YBLOCK, XBLOCK)

    X = xindex[:, None]                                  # (XBLOCK, 1)
    Y = yindex[None, :]                                  # (1, YBLOCK)
    mask_store = (X < xnumel) & (Y < ynumel)             # (XBLOCK, YBLOCK)

    z = Y // SHAPE1                                      # (1, YBLOCK)
    y_ = Y % SHAPE1                                      # (1, YBLOCK)
    offset_store = X + SHAPE2 * y_ + SHAPE2 * SHAPE1 * z # (XBLOCK, YBLOCK)

    tl.store(out_ptr + offset_store, val.permute(1, 0), mask=mask_store)


@triton.jit
def k_load_store_moddiv_noperm(
    in_ptr, out_ptr,
    ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)      # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)      # (XBLOCK,)

    y = yindex[:, None]                                  # (YBLOCK, 1)
    x = xindex[None, :]                                  # (1, XBLOCK)

    mask = (y < ynumel) & (x < xnumel)

    z = (yindex // SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_ = (yindex % SHAPE1)[:, None]                      # (YBLOCK, 1)
    y_linear = z * SHAPE1 + y_                           # (YBLOCK, 1)

    offset = x + SHAPE2 * y_linear                       # (YBLOCK, XBLOCK)

    val = tl.load(in_ptr + offset, mask=mask)
    val = val + 2
    tl.store(out_ptr + offset, val, mask=mask)


@triton.jit
def k_load_store_moddiv_perm(
    in_ptr, out_ptr, 
    ynumel, xnumel, 
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
    SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, SHAPE2: tl.constexpr
):
    # Program ID
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    # Block index calculation
    yindex = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # (YBLOCK,)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # (XBLOCK,)

    # Broadcasting to 2D
    x = xindex[:, None]  # (XBLOCK, 1)
    y = yindex[None, :]  # (1, YBLOCK)

    # Valid mask
    mask = (x < xnumel) & (y < ynumel)

    # Simulate linear index back to 3D
    z = y // SHAPE1  # (1, YBLOCK)
    y_ = y % SHAPE1  # (1, YBLOCK)

    # compute input offset (simulate [z, y_, x] access in strided format)
    offset = x + SHAPE2 * y_ + SHAPE2 * SHAPE1 * z  # (XBLOCK, YBLOCK)

    # load with implicit transpose (will be interpreted as [Y, X] then transposed)
    val = tl.load(in_ptr + offset, mask=mask)

    # apply some dummy operation
    val = val + 1

    # store it back to out_ptr with same offset
    tl.store(out_ptr + offset, val, mask=mask)


@triton.jit
def k_load_perm_scalar_ref(
    in_ptr, out_ptr,
    y1_numel, y0_numel, x2_numel,
    Y1BLOCK: tl.constexpr, Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, X2BLOCK_SUB: tl.constexpr
):
    pid_y1 = tl.program_id(0)
    pid_y0 = tl.program_id(1)

    y1_scalar = pid_y1 * Y1BLOCK

    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    base_x2 = tl.arange(0, X2BLOCK_SUB)

    y0_offset = pid_y0 * Y0BLOCK
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB

    for loop_y0 in range(loops_y0):
        y0_0 = y0_offset + loop_y0 * Y0BLOCK_SUB + base_y0[:, None]     # (Y0BLOCK_SUB, 1)
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None, :]      # (1, Y0BLOCK_SUB)
        x2_1 = base_x2[None, :]                                         # (1, X2BLOCK_SUB)
        x2 = base_x2[:, None]                                           # (X2BLOCK_SUB, 1)
        y0_0_mask = y0_0 < min(Y0BLOCK + y0_offset, y0_numel)
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        val0 = tl.load(in_ptr + (y0 + 8 * x2 + 8 * 16 * y1_scalar), mask=y0_mask)
        tl.store(out_ptr + (x2_1 + 16 * y0_0 + 8 * 16 * y1_scalar), val0.permute([1, 0]), mask=y0_0_mask)


@triton.jit
def k_load_perm_scalar(
    in_ptr, out_ptr,
    y1_numel, y0_numel, x2_numel,
    Y1BLOCK: tl.constexpr, Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, X2BLOCK_SUB: tl.constexpr
):
    pid_y1 = tl.program_id(0)
    pid_y0 = tl.program_id(1)

    y1_scalar = pid_y1 * Y1BLOCK

    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    base_x2 = tl.arange(0, X2BLOCK_SUB)

    y0_offset = pid_y0 * Y0BLOCK
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB

    for loop_y0 in range(loops_y0):
        y0_0 = y0_offset + loop_y0 * Y0BLOCK_SUB + base_y0[:, None]        # (Y0BLOCK_SUB, 1)
        x2_1 = base_x2[None, :]                                            # (1, X2BLOCK_SUB)
        y0_0_mask = y0_0 < min(Y0BLOCK + y0_offset, y0_numel)
        val0 = tl.load(in_ptr + (y0_0 + 8 * x2_1 + 8 * 16 * y1_scalar), mask=y0_0_mask)
        tl.store(out_ptr + (x2_1 + 16 * y0_0 + 8 * 16 * y1_scalar), val0, mask=y0_0_mask)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_triton_gpu_kernel(Z, Y, X, dtype, sigtype):
    shape = (Z, Y, X)
    
    a = test_common.generate_tensor(shape=(Z, Y * X), dtype=sigtype).npu()
    b = test_common.generate_tensor(shape=(Z, X, Y), dtype=sigtype).npu()

    # must set device='npu' for empty_strided, do not use out.npu() later
    out = torch.empty_strided((Z, X, Y), (X*Y, 1, X), device='npu', dtype=dtype)

    out_ref = foo(a, b, shape)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64

    if ynumel % YBLOCK != 0:
        pytest.skip(f"ynumel:{ynumel} not divisible by YBLOCK:{YBLOCK}")

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    triton_gpu[grid](
        a, b, out, 
        ynumel, xnumel, 
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X,
        enable_linearize=True
    )

    test_common.validate_cmp(sigtype, out_ref, out)


@pytest.mark.parametrize('dtype,sigtype', types_all)
@pytest.mark.parametrize('xnumel, ynumel, XBLOCK, YBLOCK', [(3, 513, 4, 64)])
def test_k_load_perm_select(xnumel, ynumel, XBLOCK, YBLOCK, dtype, sigtype):

    in_ptr = test_common.generate_tensor(shape=(ynumel * 4,), dtype=sigtype).npu()
    out_ptr = torch.zeros_like(in_ptr)


    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    k_load_perm_select[grid](
        in_ptr, out_ptr, ynumel, xnumel, 
        YBLOCK=YBLOCK, XBLOCK=XBLOCK, 
        enable_linearize=True
    )

    out_ref = torch.zeros_like(out_ptr)
    y_idx = torch.arange(ynumel).unsqueeze(1).npu()        # [ynumel, 1]
    x_idx = torch.arange(xnumel).unsqueeze(0).npu()        # [1, xnumel]
    idx = (x_idx + 4 * y_idx).reshape(-1)                  # [ynumel * xnumel]
    out_ref[idx] = in_ptr[idx]
    torch.testing.assert_close(out_ptr[idx], out_ref[idx])


@pytest.mark.parametrize('dtype,sigtype', types_all)
@pytest.mark.parametrize('xnumel, ynumel, XBLOCK, YBLOCK', [(3, 513, 4, 64)])
def test_k_store_perm_select(xnumel, ynumel, XBLOCK, YBLOCK, dtype, sigtype):
    in_ptr = test_common.generate_tensor(shape=(ynumel * 4,), dtype=sigtype).npu()
    out_ptr = torch.zeros_like(in_ptr)

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    k_store_perm_select[grid](
        in_ptr, out_ptr, ynumel, xnumel, 
        YBLOCK=YBLOCK, XBLOCK=XBLOCK, 
        enable_linearize=True
    )

    out_ref = torch.zeros_like(out_ptr)
    y_idx = torch.arange(ynumel).unsqueeze(1).npu()
    x_idx = torch.arange(xnumel).unsqueeze(0).npu()
    idx = (x_idx + 4 * y_idx).reshape(-1)
    out_ref[idx] = in_ptr[idx]
    torch.testing.assert_close(out_ptr[idx], out_ref[idx])


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_moddiv_noperm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_moddiv_noperm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X,
        enable_linearize=True
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_store_moddiv_noperm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_store_moddiv_noperm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X,
        enable_linearize=True
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_moddiv_perm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_moddiv_perm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X,
        enable_linearize=True
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_store_moddiv_perm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.zeros_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = X
    YBLOCK = 64

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_store_moddiv_perm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X,
        enable_linearize=True
    )

    torch.testing.assert_close(out, in_flat)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_store_moddiv_noperm(Z, Y, X, dtype, sigtype):
    a = test_common.generate_tensor(shape=(Z, Y, X), dtype=sigtype).npu()
    in_flat = a.contiguous().view(-1)
    out = torch.empty_like(in_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = 4
    YBLOCK = 256
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_store_moddiv_noperm[grid](
        in_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X,
        enable_linearize=True
    )

    ref = (a + 2).contiguous().view(-1)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('Z, Y, X', shapes_common)
def test_k_load_store_moddiv_perm(Z, Y, X, dtype, sigtype):
    shape = (Z, Y, X)
    numel = Z * Y * X

    a = test_common.generate_tensor(shape, dtype=sigtype).npu()
    a_flat = a.contiguous().view(-1)

    out = torch.zeros_like(a_flat)

    ynumel = Z * Y
    xnumel = X
    XBLOCK = 4
    YBLOCK = 256
    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)

    k_load_store_moddiv_perm[grid](
        a_flat, out, ynumel, xnumel,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        SHAPE0=Z, SHAPE1=Y, SHAPE2=X,
        enable_linearize=True
    )

    a_reshaped = a + 1
    out_ref = a_reshaped.contiguous().view(-1)

    torch.testing.assert_close(out, out_ref)


@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y0BLOCK_SUB, X2BLOCK_SUB', [
    (2, 8, 16, 1, 8, 4, 16),
])
def test_k_load_perm_scalar(y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y0BLOCK_SUB, X2BLOCK_SUB, dtype):
    in_ptr = torch.randn((y0_numel * 8, x2_numel), dtype=dtype, device='npu')

    out_ptr_triton_ref = torch.zeros((y1_numel, y0_numel, x2_numel), dtype=dtype, device='npu')
    out_ptr_triton = torch.zeros((y1_numel, y0_numel, x2_numel), dtype=dtype, device='npu')

    grid = (ceil_div(y1_numel, Y1BLOCK), ceil_div(y0_numel, Y0BLOCK))

    k_load_perm_scalar_ref[grid](
        in_ptr=in_ptr,
        out_ptr=out_ptr_triton_ref,
        y1_numel=y1_numel,
        y0_numel=y0_numel,
        x2_numel=x2_numel,
        Y1BLOCK=Y1BLOCK,
        Y0BLOCK=Y0BLOCK,
        Y0BLOCK_SUB=Y0BLOCK_SUB,
        X2BLOCK_SUB=X2BLOCK_SUB
    )

    k_load_perm_scalar[grid](
        in_ptr=in_ptr,
        out_ptr=out_ptr_triton,
        y1_numel=y1_numel,
        y0_numel=y0_numel,
        x2_numel=x2_numel,
        Y1BLOCK=Y1BLOCK,
        Y0BLOCK=Y0BLOCK,
        Y0BLOCK_SUB=Y0BLOCK_SUB,
        X2BLOCK_SUB=X2BLOCK_SUB,
        enable_linearize=True
    )

    torch.testing.assert_close(out_ptr_triton, out_ptr_triton_ref, rtol=1e-5, atol=1e-6)
