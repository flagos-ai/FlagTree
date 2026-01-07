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

import torch, torch_npu
import triton
import triton.language as tl
import triton.language.math as tl_math
import pytest

def test_ldst_indirect_00():

    @triton.jit
    def triton_ldst_indirect_00_kernel(
        out_ptr0, in_ptr0, in_ptr1, OFFSET0: tl.constexpr,
        XS: tl.constexpr
    ):
        pid = tl.program_id(0)
        offset1 = tl.load(in_ptr0 + OFFSET0)
        idx_in1 = offset1 + pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr1 + idx_in1, cache_modifier = ".ca", eviction_policy = "evict_first", volatile = True)
        tmp1 = tl_math.exp(tmp0)
        idx_out0 = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + idx_out0, tmp1)

    def triton_ldst_indirect_00_func(x0, x1, s, xs):
        n = x1.numel()
        ns = n - s
        assert ns == xs, "test only single core"
        y0 = torch.empty((ns,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_00_kernel[ns // xs, 1, 1](
            y0, x0, x1, OFFSET0 = s, XS = xs)
        return y0

    def torch_ldst_indirect_00_func(x0, x1, s):
        offset = x0[s]
        return torch.exp(x1[offset:])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 0
    N0, N1 = 16, 16
    blocksize = 16
    assert N0 > offset, "offset must be < N0"
    N1 = N1 + offset
    x0 = torch.arange(0, N0, dtype=torch.int32, device=DEV)
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_00_func(x0, x1, offset)
    triton_cal = triton_ldst_indirect_00_func(x0, x1, offset, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)

def test_ldst_indirect_01():

    @triton.jit
    def triton_ldst_indirect_01_kernel(
        out_ptr0, in_ptr0, in_ptr1, OFFSET0: tl.constexpr,
        XS: tl.constexpr
    ):
        pid = tl.program_id(0)
        offset1 = tl.load(in_ptr0 + OFFSET0)
        idx_in1 = offset1 + pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr1 + idx_in1)
        tmp1 = tl_math.exp(tmp0)
        idx_out0 = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + idx_out0, tmp1)

    def triton_ldst_indirect_01_func(x0, x1, s, xs):
        n = x1.numel()
        ns = n - s
        assert ns == xs, "test only single core"
        y0 = torch.empty((ns,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_01_kernel[ns // xs, 1, 1](
            y0, x0, x1, OFFSET0 = s, XS = xs)
        return y0

    def torch_ldst_indirect_01_func(x0, x1, s):
        offset = x0[s]
        return torch.exp(x1[offset:])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 0
    N0, N1 = 16, 16
    blocksize = 16
    assert N0 > offset, "offset must be < N0"
    N1 = N1 + offset
    x0 = torch.arange(0, N0, device=DEV) # int64
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_01_func(x0, x1, offset)
    triton_cal = triton_ldst_indirect_01_func(x0, x1, offset, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)

def test_ldst_indirect_02():

    @triton.jit
    def triton_ldst_indirect_02_kernel(
        out_ptr0, in_ptr0, in_ptr1,
        XS: tl.constexpr
    ):
        pid = tl.program_id(0)
        for i in tl.range(0, XS):
            tmp0 = tl.load(in_ptr0 + i)
            tmp1 = tl.load(in_ptr1 + tmp0)
            tmp2 = tl_math.exp(tmp1)
            tl.store(out_ptr0 + i, tmp2)

    def triton_ldst_indirect_02_func(x0, x1, xs):
        n0 = x0.numel()
        assert n0 == xs, "test only single core"
        y0 = torch.empty((n0,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_02_kernel[n0 // xs, 1, 1](
            y0, x0, x1, XS = xs)
        return y0

    def torch_ldst_indirect_02_func(x0, x1):
        return torch.exp(x1[x0])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 16
    assert N1 >= N0+offset, "N1 must be >= N0+offset"
    assert N0 == blocksize, "N0 must be == blocksize"
    x0 = offset + torch.arange(0, N0, device=DEV) # int64
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_02_func(x0, x1)
    triton_cal = triton_ldst_indirect_02_func(x0, x1, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)

def test_ldst_indirect_03():

    @triton.jit
    def triton_ldst_indirect_03_kernel(
        out_ptr0, in_ptr0, in_ptr1,
        XS: tl.constexpr
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp1 = tl.load(in_ptr1 + tmp0)
        tmp2 = tl_math.exp(tmp1)
        out0_idx = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_03_func(x0, x1, xs):
        n0 = x0.numel()
        assert n0 == xs, "test only single core"
        y0 = torch.empty((n0,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_03_kernel[n0 // xs, 1, 1](
            y0, x0, x1, XS = xs)
        return y0

    def torch_ldst_indirect_03_func(x0, x1):
        return torch.exp(x1[x0])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 16
    assert N1 >= N0+offset, "N1 must be >= N0+offset"
    assert N0 == blocksize, "N0 must be == blocksize"
    x0 = offset + torch.arange(0, N0, device=DEV) # int64
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_03_func(x0, x1)
    triton_cal = triton_ldst_indirect_03_func(x0, x1, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)

def test_ldst_indirect_04():

    @triton.jit
    def triton_ldst_indirect_04_kernel(
        out_ptr0, in_ptr0, in_ptr1,
        XS: tl.constexpr
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp0min = tl.min(tmp0, axis=0)
        tmp0max = tl.max(tmp0, axis=0)
        tmp0 = tmp0 * 2.0
        tmp0 = tl.clamp(tmp0, tmp0min, tmp0max)
        tmp0 = tmp0.to(tl.int32)
        tmp1 = tl.load(in_ptr1 + tmp0)
        tmp2 = tl_math.exp(tmp1)
        out0_idx = pid * XS + tl.arange(0, XS)
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_04_func(x0, x1, xs):
        n0 = x0.numel()
        assert n0 == xs, "test only single core"
        y0 = torch.empty((n0,), dtype=x1.dtype, device=x1.device)
        triton_ldst_indirect_04_kernel[n0 // xs, 1, 1](
            y0, x0, x1, XS = xs)
        return y0

    def torch_ldst_indirect_04_func(x0, x1):
        x0min = torch.min(x0)
        x0max = torch.max(x0)
        idx = torch.clamp(x0*2, x0min, x0max)
        return torch.exp(x1[idx.to(torch.int32)])

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 16
    assert N1 >= N0+offset, "N1 must be >= N0+offset"
    assert N0 == blocksize, "N0 must be == blocksize"
    x0 = offset + torch.arange(0, N0, dtype=torch.float32, device=DEV)
    x1 = torch.randn((N1,), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_04_func(x0, x1)
    triton_cal = triton_ldst_indirect_04_func(x0, x1, blocksize)
    torch.testing.assert_close(triton_cal, torch_ref)

def test_ldst_indirect_05():

    @triton.jit
    def triton_ldst_indirect_05_kernel(
        out_ptr0, in_ptr1, in_ptr2, stride_in_r,
        XS: tl.constexpr, RS: tl.constexpr
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = tl.arange(0, XS)
        tmp1 = tl.load(in_ptr1 + in_idx1)
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        tmp2 = tl_math.exp(tmp2)
        out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_05_func(xc, x2, xs, rs):
        nr = x2.size()[0]
        nc = xc.numel()
        stride_in_r = x2.stride()[0]
        assert nr == xs, "test only single core"
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_05_kernel[nr // xs, 1, 1](
            y0, xc, x2, stride_in_r, XS = xs, RS = rs)
        return y0

    def torch_ldst_indirect_05_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return torch.exp(extracted)

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 8
    lowdimsize = N0
    assert N1 >= N0+offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_05_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_05_func(xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)

def test_ldst_indirect_06():

    @triton.jit
    def triton_ldst_indirect_06_kernel(
        out_ptr0, in_ptr0, in_ptr1, in_ptr2, stride_in_r,
        XS: tl.constexpr, RS: tl.constexpr
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp1 = tl.load(in_ptr1 + in_idx1)
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        tmp2 = tl_math.exp(tmp2)
        out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_06_func(xr, xc, x2, xs, rs):
        nr = x2.size()[0]
        nc = xc.numel()
        stride_in_r = x2.stride()[0]
        assert nr == xs, "test only single core"
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_06_kernel[nr // xs, 1, 1](
            y0, xr, xc, x2, stride_in_r, XS = xs, RS = rs)
        return y0

    def torch_ldst_indirect_06_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return torch.exp(extracted)

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 4
    lowdimsize = N0
    assert N1 >= N0+offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_06_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_06_func(xr, xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)

def test_ldst_indirect_07():

    @triton.jit
    def triton_ldst_indirect_07_kernel(
        out_ptr0, in_ptr0, in_ptr1, in_ptr2, stride_in_r,
        XS: tl.constexpr, RS: tl.constexpr
    ):
        pid = tl.program_id(0)
        in_idx0 = pid * XS + tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = tl.load(in_ptr0 + in_idx0)
        tmp1 = tl.load(in_ptr1 + in_idx1)
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_07_func(xr, xc, x2, xs, rs):
        nr = x2.size()[0]
        nc = xc.numel()
        stride_in_r = x2.stride()[0]
        assert nr == xs, "test only single core"
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_07_kernel[nr // xs, 1, 1](
            y0, xr, xc, x2, stride_in_r, XS = xs, RS = rs)
        return y0

    def torch_ldst_indirect_07_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return extracted

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 4
    lowdimsize = N0
    assert N1 >= N0+offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_07_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_07_func(xr, xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_08():

    @triton.jit
    def triton_ldst_indirect_08_kernel(
        out_ptr0, in_ptr_xc, in_ptr_x2, stride_in_r,
        OUT_COLS: tl.constexpr, XS: tl.constexpr, RS: tl.constexpr
    ):
        pid = tl.program_id(0)
        row_idx_full = pid * XS + tl.arange(0, XS)
        col_pos = tl.arange(0, RS)
        xc_vals = tl.load(in_ptr_xc + col_pos)
        row_arange = tl.arange(0, XS)
        gather_flat = row_arange[:, None] * stride_in_r + xc_vals[None, :]
        vals = tl.load(in_ptr_x2 + gather_flat)
        vals = tl_math.exp(vals)
        out_flat = row_idx_full[:, None] * OUT_COLS + xc_vals[None, :]
        tl.store(out_ptr0 + out_flat, vals)

    def triton_ldst_indirect_08_func(xc, x2, xs, rs):
        nr = x2.size(0)
        out_cols = x2.size(1)
        stride_in_r = x2.stride(0)
        assert nr == xs, "test only single core"
        y0 = torch.zeros((nr, out_cols), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_08_kernel[nr // xs, 1, 1](
            y0, xc, x2, stride_in_r,
            OUT_COLS=out_cols, XS=xs, RS=rs
        )
        return y0

    def torch_ldst_indirect_08_func(xr, xc, x2):
        out = torch.zeros((xr.numel(), x2.size(1)), dtype=x2.dtype, device=x2.device)
        gathered = torch.exp(x2[xr[:, None], xc[None, :]])
        out.scatter_(1, xc.expand(xr.numel(), -1), gathered)
        return out

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 8
    lowdimsize = N0
    assert N1 >= N0 + offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_08_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_08_func(xc, x2, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)


def test_ldst_indirect_09():

    @triton.jit
    def triton_ldst_indirect_09_kernel(
        out_ptr0, in_ptr1, in_ptr2, stride_in_r,
        offset: tl.constexpr, XS: tl.constexpr, RS: tl.constexpr
    ):
        pid = tl.program_id(0)
        in_idx0 = tl.arange(0, XS)
        in_idx1 = tl.arange(0, RS)
        tmp0 = pid * XS + tl.load(in_ptr1 + in_idx0)
        tmp1 = tl.arange(0, RS) + offset
        in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
        tmp2 = tl.load(in_ptr2 + in_idx2)
        tmp2 = tl_math.exp(tmp2)
        out0_idx = pid * XS * RS + in_idx0[:, None] * RS + in_idx1[None, :]
        tl.store(out_ptr0 + out0_idx, tmp2)

    def triton_ldst_indirect_09_func(xr, x2, offset, xs, rs):
        nr = xr.numel()
        nc = rs
        stride_in_r = x2.stride()[0]
        y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
        triton_ldst_indirect_09_kernel[nr // xs, 1, 1](
            y0, xr, x2, stride_in_r, offset = offset, XS = xs, RS = rs)
        return y0

    def torch_ldst_indirect_09_func(xr, xc, x2):
        flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
        extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
        return torch.exp(extracted)

    DEV = "npu"
    DTYPE = torch.float32
    offset = 8
    N0, N1 = 16, 32
    blocksize = 8
    lowdimsize = N0
    assert N1 >= N0+offset, "N1 must be >= N0+offset"
    assert N0 == lowdimsize, "N0 must be == lowdimsize"
    xc = offset + torch.arange(0, N0, device=DEV)
    xr = torch.arange(0, blocksize, device=DEV)
    x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
    torch_ref = torch_ldst_indirect_09_func(xr, xc, x2)
    triton_cal = triton_ldst_indirect_09_func(xr, x2, offset, blocksize, lowdimsize)
    torch.testing.assert_close(triton_cal, torch_ref)

if __name__ == "__main__":
    test_ldst_indirect_05()
    print("success: test_ldst_indirect_05")