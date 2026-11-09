# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common


@triton.jit
def kernel_cal_select_mask_bool(
    Output_ptr,
    Indices_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    index = tl.load(Indices_ptr)

    true_tensor = tl.arange(0, BLOCK) < numel
    false_tensor = tl.arange(0, BLOCK) >= numel
    mask = offs < index
    res = tl.where(mask, true_tensor, false_tensor)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask_one_dim_static_offset(QK_ptr, Other_ptr, Output_ptr, BLOCK: tl.constexpr,
                                                 start_zero: tl.constexpr):
    offs = tl.arange(0, BLOCK)

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)

    if start_zero:
        mask = offs < 1
    else:
        mask = offs >= 1

    res = tl.where(mask, qk_ub, other)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask_one_dim_dyn_offset(QK_ptr, Other_ptr, Output_ptr, Indices_ptr, BLOCK: tl.constexpr,
                                              start_zero: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    index = tl.load(Indices_ptr)

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)

    if start_zero:
        mask = offs < index
    else:
        mask = offs >= index

    res = tl.where(mask, qk_ub, other)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask_one_dim_two_dyn_offset(
    QK_ptr,
    Other_ptr,
    Output_ptr,
    Indices_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    minIndex = tl.load(Indices_ptr)
    maxIndex = tl.load(Indices_ptr + 1)

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)

    mask = (offs < maxIndex) and (offs >= minIndex)
    res = tl.where(mask, qk_ub, other)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask_two_dim_static_offset(
    QK_ptr,
    Other_ptr,
    Output_ptr,
    BLOCK: tl.constexpr,
    start_zero: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    offs = rows[:, None] * BLOCK + cols[None, :]

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)
    if start_zero:
        mask_rows = rows < 1
        mask_cols = cols < 1
    else:
        mask_rows = rows >= 1
        mask_cols = cols >= 1

    res = tl.where(mask_rows[:, None] & mask_cols[None, :], qk_ub, other)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask_two_dim_one_dyn_offset(
    QK_ptr,
    Other_ptr,
    Output_ptr,
    Indices_ptr,
    BLOCK: tl.constexpr,
    start_zero: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    offs = rows[:, None] * BLOCK + cols[None, :]
    minIndex = tl.load(Indices_ptr)
    maxIndex = tl.load(Indices_ptr + 1)

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)
    if start_zero:
        mask_rows = rows < 1
    else:
        mask_rows = rows >= 1

    mask_cols = (cols < maxIndex) & (cols >= minIndex)

    res = tl.where(mask_rows[:, None] & mask_cols[None, :], qk_ub, other)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask_two_dim_all_dyn_offset(
    QK_ptr,
    Other_ptr,
    Output_ptr,
    Indices_ptr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    offs = rows[:, None] * BLOCK + cols[None, :]
    min_index1 = tl.load(Indices_ptr)
    max_index1 = tl.load(Indices_ptr + 1)
    min_index2 = tl.load(Indices_ptr + 2)
    max_index2 = tl.load(Indices_ptr + 3)

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)

    mask_rows = (rows < max_index1) & (rows >= min_index1)
    mask_cols = (cols < max_index2) & (cols >= min_index2)

    res = tl.where(mask_rows[:, None] & mask_cols[None, :], qk_ub, other)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask(
    QK_ptr,
    Other_ptr,
    Output_ptr,
    Indices_ptr,
    stride_qk: tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK) * stride_qk
    cols = tl.arange(0, BLOCK)
    offs = rows[:, None] + cols[None, :]
    row_indices = tl.load(Indices_ptr)
    col_indices = tl.load(Indices_ptr + 1)

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)
    mask_rows = rows < row_indices * stride_qk
    mask_cols = cols < col_indices

    res = tl.where(mask_rows[:, None] & mask_cols[None, :], qk_ub, other)
    tl.store(Output_ptr + offs, res)


def torch_cal_select_mask_bool(
    index: torch.Tensor,
    numel,
    BLOCK,
):
    offs = torch.arange(0, BLOCK)
    true_tensor = torch.arange(0, BLOCK) < numel
    false_tensor = torch.arange(0, BLOCK) >= numel
    mask = offs < index

    res = torch.where(mask, true_tensor, false_tensor)
    return res


def torch_cal_select_mask_one_dim_static_offset(
    QK: torch.Tensor,
    Other: torch.Tensor,
    start_zero,
):
    Output = Other.clone()
    if start_zero:
        Output[:1] = QK[:1]
    else:
        Output[1:] = QK[1:]
    return Output


def torch_cal_select_mask_one_dim_dyn_offset(
    QK: torch.Tensor,
    Other: torch.Tensor,
    Indices: torch.Tensor,
    start_zero,
):
    limit_idx = Indices[0].item()
    Output = Other.clone()
    if start_zero:
        Output[:limit_idx] = QK[:limit_idx]
    else:
        Output[limit_idx:] = QK[limit_idx:]
    return Output


def torch_cal_select_mask_one_dim_two_dyn_offset(
    QK: torch.Tensor,
    Other: torch.Tensor,
    Indices: torch.Tensor,
):
    min_idx = Indices[0].item()
    max_idx = Indices[1].item()
    Output = Other.clone()
    Output[min_idx:max_idx] = QK[min_idx:max_idx]

    return Output


def torch_cal_select_mask_two_dim_static_offset(
    QK: torch.Tensor,
    Other: torch.Tensor,
    start_zero,
):
    Output = Other.clone()
    if start_zero:
        Output[:1, :1] = QK[:1, :1]
    else:
        Output[1:, 1:] = QK[1:, 1:]

    return Output


def torch_cal_select_mask_two_dim_one_dyn_offset(
    QK: torch.Tensor,
    Other: torch.Tensor,
    Indices: torch.Tensor,
    start_zero,
):
    min_idx = Indices[0].item()
    max_idx = Indices[1].item()
    Output = Other.clone()
    if start_zero:
        Output[:1, min_idx:max_idx] = QK[:1, min_idx:max_idx]
    else:
        Output[1:, min_idx:max_idx] = QK[1:, min_idx:max_idx]

    return Output


def torch_cal_select_mask_two_dim_all_dyn_offset(
    QK: torch.Tensor,
    Other: torch.Tensor,
    Indices: torch.Tensor,
):
    min_idx1 = Indices[0].item()
    max_idx1 = Indices[1].item()
    min_idx2 = Indices[2].item()
    max_idx2 = Indices[3].item()
    Output = Other.clone()
    Output[min_idx1:max_idx1, min_idx2:max_idx2] = QK[min_idx1:max_idx1, min_idx2:max_idx2]

    return Output


def torch_cal_select_mask(
    QK: torch.Tensor,
    Other: torch.Tensor,
    Indices: torch.Tensor,
):
    row_limit_idx = Indices[0].item()
    col_limit_idx = Indices[1].item()
    Output = Other.clone()
    Output[:row_limit_idx, :col_limit_idx] = QK[:row_limit_idx, :col_limit_idx]
    return Output


@pytest.mark.parametrize('param_list', [['bool', 64, 63]])
def test_select_analysis_bool(param_list):
    dtype, SEQ_LEN, index = param_list
    assert dtype == 'bool'
    qk_cal = torch.empty(SEQ_LEN).npu()
    indices = torch.tensor([index]).npu()
    qk_ref = torch_cal_select_mask_bool(index, SEQ_LEN, SEQ_LEN)
    kernel_cal_select_mask_bool[(1, )](qk_cal, indices, SEQ_LEN, SEQ_LEN)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, True],
    ['float32', 64, False],
])
def test_select_analysis_one_dim_static_offset(param_list):
    dtype, SEQ_LEN, start_zero = param_list
    assert dtype != 'bool'
    qk = torch.rand([SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    qk_ref = torch_cal_select_mask_one_dim_static_offset(qk, other, start_zero)
    kernel_cal_select_mask_one_dim_static_offset[(1, )](qk, other, qk_cal, SEQ_LEN, start_zero)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, 62, True],
    ['float32', 64, 62, False],
])
def test_select_analysis_one_dim_dyn_offset(param_list):
    dtype, SEQ_LEN, index, start_zero = param_list
    assert dtype != 'bool'
    qk = torch.rand([SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    indices_tensor = torch.tensor([index]).npu()
    qk_ref = torch_cal_select_mask_one_dim_dyn_offset(qk, other, indices_tensor, start_zero)
    kernel_cal_select_mask_one_dim_dyn_offset[(1, )](qk, other, qk_cal, indices_tensor, SEQ_LEN, start_zero)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, 1, 62],
    ['float32', 64, 1, 62],
])
def test_select_analysis_one_dim_two_dyn_offset(param_list):
    dtype, SEQ_LEN, minIndex, maxIndex = param_list
    assert dtype != 'bool'
    qk = torch.rand([SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    indices_tensor = torch.tensor([minIndex, maxIndex]).npu()
    qk_ref = torch_cal_select_mask_one_dim_two_dyn_offset(qk, other, indices_tensor)
    kernel_cal_select_mask_one_dim_two_dyn_offset[(1, )](qk, other, qk_cal, indices_tensor, SEQ_LEN)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, True],
    ['float32', 64, False],
])
def test_select_analysis_two_dim_static_offset(param_list):
    dtype, SEQ_LEN, start_zero = param_list
    assert dtype != 'bool'
    qk = torch.rand([SEQ_LEN, SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    qk_ref = torch_cal_select_mask_two_dim_static_offset(qk, other, start_zero)
    kernel_cal_select_mask_two_dim_static_offset[(1, )](qk, other, qk_cal, SEQ_LEN, start_zero)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, 1, 62, True],
    ['float32', 64, 1, 62, False],
])
def test_select_analysis_two_dim_one_dyn_offset(param_list):
    dtype, SEQ_LEN, minIndex, maxIndex, start_zero = param_list
    assert dtype != 'bool'

    qk = torch.rand([SEQ_LEN, SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    indices_tensor = torch.tensor([minIndex, maxIndex]).npu()
    qk_ref = torch_cal_select_mask_two_dim_one_dyn_offset(qk, other, indices_tensor, start_zero)
    kernel_cal_select_mask_two_dim_one_dyn_offset[(1, )](qk, other, qk_cal, indices_tensor, SEQ_LEN, start_zero)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, 2, 62, 1, 63],
    ['float32', 64, 2, 62, 1, 63],
])
def test_select_analysis_two_dim_all_dyn_offset(param_list):
    dtype, SEQ_LEN, minIndex1, maxIndex1, minIndex2, maxIndex2 = param_list
    assert dtype != 'bool'

    qk = torch.rand([SEQ_LEN, SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    indices_tensor = torch.tensor([minIndex1, maxIndex1, minIndex2, maxIndex2]).npu()
    qk_ref = torch_cal_select_mask_two_dim_all_dyn_offset(qk, other, indices_tensor)
    kernel_cal_select_mask_two_dim_all_dyn_offset[(1, )](qk, other, qk_cal, indices_tensor, SEQ_LEN)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, 63, 62],
    ['float32', 64, 63, 62],
])
def test_select_analysis(param_list):
    dtype, SEQ_LEN, indice_x, indice_y = param_list
    assert dtype != 'bool'
    qk = torch.rand([SEQ_LEN, SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    indices_tensor = torch.tensor([indice_x, indice_y]).npu()
    qk_ref = torch_cal_select_mask(qk, other, indices_tensor)
    kernel_cal_select_mask[(1, )](qk, other, qk_cal, indices_tensor, qk.stride(0), SEQ_LEN * SEQ_LEN, SEQ_LEN)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)
