import math
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import test_common


def torch_arange(start, end):
    TRITON_MAX_TENSOR_NUMEL = 1048576
    if end < start:
        raise ValueError("arange's end argument must be greater than the start argument")
    if end - start > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}")
    return torch.arange(start, end)


def torch_arange_access(start, end):
    z = torch.zeros([end], dtype=torch.int32).npu()
    v = torch.arange(start, end).npu()
    z[start:end] = v
    return z


@triton.jit
def triton_arange(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(z + off, val)


@triton.jit
def triton_arange_access(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(START, END)
    val = tl.arange(START, END)
    tl.store(z + off, val)


@pytest.mark.parametrize('param_list',
                         [
                             [0, 128],
                             [7, 128],
                             [128, 1024],
                         ]
                         )
def test_case(param_list):
    start, end = param_list
    shape = [end - start]
    block = end - start
    dtype = 'int32'

    y_ref = torch_arange(start, end)
    y_cal = torch.zeros(shape, dtype=torch.int32).npu()

    triton_arange[(1, )](y_cal, START = start, END = end, BLOCK = block)

    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize('param_list',
                         [
                             [0, 128],
                             [7, 128],
                             [128, 1024],
                         ]
                         )
def test_case_access(param_list):
    start, end = param_list
    shape = [end]
    block = end - start
    dtype = 'int32'

    y_ref = torch_arange_access(start, end)
    y_cal = torch.zeros(shape, dtype=torch.int32).npu()

    triton_arange_access[(1, )](y_cal, START = start, END = end, BLOCK = block)

    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize('invalid_param_list',
                         [
                             [0, 10000000],
                         ]
                         )
@test_common.raises_with_match(triton.compiler.errors.CompilationError,
    "end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = 1048576")
def test_arange_invalid_range(invalid_param_list):
    start, end = invalid_param_list
    shape = [end - start]
    block = end - start

    y_cal = torch.zeros(shape, dtype=torch.int32).npu()

    triton_arange[(1, )](y_cal, START = start, END = end, BLOCK = block)


@pytest.mark.parametrize('invalid_param_list',
                         [
                             [1024, 128],
                         ]
                         )
@test_common.raises_with_match(triton.compiler.errors.CompilationError,
    "arange's end argument must be greater than the start argument")
def test_arange_invalid_revinput(invalid_param_list):
    start, end = invalid_param_list
    range = abs(end - start)
    shape = [range]
    block = range
    
    y_cal = torch.zeros(shape, dtype=torch.int32).npu()

    triton_arange[(1, )](y_cal, START = start, END = end, BLOCK = block)