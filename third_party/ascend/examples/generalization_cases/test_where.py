import logging

import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math


def torch_pointwise(x0, x1):
    res = torch.where(x0 < x1, x0, 1)
    return res

@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    tmp2 = X < Y
    ret = tl.where(tmp2, X, 1)

    tl.store(output_ptr + idx, ret)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype',
                         ['bool', 'float32', 'float16', 'bfloat16', 'int8', 'int16', 'int32', 'int64'])
def test_case2(dtype, shape):
    # 生成数据
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()
    new_shape = shape

    ans = torch_pointwise(x, y)
    output = torch.zeros_like(ans)

    if len(shape) == 1:
        fn_npu_[1, 1, shape[0]](output, x, y, z, 1, 1, 1, 1, 1, shape[0])
    elif len(shape) == 2:
        if shape[0] > shape[1]:
            fn_npu_[1, shape[0], 1](output, x, y, z, 1, 1, shape[1], 1, shape[0], shape[1])
        else:
            fn_npu_[1, 1, shape[1]](output, x, y, z, 1, shape[0], 1, 1, shape[0], shape[1])
    elif len(shape) == 3:
        if max(shape[0], shape[1], shape[2]) == shape[0]:
            fn_npu_[shape[0], 1, 1](output, x, y, z, 1, shape[1], shape[2], shape[0], shape[1], shape[2])
        else:
            fn_npu_[1, 1, shape[2]](output, x, y, z, shape[0], shape[1], 1, shape[0], shape[1], shape[2])
    else:
        fn_npu_[1, 1, 1](output, x, y, z, 1, 1, 1, 1, 1, 1)

    test_common.validate_cmp(dtype, ans, output)


@triton.jit
def fn_npu_multi_d(output_ptr, x_ptr, y_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr, DIMS: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if DIMS > 1:
        offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if DIMS > 2:
        offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if DIMS > 3:
        offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if DIMS > 4:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    X = tl.load(x_ptr + offsets)
    Y = tl.load(y_ptr + offsets)

    tmp2 = X < Y
    ret = tl.where(tmp2, X, 1)

    tl.store(output_ptr + offsets, ret)


@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('shape', [
    (4, 2, 8, 4),
    (2, 4, 2, 8, 1),

    (4, 3, 8, 1),
    (3, 4, 2, 8, 4),
])
@pytest.mark.parametrize('dtype',
                         ['float32', 'float16', 'bfloat16', 'int8', 'int16', 'int32', 'int64'])
def test_case_4d_5d(dtype, shape):
    # 生成数据
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    new_shape = shape

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()

    ans = torch_pointwise(x, y)

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    grid = (1, )
    fn_npu_multi_d[grid](output, x, y, *triton_shape, len(shape))

    test_common.validate_cmp(dtype, ans, output)
