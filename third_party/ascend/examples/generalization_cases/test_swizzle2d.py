# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import random
import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils


def swizzle2d(size_i, size_j, size_g):
    i = torch.arange(0, size_i)[:, None]
    j = torch.arange(0, size_j)[None, :]
    ij = i * size_j + j
    size_gj = size_g * size_j
    group_id = ij // size_gj
    off_i = group_id * size_g
    size_g = torch.min(size_i - off_i, torch.tensor(size_g).expand_as(off_i))
    ij = ij % size_gj
    new_i = off_i + ij % size_g
    new_j = ij // size_g
    ret = new_i * size_i + new_j
    return ret


@triton.jit
def fn_npu_(out0, out1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    i = tl.arange(0, XB)[:, None]
    j = tl.arange(0, YB)[None, :]
    ij = i * YB + j
    xx, yy = tl.swizzle2d(i, j, size_i=XB, size_j=YB, size_g=ZB)

    ptr = tl.load(out0)
    xx = tl.cast(xx, dtype=ptr.dtype)
    yy = tl.cast(yy, dtype=ptr.dtype)
    tl.store(out0 + ij, xx)
    tl.store(out1 + ij, yy)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64'])
def test_swizzle2d(shape, dtype):
    if (shape[0] > 255) or (shape[1] > 255):
        return
    size_g = random.randint(1, min(shape[0], shape[1]))
    ans = swizzle2d(shape[0], shape[1], size_g).to(eval('torch.' + dtype)).npu()

    out0 = test_common.generate_tensor(shape, dtype).npu()
    out1 = test_common.generate_tensor(shape, dtype).npu()
    fn_npu_[1, 1, 1](out0, out1, shape[0], shape[1], size_g)
    triton_ret = out0 * shape[0] + out1
    torch.testing.assert_close(triton_ret, ans)
