# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common


@triton.jit
def fn_npu_(in_ptr0, out_ptr0, xnumel, ynumel,
            XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    X = xnumel
    Y = ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)

    x0 = xindex[:, None]
    rbase = tl.arange(0, RBLOCK)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.int8)
    for roffset in range(0, ynumel, RBLOCK):
        rindex = roffset + rbase
        rmask = None
        r1 = rindex[None, :]
        tmp0 = tl.load(in_ptr0 + (r1 + (Y * x0)), rmask)
        _tmp6 = _tmp6 ^ tmp0
    tmp6 = tl.xor_sum(_tmp6, 1)

    tl.store(out_ptr0 + (xindex), tmp6, None)


def bar(tensor):
    N, M = tensor.shape
    result = torch.zeros(N, dtype=tensor.dtype, device=tensor.device)
    for i in range(N):
        row_xor_sum = 0
        for j in range(M):
            row_xor_sum ^= tensor[i, j].item()
            result[i] = row_xor_sum
    return result


@pytest.mark.parametrize('param_list',
                         [
                             ['int8', (64, 32), 64, 32],
                         ])
def test_case(param_list):
    dtype, shape, xblock, rblock = param_list
    a = test_common.generate_tensor(shape, dtype).npu()

    std_ret = bar(a)
    print(f"std_ret={std_ret}")

    value = torch.empty_strided((a.shape[0],), (1,), dtype=eval('torch.' + dtype)).npu()
    XBLOCK = xblock
    RBLOCK = rblock
    NBLOCK = a.shape[0] // XBLOCK
    fn_npu_[NBLOCK, 1, 1](a, value, a.shape[0], a.shape[1], XBLOCK, RBLOCK)
    print(f"triton_ret={value}")

    torch.testing.assert_close(value, std_ret)
