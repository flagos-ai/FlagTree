# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl

import torch
import torch_npu

NBLOCKS = 1
XS : tl.constexpr = 128
YS : tl.constexpr = 4
ZS : tl.constexpr = 8
NUMEL : tl.constexpr = XS * ZS

@triton.jit
def fn_broadcast(output_ptr, x_ptr, length):
    col_offsets = tl.arange(0, NUMEL)
    input = tl.load(x_ptr + col_offsets)
    result = input.reshape((XS, 1, ZS)).broadcast_to((XS, YS, ZS)).reshape((XS * YS * ZS))
    brc_col_offsets = tl.arange(0, NUMEL * YS)
    tl.store(output_ptr + brc_col_offsets, result)

def test_broadcast():
    length = NUMEL

    x = torch.randn((XS, 1, ZS), dtype=torch.float32).npu()
    output = torch.randn((XS, YS, ZS), dtype=torch.float32).npu()
    fn_broadcast[NBLOCKS,1,1](output, x, length, debug=True)
    assert(torch.equal(output, x.repeat(1, YS, 1)))

if __name__ == "__main__":
    test_broadcast()

