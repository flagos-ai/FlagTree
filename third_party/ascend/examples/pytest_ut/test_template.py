# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl

import time
import torch
import torch_npu
import test_common

NBLOCKS = 1
X_SIZE : tl.constexpr = 4
Y_SIZE : tl.constexpr = 64
Z_SIZE : tl.constexpr = 32
NUMEL = X_SIZE * Y_SIZE * Z_SIZE

def fn(input):
    output = input.reshape((X_SIZE, Y_SIZE, Z_SIZE)).permute((1, 0, 2)).reshape((X_SIZE * Y_SIZE * Z_SIZE))
    return output

@triton.jit
def fn_kernel(output_ptr, input_ptr):
    col_offsets = tl.arange(0, X_SIZE * Y_SIZE * Z_SIZE)
    input_local = tl.load(input_ptr + col_offsets)
    input_local = input_local.reshape((X_SIZE, Y_SIZE, Z_SIZE)).permute((1, 0, 2)).reshape((X_SIZE * Y_SIZE * Z_SIZE))  
    tl.store(output_ptr + col_offsets, input_local)


def test_cases():
    input = torch.randn(NUMEL, dtype=torch.float16).npu()
    output = torch.randn(NUMEL, dtype=torch.float16).npu()
    output2 = torch.randn(NUMEL, dtype=torch.float16).npu()
    fn_kernel[1,1,1](output, input)
    output2 = fn(input)
    test_common.validate_cmp('float16', output, output2)
    print("data validation passed")
