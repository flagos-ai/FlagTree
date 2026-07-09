# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl


@triton.jit
def sigmoid_forward(x):
    out_dtype = x.dtype
    return tl.sigmoid(x.to(tl.float32)).to(out_dtype)


@triton.jit
def exp_forward(x):
    out_dtype = x.dtype
    return tl.exp(x.to(tl.float32)).to(out_dtype)


@triton.jit
def log_forward(x):
    out_dtype = x.dtype
    return tl.log(x.to(tl.float32)).to(out_dtype)
