# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
import triton
import triton.language as tl

import torch
import torch_npu
import pytest

@triton.jit
def cat_3d_kernel(x_ptr, y_ptr, output_ptr,  # *Pointers* to input/output vector.
                  XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,  # *Shape* of input.
                  dim: tl.constexpr
                  ):
    if dim == 0:
        X_out: tl.constexpr = 2 * XB
        Y_out: tl.constexpr = YB
        Z_out: tl.constexpr = ZB
        x_idx = tl.arange(0, XB * YB * ZB)
        input_x = tl.load(x_ptr + x_idx)
        input_y = tl.load(y_ptr + x_idx)

        val = tl.cat(input_x, input_y, can_reorder=True)

        idx = tl.arange(0, X_out * Y_out * Z_out)
        tl.store(output_ptr + idx, val)

    elif dim == 1:
        X_out: tl.constexpr = XB
        Y_out: tl.constexpr = 2 * YB
        Z_out: tl.constexpr = ZB
        for idx in range(X_out * Y_out * Z_out):
            i = idx // (Y_out * Z_out)
            remainder = idx % (Y_out * Z_out)
            j = remainder // Z_out
            k = remainder % Z_out

            if j < YB:
                val = tl.load(x_ptr + i * YB * ZB + j * ZB + k)
            else:
                val = tl.load(y_ptr + i * YB * ZB + (j - YB) * ZB + k)

            tl.store(output_ptr + idx, val)

    elif dim == 2:
        X_out: tl.constexpr = XB
        Y_out: tl.constexpr = YB
        Z_out: tl.constexpr = 2 * ZB
        for idx in range(X_out * Y_out * Z_out):
            i = idx // (Y_out * Z_out)
            remainder = idx % (Y_out * Z_out)
            j = remainder // Z_out
            k = remainder % Z_out

            if k < ZB:
                val = tl.load(x_ptr + i * YB * ZB + j * ZB + k)
            else:
                val = tl.load(y_ptr + i * YB * ZB + j * ZB + (k - ZB))

            tl.store(output_ptr + idx, val)

def cat_3d(x1: torch.Tensor,
           x2: torch.Tensor,
           dim: int):
    assert x1.dim() == 3 and x2.dim() == 3, "Inputs must be 3D tensors"
    if dim < 0:
        dim += 3
    assert dim in (0, 1, 2), "Only dim=[-3, 2] supported"
    assert x1.shape[0] == x2.shape[0] and x1.shape[1] == x2.shape[1] and x1.shape[2] == x2.shape[2], \
        "tl.cat only support tensors of same shape"
    XB, YB, ZB = x1.shape
    if dim == 0:
        output_shape = (2 * XB, YB, ZB)
    elif dim == 1:
        output_shape = (XB, 2 * YB, ZB)
    elif dim == 2:
        output_shape = (XB, YB, 2 * ZB)

    output = torch.empty(output_shape, dtype=x1.dtype, device=x1.device)

    cat_3d_kernel[1,1,1](
        x1, x2, output,
        XB, YB, ZB,
        dim=dim
    )
    return output

def test_cat():
    params_list = \
        [
            ('float32', torch.float32, 2, 256, 16, 0),
            ('float32', torch.float32, 8, 8, 4, 1),
            ('float16', torch.float16, 2, 256, 16, 2),
            ('float16', torch.float16, 8, 8, 4, -3),
            ('int8', torch.int8, 2, 256, 16, -2),
            ('int8', torch.int8, 8, 8, 4, -1),
        ]
    
    for param in params_list:
        [para_type, data_type, XB, YB, ZB, dim] = param

        x = torch.full((XB, YB, ZB), 100, dtype=data_type).npu()
        y = torch.full((XB, YB, ZB), 30, dtype=data_type).npu()

        out_triton = cat_3d(x, y, dim)
        out_torch = torch.cat([x, y], dim=dim)

        assert torch.allclose(out_triton, out_torch)

    print("All tests passed! -> OK")
