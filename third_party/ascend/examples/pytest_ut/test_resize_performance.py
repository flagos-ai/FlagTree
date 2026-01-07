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
import numpy as np
import math
import pytest


@triton.jit
def nearest_resize_kernel_col_tile(
    img_src_ptr, img_dst_ptr, 
    src_rows: tl.constexpr, 
    src_cols: tl.constexpr, 
    dst_rows: tl.constexpr, 
    dst_cols: tl.constexpr,
    RR_H: tl.constexpr, 
    RR_W: tl.constexpr, 
    stride_in_h: tl.constexpr, 
    stride_in_w: tl.constexpr, 
    stride_in_c: tl.constexpr, 
    stride_out_h: tl.constexpr, 
    stride_out_w: tl.constexpr, 
    stride_out_c: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    #RR_H和RR_W分别为高和宽的缩放比例
    block_id_c = tl.program_id(0)
    block_id_h = tl.program_id(1)
    block_id_w = tl.program_id(2)

    dest_w_offs = (
        block_id_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    )

    dest_offs = (
        block_id_c[None, None] * stride_out_c
        + block_id_h[None, None] * stride_out_h
        + dest_w_offs[ None, :] * stride_out_w
    )

    fx = dest_w_offs * RR_W
    sx = tl.floor(fx)

    new_col = block_id_h * RR_H
    src_offsets = (
        block_id_c[None, None] * stride_in_c
        + new_col[None, None].to(tl.int32) * stride_in_h
        + tl.clamp(sx, 0, src_cols -1)[None, :].to(tl.int32) * stride_in_w)
    src_val = tl.load(img_src_ptr + src_offsets)
    dst_mask = dest_w_offs[None, :] < dst_cols
    tl.store(img_dst_ptr + dest_offs, src_val, mask=dst_mask)
    
    
def nearest_resize_cpu(img_src, img_dst, dst_rows, dst_cols):
    N, C, src_rows, src_cols = img_src.shape
    # 2,4 64, 32
    #RR_H和RR_W分别为高和宽的缩放比例
    RR_H =  src_rows / float(dst_rows) 
    RR_W =  src_cols / float(dst_cols)
    print("RR_H RR_W", RR_H, RR_W)
    # 2, 2
    #根据output image的坐标值(i,j)计算input image的坐标值(sy, sx)
    for i in range(dst_rows): #32
        for j in range(dst_cols): #16
            # fy = i * 2 = 0/1/..31 * 2 = 0/2/4...62
            fy = (i * RR_H)
            sy = math.floor(fy)
            # fx = j * 2 = 0/1/2..14 * 2 = 0/2/4...28
            fx = (j * RR_W)
            sx = math.floor(fx)
            src_val = img_src[0, :, np.clip(sy, 0, src_rows -1), np.clip(sx, 0, src_cols -1)]    
            # img_dst[0, :, i, j] 表示取批量中第 0 张图像、第 i 行第 j 列位置上的所有通道像素值        
            img_dst[0, :, i, j] = src_val


def test_nearest_resize():
    n, c, h, w = 1, 4, 64, 64
    img_src = torch.randint(0, 255, size=(n, c, h, w))
    dst_rows = h // 2
    dst_cols = w // 2
    img_dst_cpu = torch.randint(0, 255, size=(n, c, dst_rows, dst_cols))
    nearest_resize_cpu(img_src, img_dst_cpu, dst_rows, dst_cols)
    
    # call triton kernel
    img_src = img_src.npu()
    RR_H =  h / float(dst_rows) 
    RR_W =  w / float(dst_cols)
    img_dst_npu = torch.randint(0, 255, size=(n, c, dst_rows, dst_cols)).npu()
    stride_in_h = h
    stride_in_w = 1
    stride_in_c = h * w
    stride_out_h = dst_rows
    stride_out_w = 1
    stride_out_c = dst_rows * dst_cols
    BLOCK_SIZE = 32
    # best performance
    nearest_resize_kernel_col_tile[(4, 32, 1)](img_src, img_dst_npu, h, w, dst_rows, dst_cols, RR_H, RR_W, stride_in_h, stride_in_w, stride_in_c, stride_out_h, stride_out_w, stride_out_c, BLOCK_SIZE)
    assert torch.equal(img_dst_cpu.cpu(), img_dst_npu.cpu())