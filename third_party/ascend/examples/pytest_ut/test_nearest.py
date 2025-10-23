import torch
import torch_npu
import triton
import triton.language as tl
import math
import numpy as np
import pytest

@triton.jit
def nearest_resize_kernel(
    img_src_ptr, img_dst_ptr, src_rows, src_cols, dst_rows, dst_cols,
    RR_H, RR_W, C,
    stride_in_h, stride_in_w, stride_in_c,
    stride_out_h, stride_out_w, stride_out_c,
    BLOCK_SIZE: tl.constexpr
):
    #RR_H和RR_W分别为高和宽的缩放比例
    block_id_c = tl.program_id(0)
    block_id_h = tl.program_id(1)
    block_id_w = tl.program_id(2)
    dest_h_offs = (
        block_id_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    )
    dest_w_offs = (
        block_id_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    )
    dest_offs = (
        block_id_c[ None, None] * stride_out_c
        + dest_h_offs[ :, None] * stride_out_h
        + dest_w_offs[ None, :] * stride_out_w
    )
    #根据output image的坐标值(dest_h_offs, dest_w_offs)计算input image的坐标值(sy, sx)
    fy = dest_h_offs * RR_H
    sy = tl.floor(fy)
    fx = dest_w_offs * RR_W
    sx = tl.floor(fx)

    src_offsets = (
        block_id_c[None, None] * stride_in_c
        + tl.clamp(sy, 0, src_rows -1)[:, None].to(tl.int32) * stride_in_h
        + tl.clamp(sx, 0, src_cols -1)[None, :].to(tl.int32) * stride_in_w)
    src_val = tl.load(img_src_ptr + src_offsets)
    dst_mask = (dest_h_offs[ :, None] < dst_rows) & (dest_w_offs[None, :] < dst_cols)
    tl.store(img_dst_ptr + dest_offs, src_val, mask=dst_mask)

def triton_kernel(img_src, img_dst):
    N, C, src_rows, src_cols = img_src.shape
    _, _, dst_rows, dst_cols = img_dst.shape
    R_H = float(dst_rows) / src_rows
    R_W = float(dst_cols) / src_cols
    RR_H = 1.0 / R_H
    RR_W = 1.0 / R_W
    stride_in_n, stride_in_c, stride_in_h, stride_in_w = img_src.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = img_dst.stride()
    bs = 16
    grid = lambda meta: (
        C,
        triton.cdiv(dst_rows, meta["BLOCK_SIZE"]),
        triton.cdiv(dst_cols, meta["BLOCK_SIZE"]),
    )
    nearest_resize_kernel[grid](
        img_src, img_dst, src_rows, src_cols, dst_rows, dst_cols,
        RR_H, RR_W, C,
        stride_in_h, stride_in_w, stride_in_c,
        stride_out_h, stride_out_w, stride_out_c,
        bs)
    return img_dst

def nearest_resize_cpu(img_src, img_dst):
    N, C, src_rows, src_cols = img_src.shape
    _, _, dst_rows, dst_cols = img_dst.shape
    #RR_H和RR_W分别为高和宽的缩放比例
    RR_H =  src_rows / float(dst_rows)
    RR_W =  src_cols / float(dst_cols)
    #根据output image的坐标值(i,j)计算input image的坐标值(sy, sx)
    for i in range(dst_rows):
        for j in range(dst_cols):
            fy = (i * RR_H)
            sy = math.floor(fy)
            fx = (j * RR_W)
            sx = math.floor(fx)
            src_val = img_src[0, :, np.clip(sy, 0, src_rows -1), np.clip(sx, 0, src_cols -1)]
            img_dst[0, :, i, j] = src_val
    return img_dst

@pytest.mark.parametrize("shapes", [[360, 640, 140, 280],])
def test_nearest(shapes):
    src_rows, src_cols, dst_rows, dst_cols = shapes
    img_src = torch.rand(1, 4, src_rows, src_cols, dtype=torch.float32, device='npu')
    img_dst = torch.zeros((1, img_src.shape[1], dst_rows, dst_cols), dtype=img_src.dtype, device=img_src.device)
    torch_ref = nearest_resize_cpu(img_src.cpu(), img_dst.cpu())
    triton_cal = triton_kernel(img_src, img_dst)
    torch.testing.assert_close(torch_ref.npu(), triton_cal)

if __name__ == "__main__":
    src_rows, src_cols = 360, 640
    dst_rows, dst_cols = 140, 280
    img_src = torch.rand(1, 4, src_rows, src_cols, dtype=torch.float32, device='npu')
    img_dst = torch.zeros((1, img_src.shape[1], dst_rows, dst_cols), dtype=img_src.dtype, device=img_src.device)

    assert img_src.shape[0] == 1, "currently supports only shape[0] == 1 which does not change the functionality of thie case"
    torch_ref = nearest_resize_cpu(img_src.cpu(), img_dst.cpu())
    triton_cal = triton_kernel(img_src, img_dst)
    torch.testing.assert_close(torch_ref.npu(), triton_cal)
    print("success")