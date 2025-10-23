import torch, torch_npu
import triton
import triton.language as tl
import numpy as np
import math
import pytest


@triton.jit
def lanczos_resize_kernel(
    img_src_ptr,
    img_dst_ptr,
    img_coeffs_ptr,
    src_rows,
    src_cols,
    dst_rows,
    dst_cols,
    R_H,
    R_W,
    C,
    stride_in_h,
    stride_in_w,
    stride_in_c,
    stride_out_h,
    stride_out_w,
    stride_out_c,
    BLOCK_SIZE: tl.constexpr,
):
    block_id_c = tl.program_id(0)
    block_id_h = tl.program_id(1)
    block_id_w = tl.program_id(2)
    dest_h_offs = block_id_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dest_w_offs = block_id_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dest_offs = (
        block_id_c[None, None] * stride_out_c
        + dest_h_offs[:, None] * stride_out_h
        + dest_w_offs[None, :] * stride_out_w
    )

    RR_H = 1.0 / R_H
    RR_W = 1.0 / R_W

    fy = (dest_h_offs + 0.5) * RR_H - 0.5
    sy = tl.floor(fy)
    fx = (dest_w_offs + 0.5) * RR_W - 0.5
    sx = tl.floor(fx)

    idxY = tl.floor((fy - sy) * 24.999999).to(tl.int32)
    idxX = tl.floor((fx - sx) * 24.999999).to(tl.int32)
    tableIndex = idxY[:, None] * 25 + idxX[None, :]
    res = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)

    for ii in range(4):
        for jj in range(4):
            src_offsets = (
                block_id_c[None, None] * stride_in_c
                + (tl.clamp((sy + ii - 1), 0, src_rows - 1)).to(tl.int32)[:, None]
                * stride_in_h
                + (tl.clamp((sx + jj - 1), 0, src_cols - 1)).to(tl.int32)[None, :]
                * stride_in_w
            )
            src_val = tl.load(img_src_ptr + src_offsets)
            coeffs_offs = tableIndex[:, :] * 16 + (ii * 4 + jj)[None, None]
            coeffs = tl.load(img_coeffs_ptr + coeffs_offs)
            res = res + src_val * coeffs
    dst_mask = (dest_h_offs[:, None] < dst_rows) & (dest_w_offs[None, :] < dst_cols)
    res = tl.clamp(res, 0.0, 1.0)
    tl.store(img_dst_ptr + dest_offs, res, mask=dst_mask)


def lanczos_resize_triton(img_src, img_dst, c_lanczosCoeffs, dst_rows, dst_cols):
    N, C, src_rows, src_cols = img_src.shape
    R_H = float(dst_rows) / src_rows
    R_W = float(dst_cols) / src_cols

    stride_in_n, stride_in_c, stride_in_h, stride_in_w = img_src.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = img_dst.stride()
    BLOCK_SIZE = 16
    grid = lambda meta: (
        C,
        triton.cdiv(dst_rows, meta["BLOCK_SIZE"]),
        triton.cdiv(dst_cols, meta["BLOCK_SIZE"]),
    )
    lanczos_resize_kernel[grid](
        img_src,
        img_dst,
        c_lanczosCoeffs,
        src_rows,
        src_cols,
        dst_rows,
        dst_cols,
        R_H,
        R_W,
        C,
        stride_in_h,
        stride_in_w,
        stride_in_c,
        stride_out_h,
        stride_out_w,
        stride_out_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return img_dst


def lanczos_resize_cpu(img_src, img_dst, img_coeffs, dst_rows, dst_cols):
    N, C, src_rows, src_cols = img_src.shape
    R_H = float(dst_rows) / src_rows
    R_W = float(dst_cols) / src_cols
    for i in range(dst_rows):
        for j in range(dst_cols):
            RR_H = 1.0 / R_H
            RR_W = 1.0 / R_W
            fy = (i + 0.5) * RR_H - 0.5
            sy = math.floor(fy)
            fx = (j + 0.5) * RR_W - 0.5
            sx = math.floor(fx)
            idxY = math.floor((fy - np.floor(fy)) * 24.999999)
            idxX = math.floor((fx - np.floor(fx)) * 24.999999)
            tableIndex = idxY * 25 + idxX
            res = (0.0, 0.0, 0.0, 0.0)
            for ii in range(4):
                for jj in range(4):
                    idx_y = np.clip(sy + ii - 1, 0, src_rows - 1)
                    idx_x = np.clip(sx + jj - 1, 0, src_cols - 1)
                    src_val = img_src[0, :, idx_y, idx_x]
                    coeffs_offs = tableIndex * 16 + (ii * 4 + jj)
                    coeffs = img_coeffs[coeffs_offs]
                    res = res + src_val * coeffs

            img_dst[0, :, i, j] = np.clip(res, 0.0, 1.0)


@pytest.mark.parametrize("shapes", [[360, 640, 140, 280],])
def test_lanzcos(shapes):
    c_lanczosCoeffs = torch.randn(10000, dtype=torch.float32, device="npu") / 4.0
    src_rows, src_cols, dst_rows, dst_cols = shapes
    img_src = torch.randn(1, 4, src_rows, src_cols, dtype=torch.float32, device="npu")
    img_dst = torch.zeros(
        (1, img_src.shape[1], dst_rows, dst_cols),
        dtype=img_src.dtype,
        device=img_src.device,
    )
    resized_image = lanczos_resize_triton(
        img_src, img_dst, c_lanczosCoeffs, dst_rows, dst_cols
    )
    img_src_cpu = img_src.cpu().numpy()
    img_dst_cpu = torch.zeros(
        (1, img_src_cpu.shape[1], dst_rows, dst_cols), dtype=img_src.dtype, device="cpu"
    ).numpy()
    lanczos_resize_cpu(
        img_src_cpu, img_dst_cpu, c_lanczosCoeffs.cpu().numpy(), dst_rows, dst_cols
    )
    torch.testing.assert_close(
        resized_image.cpu(), torch.from_numpy(img_dst_cpu), atol=1.0 / 255, rtol=0
    )


def benchmark_test(
    fn_ref, fn_triton, ref_args=(), triton_args=(), name="gen_fn", times=10, repeat=10
):
    import time

    print(
        f"--------------------benchmark_{name} for {times * repeat} times--------------------"
    )
    stream = torch.npu.current_stream()
    # warm_up
    stream.synchronize()
    for _ in range(10):
        fn_triton(*triton_args)
    stream.synchronize()

    start = time.perf_counter()
    for _ in range(times * repeat):
        fn_triton(*triton_args)
    stream.synchronize()
    end = time.perf_counter()

    time_compiled = (end - start) / (times * repeat)
    time_compiled *= 1000000

    # warm_up
    stream.synchronize()
    for _ in range(10):
        std = fn_ref(*ref_args)
    stream.synchronize()

    start = time.perf_counter()
    for _ in range(times * repeat):
        std = fn_ref(*ref_args)
    stream.synchronize()
    end = time.perf_counter()
    time_eager = (end - start) / (times * repeat)
    time_eager *= 1000000

    accelerated = (time_eager - time_compiled) / time_compiled * 100
    print(
        f"Accelerated: {accelerated:.4f}% eager takes {time_eager:.3f} us, triton takes {time_compiled:.3f} us"
    )

    return accelerated, time_eager, time_compiled


if __name__ == "__main__":
    c_lanczosCoeffs = torch.randn(10000, dtype=torch.float32, device="npu") / 4.0

    src_rows, src_cols = 360, 640
    dst_rows, dst_cols = 140, 280
    img_src = torch.randn(1, 4, src_rows, src_cols, dtype=torch.float32, device="npu")

    print("==========run npu===============")
    img_dst = torch.zeros(
        (1, img_src.shape[1], dst_rows, dst_cols),
        dtype=img_src.dtype,
        device=img_src.device,
    )
    resized_image = lanczos_resize_triton(
        img_src, img_dst, c_lanczosCoeffs, dst_rows, dst_cols
    )
    resized_cpu = resized_image.cpu().numpy()
    print("==========run cpu===============")
    img_src_cpu = img_src.cpu().numpy()
    img_dst_cpu = torch.zeros(
        (1, img_src_cpu.shape[1], dst_rows, dst_cols), dtype=img_src.dtype, device="cpu"
    ).numpy()
    lanczos_resize_cpu(
        img_src_cpu, img_dst_cpu, c_lanczosCoeffs.cpu().numpy(), dst_rows, dst_cols
    )

    print("==========compare result===============")
    diff = np.abs(resized_cpu - img_dst_cpu)
    max_diff_value = np.max(diff)
    print("max diff float = ", max_diff_value)
    print("max diff * 255 int = ", int(max_diff_value * 255))
    torch.testing.assert_close(
        resized_image.cpu(), torch.from_numpy(img_dst_cpu), atol=1.0 / 255, rtol=0
    )

    print("==========profiling===============")
    accelerate, eager_time, triton_time = benchmark_test(
        lanczos_resize_cpu,
        lanczos_resize_triton,
        ref_args=(
            img_src_cpu,
            img_dst_cpu,
            c_lanczosCoeffs.cpu().numpy(),
            dst_rows,
            dst_cols,
        ),
        triton_args=(img_src, img_dst, c_lanczosCoeffs, dst_rows, dst_cols),
        name="lanzcos",
    )
