# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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
"""
Persistent Matmul
=====================
"""

import argparse
import time

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

DEV = "npu"
DTYPE = torch.float16


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def get_num_compute_cores():
    return get_npu_properties()["num_aicore"]


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    bytes_per_elem = args["c_ptr"].element_size()
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_sm = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_sm += 1

    tile_id = start_pid - NUM_SMS
    ki = -1
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_sm):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def get_configs(dtype):
    return {
        torch.float16: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 4,
            "num_stages": 2,
            "num_warps": 8,
        }
    }[dtype]


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    _, N = b.shape
    configs = get_configs(a.dtype)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=configs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs["GROUP_SIZE_M"],
        num_stages=configs["num_stages"],
        num_warps=configs["num_warps"],
    )
    return c


def matmul_persistent(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    num_sms = get_num_compute_cores()
    M, K = a.shape
    _, N = b.shape
    configs = get_configs(a.dtype)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (min(num_sms, triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])), )

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=configs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs["GROUP_SIZE_M"],
        NUM_SMS=num_sms,
        num_stages=configs["num_stages"],
        num_warps=configs["num_warps"],
    )
    return c


def torch_matmul(a, b):
    return torch.matmul(a, b)


def bench(K, reps=10):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device=DEV, dtype=DTYPE)
    b = torch.randn((K, N), device=DEV, dtype=DTYPE)

    for _ in range(reps):
        _ = torch_matmul(a, b)
        time.sleep(0.01)
    for _ in range(reps):
        _ = matmul(a, b)
        time.sleep(0.01)
    for _ in range(reps):
        _ = matmul_persistent(a, b)
        time.sleep(0.01)


def validate(M, N, K):
    a = torch.randn((M, K), device=DEV, dtype=DTYPE)
    b = torch.randn((K, N), device=DEV, dtype=DTYPE)

    torch_result = torch_matmul(a, b)
    naive_result = matmul(a, b)
    persistent_result = matmul_persistent(a, b)

    naive_vs_torch = "✅" if torch.allclose(naive_result, torch_result, atol=1.0) else "❌"
    persistent_vs_torch = "✅" if torch.allclose(persistent_result, torch_result, atol=1.0) else "❌"
    naive_vs_persistent = "✅" if torch.allclose(naive_result, persistent_result, atol=1.0) else "❌"

    print(f"M={M}, N={N}, K={K} verification naive vs torch: {naive_vs_torch} "
          f"persistent vs torch: {persistent_vs_torch} naive vs persistent: {naive_vs_persistent}")
    return torch_result, naive_result, persistent_result


def test_persistent_matmul_validate_cases():
    M, N, K = 32, 32, 32
    torch.manual_seed(0)
    torch_result, naive_result, persistent_result = validate(M, N, K)
    torch.testing.assert_close(naive_result, torch_result, atol=1.0, rtol=0)
    torch.testing.assert_close(persistent_result, torch_result, atol=1.0, rtol=0)
    torch.testing.assert_close(naive_result, persistent_result, atol=1.0, rtol=0)


if __name__ == "__main__":
    test_persistent_matmul_validate_cases()
    print("======Persistent MatMul Test Passed!======")
