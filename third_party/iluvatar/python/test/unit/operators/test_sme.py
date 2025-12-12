# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Licensed under the MIT License

import triton
import triton.language as tl
import torch
import os
from triton._C.libtriton import ir
import re
import inspect
import random
import math
import pytest

triton_cache_dir = os.environ.get('TRITON_CACHE_DIR', '/root/.triton/cache')


def print_result_decorator(func):

    def wrapper(*args, **kwargs):
        is_close, is_sme = func(*args, **kwargs)
        print(f"======> tests {func.__name__} results is close: {is_close}, sme result: {is_sme}")

    return wrapper


def get_ttgir_file(hash, name):
    return triton_cache_dir + os.path.sep + hash + os.path.sep + name + ".ttgir"


def check_dot_use_sme(lines, use_smes):
    dot_cnt = 0
    pattern = r"useSme = (\d+)"
    for line in lines:
        if "tt.dot" in line:
            if dot_cnt >= len(use_smes):
                return False
            matches = re.findall(pattern, line)
            matches = tuple([int(match) for match in matches])
            if matches != use_smes[dot_cnt]:
                print(f"expect {use_smes[dot_cnt]}, but got {matches}")
                # print(use_smes, matches)
                return False
            dot_cnt += 1
    return True


def check_mlir(F, use_smes):
    context = ir.context()
    ir.load_dialects(context)
    # backend.load_dialects(context)
    # codegen_fns = backend.get_codegen_implementation()
    path = get_ttgir_file(F.src.fn.hash_cache_file, F.src.fn.__name__)
    print(path)
    mod = ir.parse_mlir_module(path, context)
    lines = mod.str().split("\n")
    return check_dot_use_sme(lines, use_smes)


@triton.jit
def dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    K_offs = tl.arange(0, K)
    A_vals = tl.load(A + M_offs[:, None] * K + K_offs[None, :])
    B_vals = tl.load(B + K_offs[:, None] * N + N_offs[None, :])
    C_vals = tl.dot(A_vals, B_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@triton.jit
def dot_trans_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    K_offs = tl.arange(0, K)
    A_vals = tl.load(A + M_offs[:, None] * K + K_offs[None, :])
    B_vals = tl.load(B + N_offs[:, None] * K + K_offs[None, :])
    B_vals = tl.trans(B_vals)
    C_vals = tl.dot(A_vals, B_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_16x32_f16_dot():
    A = torch.randn((16, 32), dtype=torch.half, device="cuda")
    B = torch.randn((32, 16), dtype=torch.half, device="cuda")
    C = torch.zeros((16, 16), dtype=torch.half, device="cuda")
    F = dot_kernel[(1, )](A, B, C, 16, 16, 32)
    D = A @ B
    assert check_mlir(F, [(32, 0)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(32, 0)])


@triton.jit
def test_corex_sme_kernel(addrs, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, use_sme: tl.constexpr):
    A = tl.load(addrs).to(tl.pointer_type(tl.float16))
    B = tl.load(addrs + 1).to(tl.pointer_type(tl.float16))
    C = tl.load(addrs + 2).to(tl.pointer_type(tl.float16))
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    K_offs = tl.arange(0, K)
    A_offset = A + M_offs[:, None] * K + K_offs[None, :]
    if use_sme:
        tl.corex_sme(A_offset, 32)
    A_vals = tl.load(A_offset)
    B_vals = tl.load(B + K_offs[:, None] * N + N_offs[None, :])
    C_vals = tl.dot(A_vals, B_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@triton.jit
def test_corex_stride_kernel(addrs, A_stride, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    A = tl.load(addrs).to(tl.pointer_type(tl.float16))
    B = tl.load(addrs + 1).to(tl.pointer_type(tl.float16))
    C = tl.load(addrs + 2).to(tl.pointer_type(tl.float16))
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    K_offs = tl.arange(0, K)
    A_offset = A + M_offs[:, None] * K + K_offs[None, :]
    a_stride = tl.load(A_stride)
    A_vals = tl.load(A_offset, stride=a_stride)
    B_vals = tl.load(B + K_offs[:, None] * N + N_offs[None, :])
    C_vals = tl.dot(A_vals, B_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_corex_stride():
    A = torch.randn((16, 32), dtype=torch.half, device="cuda")
    B = torch.randn((32, 16), dtype=torch.half, device="cuda")
    C = torch.zeros((16, 16), dtype=torch.half, device="cuda")
    addrs = [A.data_ptr(), B.data_ptr(), C.data_ptr()]
    addr_input = torch.tensor(addrs, device="cuda")
    A_stride = torch.tensor(32, dtype=torch.int32, device="cuda")
    F = test_corex_stride_kernel[(1, )](addr_input, A_stride, 16, 16, 32)
    D = A @ B
    assert check_mlir(F, [(64, 0)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(64, 0)])


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_corex_sme():
    A = torch.randn((16, 32), dtype=torch.half, device="cuda")
    B = torch.randn((32, 16), dtype=torch.half, device="cuda")
    C = torch.zeros((16, 16), dtype=torch.half, device="cuda")
    addrs = [A.data_ptr(), B.data_ptr(), C.data_ptr()]
    addr_input = torch.tensor(addrs, device="cuda")
    O = test_corex_sme_kernel[(1, )](addr_input, 16, 16, 32, 0)
    assert check_mlir(O, [(0, 0)])
    F = test_corex_sme_kernel[(1, )](addr_input, 16, 16, 32, 1)
    assert check_mlir(F, [(32, 0)])
    D = A @ B
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(32, 0)])


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_16x32_i8_dot():
    capability = torch.cuda.get_device_capability()
    if capability[0] == 8:
        pytest.skip("tl.dot do not support int8 on QS now")
    A = torch.randint(-127, 127, (64, 64), dtype=torch.int8, device="cuda")
    B = torch.randint(-127, 127, (64, 64), dtype=torch.int8, device="cuda")
    C = torch.zeros((64, 64), dtype=torch.int32, device="cuda")
    # A = torch.randn((16, 32), dtype=torch.half, device="cuda")
    F = dot_kernel[(1, )](A, B, C, 64, 64, 64)
    D = A.to(torch.float32) @ B.to(torch.float32)
    assert check_mlir(F, [(64, 64)])
    return torch.allclose(C.to(torch.float32), D, atol=1e-3, rtol=1e-3), check_mlir(F, [(64, 64)])


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_16x32_f16_trans_dot():
    A = torch.randn((16, 32), dtype=torch.half, device="cuda")
    B = torch.randn((16, 32), dtype=torch.half, device="cuda")
    C = torch.zeros((16, 16), dtype=torch.half, device="cuda")
    F = dot_trans_kernel[(1, )](A, B, C, 16, 16, 32)
    D = A @ B.T
    assert check_mlir(F, [(32, 32)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(32, 32)])


# @print_result_decorator
# def test_16x32_i8_trans_dot():
#     A = torch.randint(-127, 127, (64, 64), dtype=torch.int8, device="cuda")
#     B = torch.randint(-127, 127, (64, 64), dtype=torch.int8, device="cuda")
#     C = torch.zeros((64, 64), dtype=torch.int32, device="cuda")
#     # A = torch.randn((16, 32), dtype=torch.half, device="cuda")
#     F = dot_trans_kernel[(1, )](A, B, C, 64, 64, 64)
#     D = A.to(torch.float32) @ B.T.to(torch.float32)
#     return torch.allclose(C.to(torch.float32), D, atol=1e-3, rtol=1e-3),  check_mlir(F, [(1, 2)])


@triton.jit
def loop_k_dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    K_offs = tl.arange(0, 32)
    N_offs = tl.arange(0, N)
    K_ITER = K // 32
    C_vals = tl.zeros((M, N), dtype=tl.float32)
    for k_iter in range(K_ITER):
        A_offs = A + k_iter * 32 + M_offs[:, None] * K + K_offs[None, :]
        B_offs = B + (k_iter * 32 + K_offs[:, None]) * N + N_offs[None, :]
        A_vals = tl.load(A_offs)
        B_vals = tl.load(B_offs)
        C_vals = tl.dot(A_vals, B_vals, C_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


# @print_result_decorator
# def test_loop_k_dot():
#     A = torch.randn((32, 256), dtype=torch.float16, device="cuda")
#     B = torch.randn((256, 32), dtype=torch.float16, device="cuda")
#     C = torch.zeros((32, 32), dtype=torch.float16, device="cuda")
#     D = A @ B
#     F = loop_k_dot_kernel[(1, )](A, B, C, 32, 32, 256, num_warps=1, num_stages=1)
#     # this case may get stride failed, cause can't get B stride
#     return torch.allclose(C, D, atol=1e-3, rtol=1e-3),  check_mlir(F, [(1, 2)])


@triton.jit
def loop_k_dot_kernel2(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    K_offs = tl.arange(0, 32)
    N_offs = tl.arange(0, N)
    K_ITER = K // 32
    C_vals = tl.zeros((M, N), dtype=tl.float32)
    for k_iter in range(K_ITER):
        A_offs = A + M_offs[:, None] * K + K_offs[None, :] + k_iter * 32
        B_offs = B + k_iter * 32 * N + K_offs[:, None] * N + N_offs[None, :]
        A_vals = tl.load(A_offs)
        B_vals = tl.load(B_offs)
        C_vals = tl.dot(A_vals, B_vals, C_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_loop_k_dot2():
    A = torch.randn((32, 256), dtype=torch.float16, device="cuda")
    B = torch.randn((256, 32), dtype=torch.float16, device="cuda")
    C = torch.zeros((32, 32), dtype=torch.float16, device="cuda")
    D = A @ B
    F = loop_k_dot_kernel2[(1, )](A, B, C, 32, 32, 256, num_warps=1, num_stages=1)
    # this case may get stride failed, cause can't get B stride
    assert check_mlir(F, [(256, 32)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(256, 32)])


@triton.jit
def loop_k_dot_kernel3(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    K_offs = tl.arange(0, 32)
    N_offs = tl.arange(0, N)
    K_ITER = K // 32
    C_vals = tl.zeros((M, N), dtype=tl.float32)
    A_base = A + M_offs[:, None] * K + K_offs[None, :]
    B_base = B + K_offs[:, None] * N + N_offs[None, :]
    for k_iter in range(K_ITER):
        A_offs = A_base + k_iter * 32
        B_offs = B_base + k_iter * 32 * N
        A_vals = tl.load(A_offs)
        B_vals = tl.load(B_offs)
        C_vals = tl.dot(A_vals, B_vals, C_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_loop_k_dot3():
    A = torch.randn((32, 256), dtype=torch.float16, device="cuda")
    B = torch.randn((256, 32), dtype=torch.float16, device="cuda")
    C = torch.zeros((32, 32), dtype=torch.float16, device="cuda")
    D = A @ B
    F = loop_k_dot_kernel3[(1, )](A, B, C, 32, 32, 256, num_warps=1, num_stages=1)
    # this case may get stride failed, cause can't get B stride
    assert check_mlir(F, [(256, 32)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(256, 32)])


@triton.jit
def loop_n_dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    K_offs = tl.arange(0, K)
    N_offs = tl.arange(0, 32)
    N_ITER = N // 32
    A_offs = A + M_offs[:, None] * K + K_offs[None, :]
    A_vals = tl.load(A_offs)
    B_base = B + K_offs[:, None] * N + N_offs[None, :]
    for n_iter in range(N_ITER):
        B_offs = B_base + n_iter * 32
        B_vals = tl.load(B_offs)
        C_vals = tl.dot(
            A_vals,
            B_vals,
        )
        tl.store(C + M_offs[:, None] * N + N_offs[None, :] + n_iter * 32, C_vals)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_loop_n_dot():
    A = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    B = torch.randn((32, 256), dtype=torch.float16, device="cuda")
    C = torch.zeros((32, 256), dtype=torch.float16, device="cuda")
    D = A @ B
    F = loop_n_dot_kernel[(1, )](A, B, C, 32, 256, 32, num_warps=1, num_stages=1)
    assert check_mlir(F, [(32, 256)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(32, 256)])


@triton.jit
def dot_stride_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    K_offs = tl.arange(0, K)
    A_vals = tl.load(A + M_offs[:, None] * K + K_offs[None, :] * 2)
    B_vals = tl.load(B + K_offs[:, None] * N + N_offs[None, :] * 2)
    C_vals = tl.dot(A_vals, B_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


# @print_result_decorator
# def test_dot_stride():
#     A = torch.randn((32, 64), dtype=torch.float16, device="cuda")
#     B = torch.randn((32, 64), dtype=torch.float16, device="cuda")
#     C = torch.zeros((32, 32), dtype=torch.float16, device="cuda")
#     D = A[:, ::2] @ B[:, ::2]
#     F = dot_stride_kernel[(1, )](A, B, C, 32, 32, 32, num_warps=1, num_stages=1)
#     return torch.allclose(C, D, atol=1e-3, rtol=1e-3),  check_mlir(F, [(0, 0)])


@triton.jit
def load_dot_kernel(A, B, C, R, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    K_offs = tl.arange(0, K)
    K_vals = tl.load(R + K_offs)
    A_offs = A + K_vals[:, None] * K + K_offs[None, :]
    B_offs = B + K_offs[:, None] * N + K_vals[None, :]
    A_vals = tl.load(A_offs)
    B_vals = tl.load(B_offs)
    C_vals = tl.dot(A_vals, B_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_load_dot():
    R = torch.arange(0, 32, dtype=torch.int32, device="cuda")
    A = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    B = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    C = torch.zeros((32, 32), dtype=torch.float16, device="cuda")
    D = A @ B
    F = load_dot_kernel[(1, )](A, B, C, R, 32, 32, 32)
    assert check_mlir(F, [(0, 0)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(0, 0)])


@triton.jit
def multi_use_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    K_offs = tl.arange(0, K)
    A_offs = A + M_offs[:, None] * K + K_offs[None, :]
    B_offs = B + K_offs[:, None] * N + N_offs[None, :]
    A_vals = tl.load(A_offs)
    B_vals = tl.load(B_offs)
    C_vals = tl.dot(A_vals, B_vals)
    C_vals += A_vals
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_multi_use_dot():
    A = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    B = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    C = torch.zeros((32, 32), dtype=torch.float16, device="cuda")
    D = A @ B + A
    F = multi_use_kernel[(1, )](A, B, C, 32, 32, 32)
    assert check_mlir(F, [(0, 32)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(0, 32)])


@triton.jit
def vllm_paged_attention_fwd(
    output,
    query,
    key_cache,
    value_cahe,
    block_tables,
    context_lens,
    query_stride,
    block_tables_stride,
    kv_block_stride,
    kv_head_stride,
    scale,
    num_kv_group: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    head_offs = tl.arange(0, head_size)
    kv_group_offs = tl.arange(0, 16)
    block_offs = tl.arange(0, block_size)
    context_len = tl.load(context_lens + batch_idx)
    num_blocks = (context_len + block_size - 1) // block_size
    # 16xhead_size
    query_vals = tl.load(query + batch_idx * query_stride + kv_head_idx * num_kv_group * head_size +
                         kv_group_offs[:, None] * head_size + head_offs[None, :])
    block_tables_base = block_tables + batch_idx * block_tables_stride
    key_cache = key_cache + kv_head_idx * kv_head_stride
    value_cache = value_cahe + kv_head_idx * kv_head_stride
    min_val = -float("inf")
    qk_max = tl.full((16, ), min_val, dtype=tl.float32)
    sum_exp = tl.zeros((16, block_size), dtype=tl.float32)
    sum_exp_dot = tl.zeros((16, head_size), dtype=tl.float32)
    key_base = key_cache + block_offs[:, None] * head_size + head_offs[None, :]
    val_base = value_cache + block_offs[:, None] * head_size + head_offs[None, :]
    # tl.store(output, num_blocks,)
    for block_idx in range(num_blocks):
        if block_idx == num_blocks - 1:
            mask = block_size - 1
            block_res = ((context_len - 1) & mask) + 1
        else:
            block_res = 16
        physical_block_idx = tl.load(block_tables_base + block_idx)
        key_offs = key_base + physical_block_idx * kv_block_stride
        # block_size x head_size
        # head_size x block_size
        key_vals = tl.trans(tl.load(key_offs, ))
        # 16 x head_size @ head_size x block_size = 16 x block_size
        qk = tl.dot(query_vals, key_vals) * scale
        qk = tl.where(block_offs[None, :] < block_res, qk, min_val)
        # 16
        new_max = tl.max(qk, axis=1)
        new_max = tl.maximum(qk_max, new_max)
        scale_factor = tl.exp(qk_max - new_max)
        qk_max = new_max
        sum_exp *= scale_factor[:, None]
        sum_exp_dot *= scale_factor[:, None]
        qk_exp = tl.exp(qk - qk_max[:, None])
        sum_exp += qk_exp
        val_offs = val_base + physical_block_idx * kv_block_stride
        # block_size x head_size
        val_vals = tl.load(val_offs)
        sum_exp_dot = tl.dot(qk_exp.to(tl.float16), val_vals, sum_exp_dot)
        # tl.store(output + kv_group_offs[:, None] * head_size + head_offs[None, :], sum_exp_dot)

    # # tl.store(output + kv_group_offs[:, None] * head_size + head_offs[None, :], sum_exp_dot)
    sum_exp_sum = tl.sum(sum_exp, axis=1)
    # tl.store(output + kv_group_offs, sum_exp_sum)
    sum_exp_dot /= sum_exp_sum[:, None]
    # tl.store(output + kv_group_offs[:, None] * head_size + head_offs[None, :], sum_exp_dot)
    tl.store(
        output + batch_idx * query_stride + (kv_head_idx * num_kv_group + kv_group_offs[:, None]) * head_size +
        head_offs[None, :], sum_exp_dot, mask=kv_group_offs[:, None] < num_kv_group)


def vllm_paged_attention_triton(
    output,
    query,  # (batch_size, num_head, head_size)
    key_cache,  # (num_blocks, num_kv_heads, block_size, head_size),
    value_cache,  # (num_blocks, num_kv_heads, block_size, head_size),
    num_kv_heads,  # (batch_size)
    scale,  # (batch_size, )
    block_tables,
    context_lens,
    block_size,
    max_context_len,
):
    batch_size, num_heads, head_size = query.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    num_kv_group = num_heads // num_kv_heads
    grid = [batch_size, num_kv_heads]
    block_tables_stride = block_tables.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride = key_cache.stride(1)
    query_stride = query.stride(0)
    num_warps = 1
    F = vllm_paged_attention_fwd[grid](output, query, key_cache, value_cache, block_tables, context_lens, query_stride,
                                       block_tables_stride, kv_block_stride, kv_head_stride, scale, num_kv_group,
                                       head_size, block_size, num_warps=num_warps, num_stages=1)
    return F


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_vllm_paged_attention():
    batch_size = 256
    num_kv_heads = 2
    num_heads = 32
    block_size = 16
    head_size = 128
    num_blocks = 81920
    scale = float(1.0 / (head_size**0.5))
    query = torch.randn(batch_size, num_heads, head_size, dtype=torch.float16, device="cuda")
    output = torch.empty_like(query)
    key_block_shape = (num_kv_heads, block_size, head_size)
    key_cache = torch.randn(size=(num_blocks, *key_block_shape), dtype=torch.float16, device="cuda")
    value_block_shape = (num_kv_heads, block_size, head_size)
    value_cache = torch.randn(size=(num_blocks, *value_block_shape), dtype=torch.float16, device="cuda")
    context_len = 4096
    block_tables = []

    for _ in range(batch_size):
        block_table = [random.randint(0, num_blocks - 1) for _ in range((context_len + block_size - 1) // block_size)]
        block_tables.append(block_table)

    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
    b_seq_len = torch.full((batch_size, ), context_len, dtype=torch.int32, device="cuda")
    F = vllm_paged_attention_triton(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        b_seq_len,
        block_size,
        context_len,
    )
    assert check_mlir(F, [(128, 128), (0, 128)])
    return True, check_mlir(F, [(128, 128), (0, 128)])


@triton.jit
def ceil_div(x, y):
    return (x + y - 1) // y


@triton.jit
def _fwd_batch_mla_paged_attention_stage1(q_nope, q_pe, ckv_cache, kpe_cache, out, kv_indices, q_indptr, kv_indptr,
                                          partial_indptr, q_start_ptr, kv_start_ptr, kv_end_ptr, work_indptr,
                                          partial_o_ptr, partial_lse_ptr, page_size: tl.constexpr,
                                          num_heads: tl.constexpr, head_dim_ckv: tl.constexpr,
                                          head_dim_kpe: tl.constexpr, scale: tl.constexpr,
                                          CTA_TILE_Q: tl.constexpr = 64, TILE_KV: tl.constexpr = 16):
    bx = tl.program_id(0)
    by = tl.program_id(1)
    cta_tile_q_off = tl.arange(0, CTA_TILE_Q)
    tile_kv_off = tl.arange(0, TILE_KV)
    head_ckv_off = tl.arange(0, head_dim_ckv)
    head_kpe_off = tl.arange(0, head_dim_kpe)
    work_start = tl.load(work_indptr + by)
    work_end = tl.load(work_indptr + by + 1)
    mask = TILE_KV - 1
    min_val = -float("inf")
    for work_idx in range(work_start, work_end):
        q_idx = tl.load(q_indptr + work_idx)
        q_start = tl.load(q_start_ptr + work_idx)
        kv_start = tl.load(kv_start_ptr + work_idx)
        kv_end = tl.load(kv_end_ptr + work_idx)
        ceil_kv_end = ceil_div(kv_end, TILE_KV) * TILE_KV
        partial_idx = tl.load(partial_indptr + work_idx)
        kv_idx = tl.load(kv_indptr + work_idx)
        batch_q_nope = tl.load(q_nope + q_idx * num_heads * head_dim_ckv + (q_start + bx * CTA_TILE_Q) * head_dim_ckv +
                               cta_tile_q_off[:, None] * head_dim_ckv + head_ckv_off[None, :])
        batch_q_pe = tl.load(q_pe + q_idx * num_heads * head_dim_kpe + (q_start + bx * CTA_TILE_Q) * head_dim_kpe +
                             cta_tile_q_off[:, None] * head_dim_kpe + head_kpe_off[None, :])
        kv_load_cnt = (ceil_kv_end - kv_start) // TILE_KV
        qk_max = tl.full(
            (CTA_TILE_Q, ),
            min_val,
            dtype=tl.float32,
        )
        sum_exp = tl.zeros(
            (CTA_TILE_Q, TILE_KV),
            dtype=tl.float32,
        )
        sum_exp_dot = tl.zeros(
            (CTA_TILE_Q, head_dim_ckv),
            dtype=tl.float32,
        )
        log2e = 1.4426950408889634073599246810019
        for kv_load_idx in range(kv_load_cnt):
            if kv_load_idx == kv_load_cnt - 1:
                block_res = ((kv_end - 1) & mask) + 1
            else:
                block_res = TILE_KV
            kv_base_off = (kv_start + kv_load_idx * TILE_KV)
            tile_loc = kv_base_off // page_size
            kv_page_idx = kv_base_off % page_size
            page_idx = tl.load(kv_indices + kv_idx + tile_loc)
            # num_pages, pages_size, head_dim_ckv
            tile_ckv = tl.load(ckv_cache + page_idx * page_size * head_dim_ckv + kv_page_idx * head_dim_ckv +
                               tile_kv_off[None, :] * head_dim_ckv + head_ckv_off[:, None])
            # num_pages, pages_size, head_dim_kpe
            tile_kpe = tl.load(kpe_cache + page_idx * page_size * head_dim_kpe + kv_page_idx * head_dim_kpe +
                               tile_kv_off[None, :] * head_dim_kpe + head_kpe_off[:, None])
            qk = tl.dot(batch_q_nope, tile_ckv)
            qk += tl.dot(batch_q_pe, tile_kpe)
            qk *= scale
            tmp = tile_kv_off[None, :] < block_res
            qk = tl.where(tmp, qk, min_val)
            new_max = tl.max(qk, axis=1)
            new_max = tl.maximum(qk_max, new_max)
            qk_diff = qk_max - new_max
            scale_factor = tl.exp(qk_diff)
            qk_max = new_max
            sum_exp *= scale_factor[:, None]
            sum_exp_dot *= scale_factor[:, None]
            qk -= qk_max[:, None]
            qk_exp = tl.exp(qk)
            sum_exp += qk_exp
            qk_exp = qk_exp.to(tile_ckv.dtype)
            sum_exp_dot = tl.dot(qk_exp, tile_ckv.T, sum_exp_dot)
        sum_exp_sum = tl.sum(sum_exp, axis=1)
        sum_exp_dot /= sum_exp_sum[:, None]
        if partial_idx == -1:
            tl.store(
                out + q_idx * num_heads * head_dim_ckv + (q_start + bx * CTA_TILE_Q) * head_dim_ckv +
                cta_tile_q_off[:, None] * head_dim_ckv + head_ckv_off[None, :], sum_exp_dot)
        else:
            tl.store(
                partial_o_ptr + partial_idx * head_dim_ckv + bx * CTA_TILE_Q * head_dim_ckv +
                cta_tile_q_off[:, None] * head_dim_ckv + head_ckv_off[None, :], sum_exp_dot)
            tl.store(partial_lse_ptr + partial_idx + bx * CTA_TILE_Q + cta_tile_q_off,
                     qk_max * log2e + tl.log(sum_exp_sum) * log2e)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_batch_mla():
    capability = torch.cuda.get_device_capability()
    if capability[0] == 8:
        pytest.skip("QS do not support load result used by both TransOp and DotOp when dtype=float32 for now.")
    kv_len = [128, 1275, 1275, 1273]
    qo_len = [125, 1, 1, 1]
    num_heads = 128
    page_size = 256
    head_dim_ckv = 512
    head_dim_kpe = 64
    dtype = torch.float16
    batch_size = len(kv_len)
    q_nope = torch.randn(sum(qo_len), num_heads, head_dim_ckv, dtype=torch.float16, device="cuda").to(dtype) * 0.1
    q_pe = torch.randn(sum(qo_len), num_heads, head_dim_kpe, dtype=torch.float16, device="cuda").to(dtype) * 0.1
    page_nums = [math.ceil(x / page_size) for x in kv_len]
    total_pages = 128
    ckv = torch.randn(
        total_pages,
        page_size,
        head_dim_ckv,
        dtype=torch.float16,
        device="cuda",
    ).to(dtype) * 0.1
    kpe = torch.randn(
        total_pages,
        page_size,
        head_dim_kpe,
        dtype=torch.float16,
        device="cuda",
    ).to(dtype) * 0.1
    sm_scale = 0.1352337788608801
    q_indptr = torch.cumsum(torch.tensor([
        0,
    ] + qo_len, dtype=torch.int32, device="cuda"), dim=0).to(torch.int32)
    kv_indptr = torch.cumsum(torch.tensor([
        0,
    ] + page_nums, dtype=torch.int32, device="cuda"), dim=0).to(torch.int32)
    kv_indices = torch.randint(0, 1, (kv_indptr[-1].item(), ), dtype=torch.int32, device="cuda")
    # kv_lens = torch.tensor(kv_len, dtype=torch.int32, device="cuda")
    # bsz_tensor = torch.tensor([batch_size, ], dtype=torch.int32, device="cuda")
    out = torch.empty_like(q_nope)
    partial_indptr = torch.full((16384, ), -1, dtype=torch.int32, device="cuda")
    q_start_ptr = torch.zeros((16384, ), dtype=torch.int32, device="cuda")
    q_start_ptr[:128] = torch.tensor([
        15872, 13952, 13440, 12288, 11776, 9856, 9344, 8192, 7680, 5760, 5248, 4096, 3584, 1664, 1152, 0, 15232, 14592,
        13056, 12672, 11136, 10496, 8960, 8576, 7040, 6400, 4864, 4480, 2944, 2304, 768, 384, 15744, 14080, 13568,
        12160, 11648, 9984, 9472, 8064, 7552, 5888, 5376, 3968, 3456, 1792, 1280, 0, 14976, 14848, 12928, 12800, 10880,
        10752, 8832, 8704, 6784, 6656, 4736, 4608, 2688, 2560, 640, 512, 15104, 14720, 13184, 12544, 11008, 10624, 9088,
        8448, 6912, 6528, 4992, 4352, 2816, 2432, 896, 256, 15360, 14464, 13312, 12416, 11264, 10368, 9216, 8320, 7168,
        6272, 5120, 4224, 3072, 2176, 1024, 128, 15616, 14208, 13696, 12032, 11520, 10112, 9600, 7936, 7424, 6016, 5504,
        3840, 3328, 1920, 1408, 0, 15488, 14336, 13824, 11904, 11392, 10240, 9728, 7808, 7296, 6144, 5632, 3712, 3200,
        2048, 1536, 0
    ], dtype=torch.int32, device="cuda")
    kv_start_ptr = torch.zeros((16384, ), dtype=torch.int32, device="cuda")
    kv_end_ptr = torch.zeros((16384, ), dtype=torch.int32, device="cuda")
    kv_end_ptr[:128] = torch.tensor([
        252, 237, 233, 224, 220, 205, 201, 192, 188, 173, 169, 160, 156, 141, 137, 128, 247, 242, 230, 227, 215, 210,
        198, 195, 183, 178, 166, 163, 151, 146, 134, 131, 251, 238, 234, 223, 219, 206, 202, 191, 187, 174, 170, 159,
        155, 142, 138, 1275, 245, 244, 229, 228, 213, 212, 197, 196, 181, 180, 165, 164, 149, 148, 133, 132, 246, 243,
        231, 226, 214, 211, 199, 194, 182, 179, 167, 162, 150, 147, 135, 130, 248, 241, 232, 225, 216, 209, 200, 193,
        184, 177, 168, 161, 152, 145, 136, 129, 250, 239, 235, 222, 218, 207, 203, 190, 186, 175, 171, 158, 154, 143,
        139, 1275, 249, 240, 236, 221, 217, 208, 204, 189, 185, 176, 172, 157, 153, 144, 140, 1273
    ], dtype=torch.int32, device="cuda")
    work_indptr = torch.zeros((16384, ), dtype=torch.int32, device="cuda")
    work_indptr[:9] = torch.tensor([
        0,
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
    ], dtype=torch.int32, device="cuda")
    partial_o_ptr = torch.zeros((7077888, ), dtype=torch.float32, device="cuda")
    partial_lse_ptr = torch.zeros((27000832, ), dtype=torch.float32, device="cuda")

    F = _fwd_batch_mla_paged_attention_stage1[2, 8](q_nope, q_pe, ckv, kpe, out, kv_indices, q_indptr, kv_indptr,
                                                    partial_indptr, q_start_ptr, kv_start_ptr, kv_end_ptr, work_indptr,
                                                    partial_o_ptr, partial_lse_ptr, page_size, num_heads, head_dim_ckv,
                                                    head_dim_kpe, sm_scale, num_warps=8, num_stages=1)
    assert check_mlir(F, [(512, 512), (64, 64), (0, 512)])
    return True, check_mlir(F, [(512, 512), (64, 64), (0, 512)])


@triton.jit
def nested_kernel_d2(A, B, C, M_offs, N_offs, K_offs, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    A_vals = tl.load(A + M_offs[:, None] * K + K_offs[None, :])
    B_vals = tl.load(B + K_offs[:, None] * N + N_offs[None, :])
    C_vals = tl.dot(A_vals, B_vals)
    tl.store(C + M_offs[:, None] * N + N_offs[None, :], C_vals)


@triton.jit
def nested_kernel_d1(A, B, C, M_offs, N_offs, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    K_offs = tl.arange(0, K)
    nested_kernel_d2(A, B, C, M_offs, N_offs, K_offs, M, N, K)


@triton.jit
def nested_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    M_offs = tl.arange(0, M)
    N_offs = tl.arange(0, N)
    nested_kernel_d1(A, B, C, M_offs, N_offs, M, N, K)


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_nested_dot():
    M, N, K = 16, 16, 32
    dtype = torch.half
    A = torch.randn((M, K), dtype=dtype, device='cuda')
    B = torch.randn((K, N), dtype=dtype, device='cuda')
    C = torch.randn((M, N), dtype=dtype, device='cuda')
    F = nested_kernel[(1, )](A, B, C, M, N, K)
    D = A @ B
    assert check_mlir(F, [(32, 0)])
    return torch.allclose(C, D, atol=1e-3, rtol=1e-3), check_mlir(F, [(32, 0)])


@triton.jit
def mask_sme_kernel(q_ptr, k_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, mask: tl.constexpr,
                    other: tl.constexpr, outTy: tl.constexpr):

    rm = tl.arange(0, M)  # (M,)
    rk = tl.arange(0, K)  # (K,)

    q_ptrs = q_ptr + rm[:, None] * K + rk[None, :]
    # With mask load
    q = tl.load(q_ptrs, mask=rk[None, :] < mask, other=other)

    cn = tl.arange(0, N)
    k_ptrs = k_ptr + rk[:, None] + cn[None, :] * K
    bk = tl.load(k_ptrs, mask=rk[:, None] < mask, other=other)

    acc = tl.dot(q, bk)  # (M, N)
    o_ptrs = o_ptr + rm[:, None] * N + cn[None, :]
    tl.store(o_ptrs, acc.to(outTy))


@pytest.mark.skip(reason="iluvatar: ir.parse_mlir_module failed in CI")
@print_result_decorator
def test_mask_sme():
    M, K, N = 64, 256, 64
    mask = K // 2
    other = 3.0
    #fp16 test case
    q = torch.randn((M, K), dtype=torch.float16, device="cuda")
    k = torch.randn((N, K), dtype=torch.float16, device="cuda")
    out = torch.empty((M, N), dtype=torch.float16, device="cuda")

    q_ref = q.clone()
    k_ref = k.clone()
    q_ref[:, mask:] = other
    k_ref[:, mask:] = other
    ref = q_ref @ k_ref.T

    F = mask_sme_kernel[(1, )](q, k, out, M, N, K, mask=mask, other=other, outTy=tl.float16)
    assert check_mlir(F, [(256, 256)])
    #int8 test case
    q_8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    k_8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    out_8 = torch.empty((M, N), dtype=torch.int32, device="cuda")
    q_ref8 = q_8.clone()
    k_ref8 = k_8.clone()
    q_ref8[:, mask:] = other
    k_ref8[:, mask:] = other
    ref_8 = q_ref8 @ k_ref8.T

    I = mask_sme_kernel[(1, )](q_8, k_8, out_8, M, N, K, mask=mask, other=other, outTy=tl.int32)
    assert check_mlir(I, [(256, 256)])
    return torch.allclose(ref, out), torch.allclose(ref_8, out_8)


if __name__ == "__main__":
    test_corex_sme()
    test_corex_stride()
    test_16x32_f16_dot()
    test_16x32_i8_dot()
    test_16x32_f16_trans_dot()
    test_load_dot()
    test_multi_use_dot()
    # this two case may coredump
    # test_16x32_i8_trans_dot()
    # test_loop_k_dot()
    test_loop_k_dot2()
    test_loop_k_dot3()
    test_loop_n_dot()
    # test_dot_stride()
    test_vllm_paged_attention()
    test_batch_mla()
    test_nested_dot()
    test_mask_sme()
