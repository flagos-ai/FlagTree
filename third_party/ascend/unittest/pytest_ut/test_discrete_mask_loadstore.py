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

# =============================================================================
# Discrete mask access conversion test suite
#
# Test matrix (mask type x operation type):
#
#  | mask type                       | load only | store only | load + store |
#  |---------------------------------|-----------|------------|--------------|
#  | single discrete mask            | (A)       | (B)        | -            |
#  | single continuous mask          | (C)       | (D)        | -            |
#  | continuous & discrete 2-way     | (E)       | (F)        | (G)          |
#  | continuous & discrete 4-way     | -         | -          | (H)          |
#  | broadcast(cont & disc) 2-D AND  | (I)       | -          | (J)          |
#
# =============================================================================

import torch
import triton
import triton.language as tl
import torch_npu
import pytest


# =============================================================================
# (A) Single discrete mask -- load only
# =============================================================================
@triton.jit
def single_disc_mask_load_kernel(
    in_ptr,
    out_ptr,
    BLOCK_N: tl.constexpr,
):
    col_offs = tl.arange(0, BLOCK_N)
    disc_mask = (col_offs * 2) < BLOCK_N
    ptr_in = in_ptr + col_offs
    ptr_out = out_ptr + col_offs
    data = tl.load(ptr_in, mask=disc_mask, other=0.0)
    tl.store(ptr_out, data)


@pytest.mark.parametrize("BLOCK_N", [8])
def test_single_discrete_mask_load(BLOCK_N):
    in_tensor = torch.arange(BLOCK_N, dtype=torch.float16, device='npu')
    out_tensor = torch.empty(BLOCK_N, dtype=torch.float16, device='npu')

    single_disc_mask_load_kernel[(1, )](in_tensor, out_tensor, BLOCK_N=BLOCK_N)

    half = BLOCK_N // 2
    expected = torch.zeros(BLOCK_N, dtype=torch.float16, device='npu')
    expected[:half] = in_tensor[:half]
    assert torch.allclose(out_tensor, expected), \
        f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}"


# =============================================================================
# (B) Single discrete mask -- store only
# =============================================================================
@triton.jit
def single_disc_mask_store_kernel(
    in_ptr,
    out_ptr,
    BLOCK_N: tl.constexpr,
):
    col_offs = tl.arange(0, BLOCK_N)
    disc_mask = (col_offs * 2) < BLOCK_N
    ptr_in = in_ptr + col_offs
    ptr_out = out_ptr + col_offs
    data = tl.load(ptr_in)
    tl.store(ptr_out, data, mask=disc_mask)


@pytest.mark.parametrize("BLOCK_N", [8])
def test_single_discrete_mask_store(BLOCK_N):
    in_tensor = torch.arange(BLOCK_N, dtype=torch.float16, device='npu')
    out_tensor = torch.full((BLOCK_N, ), -1.0, dtype=torch.float16, device='npu')

    single_disc_mask_store_kernel[(1, )](in_tensor, out_tensor, BLOCK_N=BLOCK_N)

    half = BLOCK_N // 2
    expected = torch.full((BLOCK_N, ), -1.0, dtype=torch.float16, device='npu')
    expected[:half] = in_tensor[:half]
    assert torch.allclose(out_tensor, expected), \
        f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}"


# =============================================================================
# (C) Single continuous mask -- load only
# =============================================================================
@triton.jit
def single_cont_mask_load_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    cont_mask = row_offs < M  # Continuous mask
    ptr_in = in_ptr + row_offs
    ptr_out = out_ptr + row_offs
    data = tl.load(ptr_in, mask=cont_mask, other=0.0)
    tl.store(ptr_out, data, mask=cont_mask)


@pytest.mark.parametrize("M,BLOCK_M", [(6, 8)])
def test_single_continuous_mask_load(M, BLOCK_M):
    in_tensor = torch.arange(BLOCK_M, dtype=torch.float16, device='npu')
    out_tensor = torch.full((BLOCK_M, ), -1.0, dtype=torch.float16, device='npu')

    single_cont_mask_load_kernel[(1, )](in_tensor, out_tensor, M=M, BLOCK_M=BLOCK_M)

    expected = torch.full((BLOCK_M, ), -1.0, dtype=torch.float16, device='npu')
    expected[:M] = in_tensor[:M]
    assert torch.allclose(out_tensor, expected), \
        f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}"


# =============================================================================
# (D) Single continuous mask -- store only
# =============================================================================
@triton.jit
def single_cont_mask_store_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    cont_mask = row_offs < M
    ptr_in = in_ptr + row_offs
    ptr_out = out_ptr + row_offs
    data = tl.load(ptr_in)
    tl.store(ptr_out, data, mask=cont_mask)


@pytest.mark.parametrize("M,BLOCK_M", [(6, 8)])
def test_single_continuous_mask_store(M, BLOCK_M):
    in_tensor = torch.arange(BLOCK_M, dtype=torch.float16, device='npu')
    out_tensor = torch.full((BLOCK_M, ), -1.0, dtype=torch.float16, device='npu')
    single_cont_mask_store_kernel[(1, )](in_tensor, out_tensor, M=M, BLOCK_M=BLOCK_M)
    expected = torch.full((BLOCK_M, ), -1.0, dtype=torch.float16, device='npu')
    expected[:M] = in_tensor[:M]
    assert torch.allclose(out_tensor, expected), \
        f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}"


# =============================================================================
# (E) Continuous & discrete 2-way AND -- load only
# =============================================================================
@triton.jit
def cont_disc_combined_mask_load_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    # Continuous mask
    row_boundary = row_offs < M
    # Discrete mask
    col_stride = (col_offs * 2) < BLOCK_N
    mask = row_boundary[:, None] & col_stride[None, :]
    ptr_in = in_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    ptr_out = out_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    data = tl.load(ptr_in, mask=mask, other=0.0)
    tl.store(ptr_out, data)


@pytest.mark.parametrize("M,BLOCK_M,BLOCK_N", [(6, 8, 8)])
def test_cont_disc_combined_mask_load(M, BLOCK_M, BLOCK_N):
    in_tensor = torch.ones((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    out_tensor = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')

    cont_disc_combined_mask_load_kernel[(1, )](in_tensor, out_tensor, M=M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    half_n = BLOCK_N // 2
    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    expected[:M, :half_n] = 1.0
    assert torch.allclose(out_tensor, expected), \
        f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}"


# =============================================================================
# (F) Continuous & discrete 2-way AND -- store only
# =============================================================================
@triton.jit
def cont_disc_combined_mask_store_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    row_boundary = row_offs < M  # continuous -> contLeaf
    col_stride = (col_offs * 2) < BLOCK_N  # discrete   -> discLeaf
    mask = row_boundary[:, None] & col_stride[None, :]
    ptr_in = in_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    ptr_out = out_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    data = tl.load(ptr_in)
    tl.store(ptr_out, data, mask=mask)


@pytest.mark.parametrize("M,BLOCK_M,BLOCK_N", [(6, 8, 8)])
def test_cont_disc_combined_mask_store(M, BLOCK_M, BLOCK_N):
    in_tensor = torch.ones((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    out_tensor = torch.full((BLOCK_M, BLOCK_N), -1.0, dtype=torch.float16, device='npu')
    cont_disc_combined_mask_store_kernel[(1, )](in_tensor, out_tensor, M=M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    half_n = BLOCK_N // 2
    expected = torch.full((BLOCK_M, BLOCK_N), -1.0, dtype=torch.float16, device='npu')
    expected[:M, :half_n] = 1.0
    assert torch.allclose(out_tensor, expected), \
        f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}"


# =============================================================================
# (G) Continuous & discrete 2-way AND -- load + store (complex interleave, original)
# =============================================================================
@triton.jit
def interleave_cont_disc_mask_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offs = tl.arange(0, N)
    even_col_offs = tl.arange(0, N // 2) * 2
    even_col_mask = even_col_offs < N  # discrete: cmpi(muli(range,2), N)
    row_offs = tl.arange(0, M)
    row_mask = row_offs < M  # continuous: cmpi(range_M, M)
    in_even_ptr = in_ptr + row_offs[:, None] * N + even_col_offs[None, :]
    in_odd_ptr = in_ptr + row_offs[:, None] * N + even_col_offs[None, :] + 1
    even_data = tl.load(in_even_ptr, mask=row_mask[:, None] & even_col_mask[None, :], other=0.0)
    odd_data = tl.load(in_odd_ptr)
    rotated_data = tl.interleave(-odd_data, even_data)
    out_ptr = out_ptr + row_offs[:, None] * N + col_offs[None, :]
    tl.store(out_ptr, rotated_data)


@pytest.mark.skip(reason="not supported after the NPUIR is updated in April, and will be fixed later")
@pytest.mark.parametrize("M", [4])
@pytest.mark.parametrize("N", [8])
def test_discrete_mask_load_store(M, N):
    """Regression test: mask=row_mask & even_col_mask (continuous & discrete 2-way)"""
    input_tensor = torch.arange(M * N, dtype=torch.float16, device='npu').reshape(M, N)
    output_tensor = torch.empty_like(input_tensor)
    interleave_cont_disc_mask_kernel[(1, )](input_tensor, output_tensor, M=M, N=N)
    even_cols = input_tensor[:, 0::2]
    odd_cols = input_tensor[:, 1::2]
    ref_output = torch.empty_like(input_tensor)
    ref_output[:, 0::2] = -odd_cols
    ref_output[:, 1::2] = even_cols
    assert torch.allclose(output_tensor.float(), ref_output.float())


# =============================================================================
# (H) Continuous & discrete 4-way AND -- load + store
# =============================================================================
@triton.jit
def multi_cont_disc_mask_load_store_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    row_boundary = row_offs < M  # continuous mask
    col_boundary = col_offs < N  # continuous mask
    row_stride = (row_offs * 2) < BLOCK_M  # discrete mask
    col_stride = (col_offs * 2) < BLOCK_N  # discrete mask

    mask = (row_boundary[:, None] & col_boundary[None, :] & row_stride[:, None] & col_stride[None, :])

    ptr_in = in_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    ptr_out = out_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]

    data = tl.load(ptr_in, mask=mask, other=0.0)
    result = data + 1.0
    tl.store(ptr_out, result, mask=mask)


@pytest.mark.parametrize("M,N,BLOCK_M,BLOCK_N", [
    (6, 6, 8, 8),
])
def test_multi_cont_disc_mask_load_store(M, N, BLOCK_M, BLOCK_N):
    in_tensor = torch.ones((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    out_tensor = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')

    multi_cont_disc_mask_load_store_kernel[(1, )](in_tensor, out_tensor, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    half_m = BLOCK_M // 2  # = 4
    half_n = BLOCK_N // 2  # = 4
    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    expected[:half_m, :half_n] = 2.0

    assert torch.allclose(out_tensor, expected), (f"BLOCK=({BLOCK_M},{BLOCK_N}), valid=({M},{N})\n"
                                                  f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}")


# =============================================================================
# (I) broadcast(continuous & discrete) 2-D AND -- load only
# =============================================================================
@triton.jit
def broadcast_cont_disc_2d_load_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    row_boundary = row_offs < M
    row_disc = (row_offs * 2) < BLOCK_M
    mask = row_boundary[:, None] & row_disc[:, None] & (col_offs < BLOCK_N)[None, :]

    ptr_in = in_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    ptr_out = out_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]

    data = tl.load(ptr_in, mask=mask, other=0.0)
    tl.store(ptr_out, data)


@pytest.mark.parametrize("M,BLOCK_M,BLOCK_N", [(3, 4, 8)])
def test_broadcast_cont_disc_2d_load(M, BLOCK_M, BLOCK_N):
    in_tensor = torch.ones((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    out_tensor = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')

    broadcast_cont_disc_2d_load_kernel[(1, )](in_tensor, out_tensor, M=M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    disc_true_rows = BLOCK_M // 2
    both_true_rows = min(M, disc_true_rows)

    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    expected[:both_true_rows, :] = 1.0

    assert torch.allclose(out_tensor, expected), (f"M={M}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}\n"
                                                  f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}")


# =============================================================================
# (J) broadcast(continuous & discrete) 2-D AND -- load + store
# =============================================================================
@triton.jit
def broadcast_cont_disc_2d_load_store_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    row_boundary = row_offs < M
    row_disc = (row_offs * 2) < BLOCK_M

    combined = row_boundary[:, None] & row_disc[:, None] & (col_offs < BLOCK_N)[None, :]

    ptr_in = in_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]
    ptr_out = out_ptr + row_offs[:, None] * BLOCK_N + col_offs[None, :]

    data = tl.load(ptr_in, mask=combined, other=0.0)
    tl.store(ptr_out, data, mask=combined)


@pytest.mark.parametrize("M,BLOCK_M,BLOCK_N", [(3, 4, 8)])
def test_broadcast_cont_disc_2d_load_store(M, BLOCK_M, BLOCK_N):
    in_tensor = torch.ones((BLOCK_M, BLOCK_N), dtype=torch.float16, device='npu')
    out_tensor = torch.full((BLOCK_M, BLOCK_N), -1.0, dtype=torch.float16, device='npu')

    broadcast_cont_disc_2d_load_store_kernel[(1, )](in_tensor, out_tensor, M=M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    disc_true_rows = BLOCK_M // 2
    both_true_rows = min(M, disc_true_rows)

    expected = torch.full((BLOCK_M, BLOCK_N), -1.0, dtype=torch.float16, device='npu')
    expected[:both_true_rows, :] = 1.0

    assert torch.allclose(out_tensor, expected), (f"M={M}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}\n"
                                                  f"Expected:\n{expected.cpu()}\nGot:\n{out_tensor.cpu()}")
