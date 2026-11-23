import torch
import torch_npu
import triton
import triton.language as tl
import pytest


@triton.jit
def zj_fa_fwd_pattern(in_ptr0, in_ptr1, out_ptr, M, K, N, MBLOCK: tl.constexpr, NBLOCK: tl.constexpr,
                      KBLOCK: tl.constexpr):
    a_ptr = tl.make_block_ptr(base=in_ptr0, shape=(M, K),  # 8, 3
                              strides=(K, 1), offsets=(0, 0), block_shape=(MBLOCK, KBLOCK), order=(1, 0))

    b_ptr = tl.make_block_ptr(base=in_ptr1, shape=(K, N),  # 3, 8
                              strides=(1, K), offsets=(0, 0), block_shape=(KBLOCK, NBLOCK), order=(0, 1))

    c_ptr = tl.make_block_ptr(base=out_ptr, shape=(M, N), strides=(1, M), offsets=(0, 0), block_shape=(MBLOCK, NBLOCK),
                              order=(0, 1))

    a = tl.load(a_ptr, boundary_check=(0, ), padding_option="zero")
    b = tl.load(b_ptr, boundary_check=(0, ), padding_option="zero")
    c = tl.dot(a, b)
    tl.store(c_ptr, c, boundary_check=(0, 1))


def test_permute_boundary_check():
    M = 8
    K = 3
    N = 8
    MBLOCK = 8
    NBLOCK = 8
    KBLOCK = 4
    a = torch.randn((M, K), device="npu")  # 8, 3
    b = torch.randn((N, K), device="npu")  # 8, 3
    c = torch.empty((N, M), device="npu")
    zj_fa_fwd_pattern[(1, 1, 1)](a, b, c, M, K, N, MBLOCK, NBLOCK, KBLOCK)
    std = a @ b.T
    torch.testing.assert_close(std, c.T, atol=1e-2, rtol=1e-2)
