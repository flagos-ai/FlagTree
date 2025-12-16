import torch
import torch_musa
import triton
import triton.language as tl
import numpy as np


def test_tme_1d_ld():
    device = "musa"
    SIZE = 64

    @triton.jit
    def kernel(Z, desc, SIZE: tl.constexpr):
        off_desc = 0
        off = tl.arange(0, SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [SIZE], Z.dtype.element_ty)
        tl.store(Z + off, x)

    x = torch.ones(SIZE, dtype=torch.float32, device=device)
    desc = np.empty(SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size(), desc)
    desc = torch.tensor(desc, device=device)
    z_tri = torch.zeros(SIZE, device=device)
    kernel[(1, )](z_tri, desc, SIZE=SIZE, num_warps=4)
    assert torch.equal(x, z_tri), "TME 1D load test failed"


def test_tme_2d_ld():
    device = "musa"
    TME_DESC_SIZE = 64
    M, N = 128, 128
    block_m, block_n = 128, 128

    @triton.jit
    def kernel(desc_a, desc_b, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_desc = 0
        a = tl._experimental_descriptor_load(desc_a, [off_desc, off_desc], [BLOCK_M, BLOCK_N], tl.float16)
        tl._experimental_descriptor_store(desc_b, a, [off_desc, off_desc])

    a = torch.ones((M, N), dtype=torch.float16, device=device)
    b = torch.zeros((M, N), dtype=torch.float16, device=device)
    # print("a")

    desc_a = np.empty(TME_DESC_SIZE, dtype=np.int8)
    desc_b = np.empty(TME_DESC_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), M, N, M, N, a.element_size(), desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), M, N, M, N, b.element_size(), desc_b)
    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    kernel[(1, )](desc_a, desc_b, BLOCK_M=M, BLOCK_N=N, num_warps=4)
    assert torch.equal(a, b), "TME 2D load test failed"


if __name__ == "__main__":
    test_tme_1d_ld()
    test_tme_2d_ld()
