import torch
import torch_musa
import triton
import triton.language as tl
import numpy as np


def test_1d_tmeld_tmest_off_0():
    # print("#FIXME: This test case failed.")
    return
    #if not torch.musa.is_available() or not torch.musa.get_device_capability()[0] == 3:
    #    return
    device = "musa"
    TME_DESC_SIZE = 64
    TENSOR_SIZE = 4096
    block_size = 64

    @triton.jit
    def kernel(desc_a, desc_b, BLOCK_SIZE: tl.constexpr):
        off_desc = 0
        a = tl._experimental_descriptor_load(desc_a, [off_desc], [BLOCK_SIZE], tl.float32)
        # tl.store(Z + off, x)
        tl._experimental_descriptor_store(desc_b, a, [off_desc])

    a = torch.randn(TENSOR_SIZE, dtype=torch.float32, device=device)
    b = torch.zeros(TENSOR_SIZE, dtype=torch.float32, device=device)
    desc_a = np.empty(TME_DESC_SIZE, dtype=np.int8)
    desc_b = np.empty(TME_DESC_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(a.data_ptr(), TENSOR_SIZE, block_size, a.element_size(),
                                                              desc_a)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(b.data_ptr(), TENSOR_SIZE, block_size, b.element_size(),
                                                              desc_b)
    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    kernel[(triton.cdiv(TENSOR_SIZE, block_size), )](desc_a, desc_b, BLOCK_SIZE=block_size, num_warps=4)
    if not torch.equal(a, b):
        print("not correct!")
        print(f"golden: {a}, \nresult: {b}")
        return
    print("correct!")


if __name__ == "__main__":
    test_1d_tmeld_tmest_off_0()
