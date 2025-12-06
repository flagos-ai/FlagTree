import torch
import torch_musa
import triton
import triton.language as tl
import numpy as np


def test_1d_tmeld_normalst_off():
    #if not torch.musa.is_available() or not torch.musa.get_device_capability()[0] == 3:
    #    return
    device = "musa"
    TME_DESC_SIZE = 64
    TENSOR_SIZE = 4096 * 4096
    block_size = 128

    @triton.jit
    def kernel(Z, desc, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        off_desc = pid * BLOCK_SIZE
        off = off_desc + tl.arange(0, BLOCK_SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [BLOCK_SIZE], Z.dtype.element_ty)
        tl.store(Z + off, x)

    x = torch.randn(TENSOR_SIZE, dtype=torch.float32, device=device)
    desc = np.empty(TME_DESC_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(x.data_ptr(), TENSOR_SIZE, block_size, x.element_size(),
                                                              desc)
    desc = torch.tensor(desc, device=device)
    z_tri = torch.zeros(TENSOR_SIZE, device=device)
    kernel[(triton.cdiv(TENSOR_SIZE, block_size), )](z_tri, desc, BLOCK_SIZE=block_size, num_warps=4)
    if not torch.equal(x, z_tri):
        print("not correct!")
        print(f"golden: {x}, \nresult: {z_tri}")
    else:
        print("correct!")

    assert torch.equal(x, z_tri), "The output tensor does not match the input tensor."


if __name__ == "__main__":
    test_1d_tmeld_normalst_off()
