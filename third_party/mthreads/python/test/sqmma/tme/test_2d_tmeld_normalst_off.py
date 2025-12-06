import torch
import triton
import triton.language as tl
import numpy as np


def test_2d_tmeld_normalst_off():
    #if not torch.musa.is_available() or not torch.musa.get_device_capability()[0] == 3:
    #    return
    device = "musa"
    TME_DESC_SIZE = 64
    M, N = 512, 32
    block_m, block_n = 128, 32

    @triton.jit
    def kernel(src_desc, dst_ptr, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        off_r = pid_m * BLOCK_M
        off_c = pid_n * BLOCK_N
        src = tl._experimental_descriptor_load(src_desc, [off_r, off_c], [BLOCK_M, BLOCK_N], tl.float16)
        off_rv = off_r + tl.arange(0, BLOCK_M)
        off_cv = off_c + tl.arange(0, BLOCK_N)
        dst_ptrs = dst_ptr + stride_m * off_rv[:, None] + stride_n * off_cv[None, :]
        tl.store(dst_ptrs, src)

    @triton.jit
    def kernel_golden(src, dst, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        off_r = pid_m * BLOCK_M
        off_c = pid_n * BLOCK_N
        off_rv = off_r + tl.arange(0, BLOCK_M)
        off_cv = off_c + tl.arange(0, BLOCK_N)
        src_ptrs = src + stride_m * off_rv[:, None] + stride_n * off_cv[None, :]
        src_data = tl.load(src_ptrs)
        dst_ptrs = dst + stride_m * off_rv[:, None] + stride_n * off_cv[None, :]
        tl.store(dst_ptrs, src_data)

    # v_data = torch.arange(M * N, dtype=torch.float16)
    # m_data = v_data.reshape(M, N)
    # # print(f"m_data:\n{m_data}")
    src = torch.randn((M, N), dtype=torch.float16, device=device)
    # src = m_data.to(device)
    dst = torch.zeros((M, N), dtype=torch.float16, device=device)
    src_desc = np.empty(TME_DESC_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(src.data_ptr(), M, N, block_m, block_n,
                                                              src.element_size(), src_desc)
    src_desc = torch.tensor(src_desc, device=device)
    kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](src_desc, dst, stride_m=src.stride(0),
                                                               stride_n=src.stride(1), BLOCK_M=block_m, BLOCK_N=block_n,
                                                               num_warps=4)
    # kernel_golden[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](src,
    #                                                            dst,
    #                                                            stride_m=src.stride(0),
    #                                                            stride_n=src.stride(1),
    #                                                            BLOCK_M=block_m,
    #                                                            BLOCK_N=block_n,
    #                                                            num_warps=4)
    if not torch.equal(src, dst):
        src_cpu = src.cpu()
        dst_cpu = dst.cpu()
        row_mask = ~torch.all(torch.eq(src_cpu, dst_cpu), dim=1)

        unequal_src = src_cpu[row_mask]
        unequal_dst = dst_cpu[row_mask]
        print(f"not correct!")
        for row1, row2 in zip(unequal_src, unequal_dst):
            print("src:", row1)
            print("dst:", row2)
            print("-------------------")
    else:
        # print(f"correct!\ngolden:\n{src}\nresult:\n{dst}")
        print(f"correct!")
    assert torch.equal(src, dst), "The output tensor does not match the input tensor."


if __name__ == "__main__":
    test_2d_tmeld_normalst_off()
