import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def pack_kernel(a_ptr, c_ptr, M, K, num_blocks_m, num_blocks_k, stride_im0, stride_im1, stride_om0, stride_om1,
                stride_om2, stride_om3, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, MICRO_M: tl.constexpr,
                MICRO_K: tl.constexpr):
    pid_m = tl.program_id(0)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_im0, stride_im1],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[num_blocks_m, num_blocks_k, MICRO_M, MICRO_K],
        strides=[stride_om0, stride_om1, stride_om2, stride_om3],
        offsets=[pid_m * BLOCK_SIZE_M // MICRO_M, 0, 0, 0],
        block_shape=[BLOCK_SIZE_M // MICRO_M, BLOCK_SIZE_K // MICRO_K, MICRO_M, MICRO_K],
        order=[3, 2, 1, 0],
    )

    tl.store(c_block_ptr, a, boundary_check=(0))


MICRO_M = 16
MICRO_K = 8


def triton_pack(a):
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    M, K = a.shape

    c = torch.empty((M // MICRO_M, K // MICRO_K, MICRO_M, MICRO_K), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]), )

    BLOCK_SIZE_K = triton.next_power_of_2(K)

    pack_kernel[grid](
        a,
        c,
        M,
        K,
        M // MICRO_M,
        K // MICRO_K,
        a.stride(0),
        a.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        c.stride(3),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=MICRO_M,
        MICRO_K=MICRO_K,
    )
    return c


def pack_ref(a):
    M, K = a.shape
    a_permute = torch.permute(a.view(M // MICRO_M, MICRO_M, K // MICRO_K, MICRO_K), (0, 2, 1, 3))
    return a_permute.contiguous()


if __name__ == "__main__":

    M, K = 256, 256
    A = torch.randn((M, K), dtype=torch.float16, device="cpu", requires_grad=False)

    C = triton_pack(A)

    C_ref = pack_ref(A)

    print("Max error:", torch.max(torch.abs(C - C_ref)))
    assert torch.allclose(C, C_ref, atol=0, rtol=0)
    print("Test passed!")
