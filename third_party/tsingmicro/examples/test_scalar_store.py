import torch

import triton
import triton.language as tl


@triton.jit
def reduce_kernel_2d(
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    base_ptr = output_ptr + pid0
    for i in range(0, BLOCK_SIZE):
        output = i
        tl.store(base_ptr, output)
        base_ptr += 1


def test(device):
    BLOCK_SIZE = 8
    x = torch.full([BLOCK_SIZE], -1, device=device, dtype=torch.float32)
    output = torch.full((BLOCK_SIZE,), -99, device=x.device, dtype=x.dtype)
    grid = lambda meta: (1,)

    reduce_kernel_2d[grid](output, BLOCK_SIZE=BLOCK_SIZE)
    ans = torch.arange(BLOCK_SIZE, device=device, dtype=torch.float32)
    torch.testing.assert_close(output, ans, rtol=0.001, atol=1e-5)
