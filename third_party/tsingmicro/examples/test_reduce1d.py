import torch

import triton
from triton.backends.compiler import GPUTarget
import triton.language as tl
import benchmark
benchmark.select_cpu_backend()

@triton.jit
def reduce_kernel_1d(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(x, axis=0)
    tl.store(output_ptr, acc)


def test_1d_reduce_sum(device):
    BLOCK_SIZE = 32768
    x = torch.ones([BLOCK_SIZE], device=device, dtype=torch.float32)
    output = torch.empty([1], device=device, dtype=x.dtype)
    grid = lambda meta: (1,)

    reduce_kernel_1d[grid](x, output, BLOCK_SIZE, BLOCK_SIZE=BLOCK_SIZE)
    # CPU reference
    ref = x.sum().unsqueeze(0)

    print(
        f"The maximum difference between ref and triton is "
        f"{torch.max(torch.abs(ref - output))}"
    )
    torch.testing.assert_close(output, ref, rtol=0.001, atol=1e-5)

if __name__ == "__main__":
    device = "cpu"
    test_1d_reduce_sum(device)
