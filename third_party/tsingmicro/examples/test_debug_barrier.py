import torch

import triton
import triton.language as tl


@triton.jit
def barrier(data):
    ptrs = data + tl.arange(0, 128)

    tl.debug_barrier()
    tl.store(ptrs, tl.load(ptrs) + 1.0)
    tl.debug_barrier()


def test_barrier():
    data = torch.zeros(128, dtype=torch.float32, device='cpu')

    # Launch the kernel
    grid = (1, )
    barrier[grid](data)


if __name__ == "__main__":
    test_barrier()
