"""Extern functions kernel with OOB error injection parameters.

using tl.sin as the element-wise operation for OOB DMA testing.
"""
import triton
import triton.language as tl


@triton.jit
def asin_kernel_oob(x_ptr, y_ptr, n_elements,
                     inject_pid_shift: tl.constexpr,  # 0=normal, N=shift pid by +N
                     inject_mask_mode: tl.constexpr,   # 0=normal, 1=no mask
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) + inject_pid_shift
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    if inject_mask_mode == 0:
        mask = offsets < n_elements
    elif inject_mask_mode == 1:
        mask = None

    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.sin(x)
    tl.store(y_ptr + offsets, x, mask=mask)
