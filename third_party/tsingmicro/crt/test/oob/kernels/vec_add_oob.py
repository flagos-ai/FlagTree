"""Vector add kernel with OOB error injection parameters.

"""
import triton
import triton.language as tl


@triton.jit
def add_kernel_oob(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    inject_pid_shift: tl.constexpr,     # 0=normal, 1=shift pid by +1
    inject_block_scale: tl.constexpr,   # 1=normal, 2=double block_size
    inject_mask_mode: tl.constexpr,     # 0=normal, 1=no mask, 2=wrong cmp
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid = pid + inject_pid_shift
    block_start = pid * BLOCK_SIZE * inject_block_scale
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    if inject_mask_mode == 0:
        mask = offsets < n_elements
    elif inject_mask_mode == 1:
        mask = None
    elif inject_mask_mode == 2:
        mask = offsets < n_elements + 1

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
