"""Dropout kernel with OOB error injection parameters.

"""
import triton
import triton.language as tl


@triton.jit
def dropout_kernel_oob(x_ptr, x_keep_ptr, output_ptr, n_elements, p,
                        inject_mask_mode: tl.constexpr,        # 0=normal, 1=no mask on load/store
                        inject_n_elements_scale: tl.constexpr,  # 1=normal, 2=double n_elements
                        BLOCK_SIZE: tl.constexpr,
                        seed: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    effective_n = n_elements * inject_n_elements_scale

    if inject_mask_mode == 0:
        mask = offsets < effective_n
    elif inject_mask_mode == 1:
        mask = None

    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def seeded_dropout_kernel_oob(x_ptr, output_ptr, n_elements, p,
                               inject_mask_mode: tl.constexpr,
                               inject_n_elements_scale: tl.constexpr,
                               BLOCK_SIZE: tl.constexpr,
                               seed: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    effective_n = n_elements * inject_n_elements_scale

    if inject_mask_mode == 0:
        mask = offsets < effective_n
    elif inject_mask_mode == 1:
        mask = None

    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
