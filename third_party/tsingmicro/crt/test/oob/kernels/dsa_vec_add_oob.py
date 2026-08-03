"""DSA (ping-pong / pipeline / unroll) vector add kernel with OOB injection.

Uses a manual unrolled loop with n_iter sub-blocks, mimicking DSA pipeline.
"""
import triton
import triton.language as tl


@triton.jit
def add_kernel_dsa_oob(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    inject_n_elements_scale: tl.constexpr,  # 1=normal, >1=inflate n_elements
    inject_mask_mode: tl.constexpr,          # 0=normal, 1=no mask
    inject_n_iter_scale: tl.constexpr,       # 1=normal, >1=more iters beyond BLOCK_SIZE
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # --- OOB injection: more iterations via n_iter_scale ---
    n_iter: tl.constexpr = 8 * inject_n_iter_scale  # >1 inflates iterations beyond BLOCK_SIZE
    sub_sz: tl.constexpr = BLOCK_SIZE // 8

    # --- OOB injection: inflate n_elements ---
    eff_n_elements = n_elements * inject_n_elements_scale

    for i in range(n_iter):
        offsets = block_start + i * sub_sz + tl.arange(0, sub_sz)

        # --- OOB injection: mask control ---
        if inject_mask_mode == 0:
            mask = offsets < eff_n_elements
        elif inject_mask_mode == 1:
            mask = None

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        tl.store(output_ptr + offsets, out, mask=mask)
