import torch

import triton
import triton.language as tl


@triton.jit
def splat(
    f32_val,
    f32_out,
    stride_row,
    stride_col,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    x = tl.full((2, BLOCK_SIZE_COL), f32_val, dtype=tl.float32)
    offs_row = 2 * pid0 + tl.arange(0, 2)
    offs_col = tl.arange(0, BLOCK_SIZE_COL)
    a_ptrs = f32_out + (offs_row[:, None] * stride_row + offs_col[None, :] * stride_col)
    tl.store(a_ptrs, x)


def test(device):
    n_rows = 256
    n_cols = 512
    fill_value = 123.456
    expected_result = torch.full((n_rows, n_cols), fill_value, dtype=torch.float32)
    output = torch.empty([n_rows, n_cols], device=device, dtype=expected_result.dtype)
    grid = lambda meta: (n_rows // 2,)

    splat[grid](
        fill_value,
        output,
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_ROW=n_rows,
        BLOCK_SIZE_COL=n_cols,
    )

    torch.testing.assert_close(output, expected_result, rtol=0.001, atol=1e-5)
