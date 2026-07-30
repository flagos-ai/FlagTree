import torch
import triton
import pytest

import triton.language as tl


@triton.jit
def mask_loop(
    y_ptr,
    x_ptr,
    scale_ptr,
    size: torch.int64,
    BLOCK_SIZE: tl.constexpr,
):
    bidx = tl.program_id(0)
    tidx = tl.arange(0, BLOCK_SIZE)

    grid_stride = tl.num_programs(0) * BLOCK_SIZE
    iterations = tl.cdiv(size, 4)

    idx = bidx * BLOCK_SIZE + tidx
    for it in range(iterations):
        mask = idx < size
        x = tl.load(x_ptr + idx, mask=mask).to(tl.float32)
        tl.store(y_ptr + idx, x, mask=mask)
        idx += grid_stride


@pytest.mark.parametrize(
    "b",
    [
        1,
        2,
        3,
        8,
        2048,
        4096,
    ],
)
@pytest.mark.parametrize(
    "h",
    [
        16,
        128,
        1024,
        5120,
        7680,
        8192,
    ],
)
def test_mask_loop(b, h, device):
    x = torch.randn((b, h), dtype=torch.float32, device=device)
    y = torch.empty_like(x, dtype=torch.float32, device=device)
    scale_ones = torch.ones(1, dtype=torch.float32, device=device)

    BLOCK_SIZE = 2

    grid = (2, )

    mask_loop[grid](
        y,
        x,
        scale_ones,
        x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    torch.testing.assert_close(x, y)
