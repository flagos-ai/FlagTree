"""
Vector Addition (sunrise)
=========================

Basic Triton vector-add smoke test on the sunrise backend. Rewritten to pytest
form to match the other sunrise test cases (02-06).
"""

import torch
import triton
import triton.language as tl
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y, BLOCK_SIZE=1024):
    output = torch.empty_like(x)
    assert x.device == y.device == output.device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@pytest.mark.parametrize("size", [1024, 98432, 100000])
def test_vector_add(size):
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    out = add(x, y)

    torch.testing.assert_close(out.cpu(), (x + y).cpu(), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
