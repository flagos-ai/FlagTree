import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def stacked_load_2d_kernel(x_ptr, y_ptr, M, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    m_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    m_mask = m_offsets < M
    c_offsets = m_offsets % C
    w = tl.load(x_ptr + c_offsets, mask=m_mask)
    tl.store(y_ptr + m_offsets, w, mask=m_mask)


@triton.jit
def sidebyside_load_2d_kernel(x_ptr, y_ptr, M, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    m_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m_offsets = m_offsets[None, :]
    m_mask = m_offsets < M
    c_offsets = m_offsets % C
    w = tl.load(x_ptr + c_offsets, mask=m_mask)
    tl.store(y_ptr + m_offsets, w, mask=m_mask)


@triton.jit
def sidebyside_load_1d_kernel(x_ptr, y_ptr, M, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    m_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m_mask = m_offsets < M
    c_offsets = m_offsets % C
    w = tl.load(x_ptr + c_offsets, mask=m_mask)
    tl.store(y_ptr + m_offsets, w, mask=m_mask)


configs = [
    8,  # mask < N
    16,  # N < mask = 2N < colsize
    32,  # N < mask = 3N < colsize
    64,  # N < mask = 4N = colsize
    128  # N < colsize < mask
]


@pytest.mark.parametrize("BLOCK_SIZE", configs)
def test(device, BLOCK_SIZE):
    C = 16
    B = 4
    M = B * C
    weight = torch.randn(size=(C, ), dtype=torch.float32, device=device, requires_grad=True)
    output = torch.full([B, C], -1, device=device, dtype=torch.float32)

    indices = torch.arange(M, device=device) % C
    torch_output = weight[indices].view(B, C)

    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE), )
    stacked_load_2d_kernel[grid](weight, output, M, C, BLOCK_SIZE)
    torch.testing.assert_close(output, torch_output, rtol=0.001, atol=1e-5)

    sidebyside_load_2d_kernel[grid](weight, output, M, C, BLOCK_SIZE)
    torch.testing.assert_close(output, torch_output, rtol=0.001, atol=1e-5)

    sidebyside_load_1d_kernel[grid](weight, output, M, C, BLOCK_SIZE)
    torch.testing.assert_close(output, torch_output, rtol=0.001, atol=1e-5)
