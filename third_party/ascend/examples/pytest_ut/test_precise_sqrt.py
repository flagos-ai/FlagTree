import torch
import torch_npu
import triton
import triton.language as tl

torch.set_printoptions(precision=10)
@triton.jit
def sqrtrn_kernel(x_ptr,  y_ptr,  output_ptr,  n_elements, BLOCK_SIZE: tl.constexpr ):
    id = tl.program_id(axis=0)
    start = id * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + tl.sqrt_rn(y)
    tl.store(output_ptr + offsets, output, mask=mask)

def sqrtrn(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(y)
    grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']),)
    sqrtrn_kernel[grid](x, y, output, output.numel(), BLOCK_SIZE=512)
    return output

def test_sqrtrn_fp32():
    size = 10240
    x = torch.abs(torch.randn(size, device='npu', dtype=torch.float32))
    y = torch.abs(torch.randn(size, device='npu', dtype=torch.float32))
    ref = x + torch.sqrt(y)
    cal = sqrtrn(x, y)
    torch.testing.assert_close(cal, ref, rtol=1e-06, atol=1e-06, equal_nan=True)
