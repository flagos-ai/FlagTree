import torch
import torch_npu
import triton
import triton.language as tl

# load with mask, store with scalar
@triton.jit
def sum_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # 0 / 1 / 2 * 4 + (0,1,2,3)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask).to(tl.float32)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)

def test_case():
    inp = torch.ones(16, device="npu", dtype=torch.float32)
    mid = torch.empty(4, device="npu", dtype=torch.float32)
    sum_kernel_1[(4, 1, 1)](inp, mid, 16, 4)
    ref = torch.tensor([4.0, 4.0, 4.0, 4.0], device="npu", dtype=torch.float32)
    assert torch.allclose(mid, ref,  rtol=1e-03, atol=1e-03, equal_nan=True)
