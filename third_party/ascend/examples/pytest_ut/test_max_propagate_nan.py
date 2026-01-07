import torch
import torch_npu
import triton
import triton.language as tl
import pytest

def test_max_propagate_nan():

    @triton.jit
    def func(in_ptr, out_ptr):
        a = tl.load(in_ptr + tl.arange(0, 8)[:,None] * 8 + tl.arange(0, 8)[None,:])
        a = tl.max(a, 0, propagate_nan = True)
        tl.store(out_ptr + tl.arange(0, 8), a)

    a = torch.randn((8, 8), device = "npu")
    std = a.max(0)[0]
    ans = torch.zeros((8,), dtype=torch.float32, device="npu")
    func[1,1,1](a, ans)
    torch.testing.assert_close(std, ans)