import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    M, N = 1, 32
    dtype = torch.float32
    inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device=flag_gems.device, requires_grad=True)

    ref_out = torch.outer(inp1, inp2)
    res_out = flag_gems.outer(inp1, inp2)
    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
