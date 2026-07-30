import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    dtype = torch.float32
    alpha = 0.001
    inp1 = torch.randn([100, 1], dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn([1, 100], dtype=dtype, device=flag_gems.device)

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
