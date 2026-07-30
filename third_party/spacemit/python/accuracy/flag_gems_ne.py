import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 32)
    dtype = torch.float32
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ne(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.ne(inp1, inp2)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
