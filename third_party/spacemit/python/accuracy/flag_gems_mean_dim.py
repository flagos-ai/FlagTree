import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (1, 2, 7, 7)
    dtype = torch.float32
    dim = [2, 3]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    keepdim = True

    ref_out = torch.mean(inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)
    print(res_out)
    print(ref_out)
    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
