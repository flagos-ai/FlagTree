import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    keepdim = True
    dim = 1
    shape = (2, 2, 3)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.argmax(inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)
    print(ref_out)
    print(res_out)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
