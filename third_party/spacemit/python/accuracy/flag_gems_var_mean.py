import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    dim = 1
    shape = (2, 32)
    keepdim = True
    dtype = torch.float32
    correction = 0
    if shape[0] == 1:  # TODO: res is inf, while ref is nan
        shape = (2, 2)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_var, ref_mean = torch.var_mean(inp, dim, correction=correction, keepdim=keepdim)
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(inp, dim, correction=correction, keepdim=keepdim)

    torch.testing.assert_close(res_mean, ref_mean, atol=1e-2, rtol=0)
    torch.testing.assert_close(res_var, ref_var, atol=1e-2, rtol=0)
    print("PASS")
