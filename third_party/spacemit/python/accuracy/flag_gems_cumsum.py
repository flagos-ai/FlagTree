import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 32)
    dim = 1
    dtype = torch.float32
    INT_DTYPES = [torch.int16, torch.int32, torch.int64]
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.cumsum(inp, dim=dim)

    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)
    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
