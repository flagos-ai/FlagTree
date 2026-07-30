import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 32)
    dtype = torch.float32
    FLOAT_DTYPES = [torch.float32]

    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.max(inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
