import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (16, 16)
    dim = 0
    dtype = torch.float32
    FLOAT_DTYPES = [torch.float16, torch.float32]
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(flag_gems.device) for s in shape]
    ref_out = torch.stack(inp, dim)

    with flag_gems.use_gems():
        res_out = torch.stack(inp, dim)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
