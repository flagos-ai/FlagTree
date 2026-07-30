import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 32)
    dtype = torch.float32
    INT_DTYPES = [torch.int16, torch.int32]
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=flag_gems.device).to(dtype)
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.nonzero(inp)

    res_out = flag_gems.nonzero(inp)
    print("ref_out", ref_out)
    print("res_out", res_out)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
