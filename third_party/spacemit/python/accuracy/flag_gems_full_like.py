import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    fill_value = 3.1415926
    x = torch.empty(size=shape, dtype=dtype, device="cpu")
    with flag_gems.use_gems():
        res_out = torch.full_like(x, fill_value)
    torch.testing.assert_close(res_out, torch.full_like(x, fill_value), atol=1e-2, rtol=0)

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.full_like(x, fill_value, dtype=dtype)

    torch.testing.assert_close(torch.full_like(x, fill_value, dtype=dtype), res_out, atol=1e-2, rtol=0)
    print("PASS")
