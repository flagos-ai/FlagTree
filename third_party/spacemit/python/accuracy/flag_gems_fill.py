import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    value = 0
    shape = (2, 19, 7)
    dtype = torch.float32
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)

    ref_out = torch.fill(x, value)
    with flag_gems.use_gems():
        res_out = torch.fill(x, value)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
