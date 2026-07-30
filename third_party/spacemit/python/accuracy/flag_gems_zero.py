import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32

    with flag_gems.use_gems():
        res_out = torch.zeros(shape, device=flag_gems.device)
    torch.testing.assert_close(torch.zeros(shape, device="cpu"), res_out, atol=1e-2, rtol=0)

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    torch.testing.assert_close(torch.zeros(shape, dtype=dtype, device="cpu"), res_out, atol=1e-2, rtol=0)

    print("PASS")
