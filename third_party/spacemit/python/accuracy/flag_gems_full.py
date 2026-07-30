import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    fill_value = 3.1415926
    ref_out = torch.full(shape, fill_value, device="cpu")
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, device=flag_gems.device)
    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

    # with dtype
    ref_out = torch.full(shape, fill_value, dtype=dtype, device="cpu")
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, dtype=dtype, device=flag_gems.device)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
