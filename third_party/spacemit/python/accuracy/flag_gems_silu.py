import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.nn.functional.silu(res_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
