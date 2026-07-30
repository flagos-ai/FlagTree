import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    start = 0
    step = 1
    end = 128
    dtype = torch.float32
    device = "cpu"
    pin_memory = False
    ref_out = torch.arange(start, end, step, dtype=dtype, device=device, pin_memory=pin_memory)
    with flag_gems.use_gems():
        res_out = torch.arange(start, end, step, dtype=dtype, device=device, pin_memory=pin_memory)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
