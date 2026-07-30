import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    dim = -1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = 2

    ref_out = torch.repeat_interleave(inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)

    print("PASS")
