import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    float = "none"
    shape = (20, 320, 15)
    dtype = torch.float32
    loc = (3.0 if float == "mean" else torch.full(size=shape, fill_value=3.0, dtype=dtype, device="cpu"))
    scale = (10.0 if float == "std" else torch.full(size=shape, fill_value=10.0, dtype=dtype, device="cpu"))
    with flag_gems.use_gems():
        res_out = torch.normal(loc, scale)
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1
    print("PASS")
