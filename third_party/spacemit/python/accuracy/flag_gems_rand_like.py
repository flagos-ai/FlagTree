import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (20, 320, 15)
    dtype = torch.float32
    x = torch.randn(size=shape, dtype=dtype, device="cpu")
    with flag_gems.use_gems():
        res_out = torch.randn_like(x)
    mean = torch.mean(res_out.to("cpu"))
    std = torch.std(res_out.to("cpu"))
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01
    print("PASS")
