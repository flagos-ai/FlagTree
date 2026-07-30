import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (20, 320, 15)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device="cpu")
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01
    print("PASS")
