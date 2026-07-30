import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (20, 320, 15)
    dtype = torch.float32
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.rand(shape, dtype=dtype, device="cpu")
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()
    print("PASS")
