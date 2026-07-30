import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    x1 = torch.tensor([[11, 21, 31], [21, 31, 41]], dtype=torch.float32, device=flag_gems.device)
    x2 = torch.tensor([[12, 22, 32], [22, 32, 42]], dtype=torch.float32, device=flag_gems.device)

    inputs = [x1, x2]

    ref_out = torch.cat(inputs, 0)
    with flag_gems.use_gems():
        res_out = torch.cat(inputs, 0)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
