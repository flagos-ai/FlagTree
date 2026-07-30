import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    maxi = 0.001
    mini = 0.001
    isnone = None
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None

    ref_out = torch.clamp(inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
