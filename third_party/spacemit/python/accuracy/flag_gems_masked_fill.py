import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    value = torch.tensor(1024, device=flag_gems.device)
    threshold = 0.3
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    if torch.is_tensor(value):
        ref_out = torch.masked_fill(inp, mask, value)
    else:
        ref_out = torch.masked_fill(inp, mask, value)
    with flag_gems.use_gems():
        res_out = torch.masked_fill(inp, mask, value)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
