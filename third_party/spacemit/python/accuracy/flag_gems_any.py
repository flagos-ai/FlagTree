import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

shape = (2, 32)
dtype = torch.float32
kind = "normal"
if kind == "allFalse":
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
else:
    inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)

print("inp", inp)

ref_out = torch.any(inp)
with flag_gems.use_gems():
    res_out = torch.any(inp)

torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

print("PASS")
