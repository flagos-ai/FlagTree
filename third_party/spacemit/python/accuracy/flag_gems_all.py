import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

shape = (2, 32)
dtype = torch.float32
kind = "normal"

if kind == "allTrue":
    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
else:
    inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)

ref_out = torch.all(inp)

with flag_gems.use_gems():
    res_out = torch.all(inp)

torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

print("PASS")

# dim = 1
# keepdim = True
# if kind == "allTrue":
#     inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
# else:
#     inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)

# ref_out = torch.all(inp, dim=dim, keepdim=keepdim)
# with flag_gems.use_gems():
#     res_out = torch.all(inp, dim=dim, keepdim=keepdim)

# torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
# print("PASS")
