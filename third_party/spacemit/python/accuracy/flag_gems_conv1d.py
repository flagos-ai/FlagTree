import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

shape = (1, 2, 5)
kernel = (3, 2, 3)
groups = 1
stride = 1
padding = 0
dtype = torch.float32
bias = False
dilation = 1

inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)

torch.backends.cudnn.allow_tf32 = False
weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device, requires_grad=True)
if bias is True:
    bias = torch.randn([weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True)

else:
    bias = None

ref_out = torch.nn.functional.conv1d(
    inp,
    weight,
    bias=bias,
    groups=groups,
    stride=stride,
    padding=padding,
    dilation=dilation,
).to(dtype)

import flag_gems.runtime.backend._spacemit.ops as sp

res_out = sp.conv1d(
    inp,
    weight,
    bias=bias,
    padding=padding,
    stride=stride,
    dilation=dilation,
    groups=groups,
)
print("ref_out", ref_out)
print("res_out", res_out)
torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

print("PASS")
