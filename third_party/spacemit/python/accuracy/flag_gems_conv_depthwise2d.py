import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

shape = (1, 64, 56, 56)
kernel = (64, 1, 3, 3)
groups = shape[1]
stride = (1, 1)
padding = (1, 1)
dtype = torch.float32
bias = False
dilation = (1, 1)

inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device, requires_grad=True)

if bias:
    bias_tensor = torch.randn(weight.shape[0], dtype=dtype, device=flag_gems.device, requires_grad=True)
else:
    bias_tensor = None

ref_out = torch.nn.functional.conv2d(
    inp,
    weight,
    bias=bias_tensor,
    groups=groups,
    stride=stride,
    padding=padding,
    dilation=dilation,
)

import flag_gems.runtime.backend._spacemit.ops as sp

res_out = sp._conv_depthwise2d(inp, weight, bias=bias_tensor, padding=padding, stride=stride, dilation=dilation)

torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

print("PASS")
