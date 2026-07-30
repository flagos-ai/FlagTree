import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

# shape = (1, 2, 5, 5)
# kernel =  (1, 2, 3, 3)
# groups = 1
# stride = (1,1)
# padding = (0,0)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 64, 56, 56)
# kernel =  (128, 64, 1, 1)
# groups = 1
# stride = (2,2)
# padding = (0,0)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 64, 56, 56)
# kernel =  (64, 64, 3, 3)
# groups = 1
# stride = (1,1)
# padding = (1,1)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 64, 56, 56)
# kernel =  (128, 64, 3, 3)
# groups = 1
# stride = (2,2)
# padding = (1,1)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 3, 224, 224)
# kernel =  (64, 3, 7, 7)
# groups = 1
# stride = (2,2)
# padding = (3,3)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 128, 28, 28)
# kernel =  (128, 128, 3, 3)
# groups = 1
# stride = (1,1)
# padding = (1,1)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 64, 56, 56)
# kernel =  (128, 64, 1, 1)
# groups = 1
# stride = (2,2)
# padding = (0,0)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

shape = (1, 128, 28, 28)
kernel = (256, 128, 1, 1)
groups = 1
stride = (2, 2)
padding = (0, 0)
dtype = torch.float32
bias = False
dilation = (1, 1)

inp = torch.ones(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)

torch.backends.cudnn.allow_tf32 = False
weight = torch.ones(kernel, dtype=dtype, device=flag_gems.device, requires_grad=True)
if bias is True:
    bias = torch.ones([weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True)

else:
    bias = None

ref_out = torch.nn.functional.conv2d(
    inp,
    weight,
    bias=bias,
    groups=groups,
    stride=stride,
    padding=padding,
    dilation=dilation,
).to(dtype)

# import flag_gems.runtime.backend._spacemit.ops as sp
# res_out = sp.conv2d(
#     inp,
#     weight,
#     bias=bias,
#     padding=padding,
#     stride=stride,
#     dilation=dilation,
#     groups=groups,
# )
with flag_gems.use_gems():
    res_out = torch.nn.functional.conv2d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)
print("ref_out", ref_out)
print("res_out", res_out)
torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

print("PASS")
