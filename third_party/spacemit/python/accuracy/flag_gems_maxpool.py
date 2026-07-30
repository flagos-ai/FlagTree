import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":

    shape = (1, 3, 5, 5)
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = True
    dtype = torch.float32

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)

    ref_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  ceil_mode=ceil_mode, return_indices=False)
    ref_out = ref_pool(inp)
    import flag_gems.runtime.backend._spacemit.ops as sp

    res_out = sp.maxpool2d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    torch.testing.assert_close(ref_out, res_out, atol=1e-5, rtol=1e-3)
    print("PASS")
