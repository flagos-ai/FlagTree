import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    scale = (2, 2)
    shape = (32, 16, 128, 128)
    dtype = torch.float32

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]
    ref_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    torch.testing.assert_close(res_out, ref_out, atol=1e-2, rtol=0)
    print("PASS")
