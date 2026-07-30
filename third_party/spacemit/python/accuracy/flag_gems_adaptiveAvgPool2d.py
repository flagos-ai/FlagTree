import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    m = torch.nn.AdaptiveAvgPool2d((1, 1))
    shape = (1, 2, 7, 7)
    dtype = torch.float32
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = m(inp1)
    with flag_gems.use_gems():
        res_out = flag_gems.global_avg_pool(inp1)
    print("ref_out", ref_out)
    print("res_out", res_out)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
