import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    dim = 1
    shape = (2, 32)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.nn.functional.log_softmax(inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
