import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 32)
    dim = 1
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(0, index_size, [floor(index_size * 0.8)], device=flag_gems.device)

    ref_out = torch.index_select(inp, dim, index)
    with flag_gems.use_gems():
        res_out = torch.index_select(inp, dim, index)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
