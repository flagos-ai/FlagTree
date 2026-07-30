import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems


def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp


if __name__ == "__main__":
    diagonal = -2
    shape = (1, )
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = unsqueeze_tensor(inp, 2)

    ref_out = torch.triu(inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
