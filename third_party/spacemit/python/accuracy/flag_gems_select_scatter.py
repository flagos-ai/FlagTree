import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 2)
    dim = 1
    dtype = torch.float32

    import random

    index = random.randint(0, shape[dim] - 1)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    del src_shape[dim]
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.select_scatter(inp, dim=dim, index=index, src=src)
    res_out = flag_gems.select_scatter(inp, dim=dim, index=index, src=src)

    # dim = 0
    # index = 1
    # inp = torch.randn((1, 4), device=flag_gems.device).broadcast_to((3, 4))
    # src = torch.randn((4,), device=flag_gems.device)

    # ref_out = torch.select_scatter(inp, dim=dim, index=index, src=src)
    # with flag_gems.use_gems():
    #     res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    print("ref_out", ref_out)
    print("res_out", res_out)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
