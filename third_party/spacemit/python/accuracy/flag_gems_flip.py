import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems


def get_max_ndim(shape, dims):
    max_ndim = max(len(shape), len(dims))
    for dim in dims:
        dim = dim + 1 if dim >= 0 else -dim
        if dim > max_ndim:
            max_ndim = dim
    return max_ndim


def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp


if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    dims = (0, )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    max_ndim = get_max_ndim(shape, dims)
    inp = unsqueeze_tensor(inp, max_ndim)

    with flag_gems.use_gems():
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(inp, dims)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
