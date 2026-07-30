import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    inp_shape = (32, 8, 4)
    dim = 0
    dtype = torch.float32
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    size_dim = inp_shape[dim]

    import random

    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
        random.randint(1, inp_shape[2]),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_out = torch.gather(inp, dim, index)

    with flag_gems.use_gems():
        res_out = torch.gather(inp, dim, index)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
