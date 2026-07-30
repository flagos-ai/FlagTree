import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    dim = 0
    shape = (5, 3)
    stride = (3, 1)
    dtype = torch.float32
    start = 1
    end = 4
    step = 2
    inp = torch.empty_strided(shape, stride, dtype=dtype, device=flag_gems.device)
    inp.copy_(1)

    valid_shape = list(inp.shape)
    size = valid_shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    valid_shape[dim] = (end - start + step - 1) // step

    src = torch.rand(valid_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.slice_scatter(inp, dim=dim, src=src, start=start, end=end, step=step)

    res_out = flag_gems.ops.slice_scatter(inp, dim=dim, src=src, start=start, end=end, step=step)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
