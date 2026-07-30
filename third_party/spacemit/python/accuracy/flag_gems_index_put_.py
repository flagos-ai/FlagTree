import numpy as np
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for i, shape in enumerate(indices_shape):
        index = np.random.choice(np.arange(input_shape[i]), size=shape, replace=accumulate)
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


if __name__ == "__main__":
    input_shape = (32, 32)
    indices_shape = (8, ), (8, )
    values_shape = (8, )
    dtype = torch.float32

    accumulate = True
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    inp1 = inp.clone()
    torch.index_put_(inp1, indices, values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)

    torch.testing.assert_close(inp, inp1, atol=1e-2, rtol=0)
    print("PASS")
