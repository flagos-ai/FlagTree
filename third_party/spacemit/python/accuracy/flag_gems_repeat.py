import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems


def unsqueeze_tuple(t, max_len):
    for _ in range(len(t), max_len):
        t = t + (1, )
    return t


if __name__ == "__main__":
    shape = (2, 19, 7)
    sizes = (2, 3, 4, 5)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    sizes = unsqueeze_tuple(sizes, inp.ndim)

    ref_out = inp.repeat(*sizes)
    with flag_gems.use_gems():
        res_out = inp.repeat(*sizes)

    torch.testing.assert_close(res_out, ref_out, atol=1e-2, rtol=0)
    print("PASS")
