import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    inp1 = torch.randn(shape, dtype=dtype, device="cpu")
    inp2 = torch.randn(shape, dtype=dtype, device="cpu")
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")

    import itertools

    shapes = (shape, None)
    for a_shape, b_shape, c_shape in itertools.product(shapes, shapes, shapes):
        a = inp1 if a_shape else torch.tensor(0)
        b = inp2 if b_shape else torch.tensor(1)
        c = cond if c_shape else torch.tensor(True)

        ref_out = torch.where(c, a, b)
        with flag_gems.use_gems():
            res_out = torch.where(c, a, b)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
