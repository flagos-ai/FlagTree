import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (64, 64, 64, 64)
    dtype = torch.float32
    affine = True
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device) if affine else None)
    bias = (torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device) if affine else None)

    running_mean = torch.zeros(size=(C, ), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C, ), dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    ref_out = torch.nn.functional.batch_norm(
        inp,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        eps=eps,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.batch_norm(
            inp,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            eps=eps,
        )

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

    print("PASS")
