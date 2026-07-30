import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    M, N = 256, 512
    layer_shape = [
        N,
    ]
    A = torch.randn((M, N), dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    weight = torch.randn(layer_shape, dtype=torch.float32, device=flag_gems.device, requires_grad=True)
    bias = torch.randn(layer_shape, dtype=torch.float32, device=flag_gems.device, requires_grad=True)
    eps = 1e-5

    ref_out = torch.layer_norm(
        A,
        list(layer_shape),
        weight=weight,
        bias=bias,
        eps=eps,
    )

    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            A,
            list(layer_shape),
            weight=weight,
            bias=bias,
            eps=eps,
        )

    torch.testing.assert_close(res_out, ref_out, atol=5e-2, rtol=1e-2)

    print("PASS")
