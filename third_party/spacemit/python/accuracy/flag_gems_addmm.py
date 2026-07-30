import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    M, N, K = 256, 256, 700
    A = torch.randn((M, K), dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    B = torch.randn((K, N), dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    #bias = torch.randn((N,), dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    bias = torch.zeros((N, ), dtype=torch.float32, device=flag_gems.device, requires_grad=False)

    with flag_gems.use_gems():
        with torch.no_grad():
            C = torch.addmm(bias, A, B, beta=2, alpha=3)

    C_ref = torch.addmm(bias, A, B, beta=2, alpha=3)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
