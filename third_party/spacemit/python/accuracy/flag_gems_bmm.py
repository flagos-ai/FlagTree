import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    M, N, K = 196, 256, 128
    A = torch.randn((1, M, K), dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    B = torch.randn((1, K, N), dtype=torch.float32, device=flag_gems.device, requires_grad=False)

    with flag_gems.use_gems():
        with torch.no_grad():
            C = torch.bmm(A, B)

    C_ref = torch.bmm(A, B)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)
    print("PASS")
