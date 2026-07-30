import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    M, N = 1, 32
    dtype = torch.float32
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M, ), dtype=dtype, device=flag_gems.device)

    ref_out = torch.mv(matrix, vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
