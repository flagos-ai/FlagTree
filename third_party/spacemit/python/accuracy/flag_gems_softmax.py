import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    K = 512
    inp = torch.randn([K], dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    with flag_gems.use_gems():
        with torch.no_grad():
            C = torch.softmax(inp, -1)

    C_ref = torch.softmax(inp, -1)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
