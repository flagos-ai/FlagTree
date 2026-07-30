import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 8, 8, device="cpu")
    kernel_size = (3, 5)

    pt_unfold = torch.nn.Unfold(kernel_size, padding=1, stride=2)
    ref_out = pt_unfold(input_tensor)

    res_out = flag_gems.unfold(input_tensor, kernel_size, padding=1, stride=2)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
