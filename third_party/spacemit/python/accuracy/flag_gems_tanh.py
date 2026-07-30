import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems


def generate_grid(start, end, step):
    num_points = int((end - start) / step) + 1
    grid = torch.linspace(start, end, num_points, dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    return grid


if __name__ == "__main__":
    inp = generate_grid(-10, 10, 0.0001)
    with flag_gems.use_gems():
        with torch.no_grad():
            C = torch.tanh(inp)

    C_ref = torch.tanh(inp)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
