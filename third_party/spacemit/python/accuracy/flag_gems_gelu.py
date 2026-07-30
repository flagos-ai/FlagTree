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
    inp = generate_grid(-20, 20, 0.0001)
    for approximate in ("none", "tanh"):
        with flag_gems.use_gems():
            with torch.no_grad():
                C = torch.nn.functional.gelu(inp, approximate=approximate)

        C_ref = torch.nn.functional.gelu(inp, approximate=approximate)

        torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)

        print("PASS")
