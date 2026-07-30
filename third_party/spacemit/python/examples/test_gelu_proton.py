import torch
import triton
import triton.language as tl
import triton.profiler.proton as proton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
from triton.language.extra.cpu import libdevice as tl_extra_shim


@triton.jit
def gelu_none_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GeLU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    scale: tl.constexpr = 0.7071067811  # 1 / sqrt(2)
    y = 0.5 * x * (1.0 + tl_extra_shim.erf(x * scale))
    tl.store(out_ptr + offsets, y, mask=mask)


def test_gelu_profile():
    torch.manual_seed(0)
    size = 4096 * 4
    x = torch.randn(size, dtype=torch.float32)
    output_triton = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )

    print(f"Profiling GeLU with {n_elements} elements...")

    # Warump (JIT compile)
    gelu_none_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # 2. Proton Profiler
    proton.start("gelu_profile", backend="cpu")

    # 3. Run Kernel
    # Proton will automatically hook Triton's kernel launch
    for _ in range(100):
        with proton.scope("gelu_loop"):
            gelu_none_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # 4. End the Profile and process the data
    proton.finalize()

    print("Profile finished. Check the output files (usually *.hatchet or similar).")


if __name__ == "__main__":
    test_gelu_profile()
