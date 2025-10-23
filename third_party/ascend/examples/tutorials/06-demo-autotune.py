"""
Autotune
=============
"""
import torch, torch_npu
import triton
import triton.language as tl

def test_triton_autotune():
    # Return a set of different kernel configurations for autotune
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]
    # Use @autotune decorator to automatically select the best kernel configuration
    @triton.autotune(
        configs=get_autotune_config(), # List of configurations
        key=["numel"],  # the change of numel will trigger autotuning
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr # Block size controlling how many elements each thread block processes
    ):
        pid = tl.program_id(0) # Get current program ID
        idx = pid * XS + tl.arange(0, XS) # Index range handled by current thread block
        msk = idx < numel # Mask to avoid out-of-bound access
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0) # Load x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0) # Load x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i
            tl.store(out_ptr0 + idx, tmp2, mask=msk) # Store result
    # Function to call the Triton kernel with autotuned configuration
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0
    # Reference implementation using PyTorch for correctness check
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000-1

    DEV = "npu"
    DTYPE = torch.float32
    N = 192 * 1024
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)
    torch_ref = torch_calc_func(x0, x1)
    triton_cal = triton_calc_func(x0, x1)
    torch.testing.assert_close(triton_cal, torch_ref)

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")
