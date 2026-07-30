import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = [1024, 1024]
    dtype = torch.float32
    pad_mode = "constant"
    contiguous = True

    ref_x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    if not contiguous:
        ref_x = ref_x[::2, ::2]

    if ref_x.dtype == torch.float16:
        ref_x = ref_x.to(torch.float32)

    rank = ref_x.ndim
    pad_params = list(
        torch.randint(0, 10, (rank * 2, ), dtype=torch.int32, device="cpu") if pad_mode ==
        "constant" else torch.randint(0, 10, (rank, ), dtype=torch.int32, device="cpu"))
    pad_value = float(torch.randint(0, 1024, (1, ), dtype=torch.int32, device="cpu"))

    if pad_mode != "constant":
        pad_params = [(pad_val + 2 - 1) // 2 * 2 for pad_val in pad_params]
        pad_value = None

    ref_out = torch.nn.functional.pad(ref_x, pad_params, pad_mode, pad_value)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pad(ref_x, pad_params, pad_mode, pad_value)

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
