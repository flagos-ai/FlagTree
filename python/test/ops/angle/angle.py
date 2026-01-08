import torch
import triton
import triton.language as tl

# active driver
driver = triton.runtime.driver.active
# torch.cuda, torch.aipu, torch.npu
torch_device_fn = triton.runtime.driver.active.get_device_interface()
# device
if hasattr(driver, "get_active_torch_device"):
    device = triton.runtime.driver.active.get_active_torch_device()
else:
    device = triton.runtime.driver.active.get_current_device()


@triton.jit
def angle_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    real = tl.load(x_ptr + offsets, mask=mask)
    imag = tl.load(y_ptr + offsets, mask=mask)
    real_last, imag_last = ((real.to(tl.float32), imag.to(tl.float32)) if real.dtype == tl.float16 else (real, imag))
    output = tl.math.atan2(imag_last, real_last)
    tl.store(output_ptr + offsets, output, mask=mask)


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        x = input_tensor.real
        y = input_tensor.imag
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        with torch_device_fn.device(x.device):
            angle_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def check(name, ref, res, equal_nan=False, reduce_dim=1, atol=1e-4):
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float8_e4m3fn: 1e-3,
        torch.float8_e5m2: 1e-3,
        torch.float8_e4m3fnuz: 1e-3,
        torch.float8_e5m2fnuz: 1e-3,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
    }
    res = res.cpu()
    print(f"The maximum difference out {name} between torch and triton is "
          f"{torch.max(torch.abs(ref - res))}")
    rtol = RESOLUTION[ref.dtype]
    assert torch.allclose(res, ref, atol=atol * reduce_dim, rtol=rtol), (res, ref)


if __name__ == "__main__":
    # param
    shape = (4, 13)
    dtype = torch.complex32

    # inp
    inp = torch.randn(shape, dtype=dtype, device="cpu").to(device)
    ref_inp = inp.cpu()

    # op
    #ref_out = torch.angle(ref_inp)
    res_out = angle(inp)
    #check("value", ref_out, res_out)
