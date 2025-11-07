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


@triton.jit()
def fmod_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)  #@hint: shared_memory
    y = tl.load(y_ptr + offsets, mask=mask)  #@hint: shared_memory
    z = tl.extra.aipu.libdevice.fmod(x, y)
    tl.store(y_ptr + offsets, z, mask=mask)


def test_libdevice_fmod():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, dtype=torch.float32, device=device)
    output_triton = torch.rand(size, device=device)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    fmod_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)


if __name__ == "__main__":
    test_libdevice_fmod()
