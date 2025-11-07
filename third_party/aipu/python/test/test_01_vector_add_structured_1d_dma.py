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
def add_kernel_structured_1d(
    x_ptr,
    y_ptr,
    output_ptr,
    stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    A_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(n_elements, ), strides=(stride, ), offsets=(block_start, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    B_block_ptr = tl.make_block_ptr(base=y_ptr, shape=(n_elements, ), strides=(stride, ), offsets=(block_start, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    C_block_ptr = tl.make_block_ptr(base=output_ptr, shape=(n_elements, ), strides=(stride, ), offsets=(block_start, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    A_block = tl.load(A_block_ptr)  #@hint: dma
    B_block = tl.load(B_block_ptr)  #@hint: dma
    C_block = A_block + B_block
    tl.store(C_block_ptr, C_block)


def add_structured_1d(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)

    n_elements = output.numel()
    stride = 1
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel_structured_1d[grid](x, y, output, stride, n_elements, BLOCK_SIZE=1024)
    return output.cpu()


def test_vector_add_structured_1d():
    torch.manual_seed(0)

    test_shapes = [(64), (256), (512), (63), (255), (511), (1024), (2048), (4096)]
    for size in test_shapes:
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)

        output_torch = x.cpu() + y.cpu()
        output_triton = add_structured_1d(x, y)

        print(f'The maximum difference between torch and triton is '
              f'{torch.max(torch.abs(output_torch - output_triton))}')
        assert torch.allclose(output_triton, output_torch), (output_triton, output_torch)


if __name__ == "__main__":
    test_vector_add_structured_1d()
