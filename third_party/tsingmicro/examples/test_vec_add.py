import torch

import triton
import triton.language as tl
import benchmark

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output_torch = x + y
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    # We need to preallocate the output.
    output = torch.empty_like(x)
    # assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    output = output.to("cpu")
    print(f"The maximum difference between torch and triton is "
          f"{torch.max(torch.abs(output_torch - output))}")
    return output


def test(device):
    torch.manual_seed(0)
    size = 1024
    x = torch.rand(size, device="cpu")
    y = torch.rand(size, device="cpu")
    print("x: ", x)
    print("y: ", y)
    output_torch = x + y
    x = x.to(device)
    y = y.to(device)
    output_triton = add(x, y)
    print("output_triton device: ", output_triton.device)
    # TODO: need to check some conditions otherwise the code below does not make any difference for the test
    output_triton = output_triton.to("cpu")
    print("expected", output_torch)
    print("actual", output_triton)
    print(f"The maximum difference between torch and triton is "
          f"{torch.max(torch.abs(output_torch - output_triton))}")


@benchmark.measure()
def bench_vecadd(size, provider):
    a = torch.rand(size, device='cpu', dtype=torch.float32)
    b = torch.rand(size, device='cpu', dtype=torch.float32)
    if provider == 'torch':
        a + b
    if provider == 'triton':
        add(a, b)


if __name__ == "__main__":
    # test(DEVICE)
    for X in [2**i for i in range(8, 25, 1)]:
        for provider in ['torch', 'triton']:
            bench_vecadd(X, provider)
