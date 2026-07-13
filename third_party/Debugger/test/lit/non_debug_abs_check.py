import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def abs_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    z = y + 1.0
    tl.store(y_ptr + offsets, z, mask=mask)


def main():
    n = 16
    x = torch.linspace(-8, 7, n, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    abs_kernel[(1, )](x, y, n, BLOCK_SIZE=16)
    torch_npu.npu.synchronize()
    print("non_debug_allclose", torch.allclose(y.cpu(), (torch.abs(x) + 1.0).cpu()))


if __name__ == "__main__":
    main()
