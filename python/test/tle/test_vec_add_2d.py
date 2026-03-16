# Copyright 2026- Xcoresigma Technology Co., Ltd
import torch
import torch_npu  # noqa
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               n_cols, n_rows, BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2D offsets
    block_start_m = pid_m * BLOCK_SIZE
    block_start_n = pid_n * BLOCK_SIZE
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE)

    # get address（row-major）
    x_ptrs = x_ptr + offs_m[:, None] * n_cols + offs_n[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * n_cols + offs_n[None, :]
    out_ptrs = output_ptr + offs_m[:, None] * n_cols + offs_n[None, :]

    a_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    t0 = n_elements - block_start_m
    t1 = n_elements - block_start_n
    tail_size_m = tl.minimum(t0, BLOCK_SIZE)
    tail_size_n = tl.minimum(t1, BLOCK_SIZE)

    tle.dsa.copy(x_ptrs, a_ub, [tail_size_m, tail_size_n])
    tle.dsa.copy(y_ptrs, b_ub, [tail_size_m, tail_size_n])

    tle.dsa.add(a_ub, b_ub, c_ub)

    tle.dsa.copy(c_ub, out_ptrs, [tail_size_m, tail_size_n])


def custom_func(x: torch.Tensor, y: torch.Tensor, size: int):
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 16
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    grid = (triton.cdiv(size, BLOCK_SIZE), triton.cdiv(size, BLOCK_SIZE))
    add_kernel[grid](x, y, output, n_elements, size, size - 1, BLOCK_SIZE)
    return output


def test_add():
    torch.manual_seed(0)
    size = 128
    x = torch.rand((size, size - 1), dtype=torch.float).npu()
    y = torch.rand((size, size - 1), dtype=torch.float).npu()
    output_torch = x + y
    output_triton = custom_func(x, y, size)
    print("============X===========")
    print(x)
    print("============Y===========")
    print(y)
    print("============outTorch===========")
    print(output_torch)
    print("============outTriton===========")
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')


if __name__ == "__main__":
    test_add()
