# Copyright 2026- Xcoresigma Technology Co., Ltd
import torch
import triton
import torch_npu  # noqa
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mem_addr_space is language.extra.cann.core.ascend_address_space
    with tle.dsa.hint(inter_no_alias=True):
        with tle.dsa.hint(test_k1=10):
            a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
            b_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
            c_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

        t0 = n_elements - block_start
        tail_size = tl.minimum(t0, BLOCK_SIZE)

        with tle.dsa.hint(test_k2=20):
            tle.dsa.copy(x_ptr + offsets, a_ub, [tail_size])
        tle.dsa.copy(y_ptr + offsets, b_ub, [tail_size])

    tle.dsa.add(a_ub, b_ub, c_ub)
    tle.dsa.copy(c_ub, output_ptr + offsets, [tail_size])


def custom_func(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    return output


def test_add():
    torch.manual_seed(0)
    size = 1024
    x = torch.rand(size, dtype=torch.float).npu()
    y = torch.rand(size, dtype=torch.float).npu()
    output_torch = x + y
    output_triton = custom_func(x, y)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')

    from triton.backends.ascend.testing import do_bench_npu
    bench_torch = do_bench_npu(lambda: x + y, clear_l2_cache=True, keep_res=True, collect_prof=False)
    bench_triton = do_bench_npu(lambda: custom_func(x, y), clear_l2_cache=True, keep_res=True, collect_prof=False)
    print(f"torch time : {bench_torch:.2f}")
    print(f"triton time: {bench_triton:.2f}")


if __name__ == "__main__":
    test_add()
