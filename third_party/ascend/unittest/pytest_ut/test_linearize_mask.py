import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu

types_all = [
    (torch.float32, 'float32'),
]

shapes_common = [(128, 256), (127, 256), (127, 16), (129, 256), (77, 1024), (69, 512)]

block_size = [128, 256, 1024]


def ceil_div(a, b):
    return (a + b - 1) // b


def profiler_wrapper(fn, *args):
    result_path = "./result_profiling_broadcast"
    skip_first = 10
    wait = 0
    warmup = 3
    active = 30
    repeat = 1
    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False, data_simplification=False)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat,
                                                 skip_first=skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path), record_shapes=True,
            profile_memory=False, with_stack=False, with_flops=False, with_modules=False,
            experimental_config=experimental_config) as prof:
        stream.synchronize()
        for _ in range(skip_first + (wait + warmup + active) * repeat):
            fn(*args)
            prof.step()
        stream.synchronize()


@triton.jit
def linearize_mask_broadcast_kernel(
    in_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.multiple_of(pid * BLOCK_SIZE_N, N)
    x1 = (offset + tl.arange(0, BLOCK_SIZE_N)) // N
    mask1 = (x1 < M)
    data = tl.load(in_ptr + x1 * N, mask=mask1, other=0)
    x2 = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(output_ptr + x2, data)


def torch_linearize_mask_broadcast(in_tensor):
    M = in_tensor.shape[0] // 2
    N = in_tensor.shape[1]

    output = torch.zeros_like(in_tensor)

    first_elements = in_tensor[:M, 0:1]
    output[:M] = first_elements.expand(-1, N)

    return output


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('M, N', shapes_common)
@pytest.mark.parametrize('BLOCK_SIZE_N', block_size)
def test_linearize_mask_broadcast(M, N, BLOCK_SIZE_N, dtype, sigtype):

    in_tensor = torch.randn(2 * M, N, dtype=dtype).npu()

    triton_output = torch.zeros_like(in_tensor)

    grid = (ceil_div(2 * M * N, BLOCK_SIZE_N), )

    linearize_mask_broadcast_kernel[grid](in_tensor, triton_output, N=N, M=M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                          optimize_dynamic_offset=True)

    torch_output = torch_linearize_mask_broadcast(in_tensor.clone())
    assert torch.allclose(triton_output, torch_output, rtol=1e-5, atol=1e-8)


def triton_linearize_mask_broadcast(in_tensor, BLOCK_SIZE):
    M = in_tensor.shape[0] // 2
    N = in_tensor.shape[1]

    triton_output = torch.zeros_like(in_tensor)
    grid = (ceil_div(2 * M * N, BLOCK_SIZE), )

    linearize_mask_broadcast_kernel[grid](in_tensor, triton_output, N=N, M=M, BLOCK_SIZE_N=BLOCK_SIZE,
                                          optimize_dynamic_offset=True)


@triton.jit
def rem_kernel(in_ptr0, in_ptr1, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.arange(0, BLOCK_SIZE)

    base_offset = pid * BLOCK_SIZE + x

    rem_result = base_offset % 128
    mask = rem_result < 64

    tmp0 = tl.load(in_ptr0 + base_offset, mask=mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + base_offset, mask=mask, other=0.0)
    tmp2 = tmp0 + tmp1

    tl.store(out_ptr + base_offset, tmp2, mask=mask)


def test_linearize_mask_rem():
    N = 1024
    BLOCK_SIZE = 256
    dtype = 'float32'
    shape = (N, )

    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.zeros(shape).npu()

    grid = (ceil_div(N, BLOCK_SIZE), )
    rem_kernel[grid](x0, x1, triton_res, N, BLOCK_SIZE=BLOCK_SIZE)

    base_offsets = torch.arange(N).npu()
    rem_results = base_offsets % 128
    mask_bool = rem_results < 64

    torch_res = torch.zeros((N, )).npu()
    torch_res[mask_bool] = x0[mask_bool] + x1[mask_bool]

    test_common.validate_cmp(dtype, triton_res, torch_res)


def profile_performance_test(M, N, dtype, BLOCK_SIZE):
    print(f"\nDetailed performance analysis: M={M}, N={N}, dtype={dtype}, block_size={BLOCK_SIZE}")

    in_tensor = torch.randn(2 * M, N, dtype=dtype).npu()

    def wrapper_func(x):
        triton_linearize_mask_broadcast(x, BLOCK_SIZE=BLOCK_SIZE)

    # Run performance analysis
    profiler_wrapper(wrapper_func, in_tensor)


if __name__ == "__main__":
    print("Broadcast Kernel Performance Test Suite")
    print("Function: Broadcast first element of first M rows, set remaining M rows to zero")

    # Optional: Run detailed profiler test (specific configuration)
    profile_performance_test(512, 512, torch.float32, BLOCK_SIZE=1024)

    print("\n" + "=" * 80)
    print("Test completed!")
    print(f"Detailed performance analysis results saved in: ./result_profiling_broadcast/")
    print("=" * 80)
