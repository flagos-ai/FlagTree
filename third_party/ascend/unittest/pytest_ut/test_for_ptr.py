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
    result_path = "./result_profiling_for"
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
def for_ptr_kernel(
    in_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = 2 * pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_in = in_ptr + offset
    x_out = output_ptr + offset
    for _ in range(0, 2):
        mask = (offset < M * N)
        data = tl.load(x_in, mask=mask, other=0)
        tl.store(x_out, data, mask)
        x_in += BLOCK_SIZE_N
        x_out += BLOCK_SIZE_N


@pytest.mark.parametrize('dtype, sigtype', types_all)
@pytest.mark.parametrize('M, N', shapes_common)
@pytest.mark.parametrize('BLOCK_SIZE_N', block_size)
def test_for_ptr(M, N, BLOCK_SIZE_N, dtype, sigtype):

    in_tensor = torch.randn(M, N, dtype=dtype).npu()

    triton_output = torch.zeros_like(in_tensor)

    grid = (ceil_div(2 * M * N, BLOCK_SIZE_N), )

    for_ptr_kernel[grid](in_tensor, triton_output, N=N, M=M, BLOCK_SIZE_N=BLOCK_SIZE_N, optimize_dynamic_offset=False)

    assert torch.allclose(triton_output, in_tensor, rtol=1e-5, atol=1e-8)


def triton_lfor_ptr(in_tensor, BLOCK_SIZE):
    M = in_tensor.shape[0]
    N = in_tensor.shape[1]

    triton_output = torch.zeros_like(in_tensor)
    grid = (ceil_div(2 * M * N, BLOCK_SIZE), )

    for_ptr_kernel[grid](in_tensor, triton_output, N=N, M=M, BLOCK_SIZE_N=BLOCK_SIZE, optimize_dynamic_offset=True)


def profile_performance_test(M, N, dtype, BLOCK_SIZE):
    print(f"\nDetailed performance analysis: M={M}, N={N}, dtype={dtype}, block_size={BLOCK_SIZE}")

    in_tensor = torch.randn(2 * M, N, dtype=dtype).npu()

    def wrapper_func(x):
        triton_lfor_ptr(x, BLOCK_SIZE=BLOCK_SIZE)

    # Run performance analysis
    profiler_wrapper(wrapper_func, in_tensor)


if __name__ == "__main__":
    print("For Kernel Performance Test Suite")
    print("Function: Broadcast first element")

    # Optional: Run detailed profiler test (specific configuration)
    profile_performance_test(512, 512, torch.float32, BLOCK_SIZE=1024)

    print("\n" + "=" * 80)
    print("Test completed!")
    print(f"Detailed performance analysis results saved in: ./result_profiling_for/")
    print("=" * 80)
