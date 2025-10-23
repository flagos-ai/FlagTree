import pytest
import torch
import torch_npu
import triton
import triton.language as tl


def profiler_wrapper(fn, *args):
    result_path = "./result_profiling"
    skip_first = 10
    wait = 0
    warmup = 3
    active = 30
    repeat = 1
    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat,
                                                 skip_first=skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config) as prof:
        stream.synchronize()
        for i in range(skip_first + (wait + warmup + active) * repeat):
            fn(*args)
            prof.step()
        stream.synchronize()


def test_add(x0, x1):
    def torch_func(x0, x1):
        res = x0 + x1
        return res

    @triton.jit
    def triton_kernel_add(out_ptr0, in_ptr0, in_ptr1,
                      XS: tl.constexpr):
        idx = tl.arange(0, XS)
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.load(in_ptr1 + idx)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + idx, tmp2)

    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)
        triton_kernel_add[1, 1, 1](y0, x0, x1, N)
        return y0

    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)
    torch.testing.assert_close(triton_cal, torch_ref)

    def wrapper_func(x0, x1):
        torch_ref = torch_func(x0, x1)
        triton_cal = triton_func(x0, x1)

    profiler_wrapper(wrapper_func, x0, x1)


def test_or(x0, x1):
    def torch_func(x0, x1):
        res = x0 | x1
        return res

    @triton.jit
    def triton_kernel_or(out_ptr0, in_ptr0, in_ptr1,
                      XS: tl.constexpr):
        idx = tl.arange(0, XS)
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.load(in_ptr1 + idx)
        tmp2 = tmp0 | tmp1
        tl.store(out_ptr0 + idx, tmp2)

    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)
        triton_kernel_or[1, 1, 1](y0, x0, x1, N)
        return y0

    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)
    torch.testing.assert_close(triton_cal, torch_ref)

    def wrapper_func(x0, x1):
        torch_ref = torch_func(x0, x1)
        triton_cal = triton_func(x0, x1)

    profiler_wrapper(wrapper_func, x0, x1)

def test_inductor_add(x0, x1):
    # torch_npu._inductor requires torch_npu 2.6.0+ experimental version
    import torch_npu._inductor

    def torch_func(x0, x1):
        res = x0 + x1
        return res

    compiled_func = torch.compile(torch_func, backend="inductor")
    profiler_wrapper(compiled_func, x0, x1)
    print("[INFO] Check ./result_profiling directory to find the kernel_details.csv file. "
          "       Check the columns: Input Shapes,Input Data Types,Input Formats")

if __name__ == "__main__":
    test_case_is_inductor = False
    N = 1024
    low = 1
    high = 100

    # float32
    x0_fp32 = torch.rand((N,), dtype=torch.float32).npu()
    x1_fp32 = torch.rand((N,), dtype=torch.float32).npu()

    # float16
    x0_fp16 = torch.rand((N,), dtype=torch.float16).npu()
    x1_fp16 = torch.rand((N,), dtype=torch.float16).npu()

    # bfloat16
    x0_bf16 = torch.rand((N,), dtype=torch.bfloat16).npu()
    x1_bf16 = torch.rand((N,), dtype=torch.bfloat16).npu()

    # int64
    x0_i64 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int64).npu()
    x1_i64 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int64).npu()

    # int32
    x0_i32 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int32).npu()
    x1_i32 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int32).npu()

    # int16
    x0_i16 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int16).npu()
    x1_i16 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int16).npu()

    # int8
    x0_i8 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int8).npu()
    x1_i8 = torch.randint(low=low, high=high, size=(N,), dtype=torch.int8).npu()

    # bool (i1)
    x0_i1 = torch.randint(low=0, high=2, size=(N,)).bool().npu()
    x1_i1 = torch.randint(low=0, high=2, size=(N,)).bool().npu()

    test_cases = [
        ('fp32', x0_fp32, x1_fp32),
        ('fp16', x0_fp16, x1_fp16),
        ('bf16', x0_bf16, x1_bf16),
        ('i64', x0_i64, x1_i64),
        ('i32', x0_i32, x1_i32),
        ('i16', x0_i16, x1_i16),
        ('i8', x0_i8, x1_i8),
        ('i1', x0_i1, x1_i1),
    ]

    for dtype_name, x0, x1 in test_cases:
        print(f"Running test for {dtype_name}...")
        if dtype_name != 'i1':
            if (test_case_is_inductor):
                test_inductor_add(x0, x1)
            else:
                test_add(x0, x1)
        else:
            test_or(x0, x1)
