# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
def triton_kernel_add(out_ptr0, in_ptr0, in_ptr1, XS: tl.constexpr):
    idx = tl.arange(0, XS)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + idx, tmp2)


@triton.jit
def triton_kernel_or(out_ptr0, in_ptr0, in_ptr1, XS: tl.constexpr):
    idx = tl.arange(0, XS)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 | tmp1
    tl.store(out_ptr0 + idx, tmp2)


def triton_add_func(x0, x1, N):
    y0 = torch.empty_like(x0)
    triton_kernel_add[1, 1, 1](y0, x0, x1, N)
    return y0


def triton_or_func(x0, x1, N):
    y0 = torch.empty_like(x0)
    triton_kernel_or[1, 1, 1](y0, x0, x1, N)
    return y0


# ==================== Pytest Test ====================
@pytest.mark.parametrize("dtype, low, high", [
    (torch.float32, 0, 1),
    (torch.float16, 0, 1),
    (torch.bfloat16, 0, 1),
    (torch.int64, 1, 100),
    (torch.int32, 1, 100),
    (torch.int16, 1, 100),
    (torch.int8, 1, 100),
    (torch.bool, 0, 2),
])
def test_elementwise_ops(dtype, low, high):
    N = 1024
    test_case_is_inductor = False

    if dtype == torch.bool:
        x0 = torch.randint(low=low, high=high, size=(N, )).bool().npu()
        x1 = torch.randint(low=low, high=high, size=(N, )).bool().npu()
        triton_cal = triton_or_func(x0, x1, N)
        ref = x0 | x1
    else:
        if dtype.is_floating_point:
            x0 = torch.rand((N, ), dtype=dtype).npu()
            x1 = torch.rand((N, ), dtype=dtype).npu()
        else:
            x0 = torch.randint(low=low, high=high, size=(N, ), dtype=dtype).npu()
            x1 = torch.randint(low=low, high=high, size=(N, ), dtype=dtype).npu()

        triton_cal = triton_add_func(x0, x1, N)
        ref = x0 + x1

    torch.testing.assert_close(triton_cal, ref)

    def wrapper():
        _ = triton_add_func(x0, x1, N) if dtype != torch.bool else triton_or_func(x0, x1, N)

    profiler_wrapper(wrapper)
