import triton
import triton.language as tl
import torch
import torch_npu
import pytest


@triton.jit
def triton_unk_reduce(in_ptr0, out_ptr0, y0_numel, x1_numel, X1BLOCK_SUB: tl.constexpr):
    y0_offset = tl.program_id(0)
    grid_size = tl.num_programs(0)
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for y0 in range(y0_offset, y0_numel, grid_size):
        _tmp8 = tl.full([X1BLOCK_SUB], 0, tl.float32)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1
            x1_mask = x1 < x1_numel
            tmp0 = tl.load(in_ptr0 + x1_numel * y0 + x1, x1_mask, other=0.0)
            _tmp8 += tmp0
        tmp8 = tl.sum(_tmp8, 0)
        tl.store(out_ptr0 + y0, tmp8)


def torch_reduce(arg0):
    return arg0.sum(dim=1)


@pytest.mark.parametrize(
    "param_list",
    [
        [128, 40000],
    ],
)
def test_reduce(param_list):
    y0_numel, x1_numel = param_list
    arg0_1 = torch.randn(y0_numel, x1_numel, dtype=torch.float32, device="npu")
    buf44 = torch.empty((y0_numel), dtype=torch.float32, device="npu")
    grid_size = 64
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
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        stream.synchronize()
        for _ in range(skip_first + (wait + warmup + active) * repeat):
            triton_unk_reduce[[grid_size, 1, 1]](
                arg0_1,
                buf44,
                y0_numel,
                x1_numel,
                4096,
                num_warps=32,
                force_simt_only=True,
            )
            prof.step()
        stream.synchronize()
    triton_unk_reduce[[grid_size, 1, 1]](
        arg0_1, buf44, y0_numel, x1_numel, 4096, num_warps=32, force_simt_only=True
    )
    torch_out = torch_reduce(arg0_1)
    torch.testing.assert_close(buf44, torch_out, rtol=1e-04, atol=1e-04, equal_nan=True)