import triton
import triton.language as tl
import torch
import torch_npu
import pytest


@triton.jit
def triton_unk_fused_embedding_sum_5(
    in_ptr0,
    in_ptr1,
    out_ptr0,
    y0_numel,
    x1_numel,
    r2_numel,
    X1BLOCK_SUB: tl.constexpr,
    R2BLOCK_SUB: tl.constexpr,
):
    y0_offset = tl.program_id(0)
    grid_size = tl.num_programs(0)
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    base_r2 = tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for y0 in range(y0_offset, y0_numel, grid_size):
        x1 = base_x1[None, :]
        x1_store = base_x1
        _tmp8 = tl.full([R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
        for loop_r2 in range(loops_r2):
            r2 = loop_r2 + base_r2[:, None] * loops_r2
            r2_mask = r2 < r2_numel
            tmp0 = tl.load(in_ptr0 + (r2 + r2_numel * y0), r2_mask, other=0.0)
            #embedding table length constant fold
            tmp1 = tl.full([R2BLOCK_SUB, X1BLOCK_SUB], 9000, tl.int64)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp0 < 0
            tmp4 = tl.where(tmp3, tmp2, tmp0)
            tmp6 = tl.load(in_ptr1 + (x1 + x1_numel * tmp4), r2_mask, other=0.0)
            tmp7 = tl.reshape(tmp6, [R2BLOCK_SUB, X1BLOCK_SUB])
            tmp9 = _tmp8 + tmp7
            _tmp8 = tl.where(r2_mask, tmp9, _tmp8)
        tmp8 = tl.sum(_tmp8, 0)
        tl.store(out_ptr0 + (x1_store + x1_numel * y0), tmp8)


def torch_red_fused_embedding_sum_vec(arg0, arg1):
    adjusted = torch.where(arg0 < 0, arg0 + 9000, arg0)
    emb = arg1[adjusted]
    return emb.sum(dim=1)


@pytest.mark.parametrize(
    "param_list",
    [
        [128, 128, 4000],
    ],
)
def test_embedding_sum(param_list):
    y0_numel, x1_numel, r2_numel = param_list
    arg0_1 = torch.randint(
        -1000, 9000, (y0_numel, r2_numel), dtype=torch.int64, device="npu"
    )
    arg2_1 = torch.randn(9000, x1_numel, dtype=torch.float32, device="npu")
    buf44 = torch.empty((y0_numel, x1_numel), dtype=torch.float32, device="npu")
    grid_size = 128
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
            triton_unk_fused_embedding_sum_5[[grid_size, 1, 1]](
                arg0_1,
                arg2_1,
                buf44,
                y0_numel,
                x1_numel,
                r2_numel,
                128,
                32,
                num_warps=32,
                force_simt_only=True
            )
            prof.step()
        stream.synchronize()
    triton_unk_fused_embedding_sum_5[[grid_size, 1, 1]](
        arg0_1,
        arg2_1,
        buf44,
        y0_numel,
        x1_numel,
        r2_numel,
        128,
        32,
        num_warps=32,
        force_simt_only=True
    )
    torch_out = torch_red_fused_embedding_sum_vec(arg0_1, arg2_1)
    torch.testing.assert_close(buf44, torch_out, rtol=1e-04, atol=1e-04, equal_nan=True)