import logging
import triton
import triton.language as tl
import torch
import torch_npu
import pytest


@triton.jit
def index_select_kernel_ascend_dim_0(
    inp, out, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """flaggems ascend index_select implementation on dim 0"""
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    grid_x = tl.num_programs(0)
    grid_y = tl.num_programs(1)
    for x in range(pid_x * BLOCK_M, index_len, grid_x * BLOCK_M):
        rows_offsets = x + tl.arange(0, BLOCK_M)
        indices = tl.load(index + rows_offsets, mask=(rows_offsets < index_len), other=0)
        for y in range(pid_y * BLOCK_N, N, grid_y * BLOCK_N):
            cols_offsets = y + tl.arange(0, BLOCK_N)
            cols_mask = cols_offsets < N
            inp_off = indices[:, None] * N + cols_offsets[None, :]
            out_off = rows_offsets[:, None] * N + cols_offsets[None, :]
            selected = tl.load(inp + inp_off, mask=cols_mask[None, :], other=0.0)
            tl.store(out + out_off, selected, mask=cols_mask[None, :])


def get_grid():
    import triton.runtime.driver as driver
    num_cores = driver.active.utils.get_aivector_core_num()
    logging.info("grid_M_size:%d, grid_N_size:%d", num_cores, 1)
    return num_cores, 1


@pytest.mark.parametrize(
    "param_list",
    [
        [[26, 140], 0, 23, 32, 32],
        [[3, 16], 0, 3, 32, 32],
        [[9, 6], 0, 9, 32, 32],
        [[992, 16], 0, 632, 32, 32],
        [[500000, 37], 0, 322364, 32, 32],
        [[500000, 240], 0, 375144, 64, 64],
        [[500000, 37], 0, 324344, 32, 32],
        [[500000, 240], 0, 377816, 64, 64],
    ],
)
def test_index_select_flaggems_ascend(param_list):
    shape, dim, index_size, BLOCK_M, BLOCK_N = param_list
    inp = torch.randn(shape, dtype=torch.float32, device="npu")
    index = torch.randint(0, inp.size(dim), [index_size], device="npu")
    golden = torch.index_select(inp, dim, index)
    index_len = index.numel()
    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    N = inp_shape[1]
    M = inp.numel() // N
    if dim != 0:
        logging.error("error dim:%d", dim)
        return
    grid_M_size, grid_N_size = get_grid()
    out_shape = list(inp.shape)
    out_shape[dim] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    # Perf testing
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
            index_select_kernel_ascend_dim_0[[grid_M_size, grid_N_size, ]](
                inp,
                out,
                N,
                index,
                index_len,
                BLOCK_M,
                BLOCK_N,
                num_warps=32,
                force_simt_only=True,
            )
            prof.step()
        stream.synchronize()
    # Correctness testing
    index_select_kernel_ascend_dim_0[[grid_M_size, grid_N_size, ]](
        inp,
        out,
        N,
        index,
        index_len,
        BLOCK_M,
        BLOCK_N,
        num_warps=32,
        force_simt_only=True,
    )
    torch.testing.assert_close(golden, out, rtol=1e-04, atol=1e-04, equal_nan=True)

