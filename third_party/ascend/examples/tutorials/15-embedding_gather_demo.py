# only available on 910_95
import torch
import torch_npu
from torch import empty_strided
from torch._dynamo.testing import rand_strided
import triton
import triton.language as tl

y0_numel = 128
r1_numel = 50
x2_numel = 16
embedding_size = 1353406

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

def get_autotune_config():
    return [
        triton.Config({'Y0BLOCK': 4, 'Y0BLOCK_SUB': 2, 'X2BLOCK_SUB': x2_numel, 'R1BLOCK_SUB': r1_numel, 'EMBEDDING_SIZE': embedding_size, 'multibuffer': False}),
    ]
@triton.autotune(
    configs=get_autotune_config(), # List of configurations
    key=["numel"],  # the change of numel will trigger autotuning
)
@triton.jit
def triton_unk_fused_embedding_eq_sum_where_zeros_like_0(
    in_ptr0, in_ptr1, out_ptr0, y0_numel, r1_numel, x2_numel,
    Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, X2BLOCK_SUB: tl.constexpr,
    R1BLOCK_SUB: tl.constexpr, EMBEDDING_SIZE: tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_r1 = tl.arange(0, R1BLOCK_SUB)
    base_x2 = tl.arange(0, X2BLOCK_SUB)
    r1 = base_r1[None, None, :]
    r1_mask = r1 < r1_numel
    x2 = base_x2[None, None, :]
    x2_mask = x2 < x2_numel
    # loops_x1 = (x1_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    # loops_r2 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        tmp0 = tl.load(in_ptr0 + (r1 + 50*y0), r1_mask & y0_mask, other=0.0).to(tl.int32)
        tmp1 = tl.full([1, 1, 1], -1, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.full([1, 1, 1], 0, tl.int32)
        tmp4 = tl.where(tmp2, tmp3, tmp0)
        # tmp5 = tl.full([Y0BLOCK_SUB, X2BLOCK_SUB, R1BLOCK_SUB], 1353406, tl.int32)
        # tmp6 = tmp4 + tmp5
        # tmp7 = tmp4 < 0
        # tmp8 = tl.where(tmp7, tmp6, tmp4)
        # tl.device_assert(((0 <= tmp8) & (tmp8 < 1353406)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp8 < 1353406")
        # tmp10 = tl.load(in_ptr1 + (x1 + 16*tmp8), r2_mask & x1_mask & y0_mask)
        # 用下面这行替换上述6行 SIMT
        tmp8 = tl.reshape(tmp4, [Y0BLOCK_SUB, R1BLOCK_SUB])
        tmp10 = tl.index_select(in_ptr1, tmp8, EMBEDDING_SIZE, X2BLOCK_SUB, (y0_offset + (loop_y0 * Y0BLOCK_SUB), 0, 0), (y0_numel, r1_numel, x2_numel))
        tmp14 = tl.sum(tmp10, 1).reshape(Y0BLOCK_SUB, 1, X2BLOCK_SUB)
        tl.store(out_ptr0 + (x2 + 16*y0 ), tmp14, x2_mask & y0_mask)

def triton_func(arg34_1: torch.Tensor, arg35_1: torch.Tensor, buf0: torch.Tensor):
    y0_size, _ = arg34_1.size()
    grid = lambda meta: (triton.cdiv(y0_size, meta['Y0BLOCK']), )
    triton_unk_fused_embedding_eq_sum_where_zeros_like_0[grid](arg34_1, arg35_1, buf0, y0_numel, r1_numel, x2_numel)
    return buf0


def torch_func(x0: torch.Tensor):
  return torch.sqrt(x0)

torch.manual_seed(0)

arg34_1 = rand_strided((y0_numel, r1_numel), (r1_numel, 1), device='npu', dtype=torch.int64)
arg35_1 = rand_strided((embedding_size, x2_numel), (x2_numel, 1), device='npu', dtype=torch.float32)
buf0 = empty_strided((y0_numel, x2_numel), (x2_numel, 1), device='npu', dtype=torch.float32)

output_triton = triton_func(arg34_1, arg35_1, buf0)
print("triton = ", output_triton)

# output_torch = torch_func(x0)
# print("torch = ", output_torch)
# torch.testing.assert_close(output_triton.cpu(), output_torch.cpu())

# def wrapper_func(x0, x1):
#     torch_ref = torch_func(x0, x1)
#     triton_cal = triton_func(x0, x1)

# profiler_wrapper(wrapper_func, x0, x1)
