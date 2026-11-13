import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import os
import triton.backends.ascend.runtime

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

try:
    from triton.backends.ascend.utils import is_compile_on_910_95
except Exception:

    def is_compile_on_910_95(_arch):
        return False


def get_grid():
    import triton.runtime.driver as driver
    num_cores = driver.active.utils.get_aivector_core_num()
    print(f"grid_size: {num_cores}")
    return num_cores


@triton.autotune(
    configs=[],
    key=[],
)
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


@pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="simt is support on Ascend 950",
)
@pytest.mark.parametrize(
    "param_list",
    [
        [128, 10000],
    ],
)
def test_reduce(param_list):
    y0_numel, x1_numel = param_list
    arg0_1 = torch.randn(y0_numel, x1_numel, dtype=torch.float32, device="npu")
    buf44 = torch.empty((y0_numel), dtype=torch.float32, device="npu")

    grid_size = get_grid()
    triton_unk_reduce[(grid_size, 1, 1)](
        arg0_1,
        buf44,
        y0_numel,
        x1_numel,
        compile_mode='simt_only',
    )
    torch_out = torch_reduce(arg0_1)
    torch.testing.assert_close(buf44, torch_out, rtol=1e-04, atol=1e-04, equal_nan=True)
