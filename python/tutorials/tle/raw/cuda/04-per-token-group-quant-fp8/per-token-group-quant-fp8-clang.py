from pathlib import Path

from typing import Optional
import logging
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

from triton.language.extra.cuda import libnvshmem_device

torch.cuda.set_device(1)
DEVICE = triton.runtime.driver.active.get_active_torch_device()
logger = logging.getLogger(__name__)


@dialect(name="cuda", file=(Path(__file__).parent / "per-token-group-quant-fp8.cu").resolve())
         # library={"torch": "/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/torch/"})
def edsl(*args, **kwargs):
    ...


@triton.jit
def test_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    group_size,
    num_groups,
    groups_per_block,
    eps,
    fp8_min,
    fp8_max,
):
    tle_raw.call(edsl, [x_q_ptr, x_s_ptr, x_ptr, group_size, num_groups, groups_per_block, eps, fp8_min, fp8_max])
    # libnvshmem_device.per_token_group_quant_8bit(x_ptr, x_q_ptr, x_s_ptr, group_size, num_groups, groups_per_block, eps,
    #                                              fp8_min, fp8_max)


def get_groups_per_block(num_groups: int) -> int:
    # Removing this branch gives better performance.
    # if (num_groups % 16 == 0):
    #     return 16
    if (num_groups % 8 == 0):
        return 8
    elif (num_groups % 4 == 0):
        return 4
    elif (num_groups % 2 == 0):
        return 2
    else:
        return 1


def per_token_group_quant_fp8_tle(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    # column_major_scales: bool = False,
    # scale_ue8m0: bool = False,
):
    logger.debug("GEMS PER TOKEN GROUP QUANT FP8")
    assert x.shape[-1] % group_size == 0, (f"the last dimension of `x` {x.shape[-1]} must be divisible "
                                           f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    fp8_dtype = torch.float8_e4m3fn if dtype is None else dtype
    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    shape = x.shape[:-1] + (x.shape[-1] // group_size, )
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # num_groups
    num_groups = x.numel() // group_size
    groups_per_block = get_groups_per_block(num_groups)

    # num_blocks
    THREADS_PER_GROUP = 16
    num_blocks = num_groups // groups_per_block
    num_warps = max(groups_per_block * THREADS_PER_GROUP // 32, 1)

    # The .cu device function uses `extern __shared__` for groups_per_block * group_size floats.
    # Triton's compiler cannot infer this smem requirement from the extern_call, so we patch
    # packed_metadata after warmup compilation to include the extra shared memory bytes.
    smem_bytes = groups_per_block * group_size * x.element_size()  # float32 = 4 bytes

    kernel = test_kernel.run(
        x,
        x_q,
        x_s,
        group_size,
        num_groups,
        groups_per_block,
        eps,
        fp8_min,
        fp8_max,
        grid=(num_blocks, ),
        warmup=True,
        num_warps=num_warps,
    )

    # Resolve async future if needed
    if hasattr(kernel, "result"):
        kernel = kernel.result()

    old_meta = kernel.packed_metadata
    new_shared = max(old_meta[2], smem_bytes)
    kernel.packed_metadata = old_meta[:2] + (new_shared, ) + old_meta[3:]

    test_kernel[(num_blocks, )](
        x,
        x_q,
        x_s,
        group_size,
        num_groups,
        groups_per_block,
        eps,
        fp8_min,
        fp8_max,
        num_warps=num_warps,
    )

    return x_q, x_s


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8_triton(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
):
    logger.debug("GEMS PER TOKEN GROUP QUANT FP8")
    # dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
    fp8_dtype = torch.float8_e4m3fn if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (f"the last dimension of `x` {x.shape[-1]} must be divisible "
                                           f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // group_size
    N = group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_fp8[(M, )](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[1],
        x.stride(0),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        scale_ue8m0=scale_ue8m0,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


if __name__ == "__main__":
    x = torch.randn((16384, 32768), device=DEVICE, dtype=torch.float32)
    group_size = 128

    x_q_triton, x_s_triton = per_token_group_quant_fp8_tle(x, group_size)
    # x_q_triton, x_s_triton = per_token_group_quant_fp8_triton(x, group_size)
    # x_q_tle, x_s_tle = per_token_group_quant_fp8_tle(x, group_size)

    # q_tri = x_q_triton.to(torch.float32)
    # q_tle = x_q_tle.to(torch.float32)

    # b_tri = x_q_triton.view(torch.int8).to(torch.int16)  # promote to avoid int8 overflow
    # b_tle = x_q_tle.view(torch.int8).to(torch.int16)
    # bit_diff = (b_tri - b_tle).abs()  # 0 = exact match, 1 = 1-ULP, >1 = real bug
    # num_exact = bit_diff.eq(0).sum().item()
    # num_1ulp = bit_diff.eq(1).sum().item()
    # num_beyond = bit_diff.gt(1).sum().item()
    # if num_beyond == 0:
    #     if num_1ulp == 0:
    #         print("✅ x_q Triton and TLE match (bit-exact)")
    #     else:
    #         print(f"✅ x_q Triton and TLE match (1-ULP diff={num_1ulp}, "
    #               f"expected from RTZ vs RTNE rounding)")
    # else:
    #     q_tri = x_q_triton.to(torch.float32)
    #     q_tle = x_q_tle.to(torch.float32)
    #     float_diff = (q_tri - q_tle).abs()
    #     print(f"❌ x_q Triton and TLE differ: "
    #           f"exact={num_exact}, 1-ULP={num_1ulp}, >1-ULP={num_beyond}")
    #     beyond_idx = bit_diff.gt(1).nonzero()
    #     for idx in beyond_idx[:10]:
    #         r, c = idx[0].item(), idx[1].item()
    #         group_id = r * (q_tri.shape[1] // group_size) + c // group_size
    #         pos_in_group = c % group_size
    #         print(f"  [{r},{c}] group={group_id} pos_in_group={pos_in_group}"
    #               f"  triton_bits={b_tri[r,c].item():4d}  tle_bits={b_tle[r,c].item():4d}"
    #               f"  bit_diff={bit_diff[r,c].item()}"
    #               f"  triton={q_tri[r,c].item():.1f}  tle={q_tle[r,c].item():.1f}"
    #               f"  float_diff={float_diff[r,c].item():.1f}"
    #               f"  x={x[r,c].item():.6f}  scale={x_s_triton.view(-1)[group_id].item():.8f}")

    # if torch.allclose(x_s_triton, x_s_tle, atol=0.125, rtol=0):
    #     print("✅ x_s Triton and TLE match")
    # else:
    #     print("❌ x_s Triton and TLE differ")

    # # perf
    # shapes = [(1024, 2048), (2048, 4096), (4096, 8192), (16384, 32768)]
    # group_sizes = [64, 128]
    # dtypes = [torch.float32]
    # for shape in shapes:
    #     for dtype in dtypes:
    #         for group_size in group_sizes:
    #             x = torch.rand(shape, device=DEVICE, dtype=dtype)
    #             mean_ms_triton = triton.testing.do_bench(lambda: per_token_group_quant_fp8_triton(x, group_size))
    #             mean_ms_tle = triton.testing.do_bench(lambda: per_token_group_quant_fp8_tle(x, group_size))
                
    #             print(f"\n=========  Shape: {shape}   Type: {dtype}   Group size: {group_size}  =========")
    #             print(f"Triton time: {mean_ms_triton:.4f} ms")
    #             print(f"TLE time: {mean_ms_tle:.4f} ms")
