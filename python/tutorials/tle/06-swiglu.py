# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.math as math
import numpy as np
from triton.backends.ascend.testing import do_bench_npu
import triton.experimental.tle as tle

np.random.seed(21)
DEVICE = "npu"
DEVICE_ID = 0
torch.manual_seed(20)
torch_npu.npu.set_device(int(DEVICE_ID))
torch.set_printoptions(sci_mode=False, precision=4, linewidth=300)
ascend_aiv_core_nums = triton.language.constexpr(48)


@triton.jit
def swiglu_kernel(
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    input_stride_m,
    output_stride_m,
    beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_SIZE_M
    for tile_m_idx in range(0, BLOCK_SIZE_M, TILE_SIZE_M):
        m_idx = m_start + tile_m_idx
        for tile_h_idx in range(0, H, TILE_SIZE_H):
            offs_m = m_idx + tl.arange(0, TILE_SIZE_M)
            offs_h = tile_h_idx + tl.arange(0, TILE_SIZE_H)
            mask_m = offs_m < M
            mask_h = offs_h < H
            mask = mask_m[:, None] & mask_h[None, :]
            input_offset = offs_m[:, None] * input_stride_m + offs_h[None, :]
            x_a = tl.load(input_a_ptr + input_offset, mask=mask, other=0.0)
            x_b = tl.load(input_b_ptr + input_offset, mask=mask, other=0.0)

            tmp = x_a * x_b
            sig = 1.0 / (1.0 + math.exp(-1 * x_a * beta))
            out = tmp * sig.to(tl.float16)

            output_offset = offs_m[:, None] * output_stride_m + offs_h[None, :]
            # tl.store(output_ptr + output_offset, out)
            out_buf = tle.dsa.to_buffer(out, space=tle.dsa.ascend.UB)
            with tle.dsa.hint(inter_no_alias=True):
                tle.dsa.copy(out_buf, output_ptr + output_offset, [TILE_SIZE_M, TILE_SIZE_H])


def swiglu(input_tensor: torch.Tensor, scalarValue: float) -> torch.Tensor:
    assert input_tensor.shape[-1] % 2 == 0, "The last dimension of the input tensor must be even."

    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)
    if (len(input_tensor.shape) > 2):
        input_2d = input_tensor.contiguous().view(-1, 2 * H)
    else:
        input_2d = input_tensor.contiguous()

    input_a, input_b = torch.split(input_2d, H, dim=1)
    output_2d = torch.empty(M, H, device=input_a.device, dtype=input_a.dtype)

    def get_core_num():
        try:
            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            return cores_dict["vector_core_num"]
        except (AttributeError, KeyError, TypeError):
            return None

    num_cores = 24 if get_core_num() is None else get_core_num()

    TILE_SIZE_M = min(M, 32)
    TILE_SIZE_H = min(H, 256)
    if (M * H < 256 * 64):
        num_cores = 1
    BLOCK_SIZE_M = int((M + num_cores - 1) // num_cores)
    swiglu_kernel[(num_cores, )](input_a, input_b, output_2d, M, H, input_a.stride(0), output_2d.stride(0),
                                 beta=scalarValue, BLOCK_SIZE_M=BLOCK_SIZE_M, TILE_SIZE_M=TILE_SIZE_M,
                                 TILE_SIZE_H=TILE_SIZE_H, multibuffer=True,
                                 limit_auto_multi_buffer_of_local_buffer="no-limit")
    return output_2d


def test_op(M, H, scalarValue):
    input_tensor = torch.empty((M, H), dtype=torch.float16).normal_(mean=0.0, std=0.5).requires_grad_().npu()
    triton_out = swiglu(input_tensor, scalarValue)
    npu_out = torch_npu.npu_swiglu(input_tensor.npu(), dim=-1)
    torch.testing.assert_close(triton_out, npu_out, rtol=1e-3, atol=1e-3, equal_nan=True)
    # print(npu_out)
    # print(triton_out)
    print("PASSED!")

    triton_time = do_bench_npu(lambda: swiglu(input_tensor, scalarValue), clear_l2_cache=True, collect_prof=False)
    print(f"Triton time: {triton_time:.2f} us")
    npu_time = do_bench_npu(lambda: torch_npu.npu_swiglu(input_tensor.npu(), dim=-1), clear_l2_cache=True,
                            collect_prof=False)
    print(f"NPU Swiglu time: {npu_time:.2f} us")


test_cases = [
    (4, 32),
    (256, 128),
    (8192, 128),
    (15000, 128),
    (2560, 2048),
    (15000, 2048),
]

if __name__ == "__main__":
    scalarValue = 1.0
    for i, (M, H) in enumerate(test_cases, start=1):
        print(f"====================第{i}次测试=================, M={M}, H={H}")
        test_op(M, H, scalarValue)
