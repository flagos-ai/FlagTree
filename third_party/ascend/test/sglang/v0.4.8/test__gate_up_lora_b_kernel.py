import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: python/sglang/srt/lora/triton_ops/gate_up_lora_b.py
@triton.jit
def _gate_up_lora_b_kernel(
    x,
    weights,
    output,
    K,  
    output_dim,
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    fuse_scaling_add,
    scalings,
):
    batch_id = tl.program_id(axis=2)
    gate_up_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)

    w_index = tl.load(weight_indices + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = gate_up_id * output_dim  
    rank = tl.load(lora_ranks + w_index)
    scaling = tl.load(scalings + w_index)

    K = tl.minimum(K, rank)

    num_pid_n = tl.cdiv(output_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (x + seg_start * x_stride_0 + (gate_up_id * K) * x_stride_1) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    
    for_num = tl.cdiv(K, BLOCK_K)
    k = 0
    x_tile = tl.load(
        x_ptrs,
        mask=(s_offset[:, None] < seg_len)
        and (k_offset[None, :] < K - k * BLOCK_K),
        other=0.0,
    )
    w_tile = tl.load(
        w_ptrs,
        mask=(k_offset[:, None] < K - k * BLOCK_K)
        and (n_offset[None, :] < output_dim),
        other=0.0,
    )
    partial_sum += tl.dot(x_tile, w_tile)
    output_ptr = (output + seg_start * output_stride_0 + n_start * output_stride_1) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    
    output_mask = (s_offset[:, None] < seg_len) and (n_offset[None, :] < output_dim)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def test__gate_up_lora_b_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    _gate_up_lora_b_kernel[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")
