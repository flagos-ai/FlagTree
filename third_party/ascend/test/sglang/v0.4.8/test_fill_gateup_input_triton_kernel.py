import sys
import pytest
import torch

import triton
import triton.language as tl

sys.path.append("..")
import test_common

#source python\sglang\srt\layers\moe\ep_moe\kernels.py
@triton.jit
def fill_gateup_input_triton_kernel(
    input_ptr,
    scale_ptr,
    gateup_input_ptr,
    gateup_input_scale_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    m_max,
    hidden_size,
    scale_size,
    BLOCK_SIZE: tl.constexpr,
):

    src_idx_int32 = tl.program_id(0)
    src_idx = src_idx_int32.to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    scale_src_ptr = scale_ptr + src_idx * scale_size

    vec = tl.arange(0, BLOCK_SIZE)
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        if expert_id >= start_expert_id and expert_id <= end_expert_id:
            dst_idx_int32 = tl.load(src2dst_ptr + idx)
            dst_idx = dst_idx_int32.to(tl.int64)
            dst_idx = dst_idx - start_expert_id * m_max
            dst_ptr = gateup_input_ptr + dst_idx * hidden_size
            for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                offset = start_offset + vec
                mask = offset < hidden_size
                in_data = tl.load(src_ptr + offset, mask=mask)
                tl.store(dst_ptr + offset, in_data, mask=mask)
            scale_dst_ptr = gateup_input_scale_ptr + dst_idx * scale_size
            for start_offset in tl.range(0, scale_size, BLOCK_SIZE):
                offset = start_offset + vec
                mask = offset < scale_size
                in_scale = tl.load(scale_src_ptr + offset, mask=mask)
                tl.store(scale_dst_ptr + offset, in_scale, mask=mask)
               


def test_context_fwd_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")


    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')
    fill_gateup_input_triton_kernel[data['grid']](**input_data) 
    
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")