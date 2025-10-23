import sys
import pytest
import torch

import triton
import triton.language as tl


sys.path.append("..")
import test_common


#source python\sglang\srt\layers\moe\ep_moe\kernels.py
@triton.jit
def deepgemm_compute_src2dst_triton_kernel(
    topk_ids,
    reorder_ids,
    seg_indptr,
    src2dst,
    m_max,
    num_toks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    expert_id = tl.load(topk_ids + src_id, mask=(src_id < num_toks))
    expert_dst_start = tl.load(seg_indptr + expert_id)
    expert_dst_offset = dst_id - expert_dst_start
    dst_id = expert_id * m_max + expert_dst_offset
    tl.store(src2dst + src_id, dst_id, mask=mask)


def test_context_fwd_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")


    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')
    deepgemm_compute_src2dst_triton_kernel[data['grid']](**input_data) 
    
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")