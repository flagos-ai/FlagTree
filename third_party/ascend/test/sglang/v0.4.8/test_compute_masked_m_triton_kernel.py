import sys
import pytest
import torch

import triton
import triton.language as tl

sys.path.append("..")
import test_common


#source python\sglang\srt\layers\moe\ep_moe\kernels.py
@triton.jit
def compute_masked_m_triton_kernel(seg_indptr, masked_m):
    expert_id = tl.program_id(0)
    start = tl.load(seg_indptr + expert_id)
    end = tl.load(seg_indptr + expert_id + 1)
    tl.store(masked_m + expert_id, (end - start))


def test_context_fwd_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")


    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')
    compute_masked_m_triton_kernel[data['grid']](**input_data) 
    
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")