import sys
import pytest
import torch

import triton
import triton.language as tl

sys.path.append("..")
import test_common


#source python\sglang\srt\layers\moe\ep_moe\kernels.py
@triton.jit
def compute_m_num_tiles_indptr(
    m_num_tiles_indptr, seg_indptr, batch_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    for bs in range(batch_size):
        m = tl.load(seg_indptr + bs + 1) - tl.load(seg_indptr + bs)
        cur_num_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        pre_num_tiles = tl.load(m_num_tiles_indptr + bs)
        tl.store(m_num_tiles_indptr + bs + 1, pre_num_tiles + cur_num_tiles)


def test_context_fwd_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")


    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')
    compute_m_num_tiles_indptr[data['grid']](**input_data) 
    
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")