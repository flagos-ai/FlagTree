import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common

# source : python\sglang\srt\layers\moe\ep_moe\kernels.py
@triton.jit
def create_extend_after_decode_spec_info(
    verified_id_ptr,
    seq_lens_ptr,
    accept_lens_ptr,
    positions_ptr,
    new_verified_id_ptr,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens_ptr + pid)
    accept_length = tl.load(accept_lens_ptr + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens_ptr + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions_ptr + accept_len_cumsum
    mask = offsets < accept_length
    tl.store(positions_ptr + offsets, seq_length - accept_length + offsets, mask)

    accept_len_cumsum += accept_length - 1
    verified_id_data = tl.load(verified_id_ptr + accept_len_cumsum)
    tl.store(new_verified_id_ptr + pid, verified_id_data)


def test_create_extend_after_decode_spec_info(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    # ptfile format:
    # [input_data] (dict):
    #     key : value
    # [gpu_output] (dict):
    #     key : value
    # [grid] :
    #     (1,)
    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    create_extend_after_decode_spec_info[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")