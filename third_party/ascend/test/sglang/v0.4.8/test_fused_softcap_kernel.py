import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common

#source: python\sglang\srt\layers\elementwise.py
@triton.jit
def fused_softcap_kernel(
    output_ptr,
    input_ptr,
    n_ele,
    softcap_const: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele
    x = tl.load(input_ptr + offsets, mask=mask)
    fx = x.to(tl.float32)
    fxs = fx / softcap_const
    exped = tl.exp(2 * fxs)
    top = exped - 1
    bottom = exped + 1
    output = top / bottom * softcap_const
    tl.store(output_ptr + offsets, output, mask=mask)

def test_fused_softcap_kernel(ptfile_path):
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

    fused_softcap_kernel[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")