import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: sgl-kernel/tests/test_merge_state.py
@triton.jit
def state_get_lse(o, m, d):
    return m + tl.log2(d)


@triton.jit
def _test_state_get_lse_kernel(
    m_ptr,
    d_ptr,
    out_ptr,
    n_elements: tl.constexpr,
):
    pid = tl.program_id(0)
    mask = pid < n_elements
    m = tl.load(m_ptr + pid, mask = mask)
    d = tl.load(d_ptr + pid, mask = mask)
    lse = state_get_lse(None, m, d)
    tl.store(out_ptr + pid, lse, mask = mask)


def test_context_fwd_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    # ptfile format:
    # [input_data] (dict):
    # [gpu_output] (dict):
    # [grid] :
    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    _test_state_get_lse_kernel[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")
