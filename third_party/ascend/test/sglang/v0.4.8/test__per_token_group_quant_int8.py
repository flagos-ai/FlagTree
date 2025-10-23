import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: python\sglang\srt\layers\quantization\int8_kernel.py
@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def test__per_token_group_quant_int8(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')
    _per_token_group_quant_int8[data["grid"]](**input_data)

    ref_s = data['gpu_output']['y_s_ptr'].cpu()
    cal_s = input_data['y_s_ptr'].cpu()
    torch.testing.assert_close(ref_s, cal_s, rtol=5e-03, atol=5e-03, equal_nan=True)

    #ensure the difference of y_q_ptr no more than 1
    ref_q = data['gpu_output']['y_q_ptr'].cpu()
    cal_q = input_data['y_q_ptr'].cpu()
    diff = torch.abs(ref_q - cal_q)
    diff_over_1 = diff > 1
    assert diff_over_1.sum().item() == 0