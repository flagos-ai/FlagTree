import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: python\sglang\srt\layers\elementwise.py
@triton.jit
def fused_dual_residual_rmsnorm_kernel(
    output_ptr,
    mid_ptr,
    activ_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(
        mid_ptr + input_start + offsets,
        a2r,
        mask=mask,
    )

    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)

    tl.store(
        output_ptr + input_start + offsets,
        a2r / rms2 * w2,  # implicitly casts to output dtype here
        mask=mask,
    )


def test_fused_dual_residual_rmsnorm_kernel(ptfile_path):
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

    fused_dual_residual_rmsnorm_kernel[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")