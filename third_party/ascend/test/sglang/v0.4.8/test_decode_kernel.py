import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: sgl-kernel/benchmark/bench_lightning_attention_decode.py
@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    KV,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    d_original: tl.constexpr,
    e: tl.constexpr,
    e_original: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    s = tl.load(S + off_h)
    ratio = tl.exp(-s)

    d_idx = tl.arange(0, d)
    e_idx = tl.arange(0, e)

    # Create masks for original dimensions
    d_mask = d_idx < d_original
    e_mask = e_idx < e_original

    # Load with masking
    q = tl.load(Q + qk_offset + d_idx, mask=d_mask, other=0.0)
    k = tl.load(K + qk_offset + d_idx, mask=d_mask, other=0.0)
    v = tl.load(V + v_offset + e_idx, mask=e_mask, other=0.0)

    # Load KV with 2D masking
    kv = tl.load(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        mask=(d_mask[:, None] & e_mask[None, :]),
        other=0.0,
    )

    # Compute outer product using element-wise operations
    k_v_prod = k[:, None] * v[None, :]
    kv = ratio * kv + k_v_prod

    # Store KV with 2D masking
    tl.store(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        kv.to(KV.dtype.element_ty),
        mask=(d_mask[:, None] & e_mask[None, :]),
    )

    # Compute matrix-vector multiplication using element-wise operations and reduction
    o = tl.sum(q[:, None] * kv, axis=0)

    # Store output with masking
    tl.store(Out + o_offset + e_idx, o.to(Out.dtype.element_ty), mask=e_mask)


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

    _decode_kernel[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")
