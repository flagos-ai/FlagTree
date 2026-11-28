# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block FP8 quantization with a power-of-two scale rounded upward."""


def ue8m0_quantize(grouped):
    """Quantize BF16 values carried in FP32 along the last axis.

    Retain the scale's broadcast dimension. The scale floor is applied after
    dividing the group's absolute maximum by 448. NaNs do not contribute to the maximum and clamp to -448 in the
    normalized values, matching fmin(fmax(value, -448), 448).
    """
    import torch

    value = grouped.float()
    maximum = torch.where(torch.isnan(value), 0., value.abs()).amax(-1, keepdim=True)
    raw = (maximum / maximum.new_tensor(448.)).clamp_min(1e-10)
    # Positive normal FP32 bit patterns are ordered. Clearing the fraction
    # after adding its mask implements ceil(log2) without logarithm error at
    # exact powers of two. The scale floor excludes subnormal exponents.
    scale = ((raw.contiguous().view(torch.int32) + 0x7fffff) & 0x7f800000).view(torch.float32)
    normalized = torch.nan_to_num(value / scale, nan=-448., posinf=448., neginf=-448.).clamp(-448., 448.)
    return normalized.to(torch.float8_e4m3fn), scale
