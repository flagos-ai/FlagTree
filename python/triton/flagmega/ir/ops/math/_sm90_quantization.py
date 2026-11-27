# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU bit model of SM90 reciprocal FP8 quantization for BF16 activations.

The two 128-entry tables encode the normalized-mantissa results of SM90
div.approx.ftz.f32(448, amax) and rcp.approx.ftz.f32(multiplier).
A BF16 normal amax has seven mantissa bits; exponent reconstruction plus
explicit FTZ/overflow covers its complete domain. These are arithmetic
constants, independent of model weights and workloads. Device kernels use
the corresponding instructions, not these tables. The native dense quantizer
uses a BF16 maximum, saturating FP8 conversion and FTZ float multiplication.
To regenerate the tables on SM90, enumerate BF16 bits 0x3f80 through 0x3fff,
store the two instruction results as FP32 bits, and run the complete finite
BF16-domain GPU regression before accepting regenerated constants.
"""

_MULTIPLIER_BITS = (
    0x43e00000, 0x43de4379, 0x43dc8dc9, 0x43dadec7, 0x43d9364d, 0x43d79436, 0x43d5f85c, 0x43d4629b,
    0x43d2d2d3, 0x43d148e0, 0x43cfc4a4, 0x43ce45fc, 0x43cccccd, 0x43cb58f7, 0x43c9ea5e, 0x43c880e5,
    0x43c71c72, 0x43c5bce9, 0x43c46231, 0x43c30c30, 0x43c1bacf, 0x43c06df6, 0x43bf258b, 0x43bde17c,
    0x43bca1af, 0x43bb6611, 0x43ba2e8c, 0x43b8fb0b, 0x43b7cb7d, 0x43b69fcc, 0x43b577e6, 0x43b453b9,
    0x43b33333, 0x43b21643, 0x43b0fcd7, 0x43afe6df, 0x43aed44a, 0x43adc50b, 0x43acb910, 0x43abb04a,
    0x43aaaaab, 0x43a9a824, 0x43a8a8a9, 0x43a7ac2a, 0x43a6b29b, 0x43a5bbee, 0x43a4c818, 0x43a3d70a,
    0x43a2e8ba, 0x43a1fd1c, 0x43a11423, 0x43a02dc4, 0x439f49f5, 0x439e68aa, 0x439d89d9, 0x439cad77,
    0x439bd37b, 0x439afbda, 0x439a268a, 0x43995382, 0x439882b9, 0x4397b426, 0x4396e7c0, 0x43961d7d,
    0x43955556, 0x43948f42, 0x4393cb38, 0x43930931, 0x43924924, 0x43918b0c, 0x4390cede, 0x43901495,
    0x438f5c29, 0x438ea592, 0x438df0cb, 0x438d3dcb, 0x438c8c8d, 0x438bdd09, 0x438b2f39, 0x438a8317,
    0x4389d89e, 0x43892fc6, 0x43888889, 0x4387e2e2, 0x43873ecb, 0x43869c3f, 0x4385fb37, 0x43855baf,
    0x4384bda1, 0x43842108, 0x438385df, 0x4382ec21, 0x438253c8, 0x4381bcd1, 0x43812735, 0x438092f2,
    0x43800000, 0x437edcbb, 0x437dbc09, 0x437c9de2, 0x437b823f, 0x437a6916, 0x4379525d, 0x43783e10,
    0x43772c23, 0x43761c91, 0x43750f52, 0x4374045b, 0x4372fba9, 0x4371f534, 0x4370f0f2, 0x436feedc,
    0x436eeef0, 0x436df122, 0x436cf56b, 0x436bfbca, 0x436b0433, 0x436a0ea0, 0x43691b10, 0x43682976,
    0x436739ce, 0x43664c16, 0x43656042, 0x43647650, 0x43638e3a, 0x4362a7f8, 0x4361c387, 0x4360e0e0,
)

_SCALE_BITS = (
    0x3b124925, 0x3b136db7, 0x3b149249, 0x3b15b6dc, 0x3b16db6e, 0x3b180000, 0x3b192492, 0x3b1a4925,
    0x3b1b6db7, 0x3b1c924a, 0x3b1db6db, 0x3b1edb6e, 0x3b200000, 0x3b212492, 0x3b224924, 0x3b236db7,
    0x3b249249, 0x3b25b6db, 0x3b26db6e, 0x3b280001, 0x3b292493, 0x3b2a4924, 0x3b2b6db8, 0x3b2c9249,
    0x3b2db6dc, 0x3b2edb6d, 0x3b300000, 0x3b312493, 0x3b324924, 0x3b336db7, 0x3b349249, 0x3b35b6dc,
    0x3b36db6e, 0x3b380000, 0x3b392492, 0x3b3a4925, 0x3b3b6db8, 0x3b3c9249, 0x3b3db6db, 0x3b3edb6d,
    0x3b400000, 0x3b412493, 0x3b424924, 0x3b436db7, 0x3b449249, 0x3b45b6dc, 0x3b46db6d, 0x3b480000,
    0x3b492493, 0x3b4a4924, 0x3b4b6db6, 0x3b4c9249, 0x3b4db6db, 0x3b4edb6d, 0x3b500000, 0x3b512492,
    0x3b524924, 0x3b536db6, 0x3b549249, 0x3b55b6dc, 0x3b56db6e, 0x3b580000, 0x3b592491, 0x3b5a4924,
    0x3b5b6db6, 0x3b5c9248, 0x3b5db6db, 0x3b5edb6d, 0x3b600001, 0x3b612492, 0x3b624925, 0x3b636db7,
    0x3b649249, 0x3b65b6dc, 0x3b66db6d, 0x3b680000, 0x3b692492, 0x3b6a4924, 0x3b6b6db7, 0x3b6c924a,
    0x3b6db6da, 0x3b6edb6c, 0x3b6fffff, 0x3b712492, 0x3b724924, 0x3b736db6, 0x3b749249, 0x3b75b6dc,
    0x3b76db6e, 0x3b780000, 0x3b792492, 0x3b7a4924, 0x3b7b6db7, 0x3b7c9248, 0x3b7db6dc, 0x3b7edb6c,
    0x3b800000, 0x3b809249, 0x3b812492, 0x3b81b6dc, 0x3b824925, 0x3b82db6d, 0x3b836db7, 0x3b840000,
    0x3b849249, 0x3b852493, 0x3b85b6db, 0x3b864925, 0x3b86db6e, 0x3b876db6, 0x3b87ffff, 0x3b889249,
    0x3b892492, 0x3b89b6db, 0x3b8a4925, 0x3b8adb6e, 0x3b8b6db6, 0x3b8c0001, 0x3b8c9248, 0x3b8d2492,
    0x3b8db6dc, 0x3b8e4924, 0x3b8edb6d, 0x3b8f6db7, 0x3b900000, 0x3b909249, 0x3b912492, 0x3b91b6dc,
)


def sm90_bf16_quantize(grouped):
    """Return E4M3FN codes and FP32 scales with a trailing singleton K axis."""
    import torch

    if grouped.dtype != torch.bfloat16:
        raise TypeError("SM90 reciprocal activation quantization requires BF16 input.")
    # Native BF16 hmax starts at zero and ignores a NaN operand.
    maximum = torch.nan_to_num(grouped.abs().float(), nan=0., posinf=float("inf")).amax(-1, keepdim=True)
    bits = maximum.bfloat16().view(torch.int16).to(torch.int32) & 0x7FFF
    exponent = ((bits >> 7) & 255) - 127
    mantissa = (bits & 127).long()
    multiplier_bits = torch.tensor(_MULTIPLIER_BITS, dtype=torch.int32, device=grouped.device)[mantissa]
    scale_bits = torch.tensor(_SCALE_BITS, dtype=torch.int32, device=grouped.device)[mantissa]
    multiplier_exponent = ((multiplier_bits >> 23) & 255) - exponent
    scale_exponent = ((scale_bits >> 23) & 255) + exponent
    multiplier_bits = (multiplier_bits & 0x7FFFFF) | (multiplier_exponent.clamp(1, 254) << 23)
    multiplier_bits = torch.where(multiplier_exponent >= 255, 0x7F800000, multiplier_bits)
    # Reciprocal output below the smallest normal is flushed before *448.
    multiplier_bits = torch.where(bits > 253 * 128, 0, multiplier_bits)
    multiplier_bits = torch.where(bits < 128, 0x7F800000, multiplier_bits)
    multiplier_bits = torch.where(bits == 0, 0x3F800000, multiplier_bits)
    scale_bits = (scale_bits & 0x7FFFFF) | (scale_exponent.clamp(1, 254) << 23)
    scale_bits = torch.where((scale_exponent <= 0) | (multiplier_bits == 0x7F800000), 0, scale_bits)
    scale_bits = torch.where(multiplier_bits == 0, 0x7F800000, scale_bits)
    scale_bits = torch.where(bits == 0, 0x3F800000, scale_bits)
    multiplier, scale = multiplier_bits.view(torch.float32), scale_bits.view(torch.float32)
    value = grouped.float()
    value = torch.where(value.abs() < torch.finfo(torch.float32).tiny,
                        torch.copysign(torch.zeros_like(value), value), value)
    normalized = value * multiplier
    # CUDA's saturating FP8 conversion canonicalizes the NaN sign.
    normalized = torch.where(normalized.isnan(), float("nan"), normalized)
    codes = normalized.clamp(-448., 448.).to(torch.float8_e4m3fn)
    return codes, scale

