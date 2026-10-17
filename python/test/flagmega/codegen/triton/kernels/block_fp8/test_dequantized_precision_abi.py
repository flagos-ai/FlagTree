# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Resumed kernel calls cannot ignore an operand precision contract."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_semantic_scale_geometry import _descriptor, _parameter


def _raw(attrs, *, dtype=fm.DType.BFLOAT16, variant="simt"):
    value = _descriptor("value", dtype, (1, 64))
    weight = _descriptor("weight", fm.DType.FLOAT8_E4M3FN, (64, 64))
    scale = _descriptor("scale", fm.DType.FLOAT32, (2, 2))
    output = _descriptor("output", dtype, (1, 64))
    return {"call": "linear", "family": "block_fp8", "variant": variant, "execution_kind": "local_shard",
            "parameters": {"tile_n": 16}, "semantic_attrs": {"weight_block_n": 32, "weight_block_k": 32, **attrs},
            "inputs": (_parameter("value", value), _parameter("weight", weight), _parameter("weight_scale", scale)),
            "outputs": (_parameter("result", output),), "workspaces": ()}


@pytest.mark.parametrize("attrs,dtype,variant", [
    ({"activation_quantization": "bad"}, fm.DType.BFLOAT16, "simt"),
    ({"activation_quantization": []}, fm.DType.BFLOAT16, "simt"),
    ({"weight_dequant_dtype": "float32"}, fm.DType.BFLOAT16, "simt"),
    ({"activation_quantization": "none"}, fm.DType.BFLOAT16, "mma"),
    ({"activation_quantization": "none", "weight_dequant_dtype": "bfloat16"}, fm.DType.FLOAT32, "mma"),
    ({"weight_dequant_dtype": "bfloat16"}, fm.DType.BFLOAT16, "mma"),
])
def test_raw_precision_cannot_silently_fall_back(attrs, dtype, variant):
    with pytest.raises(CodegenError):
        prepare_kernel_calls((_raw(attrs, dtype=dtype, variant=variant),), function_name="main")


def test_raw_mma_requires_and_passes_matching_16_bit_precision():
    raw = _raw({"activation_quantization": "none", "weight_dequant_dtype": "bfloat16"}, variant="mma")
    call, = prepare_kernel_calls((raw,), function_name="main")
    assert call["activation_quantization"] == "none"
    assert call["weight_dequant_type"] == "tl.bfloat16"


@pytest.mark.parametrize("splits", [0, -1, 3, True])
def test_dequantized_mma_rejects_invalid_partition_counts(splits):
    raw = _raw({"activation_quantization": "none", "weight_dequant_dtype": "bfloat16"}, variant="mma")
    raw["parameters"]["dequant_mma_splits"] = splits
    with pytest.raises(CodegenError, match="dequant_mma_splits"):
        prepare_kernel_calls((raw,), function_name="main")


@pytest.mark.parametrize("variant,attrs,splits", [
    ("simt", {"activation_quantization": "none", "weight_dequant_dtype": "bfloat16"}, 2),
    ("mma", {}, 2),
    ("mma", {"activation_quantization": "none", "weight_dequant_dtype": "bfloat16"}, 4),
])
def test_dequantized_partitions_require_matching_precision_and_whole_scale_blocks(variant, attrs, splits):
    raw = _raw(attrs, variant=variant)
    raw["parameters"]["dequant_mma_splits"] = splits
    with pytest.raises(CodegenError, match="dequant_mma_splits"):
        prepare_kernel_calls((raw,), function_name="main")


def test_dequantized_partition_choice_survives_resumed_tir_rendering():
    raw = _raw({"activation_quantization": "none", "weight_dequant_dtype": "bfloat16"}, variant="mma")
    raw["parameters"]["dequant_mma_splits"] = 2
    call, = prepare_kernel_calls((raw,), function_name="main")
    assert call["dequant_mma_splits"] == 2
