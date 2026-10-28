# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A block-FP8 GEMV result follows its declared output_data_type."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_semantic_scale_geometry import _descriptor, _parameter


def _raw(*, output_dtype, attrs=None):
    value = _descriptor("value", fm.DType.BFLOAT16, (1, 64))
    weight = _descriptor("weight", fm.DType.FLOAT8_E4M3FN, (64, 64))
    scale = _descriptor("scale", fm.DType.FLOAT32, (2, 2))
    output = _descriptor("output", output_dtype, (1, 64))
    return {"call": "linear", "family": "block_fp8", "variant": "simt", "execution_kind": "local_shard",
            "parameters": {"tile_n": 16}, "semantic_attrs": {"weight_block_n": 32, "weight_block_k": 32,
                                                              **(attrs or {})},
            "inputs": (_parameter("value", value), _parameter("weight", weight), _parameter("weight_scale", scale)),
            "outputs": (_parameter("result", output),), "workspaces": ()}


def test_float32_output_data_type_renders_with_bfloat16_activation():
    # An output head with F32 logits from BF16 activations (e.g. Qwen3.8 IQ4).
    call, = prepare_kernel_calls((_raw(output_dtype=fm.DType.FLOAT32, attrs={"output_data_type": "float32"}), ),
                                 function_name="main")
    assert call["result"]


def test_default_output_still_follows_the_activation_dtype():
    call, = prepare_kernel_calls((_raw(output_dtype=fm.DType.BFLOAT16), ), function_name="main")
    assert call["result"]


@pytest.mark.parametrize("output_dtype,attrs", [
    (fm.DType.FLOAT32, None),
    (fm.DType.BFLOAT16, {"output_data_type": "float32"}),
])
def test_output_abi_must_match_the_declared_output_dtype(output_dtype, attrs):
    with pytest.raises(CodegenError, match="output_data_type"):
        prepare_kernel_calls((_raw(output_dtype=output_dtype, attrs=attrs), ), function_name="main")
