# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Render every packed QKV branch using its selected implementation identity."""

import pytest

from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _FAMILY_ENCODERS,
    _qkv_parallel_linear_call,
)
from triton.flagmega.errors import CodegenError


@pytest.fixture
def raw_qkv_call(packed_qkv_simt_pipeline_module, monkeypatch):
    captured = []

    def capture(raw):
        captured.append(dict(raw))
        return _qkv_parallel_linear_call(raw)

    monkeypatch.setitem(_FAMILY_ENCODERS, "qkv_parallel_linear", capture)
    describe_tir_package(packed_qkv_simt_pipeline_module)
    raw, = captured
    return raw


@pytest.mark.parametrize("variant,order", [
    ("packed_fused_gemv", "relaxed"),
    ("packed_ordered_mma", "mma_k16"),
])
def test_direct_qkv_uses_selected_variant_without_requiring_pipeline(raw_qkv_call, variant, order):
    raw = {
        **raw_qkv_call, "variant": variant, "parameters": {"block_k": 128, "tile_n": 256}, "semantic_attrs":
        {**raw_qkv_call["semantic_attrs"], "reduction_order": order}
    }
    raw.pop("transfer_pipeline", None)
    raw.pop("shared_workspaces", None)
    call = _qkv_parallel_linear_call(raw)
    assert call["block_k"] == min(128, call["local_k_capacity"])
    assert call["tile_n"] == min(256, max(output["local_n_capacity"] for output in call["outputs"]))
    assert call["ordered_mma"] == (order == "mma_k16")
    assert len(call["outputs"]) == 3
    assert "host_tensor_descriptor_requests" not in call


def test_unknown_qkv_variant_is_rejected_at_the_abi_boundary(raw_qkv_call):
    with pytest.raises(CodegenError, match="has no local-shard template ABI"):
        _qkv_parallel_linear_call({**raw_qkv_call, "variant": "unsupported_qkv"})
