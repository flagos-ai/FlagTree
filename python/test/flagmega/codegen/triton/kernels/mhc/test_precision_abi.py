# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit numerical contracts cannot be silently dropped when resuming TIR."""

from copy import deepcopy

import pytest
from triton.flagmega.codegen.triton.mhc_renderer import mhc_post_call, mhc_pre_call
from triton.flagmega.errors import CodegenError

from python.test.flagmega.codegen.triton.kernels.mhc.test_mhc_packed_abi import _raw


@pytest.mark.parametrize("attrs", [{"collapsed_dtype": "int32"}, {"collapse_input_dtype": "int32"},
                                    {"collapse_reduction": "unknown"}, {"collapse_reduction": []},
                                    {"projection_reduction": "unknown"}, {"projection_splits": True},
                                    {"projection_reduction": "stream_fma_128", "projection_splits": 8}])
def test_pre_renderer_rejects_unsupported_precision_attrs(attrs):
    raw = _raw()
    raw["semantic_attrs"].update(attrs)
    with pytest.raises(CodegenError):
        mhc_pre_call(raw)


def _post_raw():
    pre = _raw(())
    values = [pre["outputs"][2], pre["inputs"][0], pre["outputs"][0], pre["outputs"][1]]
    result = deepcopy(pre["inputs"][0])
    result["formal"] = "result"
    for value, name in zip(values, ("value", "residual", "post", "comb"), strict=True):
        value["formal"] = name
    return {"inputs": values, "outputs": [result], "semantic_attrs": {},
            "parameters": {"elements_per_program": 256}}


def test_post_renderer_verifies_the_requested_store_dtype():
    raw = _post_raw()
    assert mhc_post_call(raw)["result_type"] == "tl.bfloat16"
    raw["semantic_attrs"]["output_dtype"] = "float32"
    with pytest.raises(CodegenError, match="output_dtype"):
        mhc_post_call(raw)
    raw["outputs"][0]["buffers"][0]["abi"]["scalar_dtype"] = "float32"
    assert mhc_post_call(raw)["result_type"] == "tl.float32"


@pytest.mark.parametrize("attrs", [{"output_dtype": "int32"}, {"accumulation_mode": "unknown"},
                                    {"accumulation_mode": []}])
def test_post_renderer_rejects_unsupported_precision_attrs(attrs):
    raw = _post_raw()
    raw["semantic_attrs"].update(attrs)
    with pytest.raises(CodegenError):
        mhc_post_call(raw)
