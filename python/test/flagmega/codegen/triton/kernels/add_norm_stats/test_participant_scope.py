# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError
from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model,
)


def _descriptor(name, dtype, shape):
    nbytes = dtype.itemsize
    for extent in shape:
        nbytes *= extent
    physical = fm.PhysicalBuffer(
        f"physical:{name}", "workspace", fm.dim(nbytes), 16
    )
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return fm.BufferDescriptor(
        name,
        dtype,
        shape,
        tuple(reversed(strides)),
        "workspace",
        16,
        fm.MemSpan(physical),
    )


def _parameter(formal, descriptor):
    return {
        "formal": formal,
        "buffers": ({
            "formal": formal,
            "actual": descriptor.id,
            "runtime_argument": descriptor.id,
            "runtime_value_kind": "pointer",
            "abi": describe_local_buffer_abi(descriptor),
        },),
    }


def _call(scope):
    value = _descriptor("value", fm.DType.BFLOAT16, (16,))
    stats = _descriptor("stats", fm.DType.FLOAT32, (1,))
    facts = {} if scope is None else {"participant_scope": scope}
    return prepare_kernel_calls(({
        "call": "add_stats",
        "family": "add_norm_stats",
        "variant": "persistent_rms",
        "execution_kind": "local_shard",
        "parameters": {"tile": 128, "owner_count": 128},
        "facts": facts,
        "semantic_attrs": {"axis": -1, "use_mean": False},
        "inputs": (
            _parameter("input", value),
            _parameter("addend", value),
        ),
        "outputs": (
            _parameter("result_0", value),
            _parameter("result_1", stats),
        ),
        "workspaces": (),
    },), function_name="main")[0]


def test_single_program_scope_is_an_enforced_implementation_fact():
    implementation = sm90_triton_implementation_model().implementation(
        "tir.add_norm_stats.persistent_rms"
    )

    assert implementation is not None
    assert "participant_scope" not in implementation.parameters
    assert implementation.facts["participant_scope"] == "single_program"

    template = Path(__file__).parents[6] / (
        "triton/flagmega/codegen/triton/kernels/add_norm_stats/_function.py.jinja"
    )
    source = template.read_text(encoding="utf-8")
    assert "if {{ call.participant_active }}:" in source
    assert "_flagmega_add_norm_stats_persistent_rms(" in source
    assert _call("single_program")["participant_active"] == "(shard_index == 0)"


def test_default_scope_keeps_all_programs_and_unknown_scope_is_rejected():
    assert _call(None)["participant_active"] == "True"
    with pytest.raises(CodegenError, match="participant_scope"):
        _call("one_warp_per_sm")
