# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dynamic call records retain each view's proven alignment."""
import ast

import pytest
from triton.flagmega.codegen.triton.function_pool_abi import (
    function_pool_aliases,
    plan_function_pool_abis,
)


def bindings(weight_alignment=128, scale_alignment=16):
    events = []
    for index in range(2):
        edges = []
        for name, dtype, itemsize, start, alignment in (
            ("weight", "bfloat16", 2, 128, weight_alignment),
            ("scale", "float32", 4, 256, scale_alignment),
        ):
            edges.append({"formal": name, "actual_runtime_value_kind": "pointer",
                          "actual_runtime_argument": "rdata", "actual_abi": {
                              "storage": "rdata", "pooled": True, "scalar_dtype": dtype,
                              "scalar_itemsize": itemsize, "pool_byte_offset": start + index * 256,
                              "physical_buffer": name, "alignment_bytes": alignment}})
        events.append({"kind": "function_call", "callee": "worker", "call": f"call{index}",
                       "arguments": edges, "results": [], "memory_pools": []})
    return {
        "main": {"function": "main", "arguments": [], "pools": [{"name": "rdata", "storage": "rdata"}],
                 "call_abi": {"events": events}},
        "worker": {"function": "worker", "arguments": [{"name": "weight", "buffer": "weight"},
                                                         {"name": "scale", "buffer": "scale"}],
                   "pools": [{"name": "rdata", "storage": "rdata"}], "call_abi": {"events": []}},
    }


def alignment_hint(expression):
    hints = [node for node in ast.walk(ast.parse(expression, mode="eval"))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "multiple_of"]
    return ast.literal_eval(hints[0].args[1]) if hints else 1


def test_call_record_deltas_keep_view_alignment_instead_of_anchor_alignment():
    plans, _ = plan_function_pool_abis(bindings())
    plan = plans["worker"]
    weight, scale, _ = plan["bindings"]
    assert weight["record_column"] == scale["record_column"]
    aliases = function_pool_aliases(plan)
    assert alignment_hint(aliases["weight"]) == 128
    assert alignment_hint(aliases["scale"]) == 16


def test_dynamic_bank_row_stride_reduces_the_alignment_proof():
    plans, _ = plan_function_pool_abis(bindings(), row_strides={"weight": 256, "scale": 12})
    aliases = function_pool_aliases(plans["worker"])
    assert alignment_hint(aliases["weight"]) == 128
    assert alignment_hint(aliases["scale"]) == 4


def test_weaker_caller_view_and_unknown_alignment_are_not_strengthened():
    values = bindings(2, 1)
    values["main"]["call_abi"]["events"][1]["arguments"][0]["actual_abi"]["alignment_bytes"] = 1
    plans, _ = plan_function_pool_abis(values)
    aliases = function_pool_aliases(plans["worker"])
    assert alignment_hint(aliases["weight"]) == 1
    assert alignment_hint(aliases["scale"]) == 1


@pytest.mark.parametrize("alignment", [0, 3, True])
def test_invalid_alignment_contract_is_rejected(alignment):
    from triton.flagmega.errors import CodegenError
    with pytest.raises(CodegenError, match="alignment"):
        plan_function_pool_abis(bindings(alignment))


def test_function_frames_keep_the_weakest_proven_call_alignment():
    values = bindings()
    for name in values:
        values[name]["pools"].append({"name": "workspace", "storage": "workspace"})
    for index, event in enumerate(values["main"]["call_abi"]["events"]):
        event["memory_pools"] = [{"memory_space": "workspace", "offset": 1024 * (index + 1),
                                  "alignment_bytes": 128 if index == 0 else 32}]
    plans, _ = plan_function_pool_abis(values)
    assert alignment_hint(function_pool_aliases(plans["worker"])["workspace"]) == 32


@pytest.mark.parametrize("scope_count,expected", [(1, 64), (128, 1)])
def test_call_event_serializes_the_physical_frame_alignment(scope_count, expected):
    from types import SimpleNamespace

    from triton.flagmega.codegen.triton.call_abi import _nested_event
    from triton.flagmega.ir.bufferization.memory_pool import CallMemoryPoolBinding

    plan = SimpleNamespace(physical_buffer_map={"frame": SimpleNamespace(alignment=64)})
    event = _nested_event(plan, "call", "worker", (), (),
                          (CallMemoryPoolBinding("workspace", "frame", 192, 128),),
                          {"workspace": scope_count})
    assert event["memory_pools"][0]["alignment_bytes"] == expected
