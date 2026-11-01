# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete immutable call domains permit signed int64 affine addresses."""

import ast
import copy

import pytest

from triton.flagmega.codegen.triton.function_pool_abi import function_pool_aliases, plan_function_pool_abis
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.triton.pipelines.test_function_pool_alignment import bindings


def test_complete_affine_domain_preserves_i64_offsets_and_dynamic_record():
    values = bindings()
    for event in values["main"]["call_abi"]["events"]:
        for edge in event["arguments"]:
            edge["actual_abi"]["pool_byte_offset"] += 2**40
    plans, tables = plan_function_pool_abis(values)
    plan = plans["worker"]
    table = tables[0]["values"]
    affine = [field for field in plan["bindings"] if "record_column" in field]
    assert affine and all("record_affine" in field for field in affine)
    for field in affine:
        proof = field["record_affine"]
        assert proof["base"] > 2**32 and proof["record_width"] == plan["record_width"]
        for call in plan["calls"]:
            index = (call["record_offset"] - proof["record_start"]) // proof["record_width"]
            assert proof["base"] + index * proof["stride"] == table[call["record_offset"] + field["record_column"]]
    aliases = function_pool_aliases(plan)
    for field in affine:
        assert "_flagmega_function_offsets" not in aliases[field["formal"]]
        assert "tl.cast(_flagmega_call_offsets, tl.int64)" in aliases[field["formal"]]


def test_non_affine_record_column_retains_ordinary_table_lookup():
    values = bindings()
    event = copy.deepcopy(values["main"]["call_abi"]["events"][-1])
    event["call"] = "call2"
    for edge in event["arguments"]:
        edge["actual_abi"]["pool_byte_offset"] += 384
    values["main"]["call_abi"]["events"].append(event)
    plans, _ = plan_function_pool_abis(values)
    plan = plans["worker"]
    assert plan["record_width"] == 1
    assert all("record_affine" not in field for field in plan["bindings"])
    alias = function_pool_aliases(plan)["weight"]
    assert "_flagmega_function_offsets" in alias
    loads = [node for node in ast.walk(ast.parse(alias, mode="eval"))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
    assert len(loads) == 1
    assert not any(keyword.arg == "volatile" and isinstance(keyword.value, ast.Constant)
                   and keyword.value.value for keyword in loads[0].keywords)


def test_negative_stride_uses_the_complete_call_domain():
    values = bindings()
    events = values["main"]["call_abi"]["events"]
    events[0]["arguments"], events[1]["arguments"] = events[1]["arguments"], events[0]["arguments"]
    plans, tables = plan_function_pool_abis(values)
    plan = plans["worker"]
    proof = next(field["record_affine"] for field in plan["bindings"] if "record_column" in field)
    assert proof["stride"] == -256
    for index, call in enumerate(plan["calls"]):
        assert proof["base"] + index * proof["stride"] == tables[0]["values"][call["record_offset"]]


@pytest.mark.parametrize("field,value", [("record_width", 2), ("records", 5), ("record_start", True)])
def test_inconsistent_affine_proofs_are_rejected(field, value):
    plans, _ = plan_function_pool_abis(bindings())
    plan = plans["worker"]
    binding = next(binding for binding in plan["bindings"] if "record_column" in binding)
    binding["record_affine"][field] = value
    with pytest.raises(CodegenError, match="Affine record proof"):
        function_pool_aliases(plan)
