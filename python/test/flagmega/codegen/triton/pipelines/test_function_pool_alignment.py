# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dynamic call records retain each view's proven alignment."""
import ast
from types import SimpleNamespace

import pytest
from triton.flagmega.codegen.triton.function_pool_abi import (
    bank_row_strides,
    function_pool_aliases,
    function_pool_actuals,
    plan_function_pool_abis,
)


@pytest.mark.parametrize("row_index", (None, "trip"))
def test_call_record_crosses_the_function_boundary_as_a_scalar(row_index):
    values = bindings()
    plans, _ = plan_function_pool_abis(values)
    actuals = function_pool_actuals(values["main"], SimpleNamespace(call_id="call1"),
        plans["worker"], {"rdata": "rdata"}, has_tables=True, row_index=row_index)
    record = ast.parse(actuals[-1], mode="eval").body
    assert isinstance(record, ast.Call)
    assert isinstance(record.func, ast.Attribute) and record.func.attr == "cast"
    # No table pointer arithmetic crosses the caller's boundary, including
    # dynamic bank rows; the callee resolves the scalar invocation record.
    assert "_flagmega_function_offsets" not in actuals[-1]
    assert "trip" in actuals[-1] if row_index else ast.literal_eval(record.args[0]) == 1
    alias = function_pool_aliases(plans["worker"])["weight"]
    assert "tl.cast(_flagmega_call_offsets, tl.int64)" in alias


def test_scalar_call_records_read_distinct_pooled_views_on_sm90(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    import importlib.util
    values = bindings()
    plans, tables = plan_function_pool_abis(values)
    aliases = function_pool_aliases(plans["worker"])
    # Execute the generated address expressions through a genuine noinline
    # boundary. Repeated calls retain one table base and scalar row indices.
    source = f'''import triton
import triton.language as tl
@triton.jit(noinline=True)
def role(_flagmega_call_pool_rdata, _flagmega_function_offsets, _flagmega_call_offsets, output, index):
    weight = tl.load({aliases["weight"]}).to(tl.float32)
    scale = tl.load({aliases["scale"]})
    tl.store(output + index, weight + scale)
@triton.jit
def kernel(data, offsets, output):
    for index in range(64):
        role(data, offsets, (index % 2).to(tl.int64), output, index)
'''
    path = tmp_path / "pool_record_kernel.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("pool_record_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    for offset, number in ((128, 3.5), (384, 7.0)):
        data[offset:offset + 2].copy_(torch.tensor([number], dtype=torch.bfloat16, device="cuda").view(torch.uint8))
    for offset, number in ((256, 5.0), (512, 9.0)):
        data[offset:offset + 4].copy_(torch.tensor([number], dtype=torch.float32, device="cuda").view(torch.uint8))
    offsets = torch.tensor(tables[0]["values"], dtype=torch.int64, device="cuda")
    output = torch.empty(64, dtype=torch.float32, device="cuda")
    compiled = module.kernel[(1,)](data, offsets, output, num_warps=4)
    torch.testing.assert_close(output, torch.tensor([8.5, 16.0], device="cuda").repeat(32), rtol=0, atol=0)
    assert compiled.n_spills == 0


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


def test_one_bank_field_still_carries_its_dynamic_row_address():
    values = bindings()
    values["worker"]["arguments"] = values["worker"]["arguments"][:1]
    values["worker"]["pools"] = []
    values["main"]["call_abi"]["events"] = values["main"]["call_abi"]["events"][:1]
    values["main"]["call_abi"]["events"][0]["arguments"] = values["main"]["call_abi"]["events"][0]["arguments"][:1]
    plans, _ = plan_function_pool_abis(values, row_strides={"weight": 256})
    assert "worker" in plans, "A signature-size heuristic discarded the dynamic bank-row address"
    assert "_flagmega_bank_row_index" in function_pool_aliases(plans["worker"])["weight"]


def test_dynamic_bank_row_displacement_is_added_to_a_call_record():
    plans, _ = plan_function_pool_abis(bindings(), row_strides={"weight": 256})
    alias = function_pool_aliases(plans["worker"])["weight"]
    assert "_flagmega_call_offsets" in alias and "_flagmega_bank_row_index" in alias


def test_static_checkpoint_groups_are_not_dynamic_parameter_bank_rows():
    from types import SimpleNamespace

    from triton.flagmega import ir as fm

    schema = fm.ParameterBankSchema("rows", (fm.BankField("weight", fm.tensor_type("bfloat16", (32,)), "checkpoint"),))
    module = SimpleNamespace(nodes=(fm.Node("bank", "builtin.parameter_bank", (), schema.type,
                                           attrs={"schema": schema}),))
    buffers = []
    for name in ("bank.weight", "decoder.weight"):
        buffers.append(SimpleNamespace(rdata_group=name, mem_span=SimpleNamespace(
            size=SimpleNamespace(fixed_value=64), buffer=SimpleNamespace(id=name))))
    assert bank_row_strides(module, SimpleNamespace(buffers=buffers)) == {"bank.weight": 64}


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
