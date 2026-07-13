# SPDX-License-Identifier: MIT
"""Module A: design-doc debug collect example reaches IR and debug metadata.

The example in ``docs/debugger_design.md`` shows the intended frontend path:
``tl.debug_collect_start/end`` become ``flagtree_debug.collect_begin/end`` in
TTIR, then early debug passes consume those markers for later debugger stages.
"""
from __future__ import annotations

import importlib
import inspect
import json

import pytest

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton._C.libtriton.passes import flagtree_debug as fd
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend


def _ast_source(fn, signature, constexprs):
    params = inspect.signature(ASTSource).parameters
    if "constexprs" in params:
        return ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return ASTSource(fn=fn, signature=signature, constants=constexprs)


def _codegen_implementation(backend, options):
    try:
        return backend.get_codegen_implementation(options)
    except TypeError:
        return backend.get_codegen_implementation()


def _make_ir(source, target, options, codegen_fns, module_map, context):
    params = inspect.signature(source.make_ir).parameters
    if "target" in params:
        return source.make_ir(target, options, codegen_fns, module_map, context)
    return source.make_ir(options, codegen_fns, module_map, context)


def _add_stages(backend, stages, options, source):
    if "language" in inspect.signature(backend.add_stages).parameters:
        backend.add_stages(stages, options, source.language)
    else:
        backend.add_stages(stages, options)


def _find_tracked_op(rows, access_type):
    matches = [row for row in rows if row.get("accessType") == access_type]
    assert len(matches) == 1
    return matches[0]


@triton.jit
def _design_debug_kernel(x_ptr, y_ptr, a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    tl.debug_collect_start(level=1)

    y = a * b
    z = x + y

    tl.debug_collect_end()

    tl.store(y_ptr + offsets, z, mask=mask)


@triton.jit
def _memory_debug_kernel(x_ptr):
    offsets = tl.arange(0, 4)
    tl.debug_collect_start(level=1)
    x = tl.load(x_ptr + offsets)
    tl.store(x_ptr + offsets, x)
    tl.debug_collect_end()


@pytest.mark.module_a
@pytest.mark.module_a_a1
@pytest.mark.module_a_a2
def test_module_a_design_example_debug_flag_reaches_ascend_ir():
    importlib.import_module("triton.backends.ascend.compiler")

    target = GPUTarget("npu", "Ascend910B", 0)
    backend = make_backend(target)
    source = _ast_source(
        _design_debug_kernel,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "n": "i32",
        },
        {"BLOCK_SIZE": 16},
    )
    options = backend.parse_options(source.parse_options())
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    mod = _make_ir(source, target, options, _codegen_implementation(backend, options), backend.get_module_map(),
                   context)
    pre_debug_ttir = str(mod)

    assert fd.has_debug_collect_markers(mod) is True
    assert "flagtree_debug.collect_begin" in pre_debug_ttir
    assert "flagtree_debug.collect_end" in pre_debug_ttir
    assert "level = 1" in pre_debug_ttir

    metadata = {
        "hash": "module-a-design-example-ir-flag",
        "target": target,
        **options.__dict__,
    }
    stages = {}
    _add_stages(backend, stages, options, source)
    assert "ttir" in stages
    ttir_mod = stages["ttir"](mod, metadata)
    post_debug_ttir = str(ttir_mod)

    assert metadata["debug_enabled"] is True
    assert metadata["debug_protocol_version"] == 2
    assert metadata["debug_record_level"] == 1
    assert metadata["debug_addr_level"] == 0
    assert metadata["debug_export_mode"] == "POST_KERNEL_EXPORT"
    assert isinstance(metadata["debug_kernel_id"], int)
    assert metadata["debug_kernel_id"] != 0
    assert metadata["debug_launch_hidden_arg"] is False

    tracked_table = metadata["debug_tracked_table"]
    assert isinstance(tracked_table, list)
    assert tracked_table
    assert all("opId" in row and "scopeId" in row for row in tracked_table)

    assert fd.has_debug_collect_markers(ttir_mod) is False
    assert "flagtree_debug.collect_begin" not in post_debug_ttir
    assert "flagtree_debug.collect_end" not in post_debug_ttir
    assert "flagtree.debug.op_id" in post_debug_ttir
    assert "flagtree.debug.instrumented" in post_debug_ttir
    assert "flagtree.debug.enable_hidden_arg_abi = false" in post_debug_ttir
    assert "flagtree_debug.record_" not in post_debug_ttir

    expected_statements = {
        "arith.mulf": "y = a * b",
        "arith.addf": "z = x + y",
    }
    tracked_by_op_name = {row["mlirOpName"]: row for row in tracked_table}
    for op_name, statement in expected_statements.items():
        assert op_name in tracked_by_op_name
        assert tracked_by_op_name[op_name]["tritonStatement"] == statement
        assert tracked_by_op_name[op_name]["opCategory"] == ""
        assert tracked_by_op_name[op_name]["role"] == ""


@pytest.mark.module_a
@pytest.mark.module_a_a3
def test_module_a_hidden_arg_abi_flag_adds_tt_func_argument(monkeypatch):
    monkeypatch.setenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", "1")
    importlib.import_module("triton.backends.ascend.compiler")

    target = GPUTarget("npu", "Ascend910B", 0)
    backend = make_backend(target)
    source = _ast_source(
        _design_debug_kernel,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "n": "i32",
        },
        {"BLOCK_SIZE": 16},
    )
    options = backend.parse_options(source.parse_options())
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    mod = _make_ir(source, target, options, _codegen_implementation(backend, options), backend.get_module_map(),
                   context)
    metadata = {
        "hash": "module-a-hidden-arg-abi",
        "target": target,
        **options.__dict__,
    }
    stages = {}
    _add_stages(backend, stages, options, source)
    ttir_mod = stages["ttir"](mod, metadata)
    post_debug_ttir = str(ttir_mod)

    assert metadata["debug_enabled"] is True
    assert metadata["debug_launch_hidden_arg"] is True
    assert "flagtree.debug.enable_hidden_arg_abi = true" in post_debug_ttir
    assert 'flagtree.debug.hidden_arg = "__debug_ctrl_ptr"' in post_debug_ttir
    assert "flagtree.debug.hidden_arg_index" in post_debug_ttir
    assert 'flagtree.debug.hidden_arg_type = "!tt.ptr<i32>"' in post_debug_ttir
    assert '!tt.ptr<i32> {flagtree.debug.hidden_arg = "__debug_ctrl_ptr"}' in post_debug_ttir
    assert "flagtree.debug.record_size = 64 : i32" in post_debug_ttir
    assert "flagtree.debug.records_per_instance = 10 : i32" in post_debug_ttir
    assert "tt.atomic_rmw" not in post_debug_ttir
    assert "tensor<8x!tt.ptr<i32>>" not in post_debug_ttir
    assert "tt.store" in post_debug_ttir
    assert "tt.get_program_id" in post_debug_ttir
    assert "tt.get_num_programs" in post_debug_ttir
    assert "flagtree_debug.record_summary" not in post_debug_ttir


@pytest.mark.module_a
@pytest.mark.module_a_a3
def test_module_c_d_hidden_arg_instrumentation_lowers_through_ascend_ttadapter(monkeypatch):
    monkeypatch.setenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", "1")
    importlib.import_module("triton.backends.ascend.compiler")

    target = GPUTarget("npu", "Ascend910B", 0)
    backend = make_backend(target)
    source = _ast_source(
        _design_debug_kernel,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "n": "i32",
        },
        {"BLOCK_SIZE": 16},
    )
    options = backend.parse_options(source.parse_options())
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    current = _make_ir(source, target, options, _codegen_implementation(backend, options), backend.get_module_map(),
                       context)
    metadata = {
        "hash": "module-cd-hidden-arg-ttadapter",
        "target": target,
        **options.__dict__,
    }
    stages = {}
    _add_stages(backend, stages, options, source)
    assert "ttir" in stages
    assert "ttadapter" in stages

    current = stages["ttir"](current, metadata)
    ttir = str(current)
    assert metadata["debug_enabled"] is True
    assert metadata["debug_launch_hidden_arg"] is True
    assert "flagtree_debug.record_summary" not in ttir
    assert "flagtree.debug.record_size = 64 : i32" in ttir
    assert "flagtree.debug.records_per_instance = 10 : i32" in ttir
    assert "tt.atomic_rmw" not in ttir
    assert "tensor<8x!tt.ptr<i32>>" not in ttir
    assert "tt.store" in ttir
    assert 'flagtree.debug.hidden_arg_type = "!tt.ptr<i32>"' in ttir

    current = stages["ttadapter"](current, metadata)
    ttadapter_ir = str(current)
    assert "flagtree_debug.record_summary" not in ttadapter_ir
    assert "flagtree_debug.record_memory_event" not in ttadapter_ir
    assert "flagtree_debug.capture_memory_address" not in ttadapter_ir
    assert "flagtree_debug.record_full_value_ref" not in ttadapter_ir
    assert "tt.atomic_rmw" not in ttadapter_ir
    assert "scf.if" in ttadapter_ir


@pytest.mark.module_a
@pytest.mark.module_a_a1
@pytest.mark.module_a_a2
def test_module_a_to_b_memory_metadata_from_frontend_markers():
    importlib.import_module("triton.backends.ascend.compiler")

    target = GPUTarget("npu", "Ascend910B", 0)
    backend = make_backend(target)
    source = _ast_source(_memory_debug_kernel, {"x_ptr": "*fp32"}, {})
    options = backend.parse_options(source.parse_options())
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    mod = _make_ir(source, target, options, _codegen_implementation(backend, options), backend.get_module_map(),
                   context)
    pre_debug_ttir = str(mod)

    assert fd.has_debug_collect_markers(mod) is True
    assert "flagtree_debug.collect_begin" in pre_debug_ttir
    assert "flagtree_debug.collect_end" in pre_debug_ttir

    metadata = {
        "hash": "module-a-to-b-memory-metadata",
        "target": target,
        **options.__dict__,
    }
    stages = {}
    _add_stages(backend, stages, options, source)
    assert "ttir" in stages
    ttir_mod = stages["ttir"](mod, metadata)
    post_debug_ttir = str(ttir_mod)

    assert fd.has_debug_collect_markers(ttir_mod) is False
    assert "flagtree_debug.collect_begin" not in post_debug_ttir
    assert "flagtree_debug.collect_end" not in post_debug_ttir
    assert "flagtree.debug.scope_id" in post_debug_ttir
    assert "flagtree.debug.op_id" in post_debug_ttir
    assert "flagtree.debug.enable_hidden_arg_abi = false" in post_debug_ttir
    assert "flagtree_debug.record_" not in post_debug_ttir

    assert metadata["debug_enabled"] is True
    assert metadata["debug_protocol_version"] == 2
    assert metadata["debug_record_level"] == 1
    assert metadata["debug_addr_level"] == 0
    assert metadata["debug_export_mode"] == "POST_KERNEL_EXPORT"
    assert isinstance(metadata["debug_kernel_id"], int)
    assert metadata["debug_kernel_id"] != 0
    assert metadata["debug_launch_hidden_arg"] is False

    tracked_table = metadata["debug_tracked_table"]
    metadata_json = json.loads(metadata["debug_metadata_json"])
    assert metadata_json["debugKernelId"] == metadata["debug_kernel_id"]
    assert metadata_json["scopeCount"] == 1
    assert metadata_json["trackedOpCount"] == len(tracked_table)
    assert metadata_json["trackedOps"] == tracked_table

    op_ids = [row["opId"] for row in tracked_table]
    assert op_ids == list(range(1, len(tracked_table) + 1))
    assert {row["scopeId"] for row in tracked_table} == {1}

    load = _find_tracked_op(tracked_table, "load")
    store = _find_tracked_op(tracked_table, "store")
    assert load["mlirOpName"] == "tt.load"
    assert store["mlirOpName"] == "tt.store"
    for row in (load, store):
        assert row["isMemoryOp"] is True
        assert row["opCategory"] == row["accessType"]
        assert row["role"] == row["accessType"]
        assert row["addrSpace"] == "global"
        assert row["accessBytes"] == 4
        assert row["alignmentRequired"] == 4
        assert row["result"]["elementDtype"] == "f32"
        assert row["result"]["elementBits"] == 32
        assert "test_module_a_design_example_ir_flag.py" in row["sourceLoc"]
        assert any(operand["operandRole"] == "ptr" and operand["value"]["addrSpace"] == "global"
                   for operand in row["operands"])

    store_value = [operand for operand in store["operands"] if operand["operandRole"] == "value"]
    assert len(store_value) == 1
    assert store_value[0]["producerOpId"] == load["opId"]
