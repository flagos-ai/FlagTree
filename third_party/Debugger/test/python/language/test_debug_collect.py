import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir, passes
from triton._C.libtriton.passes import flagtree_debug as fd
import pytest
import inspect
import importlib.util
import json
import os


def _source(fn):
    ast_params = inspect.signature(ASTSource).parameters
    if "constexprs" in ast_params:
        return ASTSource(fn=fn, signature={"x_ptr": "*fp32"}, constexprs={})
    return ASTSource(fn=fn, signature={"x_ptr": "*fp32"}, constants={})


def _target():
    if os.environ.get("FLAGTREE_BACKEND") == "ascend":
        return GPUTarget("npu", os.environ.get("ASCEND_TEST_ARCH", "Ascend910B"), 0)
    return GPUTarget("cuda", 80, 32)


def _require_compile_runtime():
    if os.environ.get("FLAGTREE_BACKEND") != "ascend":
        return
    if importlib.util.find_spec("mindspore") or importlib.util.find_spec("torch_npu"):
        return
    pytest.skip("Ascend full compile requires mindspore or torch_npu runtime package")


def _find_tracked_op(rows, access_type):
    matches = [row for row in rows if row.get("accessType") == access_type]
    assert len(matches) == 1
    return matches[0]


def _run_pm(pm, mod):
    try:
        pm.run(mod, "test_debug_collect")
    except TypeError:
        pm.run(mod)


def test_debug_collect_markers_stripped_from_persisted_ttir():
    """Markers are detected (metadata), then erased in ResolveDebugScopePass stub."""
    _require_compile_runtime()

    @triton.jit
    def kernel(x_ptr):
        offsets = tl.arange(0, 4)
        tl.debug_collect_start(level=1)
        x = tl.load(x_ptr + offsets)
        tl.store(x_ptr + offsets, x)
        tl.debug_collect_end()

    out = triton.compile(_source(kernel), target=_target())
    ttir = out.asm["ttir"]
    assert "flagtree.debug.collect_begin" not in ttir
    assert "flagtree.debug.collect_end" not in ttir
    assert "flagtree.debug.scope_id" in ttir
    assert "flagtree.debug.op_id" in ttir
    assert out.metadata.debug_enabled is True


def test_debug_compile_metadata_keys():
    _require_compile_runtime()

    @triton.jit
    def kernel(x_ptr):
        offsets = tl.arange(0, 4)
        tl.debug_collect_start(level=1)
        x = tl.load(x_ptr + offsets)
        tl.store(x_ptr + offsets, x)
        tl.debug_collect_end()

    out = triton.compile(_source(kernel), target=_target())
    md = out.metadata
    assert md.debug_enabled is True
    assert md.debug_protocol_version == 2
    assert md.debug_record_level == 1
    assert md.debug_addr_level == 0
    assert md.debug_export_mode == "POST_KERNEL_EXPORT"
    assert isinstance(md.debug_kernel_id, int)
    assert md.debug_kernel_id != 0
    assert isinstance(md.debug_tracked_table, list)
    assert len(md.debug_tracked_table) > 0
    assert isinstance(md.debug_tracked_table[0], dict)
    assert md.debug_launch_hidden_arg is False

    metadata_json = json.loads(md.debug_metadata_json)
    assert metadata_json["debugKernelId"] == md.debug_kernel_id
    assert metadata_json["scopeCount"] == 1
    assert metadata_json["trackedOpCount"] == len(md.debug_tracked_table)
    assert metadata_json["trackedOps"] == md.debug_tracked_table

    op_ids = [row["opId"] for row in md.debug_tracked_table]
    assert op_ids == list(range(1, len(md.debug_tracked_table) + 1))
    assert {row["scopeId"] for row in md.debug_tracked_table} == {1}

    load = _find_tracked_op(md.debug_tracked_table, "load")
    store = _find_tracked_op(md.debug_tracked_table, "store")
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
        assert any(operand["operandRole"] == "ptr" and operand["value"]["addrSpace"] == "global"
                   for operand in row["operands"])

    store_value = [operand for operand in store["operands"] if operand["operandRole"] == "value"]
    assert len(store_value) == 1
    assert store_value[0]["producerOpId"] == load["opId"]


def test_debug_collect_skips_call_ops_in_metadata(tmp_path):
    ctx = ir.context()
    ir.load_dialects(ctx)
    path = tmp_path / "debug_call_skip.mlir"
    path.write_text(
        """
        module {
          tt.func @helper(%arg0: i32) -> i32 {
            %h = arith.addi %arg0, %arg0 {flagtree.debug.scope_id = 2 : i32} : i32
            tt.return %h : i32
          }
          tt.func @kernel(%arg0: i32) -> i32 {
            %0 = tt.call @helper(%arg0) {flagtree.debug.scope_id = 1 : i32} : (i32) -> i32
            %1 = arith.addi %0, %arg0 {flagtree.debug.scope_id = 1 : i32} : i32
            tt.return %1 : i32
          }
        }
        """,
        encoding="utf-8",
    )
    mod = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    pm.enable_debug()
    fd.add_assign_debug_op_id(pm)
    _run_pm(pm, mod)

    rows = json.loads(fd.get_debug_tracked_op_table_json(mod))
    assert [row["mlirOpName"] for row in rows] == ["arith.addi"]
    text = mod.str()
    assert "tt.call @helper" in text
    call_line = text.split("tt.call @helper", 1)[1].split("\n", 1)[0]
    assert "flagtree.debug.scope_id" not in call_line
    assert "flagtree.debug.op_id" not in call_line
    helper_line = text.split("arith.addi %arg0, %arg0", 1)[1].split("\n", 1)[0]
    assert "flagtree.debug.scope_id" not in helper_line
    assert "flagtree.debug.op_id" not in helper_line


def test_debug_collect_does_not_penetrate_called_helpers(tmp_path):
    ctx = ir.context()
    ir.load_dialects(ctx)
    path = tmp_path / "debug_call_no_penetration.mlir"
    path.write_text(
        """
        module {
          tt.func private @helper(%arg0: i32) -> i32 {
            "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
            %h = arith.addi %arg0, %arg0 : i32
            "flagtree_debug.collect_end"() : () -> ()
            tt.return %h : i32
          }
          tt.func @kernel(%arg0: i32) -> i32 {
            "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
            %0 = tt.call @helper(%arg0) : (i32) -> i32
            %1 = arith.addi %0, %arg0 : i32
            "flagtree_debug.collect_end"() : () -> ()
            tt.return %1 : i32
          }
        }
        """,
        encoding="utf-8",
    )
    mod = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    pm.enable_debug()
    fd.add_resolve_debug_scope(pm)
    fd.add_assign_debug_op_id(pm)
    _run_pm(pm, mod)

    rows = json.loads(fd.get_debug_tracked_op_table_json(mod))
    assert [row["mlirOpName"] for row in rows] == ["arith.addi"]
    assert [row["scopeId"] for row in rows] == [1]
    text = mod.str()
    assert "flagtree_debug.collect_begin" not in text
    assert "flagtree_debug.collect_end" not in text
    call_line = text.split("tt.call @helper", 1)[1].split("\n", 1)[0]
    assert "flagtree.debug.scope_id" not in call_line
    assert "flagtree.debug.op_id" not in call_line
    helper_line = text.split("arith.addi %arg0, %arg0", 1)[1].split("\n", 1)[0]
    assert "flagtree.debug.scope_id" not in helper_line
    assert "flagtree.debug.op_id" not in helper_line


def test_debug_collect_records_outer_reduce_not_combiner_body(tmp_path):
    ctx = ir.context()
    ir.load_dialects(ctx)
    path = tmp_path / "debug_reduce_outer_only.mlir"
    path.write_text(
        """
        module {
          tt.func @kernel(%arg0: tensor<4xf32>) -> f32 {
            "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
            %0 = "tt.reduce"(%arg0) ({
            ^bb0(%lhs: f32, %rhs: f32):
              %cmp = arith.cmpf ogt, %lhs, %rhs : f32
              %sel = arith.select %cmp, %lhs, %rhs : f32
              tt.reduce.return %sel : f32
            }) {axis = 0 : i32} : (tensor<4xf32>) -> f32
            %1 = arith.addf %0, %0 : f32
            "flagtree_debug.collect_end"() : () -> ()
            tt.return %1 : f32
          }
        }
        """,
        encoding="utf-8",
    )
    mod = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    pm.enable_debug()
    fd.add_resolve_debug_scope(pm)
    fd.add_assign_debug_op_id(pm)
    fd.add_insert_instrumentation(pm)
    _run_pm(pm, mod)

    rows = json.loads(fd.get_debug_tracked_op_table_json(mod))
    assert [row["mlirOpName"] for row in rows] == ["tt.reduce", "arith.addf"]
    assert all(row["scopeId"] == 1 for row in rows)

    text = mod.str()
    reduce_body = text.split("^bb0", 1)[1].split("tt.reduce.return", 1)[0]
    assert "flagtree.debug.scope_id" not in reduce_body
    assert "flagtree.debug.op_id" not in reduce_body
    assert "flagtree_debug.record" not in reduce_body
    assert text.count("flagtree_debug.record_summary_bundle") >= 2


def test_debug_collect_tensor_pointer_ops_are_metadata_only(tmp_path):
    from triton.compiler.flagtree_debug import run_ttir_debug_passes_if_needed

    ctx = ir.context()
    ir.load_dialects(ctx)
    path = tmp_path / "debug_tensor_pointer_metadata_only.mlir"
    path.write_text(
        """
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
            %c0_i32 = arith.constant 0 : i32
            %c1_i64 = arith.constant 1 : i64
            %c16_i64 = arith.constant 16 : i64
            "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
            %0 = tt.make_tensor_ptr %arg0, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
            %1 = tt.make_tensor_ptr %arg1, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
            %2 = tt.load %0 : !tt.ptr<tensor<16xf32>>
            tt.store %1, %2 : !tt.ptr<tensor<16xf32>>
            "flagtree_debug.collect_end"() : () -> ()
            tt.return
          }
        }
        """,
        encoding="utf-8",
    )
    mod = ir.parse_mlir_module(str(path), ctx)
    metadata = {
        "hash": "tensor-pointer-debug-fallback",
        "debug_record_level": 1,
        "debug_addr_level": 1,
    }
    run_ttir_debug_passes_if_needed(mod, metadata)

    rows = metadata["debug_tracked_table"]
    assert [row["mlirOpName"] for row in rows] == [
        "tt.make_tensor_ptr",
        "tt.make_tensor_ptr",
        "tt.load",
        "tt.store",
    ]
    assert all(row["scopeId"] == 1 for row in rows)

    text = mod.str()
    assert "flagtree_debug.record_summary" not in text
    assert "flagtree_debug.record_summary_bundle" not in text
    assert "flagtree_debug.capture_memory_address" not in text
    assert "flagtree.debug.hidden_arg" not in text
    assert "flagtree_debug.collect_begin" not in text
    assert "flagtree_debug.collect_end" not in text
    assert metadata["debug_records_per_instance"] == 0
    assert metadata["debug_launch_hidden_arg"] is False
    assert metadata["debug_metadata_only_reason"] == "triton_tensor_pointer"


def test_debug_collect_large_store_level1_keeps_representative_memory_event(tmp_path):
    ctx = ir.context()
    ir.load_dialects(ctx)
    path = tmp_path / "debug_large_store_metadata_only.mlir"
    path.write_text(
        """
        module attributes {flagtree.debug.addr_level = 1 : i32} {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
            %value = arith.constant dense<0.000000e+00> : tensor<512xf32>
            "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
            tt.store %base, %value : tensor<512x!tt.ptr<f32>>
            "flagtree_debug.collect_end"() : () -> ()
            tt.return
          }
        }
        """,
        encoding="utf-8",
    )
    mod = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    pm.enable_debug()
    fd.add_resolve_debug_scope(pm)
    fd.add_assign_debug_op_id(pm)
    fd.add_insert_instrumentation(pm)
    _run_pm(pm, mod)

    rows = json.loads(fd.get_debug_tracked_op_table_json(mod))
    assert [row["mlirOpName"] for row in rows] == ["tt.store"]

    text = mod.str()
    assert "flagtree_debug.collect_begin" not in text
    assert "flagtree_debug.collect_end" not in text
    assert "flagtree_debug.capture_memory_address" in text
    assert 'event_kind = "BASE_ALIGNED_ADDR"' in text
    assert "flagtree_debug.record_memory_event" not in text
    assert "flagtree.debug.instrumented" in text
    assert fd.get_debug_records_per_instance(mod) == 1


def test_debug_collect_second_inliner_restores_call_free_helper(tmp_path):
    ctx = ir.context()
    ir.load_dialects(ctx)
    path = tmp_path / "debug_call_second_inline.mlir"
    path.write_text(
        """
        module {
          tt.func private @helper(%arg0: i32) -> i32 {
            "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
            %h = arith.addi %arg0, %arg0 : i32
            "flagtree_debug.collect_end"() : () -> ()
            tt.return %h : i32
          }
          tt.func @kernel(%arg0: i32) -> i32 {
            "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
            %0 = tt.call @helper(%arg0) : (i32) -> i32
            %1 = arith.addi %0, %arg0 : i32
            "flagtree_debug.collect_end"() : () -> ()
            tt.return %1 : i32
          }
        }
        """,
        encoding="utf-8",
    )
    mod = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    pm.enable_debug()
    fd.add_resolve_debug_scope(pm)
    fd.add_assign_debug_op_id(pm)
    passes.common.add_inliner(pm)
    passes.common.add_symbol_dce(pm)
    _run_pm(pm, mod)

    rows = json.loads(fd.get_debug_tracked_op_table_json(mod))
    assert [row["mlirOpName"] for row in rows] == ["arith.addi"]
    text = mod.str()
    assert "tt.call @helper" not in text
    assert "tt.func private @helper" not in text
    assert "flagtree_debug.collect_begin" not in text
    assert "flagtree_debug.collect_end" not in text


def test_no_collect_debug_disabled():
    _require_compile_runtime()

    @triton.jit
    def kernel(x_ptr):
        tl.store(x_ptr + tl.arange(0, 4), tl.zeros([4], dtype=tl.float32))

    out = triton.compile(_source(kernel), target=_target())
    assert out.metadata.debug_enabled is False
    assert out.metadata.debug_launch_hidden_arg is False


def test_debug_collect_illegal_nesting_raises():

    @triton.jit
    def kernel(x_ptr):
        tl.debug_collect_start(level=1)
        tl.debug_collect_start(level=1)
        tl.debug_collect_end()
        tl.debug_collect_end()

    with pytest.raises(Exception):
        triton.compile(_source(kernel), target=_target())


def test_debug_collect_missing_end_raises():

    @triton.jit
    def kernel(x_ptr):
        tl.debug_collect_start(level=1)
        tl.store(x_ptr + tl.arange(0, 1), tl.zeros([1], dtype=tl.float32))

    with pytest.raises(Exception):
        triton.compile(_source(kernel), target=_target())
