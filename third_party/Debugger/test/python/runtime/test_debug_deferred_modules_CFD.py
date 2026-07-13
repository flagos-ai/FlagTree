from __future__ import annotations

import json
import struct
import subprocess
import importlib
import inspect
import shutil
import sys
import ctypes
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
TRITON_OPT = ROOT / "python" / "build" / "cmake.linux-aarch64-cpython-3.11" / "bin" / "triton-opt"


def _static_value(
    *,
    dtype: str = "tensor<16x16xf32>",
    element_dtype: str = "f32",
    shape: str = "[16,16]",
    layout: str = "blocked",
    addr_space: str = "",
    rank: int = 2,
    element_bits: int = 32,
) -> dict[str, object]:
    return {
        "valueKind": "tensor",
        "dtype": dtype,
        "elementDtype": element_dtype,
        "shape": shape,
        "stride": "",
        "layout": layout,
        "encoding": "",
        "addrSpace": addr_space,
        "rank": rank,
        "elementBits": element_bits,
        "vecWidth": 1,
    }


def _tracked_op(
    op_id: int,
    mlir_op_name: str,
    op_category: str,
    triton_statement: str,
    *,
    role: str = "",
) -> dict[str, object]:
    return {
        "opId": op_id,
        "scopeId": 1,
        "resultIndex": 0,
        "isMemoryOp": False,
        "opCategory": op_category,
        "role": role,
        "mlirOpName": mlir_op_name,
        "sourceLoc": f"example.py:{8 + op_id}",
        "tritonStatement": triton_statement,
        "inlineCallPath": "",
        "result": _static_value(),
        "operands": [],
        "addrSpace": "",
        "accessType": "",
        "accessBytes": 0,
        "alignmentRequired": 0,
        "hasMask": False,
        "maskDtype": "",
        "cacheModifier": "",
        "evictionPolicy": "",
        "isVolatile": False,
        "boundaryCheckPolicy": "",
        "paddingSemantics": "",
    }


def _doc_example_metadata_json(*, backend_name: str = "cuda") -> str:
    return json.dumps({
        "debugKernelId":
        99,
        "kernelName":
        "doc_example_kernel",
        "backendName":
        backend_name,
        "targetName":
        "Ascend910B4-1" if backend_name == "ascend" else "host",
        "scopeCount":
        1,
        "trackedOpCount":
        2,
        "trackedOps": [
            _tracked_op(1, "tt.dot", "", "y = tl.dot(a, b)"),
            _tracked_op(2, "arith.addf", "", "z = x + y"),
        ],
    })


def _doc_example_summary_buffer(*, capacity: int = 64) -> bytes:
    record_size = 32
    payload_offset = 32 + capacity * record_size
    header = struct.pack("<IIIIIIII", 3, capacity, 0, 0, record_size, payload_offset, 0, 0)
    records = [
        struct.pack("<HHIQHHIQ", 1, 0, 1, 0, 6, 1, 0, 256),
        struct.pack("<HHIQHHId", 1, 0, 1, 0, 3, 3, 0, 0.125),
        struct.pack("<HHIQHHId", 1, 0, 2, 0, 5, 3, 0, 8.5),
    ]
    return header + b"".join(records)


def _summary_u64(op_id: int, collector: int, value: int, *, instance: int = 0) -> bytes:
    return struct.pack("<HHIQHHIQ", 1, 0, op_id, instance, collector, 1, 0, value)


def _summary_f32(op_id: int, collector: int, value: float, *, instance: int = 0) -> bytes:
    return (struct.pack("<HHIQHHI", 1, 0, op_id, instance, collector, 2, 0) + struct.pack("<fI", value, 0))


def _summary_count_bundle(op_id: int, *, nan: int, inf: int, zero: int, element: int, instance: int = 0) -> bytes:
    return struct.pack(
        "<HHIQQQQQQQ",
        4,
        0,
        op_id,
        instance,
        nan,
        inf,
        zero,
        element,
        0,
        0,
    )


def _summary_value_bundle(op_id: int, *, mean: float, min_val: float, max_val: float, l2_norm: float,
                          instance: int = 0) -> bytes:
    return struct.pack(
        "<HHIQffffIIIIIIII",
        5,
        0,
        op_id,
        instance,
        mean,
        min_val,
        max_val,
        l2_norm,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def _compact_summary_count_bundle(*, nan: int, inf: int, zero: int, element: int) -> bytes:
    record = bytearray(64)
    struct.pack_into("<QQQQ", record, 16, nan, inf, zero, element)
    return bytes(record)


def _compact_summary_value_bundle(*, mean: float, min_val: float, max_val: float, l2_norm: float) -> bytes:
    record = bytearray(64)
    struct.pack_into("<ffff", record, 16, mean, min_val, max_val, l2_norm)
    return bytes(record)


def _require_acl_device():
    acl = pytest.importorskip("acl", reason="CANN Python ACL runtime is unavailable")
    try:
        ret = acl.init()
    except Exception as exc:
        pytest.skip(f"acl.init failed: {exc}")
    # 100002 is returned by the CANN Python binding when ACL was already
    # initialized by an earlier test in the same process.
    if ret not in (0, None, 100002):
        pytest.skip(f"acl.init failed with code {ret}")

    try:
        ret = acl.rt.set_device(0)
    except Exception as exc:
        pytest.skip(f"acl.rt.set_device(0) failed: {exc}")
    if ret not in (0, None):
        pytest.skip(f"acl.rt.set_device(0) failed with code {ret}")
    return acl


def _run_triton_opt(input_file: str, *args: str) -> str:
    if not TRITON_OPT.exists():
        pytest.skip(f"triton-opt is not built: {TRITON_OPT}")
    result = subprocess.run(
        [
            str(TRITON_OPT),
            str(ROOT / "third_party" / "Debugger" / "test" / "lit" / input_file),
            *args,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _compile_design_example_with_hidden_debug_arg(monkeypatch):
    monkeypatch.setenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", "1")

    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend

    sys.path.insert(
        0,
        str(ROOT / "third_party" / "Debugger" / "test" / "python" / "language"),
    )
    design_module = importlib.import_module("test_module_a_design_example_ir_flag")
    kernel = design_module._design_debug_kernel

    importlib.import_module("triton.backends.ascend.compiler")
    target = GPUTarget("npu", "Ascend910B", 0)
    backend = make_backend(target)
    params = inspect.signature(ASTSource).parameters
    kwargs = {
        "fn": kernel,
        "signature": {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "n": "i32",
        },
    }
    kwargs["constexprs" if "constexprs" in params else "constants"] = {"BLOCK_SIZE": 16}
    source = ASTSource(**kwargs)
    options = backend.parse_options(source.parse_options())
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    try:
        codegen = backend.get_codegen_implementation(options)
    except TypeError:
        codegen = backend.get_codegen_implementation()

    make_ir_params = inspect.signature(source.make_ir).parameters
    if "target" in make_ir_params:
        current = source.make_ir(target, options, codegen, backend.get_module_map(), context)
    else:
        current = source.make_ir(options, codegen, backend.get_module_map(), context)

    metadata = {
        "hash": "module-cfd-compiled-design-example",
        "target": target,
        **options.__dict__,
    }
    stages = {}
    add_stages_params = inspect.signature(backend.add_stages).parameters
    if "language" in add_stages_params:
        backend.add_stages(stages, options, source.language)
    else:
        backend.add_stages(stages, options)
    current = stages["ttir"](current, metadata)
    ttir = str(current)
    current = stages["ttadapter"](current, metadata)
    ttadapter_ir = str(current)
    return metadata, ttir, ttadapter_ir, stages


def _compiled_design_example_buffer(tracked_ops: list[dict[str, object]], *, capacity: int = 64) -> bytes:
    record_size = 64
    payload_offset = 32 + capacity * record_size
    records = []
    for row in tracked_ops:
        assert row["scopeId"] == 1
        synthetic = bool(row["isSyntheticStatementCapture"])
        assert row["opCategory"] == ("statement_operand_capture" if synthetic else "")
        assert row["role"] == ("operand" if synthetic else "")
        op_id = int(row["opId"])
        records.extend([
            _compact_summary_count_bundle(nan=0, inf=0, zero=0, element=16),
            _compact_summary_value_bundle(
                mean=24.5 if op_id == 1 else 26.5,
                min_val=2.0 if op_id == 1 else 4.0,
                max_val=47.0 if op_id == 1 else 49.0,
                l2_norm=110.0 if op_id == 1 else 118.0,
            ),
        ])
    header = struct.pack(
        "<IIIIIIII",
        0,
        capacity,
        0,
        0,
        record_size,
        payload_offset,
        0,
        0,
    )
    raw = header + b"".join(records)
    return raw + b"\0" * (payload_offset - len(raw))


def _write_host_debug_buffer(hidden_arg_value: int, staging: bytes) -> None:
    assert hidden_arg_value != 0
    ctypes.memmove(hidden_arg_value, staging, len(staging))


def test_module_c_instrumentation_marks_summary_memory_and_full_value_records():
    output = _run_triton_opt(
        "insert-instrumentation.mlir",
        "-split-input-file",
        "--flagtree-insert-debug-records",
    )

    assert 'flagtree.debug.hidden_arg = "__debug_ctrl_ptr"' in output
    assert ('flagtree.debug.logical_instance_id_formula = '
            '"pid0 + pid1 * num_programs0 + pid2 * num_programs0 * num_programs1"') in output
    assert "flagtree.debug.instrumented = true" in output
    assert 'flagtree.debug.record_kinds = ["summary", "memory_event"]' in output
    assert 'flagtree.debug.record_kinds = ["memory_event"]' in output
    assert 'flagtree.debug.record_kinds = ["summary", "memory_event", "full_value"]' in output
    assert 'flagtree.debug.memory_event_kind = "LAST_ALIGNED_ADDR"' in output
    assert "flagtree.debug.full_value_ref = true" in output
    assert '"flagtree_debug.record_summary_bundle"' in output
    assert "flagtree_debug.capture_memory_address" in output
    assert "flagtree_debug.record_full_value_ref" in output
    assert '"flagtree_debug.capture_memory_address"(%arg1)' in output


def test_module_c_simplifies_only_debug_hidden_arg_memref_writes():
    output = _run_triton_opt(
        "simplify-record-memref-writes.mlir",
        "--allow-unregistered-dialect",
        "--flagtree-simplify-debug-record-memref-writes",
    )

    assert 'flagtree.debug.hidden_arg = "__debug_ctrl_ptr"' in output
    assert output.count("memref.store") == 1
    assert "memref.store %c7_i32" in output
    assert output.count("bufferization.materialize_in_destination") == 1
    assert "linalg.fill" in output


def test_module_c_pipeline_consumes_module_b_ids():
    output = _run_triton_opt(
        "b-to-c-pipeline.mlir",
        "--flagtree-resolve-debug-scope",
        "--flagtree-assign-debug-op-id",
        "--flagtree-insert-debug-records",
    )

    assert "flagtree.debug.op_id = 1 : i32" in output
    assert "flagtree.debug.op_id = 2 : i32" in output
    assert 'flagtree.debug.record_kinds = ["summary", "memory_event"]' in output
    assert 'flagtree.debug.record_kinds = ["summary"]' in output
    assert '"flagtree_debug.record_summary_bundle"' in output
    assert "flagtree_debug.capture_memory_address" in output
    assert "scope_id = 1 : i32" in output


def test_module_c_doc_example_instrumented_ir():
    output = _run_triton_opt(
        "doc-example-instrumentation.mlir",
        "--flagtree-insert-debug-records",
    )

    assert 'flagtree.debug.hidden_arg = "__debug_ctrl_ptr"' in output
    assert "flagtree.debug.op_id = 1 : i32" in output
    assert "flagtree.debug.op_id = 2 : i32" in output
    assert 'flagtree.debug.triton_statement = "y = tl.dot(a, b)"' in output
    assert 'flagtree.debug.triton_statement = "z = x + y"' in output
    assert 'flagtree.debug.record_kinds = ["summary"]' in output
    assert output.count("flagtree_debug.record_summary_bundle") == 2
    assert "op_id = 1 : i32" in output
    assert "op_id = 2 : i32" in output


def test_module_f_exported_buffer_decodes_through_module_d():
    from triton._C.libtriton import debugger as dbg

    handle = dbg.prepare_launch(
        {
            "debug_enabled": True,
            "debug_kernel_id": 17,
            "debug_kernel_name": "cfd_empty_kernel",
            "debug_backend_name": "cuda",
            "debug_record_level": 1,
            "debug_export_mode": "POST_KERNEL_EXPORT",
            "debug_record_capacity": 8,
        },
        0,
        None,
    )
    assert int(handle.hidden_arg_value) != 0

    exported = handle.finish()
    decoded = dbg.decode_exported_run(exported)

    assert decoded["meta"]["kernel_id"] == 17
    assert decoded["header"]["capacity"] == 64
    assert decoded["header"]["record_size"] == 32
    assert decoded["records"] == []


def test_module_d_decodes_summary_record_and_exports_text_report():
    from triton._C.libtriton import debugger as dbg

    header = struct.pack("<IIIIIIII", 1, 1, 0, 0, 32, 64, 0, 0)
    summary = struct.pack("<HHIQHHId", 1, 0, 7, 42, 3, 3, 0, 3.5)
    exported = {
        "meta": {
            "run_id": 3,
            "device_id": 0,
            "kernel_id": 17,
            "protocol_version": 1,
            "record_level": 1,
            "export_mode": 1,
            "backend_kind": 1,
        },
        "runtime_metadata": {"buffers": [], "tensors": []},
        "raw_buffer": header + summary,
    }
    metadata_json = json.dumps({
        "debugKernelId": 17,
        "kernelName": "cfd_summary_kernel",
        "backendName": "cuda",
        "targetName": "host",
        "scopeCount": 1,
        "trackedOpCount": 0,
        "trackedOps": [],
    })

    decoded = dbg.decode_exported_run(exported)
    assert decoded["records"][0]["record_kind"] == "SUMMARY"
    assert decoded["records"][0]["op_id"] == 7
    assert decoded["records"][0]["logical_instance_id"] == 42
    assert decoded["records"][0]["f64_value"] == 3.5

    report = dbg.render_text_report(exported, metadata_json)
    assert "FlagTree Debug Report" in report
    assert "kernel_name: cfd_summary_kernel" in report
    assert "record_count: 1" in report
    assert "summary_records=1" not in report
    assert "mean: [3.5 (F64)]" in report
    assert "latest.mean=3.5" not in report


def test_module_f_cann_export_decodes_to_module_d_report_for_doc_example():
    from triton._C.libtriton import debugger as dbg

    acl = _require_acl_device()
    metadata_json = _doc_example_metadata_json(backend_name="ascend")
    handle = dbg.prepare_launch(
        {
            "debug_enabled": True,
            "debug_kernel_id": 99,
            "debug_kernel_name": "doc_example_kernel",
            "debug_backend_name": "ascend",
            "debug_target_name": "Ascend910B4-1",
            "debug_record_level": 1,
            "debug_export_mode": "POST_KERNEL_EXPORT",
            "debug_record_capacity": 8,
            "debug_metadata_json": metadata_json,
        },
        0,
        {"buffers": [], "tensors": []},
    )
    assert int(handle.hidden_arg_value) != 0

    staging = _doc_example_summary_buffer()
    ret = acl.rt.memcpy(
        int(handle.hidden_arg_value),
        len(staging),
        acl.util.bytes_to_ptr(staging),
        len(staging),
        1,
    )
    assert ret in (0, None)

    exported = handle.finish()
    decoded = dbg.decode_exported_run(exported)
    report = dbg.render_text_report(exported, metadata_json)

    assert decoded["meta"]["kernel_id"] == 99
    assert decoded["meta"]["backend_kind"] == 4
    assert decoded["header"]["write_idx"] == 3
    assert decoded["header"]["capacity"] == 64
    assert len(decoded["records"]) == 3
    assert decoded["records"][0]["op_id"] == 1
    assert decoded["records"][2]["op_id"] == 2

    assert "FlagTree Debug Report" in report
    assert "backend: CANN" in report
    assert "kernel_name: doc_example_kernel" in report
    assert "record_count: 3" in report
    assert "triton_statement: y = tl.dot(a, b)" in report
    assert "triton_statement: z = x + y" in report
    assert "element_count: [256 (U64)]" in report
    assert "mean         : [0.125 (F64)]" in report
    assert "max: [8.5 (F64)]" in report
    assert "latest.element_count=256" not in report


def test_compiled_design_example_instrumented_ir_exports_final_debugger_report(monkeypatch):
    from triton._C.libtriton import debugger as dbg

    metadata, instrumented_ttir, ttadapter_ir, _ = _compile_design_example_with_hidden_debug_arg(monkeypatch)
    compiled_metadata = json.loads(metadata["debug_metadata_json"])
    tracked_ops = metadata["debug_tracked_table"]

    assert metadata["debug_enabled"] is True
    assert metadata["debug_launch_hidden_arg"] is True
    assert metadata["debug_record_layout"] == "deterministic_compact_v1"
    assert metadata["debug_records_per_instance"] == len(metadata["debug_record_plan"])
    assert compiled_metadata["debugKernelId"] == metadata["debug_kernel_id"]
    assert compiled_metadata["trackedOps"] == tracked_ops
    assert [row["mlirOpName"] for row in tracked_ops] == [
        "arith.mulf",
        "flagtree.debug.operand_capture",
        "flagtree.debug.operand_capture",
        "arith.addf",
        "flagtree.debug.operand_capture",
    ]
    assert [row["opId"] for row in tracked_ops] == [1, 3, 4, 2, 5]
    assert [row["isSyntheticStatementCapture"] for row in tracked_ops] == [
        False,
        True,
        True,
        False,
        True,
    ]

    assert "flagtree_debug.record_summary" not in instrumented_ttir
    assert "flagtree.debug.record_size = 64 : i32" in instrumented_ttir
    assert "flagtree.debug.records_per_instance = 10 : i32" in instrumented_ttir
    assert "tt.atomic_rmw" not in instrumented_ttir
    assert "tensor<8x!tt.ptr<i32>>" in instrumented_ttir
    assert "tt.store" in instrumented_ttir
    assert 'flagtree.debug.hidden_arg_type = "!tt.ptr<i32>"' in instrumented_ttir
    assert "flagtree_debug.record_summary" not in ttadapter_ir
    assert "tt.atomic_rmw" not in ttadapter_ir

    handle = dbg.prepare_launch(
        {
            "debug_enabled": True,
            "debug_kernel_id": metadata["debug_kernel_id"],
            "debug_kernel_name": compiled_metadata["kernelName"],
            "debug_backend_name": "cuda",
            "debug_target_name": compiled_metadata["targetName"],
            "debug_record_level": metadata["debug_record_level"],
            "debug_export_mode": metadata["debug_export_mode"],
            "debug_record_capacity": 64,
            "debug_record_size": metadata["debug_record_size"],
            "debug_metadata_json": metadata["debug_metadata_json"],
        },
        0,
        {
            "buffers": [],
            "tensors": [],
            "grid": (1, 1, 1),
            "records_per_instance": metadata["debug_records_per_instance"],
            "record_layout": metadata["debug_record_layout"],
            "record_plan": metadata["debug_record_plan"],
        },
    )
    assert int(handle.hidden_arg_value) != 0

    staging = _compiled_design_example_buffer(tracked_ops)
    _write_host_debug_buffer(int(handle.hidden_arg_value), staging)

    exported = handle.finish()
    decoded = dbg.decode_exported_run(exported)
    report = dbg.render_text_report(exported, metadata["debug_metadata_json"])

    assert decoded["meta"]["kernel_id"] == metadata["debug_kernel_id"]
    assert decoded["meta"]["backend_kind"] == 1
    assert decoded["header"]["write_idx"] == 10
    assert decoded["header"]["capacity"] == 64
    assert decoded["header"]["record_size"] == 64
    assert [record["op_id"] for record in decoded["records"]] == [
        1,
        1,
        3,
        3,
        4,
        4,
        2,
        2,
        5,
        5,
    ]
    assert [record["record_kind"] for record in decoded["records"]
            ] == [kind for _ in tracked_ops for kind in ("SUMMARY_COUNT_BUNDLE_U64", "SUMMARY_VALUE_BUNDLE_F32")]
    count_records = [record for record in decoded["records"] if record["record_kind"] == "SUMMARY_COUNT_BUNDLE_U64"]
    assert [record["element_count"] for record in count_records] == [16] * 5
    assert [record["nan_count"] for record in count_records] == [0] * 5

    assert "FlagTree Debug Report" in report
    assert f"kernel_id: {metadata['debug_kernel_id']}" in report
    assert "backend: CUDA" in report
    assert "record_count: 10" in report
    assert "IR Op Log Records" in report
    assert "mlir_op: arith.mulf" in report
    assert "mlir_op: arith.addf" in report
    assert "mlir_op: flagtree.debug.operand_capture" in report
    # The combined report renders each tracked value once in the statement
    # view and once in the IR op log view.
    assert report.count("summary:") == 10
    assert report.count("element_count: [16 (U64)]") == 10
    assert report.count("zero_count   : [0 (U64)]") == 10
    assert report.count("inf_count    : [0 (U64)]") == 10
    assert report.count("mean         :") == 10
    assert report.count("l2_norm      :") == 10
    assert "Aggregates" not in report
    assert "latest.element_count=16" not in report


def test_compiled_design_example_ttadapter_reaches_bishengir_binary_stage(monkeypatch):
    compiler = shutil.which("bishengir-compile")
    if compiler is None:
        pytest.skip("bishengir-compile is unavailable")

    metadata, _, binary_source, stages = _compile_design_example_with_hidden_debug_arg(monkeypatch)
    if "mlirbc" in stages:
        binary_source = stages["mlirbc"](binary_source, metadata)
        binary_source = stages["bcmlir"](binary_source, metadata)
    binary = stages["npubin"](binary_source, metadata)
    assert isinstance(binary, bytes)
    assert binary
