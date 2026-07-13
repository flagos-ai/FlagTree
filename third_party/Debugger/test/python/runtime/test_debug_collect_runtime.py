import importlib.util
import pytest
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    from triton.compiler.flagtree_debug import prepare_launch_debug_ctrl
except Exception as exc:
    pytest.skip(f"debug collect runtime tests require importable triton runtime: {exc}", allow_module_level=True)


def test_prepare_launch_debug_ctrl_sets_ptr():
    k = SimpleNamespace()
    k._run = SimpleNamespace()
    k._run.debug_launch_hidden_arg = True
    k._debug_ctrl_ptr = 0xDEADBEEF
    prepare_launch_debug_ctrl(k, stream=None)
    assert k._run.debug_ctrl_ptr == 0xDEADBEEF


def test_prepare_launch_skips_when_run_disabled():
    k = SimpleNamespace()
    k._run = SimpleNamespace()
    k._run.debug_launch_hidden_arg = False
    k._run.debug_ctrl_ptr = 0
    prepare_launch_debug_ctrl(k, stream=None)
    assert k._run.debug_ctrl_ptr == 0


def test_debug_collect_runtime_uses_flagtree_backend_for_backend_name(monkeypatch):
    from triton.runtime.debug_collect_runtime import DebugCollectRuntime

    monkeypatch.setenv("FLAGTREE_BACKEND", "ascend")
    metadata = {
        "target": SimpleNamespace(backend="npu", arch="Ascend910B4"),
        "debug_target_name": "Ascend910B4",
    }
    normalized = DebugCollectRuntime._normalize_launch_metadata(metadata)
    assert normalized["debug_backend_name"] == "ascend"
    assert normalized["debug_target_name"] == "Ascend910B4"


def test_debug_collect_runtime_does_not_infer_backend_from_npu_target(monkeypatch):
    from triton.runtime.debug_collect_runtime import DebugCollectRuntime

    monkeypatch.delenv("FLAGTREE_BACKEND", raising=False)
    metadata = {
        "target": SimpleNamespace(backend="npu", arch="Ascend910B4"),
        "debug_target_name": "Ascend910B4",
    }
    normalized = DebugCollectRuntime._normalize_launch_metadata(metadata)
    assert normalized["debug_backend_name"] == "npu"


def test_debug_collect_runtime_prepare_export_decodes_header():
    from triton.runtime.debug_collect_runtime import default_debug_collect_runtime

    md = SimpleNamespace()
    md.debug_kernel_id = 7
    md.debug_enabled = True
    md.debug_protocol_version = 1
    md.debug_record_level = 1
    md.debug_export_mode = "POST_KERNEL_EXPORT"
    md.debug_record_capacity = 8
    md.debug_backend_name = "cuda"
    md.debug_metadata_json = json.dumps({
        "debugKernelId": 7,
        "kernelName": "unit_empty_kernel",
        "backendName": "cuda",
        "targetName": "host",
        "scopeCount": 0,
        "trackedOpCount": 0,
        "trackedOps": [],
    })
    default_debug_collect_runtime.clear_exported_runs()
    ctx = default_debug_collect_runtime.prepare(md, stream=None)
    assert ctx.debug_kernel_id == 7
    assert default_debug_collect_runtime.hidden_arg(ctx) != 0
    run = default_debug_collect_runtime.export(ctx, stream=None)
    assert run.debug_kernel_id == 7
    assert len(run.raw_buffer) >= 32
    assert run.decoded["header"]["capacity"] == 64
    assert run.decoded["header"]["record_size"] == 32
    assert run.decoded["records"] == []
    assert "FlagTree Debug Report" in run.report
    assert "kernel_name: unit_empty_kernel" in run.report
    assert "record_count: 0" in run.report
    assert default_debug_collect_runtime.peek_exported_runs() == [run]
    assert default_debug_collect_runtime.take_exported_runs() == [run]
    assert default_debug_collect_runtime.peek_exported_runs() == []


def test_debugger_binding_decodes_and_reports_summary_record():
    from triton._C.libtriton import debugger as dbg

    header = struct.pack("<IIIIIIII", 1, 1, 0, 0, 32, 64, 0, 0)
    summary = struct.pack("<HHIQHHId", 1, 0, 1, 42, 6, 3, 0, 3.5)
    exported = {
        "meta": {
            "run_id": 9,
            "device_id": 0,
            "kernel_id": 7,
            "protocol_version": 1,
            "record_level": 1,
            "export_mode": 1,
            "backend_kind": 1,
        },
        "runtime_metadata": {"buffers": [], "tensors": []},
        "raw_buffer": header + summary,
    }
    metadata_json = json.dumps({
        "debugKernelId": 7,
        "kernelName": "unit_kernel",
        "backendName": "cuda",
        "targetName": "host",
        "scopeCount": 1,
        "trackedOpCount": 0,
        "trackedOps": [],
    })

    decoded = dbg.decode_exported_run(exported)
    assert decoded["header"]["write_idx"] == 1
    assert decoded["records"][0]["record_kind"] == "SUMMARY"
    assert decoded["records"][0]["op_id"] == 1
    assert decoded["records"][0]["logical_instance_id"] == 42

    report = dbg.render_text_report(exported, metadata_json)
    assert "FlagTree Debug Report" in report
    assert "record_count: 1" in report
    assert "IR Op Log Records" in report
    assert "summary:" in report
    assert "element_count: [3.5 (F64)]" in report

    json_report = dbg.render_json_report(exported, metadata_json)
    assert '"records_by_op"' in json_report
    assert '"instances":[42]' in json_report
    assert '"summary"' in json_report


def test_debugger_binding_decodes_deterministic_compact_bundle_records():
    from triton._C.libtriton import debugger as dbg

    record_size = 64
    capacity = 4
    payload_offset = 32 + capacity * record_size
    header = struct.pack("<IIIIIIII", 4, capacity, 0, 0, record_size, payload_offset, 0, 0)
    records = [bytearray(record_size) for _ in range(capacity)]
    struct.pack_into("<QQQQ", records[0], 16, 0, 0, 1, 16)
    struct.pack_into("<ffff", records[1], 16, 4.0, 0.0, 8.0, 18.5)
    struct.pack_into("<QQQQ", records[2], 16, 1, 2, 3, 32)
    struct.pack_into("<ffff", records[3], 16, 5.0, 1.0, 9.0, 22.0)
    exported = {
        "meta": {
            "run_id": 10,
            "device_id": 0,
            "kernel_id": 8,
            "protocol_version": 2,
            "record_level": 1,
            "export_mode": 1,
            "backend_kind": 1,
        },
        "runtime_metadata": {
            "buffers": [],
            "tensors": [],
            "records_per_instance":
            2,
            "record_layout":
            "deterministic_compact_v1",
            "record_plan": [
                {
                    "record_index": 0,
                    "op_id": 7,
                    "scope_id": 1,
                    "record_kind": 4,
                    "collector_kind": 0,
                    "result_type": 0,
                    "event_kind": 0,
                },
                {
                    "record_index": 1,
                    "op_id": 7,
                    "scope_id": 1,
                    "record_kind": 5,
                    "collector_kind": 0,
                    "result_type": 0,
                    "event_kind": 0,
                },
            ],
        },
        "raw_buffer": header + b"".join(bytes(record) for record in records),
    }

    decoded = dbg.decode_exported_run(exported)
    assert [record["record_kind"] for record in decoded["records"]] == [
        "SUMMARY_COUNT_BUNDLE_U64",
        "SUMMARY_VALUE_BUNDLE_F32",
        "SUMMARY_COUNT_BUNDLE_U64",
        "SUMMARY_VALUE_BUNDLE_F32",
    ]
    assert [record["op_id"] for record in decoded["records"]] == [7, 7, 7, 7]
    assert [record["logical_instance_id"] for record in decoded["records"]] == [0, 0, 1, 1]
    assert decoded["records"][0]["nan_count"] == 0
    assert decoded["records"][0]["zero_count"] == 1
    assert decoded["records"][2]["nan_count"] == 1
    assert decoded["records"][3]["mean"] == pytest.approx(5.0)


def test_debugger_binding_decodes_deterministic_compact_timeline_record():
    from triton._C.libtriton import debugger as dbg

    record_size = 64
    capacity = 1
    payload_offset = 32 + capacity * record_size
    header = struct.pack("<IIIIIIII", 1, capacity, 0, 0, record_size, payload_offset, 0, 0)
    record = bytearray(record_size)
    struct.pack_into("<QQQ", record, 16, 100, 145, 45)
    exported = {
        "meta": {
            "run_id": 11,
            "device_id": 0,
            "kernel_id": 9,
            "protocol_version": 2,
            "record_level": 1,
            "export_mode": 1,
            "backend_kind": 4,
        },
        "runtime_metadata": {
            "buffers": [],
            "tensors": [],
            "records_per_instance":
            1,
            "record_layout":
            "deterministic_compact_v1",
            "record_plan": [
                {
                    "record_index": 0,
                    "op_id": 7,
                    "scope_id": 1,
                    "record_kind": 6,
                    "collector_kind": 0,
                    "result_type": 0,
                    "event_kind": 0,
                },
            ],
        },
        "raw_buffer": header + bytes(record),
    }

    decoded = dbg.decode_exported_run(exported)
    assert decoded["records"] == [{
        "record_kind": "TIMELINE",
        "op_id": 7,
        "logical_instance_id": 0,
        "start_cycle": 100,
        "end_cycle": 145,
        "duration_cycle": 45,
    }]


def test_ascend_launcher_appends_debug_hidden_arg_when_enabled():
    from triton.backends.ascend.driver import NPULauncher

    launcher = object.__new__(NPULauncher)
    launcher.compile_only = False
    launcher.enable_msprof_register_tensor = False
    launcher.debug_launch_hidden_arg = True
    launcher.debug_ctrl_ptr = 0xAABBCCDD
    seen = {}

    def launch(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return 0

    launcher.launch = launch
    launcher(
        1,
        1,
        1,
        0,
        0x1234,
        {"hash": "unit"},
        None,
        None,
        None,
        99,
    )

    assert seen["args"][-2:] == (99, 0xAABBCCDD)


def test_ascend_spec_compiled_kernel_launch_metadata_includes_grid(monkeypatch):
    import triton

    compiler_path = (Path(triton.__file__).parent / "spec" / "ascend" / "compiler" / "compiler.py")
    spec = importlib.util.spec_from_file_location(
        "triton.compiler._ascend_spec_compiler_under_test",
        compiler_path,
    )
    ascend_compiler = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, ascend_compiler)
    spec.loader.exec_module(ascend_compiler)

    kernel = SimpleNamespace(
        metadata=SimpleNamespace(debug_enabled=True),
        name="debug_grid_kernel",
        function=0x1234,
        src=object(),
        _init_handles=lambda: None,
    )

    launch_metadata = ascend_compiler.CompiledKernel.launch_metadata(kernel, (7, ), 99).get()

    assert launch_metadata["name"] == "debug_grid_kernel"
    assert launch_metadata["function"] == 0x1234
    assert launch_metadata["stream"] == 99
    assert launch_metadata["grid"] == (7, 1, 1)


def test_ascend_spec_jit_prepares_and_finalizes_debug_hidden_arg(monkeypatch):
    import triton
    import triton.backends.ascend as ascend_backend
    from triton.runtime import debugger

    jit_path = Path(triton.__file__).parent / "spec" / "ascend" / "runtime" / "jit.py"
    if not jit_path.exists():
        jit_path = (Path(ascend_backend.__file__).parent / "spec" / "triton" / "runtime" / "jit.py")
    spec = importlib.util.spec_from_file_location(
        "triton.runtime._ascend_spec_jit_under_test",
        jit_path,
    )
    ascend_jit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, ascend_jit)
    spec.loader.exec_module(ascend_jit)

    calls = []

    def finalize(error):
        calls.append(("finalize", error))

    def prepare_kernel_launch(metadata, stream, launch_metadata, kernel_args):
        calls.append((metadata.name, stream, launch_metadata, kernel_args))
        return debugger.PreparedKernelLaunch(
            kernel_args=(0x12345678, ),
            finalize=finalize,
        )

    monkeypatch.setattr(debugger, "prepare_kernel_launch", prepare_kernel_launch)
    monkeypatch.setattr(
        debugger,
        "finalize_prepared_launch",
        lambda prepared, error: prepared.finalize(error),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: calls.append("sync"))),
    )

    kernel = SimpleNamespace(
        metadata=SimpleNamespace(
            debug_launch_hidden_arg=True,
            debug_records_per_instance=1,
            name="ascend_debug_kernel",
        ),
        run=SimpleNamespace(debug_ctrl_ptr=0),
    )

    prepared = ascend_jit._prepare_flagtree_debug_launch(kernel, 99, {"grid": (1, 1, 1)}, ("x", "y"))
    assert kernel.run.debug_ctrl_ptr == 0x12345678

    ascend_jit._finalize_flagtree_debug_launch(prepared, None)
    assert calls == [
        ("ascend_debug_kernel", 99, {"grid": (1, 1, 1)}, ("x", "y")),
        "sync",
        ("finalize", None),
    ]


def test_ascend_spec_jit_applies_instrumentation_mode_to_compile_options(monkeypatch, ):
    import triton
    import triton.backends.ascend as ascend_backend

    jit_path = Path(triton.__file__).parent / "spec" / "ascend" / "runtime" / "jit.py"
    if not jit_path.exists():
        jit_path = (Path(ascend_backend.__file__).parent / "spec" / "triton" / "runtime" / "jit.py")
    spec = importlib.util.spec_from_file_location(
        "triton.runtime._ascend_spec_jit_instrumentation_under_test",
        jit_path,
    )
    ascend_jit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, ascend_jit)
    spec.loader.exec_module(ascend_jit)

    monkeypatch.setattr(
        ascend_jit.knobs.compilation,
        "instrumentation_mode",
        "debugger",
    )
    kwargs = {}
    ascend_jit._apply_compilation_instrumentation_mode(kwargs)
    assert kwargs == {"instrumentation_mode": "debugger"}

    explicit = {"instrumentation_mode": "custom"}
    ascend_jit._apply_compilation_instrumentation_mode(explicit)
    assert explicit == {"instrumentation_mode": "custom"}


def test_ascend_options_hash_includes_instrumentation_mode():
    from triton.backends.ascend.compiler import NPUOptions

    plain = NPUOptions(arch="Ascend910B4")
    instrumented = NPUOptions(
        arch="Ascend910B4",
        instrumentation_mode="debugger",
    )

    assert plain.hash() != instrumented.hash()


def test_ascend_spec_jit_exports_metadata_only_when_no_debug_records(monkeypatch):
    import triton
    import triton.backends.ascend as ascend_backend
    from triton.runtime import debugger

    jit_path = Path(triton.__file__).parent / "spec" / "ascend" / "runtime" / "jit.py"
    if not jit_path.exists():
        jit_path = (Path(ascend_backend.__file__).parent / "spec" / "triton" / "runtime" / "jit.py")
    spec = importlib.util.spec_from_file_location(
        "triton.runtime._ascend_spec_jit_zero_records_under_test",
        jit_path,
    )
    ascend_jit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, ascend_jit)
    spec.loader.exec_module(ascend_jit)

    def prepare_kernel_launch(*_args, **_kwargs):
        raise AssertionError("zero-record debug kernels must not prepare hidden arg launch")

    calls = []

    def prepare_metadata_only_kernel_launch(metadata, stream, launch_metadata, kernel_args):
        calls.append((metadata.name, stream, launch_metadata, kernel_args))
        return debugger.PreparedKernelLaunch(
            kernel_args=(),
            finalize=lambda error: calls.append(("finalize", error)),
        )

    monkeypatch.setattr(debugger, "prepare_kernel_launch", prepare_kernel_launch)
    monkeypatch.setattr(
        debugger,
        "prepare_metadata_only_kernel_launch",
        prepare_metadata_only_kernel_launch,
    )
    monkeypatch.setattr(
        debugger,
        "finalize_prepared_launch",
        lambda prepared, error: prepared.finalize(error),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: calls.append("sync"))),
    )
    kernel = SimpleNamespace(
        metadata=SimpleNamespace(
            debug_enabled=True,
            debug_launch_hidden_arg=True,
            debug_records_per_instance=0,
            name="ascend_zero_record_debug_kernel",
        ),
        run=SimpleNamespace(debug_ctrl_ptr=0, debug_launch_hidden_arg=True),
    )

    prepared = ascend_jit._prepare_flagtree_debug_launch(kernel, 99, {"grid": (1, 1, 1)}, ("x", "y"))
    assert prepared is not None
    assert prepared.kernel_args == ()
    assert kernel.run.debug_ctrl_ptr == 0
    assert kernel.run.debug_launch_hidden_arg is False

    ascend_jit._finalize_flagtree_debug_launch(prepared, None)
    assert calls == [
        ("ascend_zero_record_debug_kernel", 99, {"grid": (1, 1, 1)}, ("x", "y")),
        "sync",
        ("finalize", None),
    ]
