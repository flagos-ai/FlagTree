"""
Minimal smoke test for Proton's CANN vendor backend.

Run on the server with:

    python -m pytest -q third_party/proton/flagtree_profiler/test/test_cann_smoke.py -s

This test intentionally uses a host-timing fallback op instead of declaring a
GPU kernel. It validates the public API and artifact/degradation contract before
running a larger Ascend workload.
"""

import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest
import triton.knobs as knobs
import triton.profiler as proton
from triton.compiler import LazyDict
from triton._C.libproton import proton as libproton

proton_profile = importlib.import_module("triton.profiler.profile")


@pytest.fixture(autouse=True)
def _use_legacy_cann_triton_hook(monkeypatch):
    monkeypatch.setenv("PROTON_CANN_TRITON_HOOK_LEGACY", "1")


def _finalize_cann_import(base, import_root, monkeypatch, metrics="aicore"):
    monkeypatch.setenv("PROTON_CANN_IMPORT_PATH", str(import_root))
    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=("runtime_base:"
              f"vendor_metrics={metrics}:"
              f"aclprof_output_path={import_root}:"
              "runtime_host_timing_fallback=false:"
              "aclprof_runtime_enabled=false:"
              "aclprof_auto_export=false:"
              "mstx_enabled=false:"
              "aclprof_msproftx_enabled=false"),
    )
    proton.finalize(session_id)
    return json.loads(base.with_suffix(".vendor.json").read_text())


def _op_summary_associations(vendor_json):
    return [assoc for assoc in vendor_json.get("associations", []) if assoc.get("source") == "aclprof_op_summary"]


def _require_real_cann_environment():
    if shutil.which("msprof") is None:
        pytest.skip("msprof is not available")
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torch_npu is not available: {exc!r}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch_npu is installed, but no NPU is available")


def test_ir_record_buffer_capacity(monkeypatch):
    monkeypatch.delenv("PROTON_IR_RECORD_BUFFER_MB", raising=False)
    assert proton_profile._instrumentation_record_capacity() == 524288

    monkeypatch.setenv("PROTON_IR_RECORD_BUFFER_MB", "64")
    assert proton_profile._instrumentation_record_capacity() == 1048576

    monkeypatch.setenv("PROTON_IR_RECORD_BUFFER_MB", "0")
    with pytest.raises(ValueError, match="must be a positive integer"):
        proton_profile._instrumentation_record_capacity()


def test_ir_record_capacity_is_part_of_cache_mode(monkeypatch):
    monkeypatch.setenv("PROTON_IR_RECORD_BUFFER_MB", "32")
    proton_profile._activate_instrumentation()
    try:
        assert knobs.compilation.instrumentation_mode == ("debugger:record_capacity=524288")
    finally:
        proton_profile._deactivate_instrumentation()
    assert knobs.compilation.instrumentation_mode == ""


@pytest.fixture(scope="session")
def real_cann_direct_run(tmp_path_factory):
    _require_real_cann_environment()
    repo = pathlib.Path(__file__).resolve().parents[4]
    out = tmp_path_factory.mktemp("proton_cann_direct_real")
    profile_base = out / "profile"
    msprof_out = out / "msprof"
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    env.setdefault("ASCEND_VISIBLE_DEVICES", "0")
    env.setdefault("PROTON_CANN_TRITON_HOOK_LEGACY", "1")
    cmd = [
        sys.executable,
        str(repo / "third_party/proton/flagtree_profiler/scripts/cann_operator_profile_suite.py"),
        "--workload",
        "--name",
        str(profile_base),
        "--vendor-output",
        str(msprof_out),
        "--operator",
        "triton_vector_add_fp32",
        "--device",
        env.get("PROTON_CANN_TEST_DEVICE", "0"),
        "--iters",
        env.get("PROTON_CANN_TEST_ITERS", "3"),
        "--warmup",
        env.get("PROTON_CANN_TEST_WARMUP", "1"),
    ]
    result = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(env.get("PROTON_CANN_TEST_TIMEOUT", "240")),
    )
    if result.returncode != 0:
        pytest.fail(result.stdout)

    return {
        "out": out,
        "stdout": result.stdout,
        "meta": json.loads(profile_base.with_suffix(".meta.json").read_text()),
        "vendor": json.loads(profile_base.with_suffix(".vendor.json").read_text()),
        "timeline": json.loads(profile_base.with_suffix(".timeline.json").read_text().splitlines()[0]),
    }


def test_cann_backend_smoke(tmp_path):
    base = tmp_path / "profile_run"
    vendor_output = tmp_path / "proton_cann_profile"
    vendor_output.mkdir(parents=True, exist_ok=True)

    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=("runtime_base:"
              "vendor_metrics=aicore,bandwidth:"
              f"aclprof_output_path={vendor_output}:"
              "runtime_host_timing_fallback=true:"
              "aclprof_runtime_enabled=false:"
              "aclprof_auto_export=false"),
    )

    scope_id = libproton.record_scope()
    libproton.enter_op(scope_id, "cann_smoke_kernel")
    time.sleep(0.001)
    libproton.exit_op(scope_id, "cann_smoke_kernel")

    proton.finalize(session_id)

    hatchet = base.with_suffix(".hatchet")
    timeline = base.with_suffix(".timeline.json")
    meta = base.with_suffix(".meta.json")
    vendor = base.with_suffix(".vendor.json")

    assert hatchet.exists()
    assert timeline.exists()
    assert meta.exists()
    assert vendor.exists()

    meta_json = json.loads(meta.read_text())
    vendor_json = json.loads(vendor.read_text())
    timeline_json = json.loads(timeline.read_text().splitlines()[0])

    assert meta_json["backend"] == "cann"
    assert meta_json["runtime_base_enabled"] is True
    assert "aicore" in meta_json["vendor_metrics_enabled"]
    assert "bandwidth" in meta_json["vendor_metrics_enabled"]
    assert isinstance(meta_json["degrade_reasons"], list)
    assert isinstance(vendor_json.get("degrade_reasons", []), list)
    assert isinstance(vendor_json.get("associations", []), list)
    assert isinstance(timeline_json.get("traceEvents", []), list)

    config = meta_json["config"]
    assert "vendor_runtime_metric_overlays" in config
    assert "vendor_association_collected" in config
    assert "runtime_base_host_fallback_associations" in config


def test_cann_imports_exported_msprof_tx_csv(tmp_path, monkeypatch):
    import_dir = tmp_path / "msprof" / "PROF_000001" / "mindstudio_profiler_output"
    import_dir.mkdir(parents=True, exist_ok=True)
    msprof_tx_csv = import_dir / "msprof_tx_0.csv"
    msprof_tx_csv.write_text("\n".join([
        "Name,Start Time(us),Duration(us),Domain",
        "proton_cann_mstx_range,1000,25,proton",
    ]))
    monkeypatch.setenv("PROTON_CANN_IMPORT_PATH", str(tmp_path / "msprof"))

    base = tmp_path / "profile_run_msprof_tx"
    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=("runtime_base:"
              "vendor_metrics=aicore:"
              "runtime_host_timing_fallback=false:"
              "aclprof_runtime_enabled=false:"
              "aclprof_auto_export=false"),
    )
    proton.finalize(session_id)

    vendor = base.with_suffix(".vendor.json")
    vendor_json = json.loads(vendor.read_text())
    assert any("msprof_tx_0.csv" in path for path in vendor_json.get("raw_inputs", []))
    assert any(assoc.get("source") == "msprof_mstx" for assoc in vendor_json.get("associations", []))


def test_cann_defaults_host_timing_fallback_and_temporary_output(tmp_path, monkeypatch):
    monkeypatch.delenv("PROTON_CANN_RUNTIME_HOST_FALLBACK", raising=False)
    monkeypatch.delenv("PROTON_CANN_PROFILE_OUTPUT", raising=False)

    base = tmp_path / "profile_run_defaults"
    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=("runtime_base:"
              "vendor_metrics=aicore:"
              "aclprof_runtime_enabled=false:"
              "aclprof_auto_export=false:"
              "mstx_enabled=false"),
    )
    proton.finalize(session_id)

    meta_json = json.loads(base.with_suffix(".meta.json").read_text())
    config = meta_json["config"]
    assert config["runtime_host_timing_fallback"] == "true"
    assert config["aclprof_output_path_temporary"] == "true"


def test_cann_import_path_does_not_scan_session_parent(tmp_path, monkeypatch):
    import_dir = tmp_path / "import_only"
    import_dir.mkdir()
    (import_dir / "op_summary_0.csv").write_text("\n".join([
        "Op Name,Task ID,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "wanted_kernel,1,1,0,1000,10",
    ]))
    (tmp_path / "op_summary_stale.csv").write_text("\n".join([
        "Op Name,Task ID,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "stale_kernel,2,1,0,2000,10",
    ]))

    vendor_json = _finalize_cann_import(tmp_path / "profile_run_import_isolated", import_dir, monkeypatch)
    raw_inputs = vendor_json.get("raw_inputs", [])
    assert raw_inputs
    assert all(str(import_dir) in path for path in raw_inputs)
    assert not any("op_summary_stale.csv" in path for path in raw_inputs)


def test_cann_backend_rejects_overlapping_sessions(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    mode = ("runtime_base:"
            "vendor_metrics=aicore:"
            f"aclprof_output_path={tmp_path / 'msprof'}:"
            "runtime_host_timing_fallback=false:"
            "aclprof_runtime_enabled=false:"
            "aclprof_auto_export=false:"
            "mstx_enabled=false")
    session_id = proton.start(
        name=str(first),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=mode,
    )
    try:
        with pytest.raises(RuntimeError, match="overlapping sessions"):
            proton.start(
                name=str(second),
                context="shadow",
                data="tree",
                hook="triton",
                backend="cann",
                mode=mode,
            )
    finally:
        proton.finalize(session_id)


def test_finalize_session_restores_triton_hook(tmp_path):
    original_enter = list(knobs.runtime.launch_enter_hook.calls)
    original_exit = list(knobs.runtime.launch_exit_hook.calls)
    base = tmp_path / "profile_run_hook_cleanup"
    mode = ("runtime_base:"
            "vendor_metrics=aicore:"
            f"aclprof_output_path={tmp_path / 'msprof'}:"
            "runtime_host_timing_fallback=false:"
            "aclprof_runtime_enabled=false:"
            "aclprof_auto_export=false:"
            "mstx_enabled=false")
    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=mode,
    )
    try:
        assert knobs.runtime.launch_enter_hook.calls != original_enter
        assert knobs.runtime.launch_exit_hook.calls != original_exit
    finally:
        proton.finalize(session_id)
    assert knobs.runtime.launch_enter_hook.calls == original_enter
    assert knobs.runtime.launch_exit_hook.calls == original_exit


def test_triton_hook_chains_and_restores_existing_hooks_after_failed_start(tmp_path):
    original_enter = list(knobs.runtime.launch_enter_hook.calls)
    original_exit = list(knobs.runtime.launch_exit_hook.calls)
    calls = []

    def previous_enter(*args):
        calls.append(("enter", args))

    def previous_exit(*args):
        calls.append(("exit", args))

    knobs.runtime.launch_enter_hook.add(previous_enter)
    knobs.runtime.launch_exit_hook.add(previous_exit)

    mode = ("runtime_base:"
            "vendor_metrics=aicore:"
            f"aclprof_output_path={tmp_path / 'msprof'}:"
            "runtime_host_timing_fallback=false:"
            "aclprof_runtime_enabled=false:"
            "aclprof_auto_export=false:"
            "mstx_enabled=false")
    session_id = None
    try:
        session_id = proton.start(
            name=str(tmp_path / "first"),
            context="shadow",
            data="tree",
            hook="triton",
            backend="cann",
            mode=mode,
        )
        metadata = LazyDict({"name": "hook_chain_probe"})
        knobs.runtime.launch_enter_hook(metadata)
        knobs.runtime.launch_exit_hook(metadata)
        with pytest.raises(RuntimeError, match="overlapping sessions"):
            proton.start(
                name=str(tmp_path / "second"),
                context="shadow",
                data="tree",
                hook="triton",
                backend="cann",
                mode=mode,
            )
        assert previous_enter in knobs.runtime.launch_enter_hook.calls
        assert len(knobs.runtime.launch_enter_hook.calls) > len(original_enter) + 1
        proton.finalize(session_id)
        session_id = None
        assert knobs.runtime.launch_enter_hook.calls == original_enter + [previous_enter]
        assert knobs.runtime.launch_exit_hook.calls == original_exit + [previous_exit]
    finally:
        if session_id is not None:
            proton.finalize(session_id)
        knobs.runtime.launch_enter_hook.remove(previous_enter)
        knobs.runtime.launch_exit_hook.remove(previous_exit)

    assert calls[0][0] == "enter"
    assert calls[1][0] == "exit"


def test_cann_correlates_op_summary_by_correlation_id(tmp_path, monkeypatch):
    csv_dir = tmp_path / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "op_summary_0.csv").write_text("\n".join([
        "Op Name,Correlation ID,Task ID,Stream ID,Device ID,Task Start Time(us),Task Duration(us),aicoreTimeMs,totalCycle",
        "summary_kernel,9001,77,1,0,2000,50,0.2,12345",
    ]))
    (csv_dir / "task_time_0.csv").write_text("\n".join([
        "Kernel Name,Correlation ID,Task ID,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "runtime_kernel_by_corr,9001,999,8,0,900000,10",
    ]))

    vendor_json = _finalize_cann_import(tmp_path / "profile_run_corr", csv_dir, monkeypatch)
    associations = _op_summary_associations(vendor_json)
    assert len(associations) == 1
    assoc = associations[0]
    assert assoc["state"] == "collected"
    assert "correlation_id" in assoc["note"]
    assert assoc["metrics"]["aicore_time_ms"] == 0.2
    assert assoc["metrics"]["totalcycle"] == 12345


def test_cann_derives_bandwidth_from_op_summary_bytes(tmp_path, monkeypatch):
    csv_dir = tmp_path / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "op_summary_0.csv").write_text("\n".join([
        "Op Name,Task ID,Stream ID,Device ID,Task Start Time(us),Task Duration(us),HBM Read(MB),HBM Write(MB)",
        "bandwidth_kernel,11,1,0,1000,10,2,1",
    ]))

    base = tmp_path / "profile_run_bandwidth"
    vendor_json = _finalize_cann_import(base, csv_dir, monkeypatch, "aicore,bandwidth")
    associations = _op_summary_associations(vendor_json)
    assert len(associations) == 1
    metrics = associations[0]["metrics"]
    assert metrics["memory_read_bytes"] == 2_000_000
    assert metrics["memory_write_bytes"] == 1_000_000
    assert metrics["memory_access_bytes"] == 3_000_000
    assert metrics["bandwidth_gb_s"] == 300
    assert metrics["bandwidth_source"].startswith("derived_from_bytes_and_task_duration")

    timeline = json.loads(base.with_suffix(".timeline.json").read_text())
    event = next(event for event in timeline["traceEvents"]
                 if str(event.get("name", "")).startswith("bandwidth_kernel"))
    assert event["cat"] == "cann_runtime:aclprof_op_summary"
    assert event["args"]["cann.bandwidth_gb_s"] == 300
    assert event["args"]["metrics"]["cann.memory_access_bytes"] == 3_000_000

    hatchet = json.loads(base.with_suffix(".hatchet").read_text())
    nodes = []

    def collect(node):
        nodes.append(node)
        for child in node.get("children", []):
            collect(child)

    collect(hatchet[0])
    bandwidth_node = next(node for node in nodes
                          if str(node.get("frame", {}).get("name", "")).startswith("bandwidth_kernel"))
    assert bandwidth_node["metrics"]["cann.bandwidth_gb_s"] == 300
    assert all(not isinstance(value, str) for node in nodes for value in node.get("metrics", {}).values())


def test_cann_imports_direct_bandwidth_supplemental_csv(tmp_path, monkeypatch):
    csv_dir = tmp_path / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "bandwidth_0.csv").write_text("\n".join([
        "Kernel Name,Task Start Time(us),Task Duration(us),Memory Bandwidth(GB/s)",
        "bandwidth_direct_kernel,1000,20,123.5",
    ]))

    vendor_json = _finalize_cann_import(
        tmp_path / "profile_run_direct_bandwidth",
        csv_dir,
        monkeypatch,
        "bandwidth",
    )
    associations = [assoc for assoc in vendor_json.get("associations", []) if assoc.get("source") == "msprof_bandwidth"]
    assert len(associations) == 1
    metrics = associations[0]["metrics"]
    assert metrics["memory_bandwidth_gb_s"] == 123.5
    assert metrics["bandwidth_gb_s"] == 123.5
    assert metrics["bandwidth_source"] == "direct_csv_column:memory_bandwidth_gb_s"


def test_cann_imports_hbm_read_write_bandwidth_csv(tmp_path, monkeypatch):
    csv_dir = tmp_path / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "hbm_0.csv").write_text("\n".join([
        "Device_id,Metric,Read(MB/s),Write(MB/s)",
        "0,Average,11.25,7.75",
    ]))

    vendor_json = _finalize_cann_import(
        tmp_path / "profile_run_hbm_bandwidth",
        csv_dir,
        monkeypatch,
        "bandwidth",
    )
    associations = [assoc for assoc in vendor_json.get("associations", []) if assoc.get("source") == "msprof_bandwidth"]
    assert len(associations) == 1
    metrics = associations[0]["metrics"]
    assert metrics["memory_read_bandwidth_gb_s"] == 0.01125
    assert metrics["memory_write_bandwidth_gb_s"] == 0.00775
    assert metrics["bandwidth_gb_s"] == 0.019
    assert metrics["bandwidth_source"] == "direct_csv_column:memory_read_write_bandwidth"


def test_cann_correlates_op_summary_by_task_id(tmp_path, monkeypatch):
    csv_dir = tmp_path / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "op_summary_0.csv").write_text("\n".join([
        "Op Name,Task ID,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "summary_kernel_task,7007,4,0,1200,30",
    ]))
    (csv_dir / "task_time_0.csv").write_text("\n".join([
        "Kernel Name,Task ID,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "runtime_kernel_by_task,7007,9,0,800000,15",
    ]))

    vendor_json = _finalize_cann_import(tmp_path / "profile_run_task", csv_dir, monkeypatch)
    associations = _op_summary_associations(vendor_json)
    assert len(associations) == 1
    assert associations[0]["state"] == "collected"
    assert "task_id" in associations[0]["note"]


def test_cann_correlates_op_summary_by_strict_runtime_key(tmp_path, monkeypatch):
    csv_dir = tmp_path / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "op_summary_0.csv").write_text("\n".join([
        "Op Name,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "strict_kernel,3,2,1000,40",
    ]))
    (csv_dir / "task_time_0.csv").write_text("\n".join([
        "Kernel Name,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "strict_kernel,3,2,1001,39",
    ]))

    vendor_json = _finalize_cann_import(tmp_path / "profile_run_strict", csv_dir, monkeypatch)
    associations = _op_summary_associations(vendor_json)
    assert len(associations) == 1
    assert associations[0]["state"] == "collected"
    assert "device_id/stream_id/op_name" in associations[0]["note"]


def test_cann_fuzzy_match_requires_timestamp_window(tmp_path, monkeypatch):
    csv_dir = tmp_path / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "op_summary_0.csv").write_text("\n".join([
        "Op Name,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "vendor_only_kernel,1,0,1000,30",
    ]))
    (csv_dir / "task_time_0.csv").write_text("\n".join([
        "Kernel Name,Stream ID,Device ID,Task Start Time(us),Task Duration(us)",
        "different_runtime_kernel,9,0,50000,20",
    ]))

    vendor_json = _finalize_cann_import(tmp_path / "profile_run_unmatched", csv_dir, monkeypatch)
    associations = _op_summary_associations(vendor_json)
    assert len(associations) == 1
    assert associations[0]["state"] == "unmatched"
    assert "No runtime event matched" in associations[0]["note"]


def test_cann_real_direct_exports_bandwidth(real_cann_direct_run):
    vendor_json = real_cann_direct_run["vendor"]
    raw_inputs = vendor_json.get("raw_inputs", [])
    assert any("op_summary" in path for path in raw_inputs)
    assert any("hbm" in path for path in raw_inputs)

    bandwidth_associations = [
        assoc for assoc in vendor_json.get("associations", []) if "bandwidth_gb_s" in assoc.get("metrics", {})
    ]
    assert bandwidth_associations
    assert any(
        assoc.get("source") == "aclprof_op_summary" and "memory_access_bytes" in assoc.get("metrics", {})
        for assoc in bandwidth_associations)
    assert any(assoc.get("source") == "msprof_bandwidth" for assoc in bandwidth_associations)


def test_cann_real_direct_imports_aicore_op_summary(real_cann_direct_run):
    associations = _op_summary_associations(real_cann_direct_run["vendor"])
    assert associations
    triton_associations = [
        assoc for assoc in associations if assoc.get("metrics", {}).get("op_type") == "_vector_add_kernel"
    ]
    assert triton_associations
    assert any("aicore_time_us" in assoc.get("metrics", {}) or "aicore_time_ms" in assoc.get("metrics", {})
               for assoc in triton_associations)


def test_cann_real_direct_imports_mstx_timeline(real_cann_direct_run):
    vendor_json = real_cann_direct_run["vendor"]
    assert any(
        assoc.get("source") == "msprof_mstx"
        and assoc.get("metrics", {}).get("message") == "proton_cann_triton::triton_vector_add_fp32"
        for assoc in vendor_json.get("associations", []))

    events = real_cann_direct_run["timeline"].get("traceEvents", [])
    assert any(str(event.get("name", "")).startswith("proton_cann_triton::triton_vector_add_fp32") for event in events)
    assert any(assoc.get("source") == "aclprof_op_summary" for assoc in vendor_json.get("associations", []))
