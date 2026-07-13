#!/usr/bin/env python3
"""Run a complex real Ascend debugger instrumentation probe.

This script intentionally avoids MindSpore / torch_npu tensor dependencies.  It
uses ACL for device memory and registers a minimal local Ascend launcher backend
strategy so that the debugger hidden argument can be exercised with a real CANN
kernel launch.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(os.environ.get("FLAGTREE_DEBUGGER_COMPLEX_OUT", "/tmp/flagtree_debugger_complex_e2e"))
BLOCK_SIZE = 16
COLLECTOR_NAMES = {
    1: "nan_count",
    2: "inf_count",
    3: "mean",
    4: "min",
    5: "max",
    6: "element_count",
    7: "zero_count",
    8: "l2_norm",
}


def ensure_import_paths() -> None:
    build_lib = ROOT / "python" / "build" / "lib.linux-aarch64-cpython-311"
    paths = [
        build_lib,
        ROOT / "python",
        Path("/usr/local/Ascend/ascend-toolkit/latest/python/site-packages"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"),
    ]
    for path in reversed(paths):
        if path.exists():
            sys.path.insert(0, str(path))


ensure_import_paths()

import triton
import triton.language as tl


@triton.jit
def _complex_debug_kernel(x_ptr, y_ptr, a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)

    tl.debug_collect_start(level=1)

    p = a * b
    q = p + x
    r = q - c
    s = r * r
    y = s + q

    tl.debug_collect_end()

    tl.store(y_ptr + offsets, y, mask=mask)


def register_acl_backend_strategy() -> None:
    from triton.backends.ascend.backend_register import backend_strategy_registry
    import triton.backends.ascend.utils as ascend_utils

    registry = backend_strategy_registry._get_instance()
    registry.strategies.setdefault("acl", {})
    registry.strategies["acl"].update({
        "version_hash": lambda: ["acl-complex-probe"],
        "cxx_abi": lambda: 1,
        "get_cc_cmd": lambda build_pch: [],
        "get_current_device": lambda: 0,
        "set_current_device": lambda device_id: 0,
        "get_current_stream": lambda device=None: 0,
        "header_file": lambda enable_taskqueue: "",
        "pre_launch": lambda: "",
        "async_launch": lambda func: f"{func}();",
        "allocate_memory": lambda size, stream: "nullptr",
        "allocate_sync_block_lock": lambda size, stream: "nullptr",
        "get_empty_tensor": lambda size: None,
        "get_tensor_params_shape": lambda *args: [],
        "get_device_interface": lambda: None,
    })
    ascend_utils.backend_policy = "acl"


def patch_launcher_build():
    import triton.backends.ascend.driver as ascend_driver
    import triton.backends.ascend.utils as ascend_utils

    def make_launcher_no_pch(header_src, wrapper_src, debug=False):
        del debug
        build_dir = OUT / "launcher_build"
        build_dir.mkdir(parents=True, exist_ok=True)
        header_path = build_dir / "precompiled.h"
        src_path = build_dir / "launcher_no_pch.cxx"
        header_path.write_text(header_src, encoding="utf-8")
        src_path.write_text(wrapper_src, encoding="utf-8")
        return ascend_utils._build_npu_ext("launcher_no_pch", str(header_path), str(src_path), precompile=False)

    ascend_driver.make_npu_launcher_stub = make_launcher_no_pch
    return make_launcher_no_pch


def ast_source():
    from triton.compiler import ASTSource

    params = inspect.signature(ASTSource).parameters
    kwargs = {
        "fn": _complex_debug_kernel,
        "signature": {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "c_ptr": "*fp32",
            "n": "i32",
        },
    }
    kwargs["constants" if "constants" in params else "constexprs"] = {"BLOCK_SIZE": BLOCK_SIZE}
    return ASTSource(**kwargs)


def get_codegen(backend, options):
    try:
        return backend.get_codegen_implementation(options)
    except TypeError:
        return backend.get_codegen_implementation()


def make_ir(source, target, options, codegen, module_map, context):
    params = inspect.signature(source.make_ir).parameters
    if "target" in params:
        return source.make_ir(target, options, codegen, module_map, context)
    return source.make_ir(options, codegen, module_map, context)


def compile_kernel(log_lines: list[str]):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    from triton.backends.ascend.compiler import _parse_linalg_metadata

    try:
        from triton._C.libtriton import buffer_ir
        from triton._C.libtriton.ascend import ir as ascend_ir
    except Exception:
        buffer_ir = None
        ascend_ir = None

    target = GPUTarget("npu", "Ascend910B", 0)
    backend = make_backend(target)
    source = ast_source()
    options = backend.parse_options(source.parse_options())
    context = ir.context()
    ir.load_dialects(context)
    if buffer_ir is not None:
        buffer_ir.load_dialects(context)
    if ascend_ir is not None:
        ascend_ir.load_dialects(context)
    backend.load_dialects(context)

    mod = make_ir(source, target, options, get_codegen(backend, options), backend.get_module_map(), context)
    metadata = {"hash": "complex-ascend-debugger-evaluation", "target": target, **options.__dict__}
    stages = {}
    if "language" in inspect.signature(backend.add_stages).parameters:
        backend.add_stages(stages, options, source.language)
    else:
        backend.add_stages(stages, options)
    ttir_mod = stages["ttir"](mod, metadata)
    ttir_text = str(ttir_mod)
    ttadapter_text = stages["ttadapter"](ttir_mod, metadata)
    parsed_ttadapter, metadata = _parse_linalg_metadata(ttadapter_text, metadata)
    compile_ttadapter = parsed_ttadapter.replace(', hacc.target = #hacc.target<"Ascend910B">', "")
    compile_ttadapter = compile_ttadapter.replace(', hacc.target = #hacc.target<"Ascend910B4-1">', "")

    work = OUT / "manual_compile"
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernel.ttadapter.mlir").write_text(compile_ttadapter, encoding="utf-8")

    compiler = shutil.which("bishengir-compile") or "/usr/local/Ascend/ascend-toolkit/latest/bin/bishengir-compile"
    cmd = [
        compiler,
        str(work / "kernel.ttadapter.mlir"),
        "--enable-auto-multi-buffer",
        "--enable-hfusion-compile",
        "--enable-hivm-compile",
        "--enable-triton-kernel-compile",
        "-o",
        str(work / "kernel"),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (work / "compile.stdout").write_text(result.stdout, encoding="utf-8")
    (work / "compile.stderr").write_text(result.stderr, encoding="utf-8")
    log_lines.append("compile_command: " + " ".join(cmd))
    log_lines.append(f"compile_returncode: {result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    npubin_path = work / "kernel.o"
    if not npubin_path.exists():
        raise RuntimeError(f"expected binary missing: {npubin_path}")

    metadata_json = metadata["debug_metadata_json"]
    (OUT / "metadata.json").write_text(json.dumps(metadata, default=vars, indent=2), encoding="utf-8")
    (OUT / "instrumented.ttir.mlir").write_text(ttir_text, encoding="utf-8")
    (OUT / "ttadapter.mlir").write_text(parsed_ttadapter, encoding="utf-8")
    (OUT / "kernel.npubin").write_bytes(npubin_path.read_bytes())
    log_lines.append(f"debug_kernel_id: {metadata['debug_kernel_id']}")
    log_lines.append(f"tracked_ops: {len(metadata['debug_tracked_table'])}")
    return source, backend, metadata, npubin_path.read_bytes(), metadata_json


def input_arrays():
    x = np.arange(BLOCK_SIZE, dtype=np.float32)
    a = np.array(
        [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, np.inf, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0, 8.0],
        dtype=np.float32,
    )
    b = np.array(
        [2.0, 0.0, 3.0, -1.0, 2.0, -2.0, 1.0, 0.5, 2.0, -1.0, -1.0, 0.0, 2.0, 1.0, -0.5, 0.25],
        dtype=np.float32,
    )
    p = a * b
    q = p + x
    c = np.where(np.isfinite(q), q, 0.0).astype(np.float32)
    return x, a, b, c


def expected_tensors(x, a, b, c):
    p = a * b
    q = p + x
    r = q - c
    s = r * r
    y = s + q
    return [p, q, r, s, y]


def summary_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64)
    stats: dict[str, float | int] = {
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "zero_count": int((finite & (values == 0.0)).sum()),
        "element_count": int(values.size),
        "mean": 0.0,
        "min": 0.0,
        "max": 0.0,
        "l2_norm": 0.0,
    }
    if finite_values.size:
        stats["mean"] = float(finite_values.mean())
        stats["min"] = float(finite_values.min())
        stats["max"] = float(finite_values.max())
        stats["l2_norm"] = float(math.sqrt(float(np.square(finite_values).sum())))
    return stats


def check_close(name: str, actual: float, expected: float, *, atol: float = 2e-3) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-4, abs_tol=atol):
        raise AssertionError(f"{name}: actual={actual} expected={expected}")


def validate_records(decoded: dict, metadata: dict, expected_values: list[np.ndarray], log_lines: list[str]) -> None:
    records = decoded["records"]
    tracked = metadata["debug_tracked_table"]
    if len(tracked) != len(expected_values):
        raise AssertionError(f"tracked op count mismatch: {len(tracked)} vs {len(expected_values)}")
    expected_record_count = len(tracked) * 8
    if len(records) != expected_record_count:
        raise AssertionError(f"record count mismatch: {len(records)} vs {expected_record_count}")

    grouped: dict[int, dict[str, float | int]] = {}
    for record in records:
        metric = COLLECTOR_NAMES[int(record["collector_kind"])]
        if int(record["result_type"]) == 1:
            value: float | int = int(record["u64_value"])
        else:
            value = float(record["f32_value"])
        grouped.setdefault(int(record["op_id"]), {})[metric] = value

    for index, row in enumerate(tracked):
        op_id = int(row["opId"])
        expected = summary_stats(expected_values[index])
        actual = grouped.get(op_id, {})
        for metric in ("nan_count", "inf_count", "zero_count", "element_count"):
            if int(actual.get(metric, -1)) != int(expected[metric]):
                raise AssertionError(f"op {op_id} {metric}: actual={actual.get(metric)} expected={expected[metric]}")
        for metric in ("mean", "min", "max", "l2_norm"):
            check_close(f"op {op_id} {metric}", float(actual.get(metric, float('nan'))), float(expected[metric]))
        log_lines.append("validated_op: "
                         f"op_id={op_id} mlir_op={row['mlirOpName']} "
                         f"nan={actual['nan_count']} inf={actual['inf_count']} zero={actual['zero_count']} "
                         f"mean={actual['mean']} l2={actual['l2_norm']} elements={actual['element_count']}")


def run_probe() -> dict:
    os.environ["TRITON_FLAGTREE_DEBUG_LAUNCH_PTR"] = "1"
    register_acl_backend_strategy()
    make_launcher_no_pch = patch_launcher_build()

    import acl
    from triton._C.libtriton import debugger as dbg
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver

    active_launcher_module = sys.modules.get(driver.active.launcher_cls.__module__)
    if active_launcher_module is not None:
        active_launcher_module.make_npu_launcher_stub = make_launcher_no_pch

    log_lines: list[str] = []
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("*"):
        if path.is_file():
            path.unlink()

    source, backend, metadata, npubin, metadata_json = compile_kernel(log_lines)

    ret = acl.init()
    if ret not in (0, None, 100002):
        raise RuntimeError(f"acl.init failed: {ret}")
    ret = acl.rt.set_device(0)
    if ret not in (0, None):
        raise RuntimeError(f"acl.rt.set_device(0) failed: {ret}")
    stream, ret = acl.rt.create_stream()
    if ret not in (0, None):
        raise RuntimeError(f"acl.rt.create_stream failed: {ret}")

    kernel_metadata = dict(metadata)
    target = kernel_metadata["target"]
    if isinstance(target, dict):
        kernel_metadata["target"] = GPUTarget(target["backend"], target["arch"], target["warp_size"])
    kernel_tuple = namedtuple("KernelMetadata", sorted(kernel_metadata.keys()))
    metadata_obj = kernel_tuple(**{k: kernel_metadata[k] for k in sorted(kernel_metadata.keys())})
    packed_metadata = backend.pack_metadata(metadata_obj)
    launcher = driver.active.launcher_cls(source, metadata_obj)
    loaded = driver.active.utils.load_binary(metadata_obj.name, npubin, metadata_obj.shared, 0)
    function = int(loaded[1])

    handle = dbg.prepare_launch(
        {
            "debug_enabled": True,
            "debug_kernel_id": metadata["debug_kernel_id"],
            "debug_kernel_name": json.loads(metadata_json)["kernelName"],
            "debug_backend_name": "ascend",
            "debug_target_name": acl.get_soc_name(),
            "debug_record_level": metadata["debug_record_level"],
            "debug_export_mode": metadata["debug_export_mode"],
            "debug_record_capacity": 128,
            "debug_metadata_json": metadata_json,
        },
        int(stream),
        {"buffers": [], "tensors": []},
    )
    launcher.debug_ctrl_ptr = int(handle.hidden_arg_value)
    log_lines.append(f"stream: {int(stream)}")
    log_lines.append(f"debug_hidden_arg: {launcher.debug_ctrl_ptr}")

    def dev_malloc(num_bytes: int) -> int:
        ptr, rc = acl.rt.malloc(num_bytes, 0)
        if rc not in (0, None):
            raise RuntimeError(f"acl.rt.malloc failed: {rc}")
        return int(ptr)

    def h2d(dst: int, array: np.ndarray) -> None:
        data = np.ascontiguousarray(array).tobytes()
        rc = acl.rt.memcpy(dst, len(data), acl.util.bytes_to_ptr(data), len(data), 1)
        if rc not in (0, None):
            raise RuntimeError(f"H2D memcpy failed: {rc}")

    def copy_from_device(src: int, num_bytes: int) -> bytes:
        host, rc = acl.rt.malloc_host(num_bytes)
        if rc not in (0, None):
            raise RuntimeError(f"malloc_host failed: {rc}")
        try:
            rc = acl.rt.memcpy(host, num_bytes, src, num_bytes, 2)
            if rc not in (0, None):
                raise RuntimeError(f"D2H memcpy failed: {rc}")
            return bytes(acl.util.ptr_to_bytes(host, num_bytes))
        finally:
            acl.rt.free_host(host)

    def d2h(src: int, count: int) -> np.ndarray:
        return np.frombuffer(copy_from_device(src, count * 4), dtype=np.float32).copy()

    x, a, b, c = input_arrays()
    expected = expected_tensors(x, a, b, c)
    y_init = np.full(BLOCK_SIZE, -999.0, dtype=np.float32)
    ptrs: list[int] = []
    try:
        size = BLOCK_SIZE * 4
        x_dev = dev_malloc(size)
        y_dev = dev_malloc(size)
        a_dev = dev_malloc(size)
        b_dev = dev_malloc(size)
        c_dev = dev_malloc(size)
        ptrs.extend([x_dev, y_dev, a_dev, b_dev, c_dev])
        for ptr, array in ((x_dev, x), (y_dev, y_init), (a_dev, a), (b_dev, b), (c_dev, c)):
            h2d(ptr, array)

        before = copy_from_device(launcher.debug_ctrl_ptr, 64)
        (OUT / "device_header_before_launch.bin").write_bytes(before)
        log_lines.append("header_before_words: " + repr(list(struct.unpack("<16I", before))))

        launcher(
            1,
            1,
            1,
            int(stream),
            function,
            packed_metadata,
            None,
            None,
            None,
            x_dev,
            y_dev,
            a_dev,
            b_dev,
            c_dev,
            BLOCK_SIZE,
        )
        rc = acl.rt.synchronize_stream(stream)
        if rc not in (0, None):
            raise RuntimeError(f"acl.rt.synchronize_stream failed: {rc}")

        after = copy_from_device(launcher.debug_ctrl_ptr, 160)
        (OUT / "device_header_after_launch.bin").write_bytes(after)
        log_lines.append("header_after_words: " + repr(list(struct.unpack("<40I", after))))
        y_out = d2h(y_dev, BLOCK_SIZE)
    finally:
        for ptr in ptrs:
            acl.rt.free(ptr)

    exported = handle.finish()
    raw = bytes(exported.get("raw_buffer", b""))
    decoded = dbg.decode_exported_run(exported)
    report = dbg.render_text_report(exported, metadata_json)
    validate_records(decoded, metadata, expected, log_lines)

    output_matches = bool(np.allclose(y_out, expected[-1], equal_nan=True))
    if not output_matches:
        raise AssertionError(f"output mismatch: y_out={y_out.tolist()} expected={expected[-1].tolist()}")
    log_lines.append("output_matches: true")
    log_lines.append(f"record_count: {len(decoded['records'])}")
    log_lines.append(f"decoded_header: {decoded['header']}")

    (OUT / "raw_buffer.bin").write_bytes(raw)
    (OUT / "decoded.json").write_text(json.dumps(decoded, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "debugger_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "output_matches": output_matches,
        "record_count": len(decoded["records"]),
        "header": decoded["header"],
        "tracked_ops": metadata["debug_tracked_table"],
        "x": x.tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "c": c.tolist(),
        "expected_outputs": [array.tolist() for array in expected],
        "y_out": y_out.tolist(),
        "report_path": str(OUT / "debugger_report.txt"),
    }
    (OUT / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "evaluation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    try:
        acl.rt.destroy_stream(stream)
    except Exception:
        pass
    return summary


if __name__ == "__main__":
    result = run_probe()
    print(
        json.dumps(
            {
                "output_matches": result["output_matches"], "record_count": result["record_count"], "header":
                result["header"]
            }, indent=2))
