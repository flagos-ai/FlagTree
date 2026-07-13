from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import pprint
import re
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PreparedKernelLaunch:
    kernel_args: tuple[int, ...] = ()
    finalize: Optional[Callable[[Optional[BaseException]], None]] = None


@dataclass(frozen=True)
class DebuggerConfig:
    enabled: bool = False
    record_level: int = 1
    addr_level: int = 0
    export_mode: str = "POST_KERNEL_EXPORT"
    record_capacity: int = 1024
    export_on_error: bool = False
    output_dir: str | None = "/tmp/flagtree_debugger_manual"
    export_raw_records: bool = False
    runtime_metadata_builder: Optional[Callable[[Any, Any, Sequence[Any]], Any]] = None
    export_handler: Optional[Callable[[dict[str, Any]], None]] = None


_DEFAULT_OUTPUT_DIR = Path("/tmp/flagtree_debugger_manual")
_DEFAULT_RECORD_CAPACITY = 1024
_DEFAULT_ADDR_LEVEL = 0
_DEFAULT_EXPORT_MODE = "POST_KERNEL_EXPORT"
_DEFAULT_EXPORT_ON_ERROR = False
_USE_CURRENT_CONFIG = object()
_USE_CURRENT_OUTPUT_DIR = _USE_CURRENT_CONFIG
_launch_prepare_hook = None
_output_dir: Path | None = _DEFAULT_OUTPUT_DIR
_record_capacity = _DEFAULT_RECORD_CAPACITY
_export_mode = _DEFAULT_EXPORT_MODE
_export_on_error = _DEFAULT_EXPORT_ON_ERROR
_raw_record_export_enabled = False
_active_config = DebuggerConfig()
_exported_runs: list[dict[str, Any]] = []
_CONFIG_KEYS = frozenset({
    "output_dir",
    "record_capacity",
    "export_mode",
    "export_on_error",
    "export_raw_records",
})
_DISABLED_BUILD_MESSAGE = ("FlagTree debugger support is not available in this build; rebuild with "
                           "-DFLAGTREE_ENABLE_DEBUGGER=ON")


def _normalize_kernel_args(kernel_args: Any) -> tuple[int, ...]:
    if kernel_args is None:
        return ()
    if isinstance(kernel_args, int):
        return (int(kernel_args), )
    if isinstance(kernel_args, Sequence):
        return tuple(int(arg) for arg in kernel_args)
    raise TypeError("debugger launch hook must return an int, a sequence of ints, or "
                    "PreparedKernelLaunch")


def _wrap_launch_prepare_hook(hook: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(hook)
    positional = [
        parameter for parameter in signature.parameters.values() if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL
                          for parameter in signature.parameters.values())
    if accepts_varargs or len(positional) >= 4:
        return hook
    if len(positional) == 3:
        return lambda metadata, stream, launch_metadata, kernel_args: hook(metadata, stream, launch_metadata)
    raise TypeError("debugger launch hook must accept "
                    "(metadata, stream, launch_metadata[, kernel_args])")


def _load_binding():
    return importlib.import_module("triton._C.libtriton").debugger


def is_available() -> bool:
    try:
        libtriton = importlib.import_module("triton._C.libtriton")
    except (ImportError, OSError):
        return False
    native_debugger = getattr(libtriton, "debugger", None)
    native_passes = getattr(getattr(libtriton, "passes", None), "flagtree_debug", None)
    return native_debugger is not None and native_passes is not None


def _require_available() -> None:
    if not is_available():
        raise RuntimeError(_DISABLED_BUILD_MESSAGE)


def _normalize_output_dir(path: Any) -> Path | None:
    if path is None:
        return None
    return Path(os.fspath(path)).expanduser()


def configure(config: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
    """Update debugger defaults used by the next ``enable_debug(level=...)``.

    Supported keys are ``output_dir``, ``record_capacity``, ``export_mode``,
    ``export_on_error``, and ``export_raw_records``. Keys not provided keep
    their current values.
    """
    global _output_dir, _record_capacity, _export_mode
    global _export_on_error, _raw_record_export_enabled

    updates = {}
    if config is not None:
        updates.update(dict(config))
    updates.update(kwargs)

    unknown = sorted(set(updates) - _CONFIG_KEYS)
    if unknown:
        raise TypeError(f"unknown debugger config key(s): {', '.join(unknown)}")

    if "output_dir" in updates:
        _output_dir = _normalize_output_dir(updates["output_dir"])
    if "record_capacity" in updates:
        capacity = int(updates["record_capacity"])
        if capacity <= 0:
            raise ValueError("debugger record capacity must be positive")
        _record_capacity = capacity
    if "export_mode" in updates:
        _export_mode = _normalize_export_mode(updates["export_mode"])
    if "export_on_error" in updates:
        _export_on_error = bool(updates["export_on_error"])
    if "export_raw_records" in updates:
        _raw_record_export_enabled = bool(updates["export_raw_records"])


def reset_config() -> None:
    """Restore debugger defaults used by ``enable_debug(level=...)``."""
    configure(
        output_dir=_DEFAULT_OUTPUT_DIR,
        record_capacity=_DEFAULT_RECORD_CAPACITY,
        export_mode=_DEFAULT_EXPORT_MODE,
        export_on_error=_DEFAULT_EXPORT_ON_ERROR,
        export_raw_records=False,
    )


def get_config() -> dict[str, Any]:
    """Return the current debugger defaults."""
    return {
        "output_dir": get_output_dir(),
        "record_capacity": int(_record_capacity),
        "export_mode": str(_export_mode),
        "export_on_error": bool(_export_on_error),
        "export_raw_records": bool(_raw_record_export_enabled),
    }


def set_output_dir(path: str | os.PathLike | None) -> None:
    """Compatibility wrapper for ``configure(output_dir=...)``."""
    configure(output_dir=path)


def get_output_dir() -> str | None:
    """Return the current automatic report export directory, or ``None``."""
    if _output_dir is None:
        return None
    return str(_output_dir)


def _normalize_export_mode(export_mode: str | int) -> str:
    if isinstance(export_mode, int):
        return "STREAMING_EXPORT" if export_mode == 2 else "POST_KERNEL_EXPORT"
    normalized = str(export_mode).strip().replace("-", "_").upper()
    if normalized in {"STREAMING", "STREAMING_EXPORT"}:
        return "STREAMING_EXPORT"
    return "POST_KERNEL_EXPORT"


def _normalize_addr_level(addr_level: int) -> int:
    value = int(addr_level)
    if value < 0 or value > 2:
        raise ValueError("debugger addr_level must be 0, 1, or 2")
    return value


def _derive_kernel_id(metadata_dict: dict[str, Any]) -> int:
    kernel_hash = metadata_dict.get("hash")
    if isinstance(kernel_hash, str) and kernel_hash:
        kernel_id = int(kernel_hash[:8], 16)
        return kernel_id or 1

    digest = hashlib.sha256(repr(sorted(metadata_dict.items())).encode("utf-8")).hexdigest()
    kernel_id = int(digest[:8], 16)
    return kernel_id or 1


def _target_to_name(target: Any) -> str:
    if isinstance(target, dict):
        arch = target.get("arch")
        if arch is not None:
            return str(arch)
        backend = target.get("backend")
        if backend is not None:
            return str(backend)
        return ""
    arch = getattr(target, "arch", None)
    if arch is not None:
        return str(arch)
    backend = getattr(target, "backend", None)
    if backend is not None:
        return str(backend)
    return ""


def _target_backend(target: Any) -> str:
    if isinstance(target, dict):
        backend = target.get("backend")
        return "" if backend is None else str(backend)
    backend = getattr(target, "backend", None)
    return "" if backend is None else str(backend)


def _normalize_backend_name(backend_name: Any) -> str:
    name = "" if backend_name is None else str(backend_name)
    return name


def _safe_filename_component(value: Any, fallback: str) -> str:
    text = "" if value is None else str(value)
    text = text.strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    safe = safe.strip("._-")
    return safe or fallback


def _current_script_stem() -> str:
    script = sys.argv[0] if sys.argv else ""
    if not script:
        return "interactive"
    stem = Path(script).stem
    return _safe_filename_component(stem, "interactive")


def _exported_run_meta(exported_run: dict[str, Any]) -> dict[str, Any]:
    meta = exported_run.get("meta")
    return meta if isinstance(meta, dict) else {}


def _render_export_summary(exported_run: dict[str, Any], decoded: dict[str, Any] | None,
                           metadata_dict: dict[str, Any]) -> str:
    meta = _exported_run_meta(exported_run)
    raw_buffer = exported_run.get("raw_buffer", b"")
    raw_size = len(raw_buffer) if hasattr(raw_buffer, "__len__") else 0
    lines = [
        "FlagTree Debug Export",
        f"kernel_name: {metadata_dict.get('debug_kernel_name') or metadata_dict.get('name') or '<unknown>'}",
        f"kernel_id: {meta.get('kernel_id', metadata_dict.get('debug_kernel_id', 0))}",
        f"run_id: {meta.get('run_id', '<unknown>')}",
        f"backend: {metadata_dict.get('debug_backend_name', '')}",
        f"target: {metadata_dict.get('debug_target_name', '')}",
        f"raw_buffer_bytes: {raw_size}",
    ]
    if decoded is not None:
        lines.extend([
            "",
            "Decoded Header",
            pprint.pformat(decoded.get("header", {}), sort_dicts=True),
        ])
    return "\n".join(lines)


def _build_report_path(output_dir: Path, exported_run: dict[str, Any], metadata_dict: dict[str, Any]) -> Path:
    meta = _exported_run_meta(exported_run)
    script_stem = _current_script_stem()
    kernel_name = _safe_filename_component(
        metadata_dict.get("debug_kernel_name") or metadata_dict.get("name"),
        "kernel",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_id = _safe_filename_component(meta.get("run_id", "0"), "0")
    return output_dir / f"{script_stem}_{kernel_name}_{timestamp}_run{run_id}.txt"


def _render_raw_records(exported_run: dict[str, Any], decoded: dict[str, Any], metadata_dict: dict[str, Any]) -> str:
    meta = _exported_run_meta(exported_run)
    lines = [
        "FlagTree Debug Raw Records",
        f"kernel_name: {metadata_dict.get('debug_kernel_name') or metadata_dict.get('name') or '<unknown>'}",
        f"kernel_id: {meta.get('kernel_id', metadata_dict.get('debug_kernel_id', 0))}",
        f"run_id: {meta.get('run_id', '<unknown>')}",
        "",
        "Decoded Header",
        pprint.pformat(decoded.get("header", {}), sort_dicts=True),
        "",
        "Decoded Records",
        pprint.pformat(decoded.get("records", []), sort_dicts=True),
    ]
    return "\n".join(lines)


def _is_full_dump_run(metadata_dict: dict[str, Any]) -> bool:
    return (int(metadata_dict.get("debug_record_level", 1)) == 2
            and int(metadata_dict.get("debug_full_dump_payload_bytes_per_instance", 0)) > 0
            and bool(metadata_dict.get("debug_full_dump_plan")))


def _record_level_id(value: Any) -> int:
    if isinstance(value, int):
        return 2 if value == 2 else 1
    text = str(value)
    return 2 if text in {"2", "LEVEL_TENSOR_FULL"} else 1


def _export_mode_id(value: Any) -> int:
    if isinstance(value, int):
        return 2 if value == 2 else 1
    text = str(value)
    return 2 if text == "STREAMING_EXPORT" else 1


def _empty_debug_raw_buffer(metadata_dict: Mapping[str, Any]) -> bytes:
    record_size = int(metadata_dict.get("debug_record_size", 64) or 64)
    fields = (
        0,  # writeIdx
        0,  # capacity
        0,  # overflowCount
        0,  # flags
        record_size,
        32,  # payloadOffset
        0,
        0,
    )
    return b"".join(int(field).to_bytes(4, "little", signed=False) for field in fields)


def _metadata_only_exported_run(metadata_dict: dict[str, Any], runtime_metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "meta": {
            "run_id": len(_exported_runs) + 1,
            "device_id": int(metadata_dict.get("debug_device_id", 0) or 0),
            "kernel_id": int(metadata_dict.get("debug_kernel_id", _derive_kernel_id(metadata_dict))),
            "protocol_version": int(metadata_dict.get("debug_protocol_version", 2) or 2),
            "record_level": _record_level_id(metadata_dict.get("debug_record_level", 1)),
            "export_mode": _export_mode_id(metadata_dict.get("debug_export_mode", _active_config.export_mode)),
            "backend_kind": 0,
        },
        "runtime_metadata": dict(runtime_metadata),
        "raw_buffer": _empty_debug_raw_buffer(metadata_dict),
    }


def _npy_dtype_descriptor(dtype: str) -> str:
    if dtype == "float32":
        return "<f4"
    if dtype == "float64":
        return "<f8"
    if dtype == "int64":
        return "<i8"
    if dtype == "uint64":
        return "<u8"
    raise ValueError(f"unsupported debugger artifact dtype: {dtype}")


def _npy_dtype_element_bytes(dtype: str) -> int:
    if dtype in {"float32"}:
        return 4
    if dtype in {"float64", "int64", "uint64"}:
        return 8
    raise ValueError(f"unsupported debugger artifact dtype: {dtype}")


def _write_npy(path: Path, payload: bytes, dtype: str, shape: Sequence[int]) -> None:
    descr = _npy_dtype_descriptor(dtype)
    dims = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in dims):
        raise ValueError("debugger artifact shape cannot contain negative dimensions")
    element_count = 1
    for dim in dims:
        element_count *= dim
    expected_bytes = element_count * _npy_dtype_element_bytes(dtype)
    if len(payload) != expected_bytes:
        raise RuntimeError("debugger artifact payload size does not match planned dtype/shape")
    if len(dims) == 1:
        shape_repr = f"({dims[0]},)"
    else:
        shape_repr = "(" + ", ".join(str(dim) for dim in dims) + ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, }}"
    header_bytes = header.encode("latin1")
    padding = (16 - ((10 + len(header_bytes) + 1) % 16)) % 16
    header_bytes = header_bytes + b" " * padding + b"\n"
    if len(header_bytes) > 0xFFFF:
        raise ValueError("debugger artifact .npy header is too large")
    path.write_bytes(b"\x93NUMPY\x01\x00" + len(header_bytes).to_bytes(2, "little") + header_bytes + payload)


def _full_dump_plan_by_record(metadata_dict: dict[str, Any]) -> dict[int, dict[str, Any]]:
    plan = metadata_dict.get("debug_full_dump_plan") or []
    result = {}
    for entry in plan:
        if not isinstance(entry, Mapping):
            continue
        result[int(entry.get("record_index", 0))] = dict(entry)
    return result


def _record_index_for_slot(slot_index: int, runtime_metadata: Mapping[str, Any]) -> int:
    records_per_instance = int(runtime_metadata.get("records_per_instance") or 0)
    if records_per_instance <= 0:
        return slot_index
    return slot_index % records_per_instance


def _write_full_dump_artifacts(report_path: Path, exported_run: dict[str, Any], decoded: dict[str, Any],
                               metadata_dict: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_metadata = dict(exported_run.get("runtime_metadata") or {})
    plan_by_record = _full_dump_plan_by_record(metadata_dict)
    raw_buffer = bytes(exported_run.get("raw_buffer", b""))
    header = decoded.get("header", {})
    if int(header.get("overflow_count", 0)) != 0 or int(header.get("flags", 0)) & 1:
        raise RuntimeError("level-2 debugger full dump cannot export from an overflowed debug buffer")

    artifact_dir = report_path.with_suffix("")
    artifact_dir = artifact_dir.with_name(f"{artifact_dir.name}_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    records = decoded.get("records", [])
    for slot_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("record_kind") != "FULL_VALUE":
            continue
        record_index = _record_index_for_slot(slot_index, runtime_metadata)
        plan = plan_by_record.get(record_index)
        if plan is None:
            raise RuntimeError(f"missing full-dump plan for record_index={record_index}")
        payload_offset = int(record.get("payload_offset", 0))
        payload_length = int(record.get("payload_length", 0))
        if payload_length <= 0:
            raise RuntimeError(f"empty full-dump payload for record_index={record_index}")
        if payload_offset < 0 or payload_offset + payload_length > len(raw_buffer):
            raise RuntimeError(f"full-dump payload range is outside raw buffer for record_index={record_index}")
        payload = raw_buffer[payload_offset:payload_offset + payload_length]
        dtype = str(plan.get("artifact_dtype", ""))
        shape = plan.get("shape") or [int(plan.get("element_count", 0))]
        kind = str(plan.get("kind", "value"))
        op_id = int(record.get("op_id", plan.get("op_id", 0)))
        instance_id = int(record.get("logical_instance_id", 0))
        stem = (f"op{op_id}_inst{instance_id}_rec{record_index}_"
                f"{_safe_filename_component(kind, 'dump')}.npy")
        artifact_path = artifact_dir / stem
        _write_npy(artifact_path, payload, dtype, shape)
        artifacts.append({
            "op_id": op_id,
            "logical_instance_id": instance_id,
            "record_index": record_index,
            "kind": kind,
            "source": plan.get("source", ""),
            "artifact_dtype": dtype,
            "shape": list(shape),
            "payload_offset": payload_offset,
            "payload_length": payload_length,
            "path": str(artifact_path),
        })

    expected_records = len(plan_by_record)
    if expected_records and not artifacts:
        raise RuntimeError("level-2 debugger did not produce any full-dump artifacts")

    index_path = artifact_dir / "tensor_index.json"
    index = {
        "kernel_name": metadata_dict.get("debug_kernel_name") or metadata_dict.get("name") or "",
        "kernel_id": metadata_dict.get("debug_kernel_id", 0),
        "run_id": _exported_run_meta(exported_run).get("run_id", 0),
        "artifacts": artifacts,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    runtime_metadata["full_dump_artifacts"] = artifacts
    exported_run["runtime_metadata"] = runtime_metadata
    exported_run["full_dump_artifact_dir"] = str(artifact_dir)
    exported_run["full_dump_index_path"] = str(index_path)
    return artifacts


def _finalize_exported_run(exported_run: dict[str, Any], metadata_dict: dict[str, Any]) -> dict[str, Any]:
    exported_run["debug_kernel_name"] = str(metadata_dict.get("debug_kernel_name") or metadata_dict.get("name") or "")
    tracked_table = metadata_dict.get("debug_tracked_table")
    if isinstance(tracked_table, Sequence) and not isinstance(tracked_table, (str, bytes, bytearray)):
        exported_run["debug_tracked_table"] = list(tracked_table)

    binding = _load_binding()
    decoded = binding.decode_exported_run(exported_run)
    exported_run["decoded"] = decoded

    output_dir = _output_dir
    report_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = _build_report_path(output_dir, exported_run, metadata_dict)

    if _is_full_dump_run(metadata_dict):
        if report_path is None:
            raise RuntimeError("level-2 debugger full dump requires debugger output_dir")
        _write_full_dump_artifacts(report_path, exported_run, decoded, metadata_dict)

    summary = _render_export_summary(exported_run, decoded, metadata_dict)
    report = ""
    json_report = ""
    op_log_report = ""
    op_log_json_report = ""
    metadata_json = metadata_dict.get("debug_metadata_json")
    if metadata_json:
        render_text_statement_report = getattr(binding, "render_text_statement_report", None)
        if callable(render_text_statement_report):
            report = render_text_statement_report(exported_run, str(metadata_json))
        else:
            report = binding.render_text_report(exported_run, str(metadata_json))
        exported_run["report"] = report

        render_text_op_log_report = getattr(binding, "render_text_op_log_report", None)
        if callable(render_text_op_log_report):
            op_log_report = render_text_op_log_report(exported_run, str(metadata_json))
            exported_run["op_log_report"] = op_log_report

        render_json_statement_report = getattr(binding, "render_json_statement_report", None)
        render_json_report = (render_json_statement_report if callable(render_json_statement_report) else getattr(
            binding, "render_json_report", None))
        if callable(render_json_report):
            json_report = render_json_report(exported_run, str(metadata_json))
            exported_run["json_report"] = json_report

        render_json_op_log_report = getattr(binding, "render_json_op_log_report", None)
        if callable(render_json_op_log_report):
            op_log_json_report = render_json_op_log_report(exported_run, str(metadata_json))
            exported_run["op_log_json_report"] = op_log_json_report

    report_text = summary
    if report:
        report_text += "\n\n"
        report_text += report.lstrip("\n")

    op_log_report_text = summary
    if op_log_report:
        op_log_report_text += "\n\n"
        op_log_report_text += op_log_report.lstrip("\n")

    if report_path is not None:
        report_path.write_text(report_text)
        exported_run["report_path"] = str(report_path)
        if op_log_report:
            op_log_report_path = report_path.with_name(f"{report_path.stem}_op_log.txt")
            op_log_report_path.write_text(op_log_report_text)
            exported_run["op_log_report_path"] = str(op_log_report_path)
        if json_report:
            json_report_path = report_path.with_suffix(".json")
            json_report_path.write_text(json_report)
            exported_run["json_report_path"] = str(json_report_path)
        if op_log_json_report:
            op_log_json_report_path = report_path.with_name(f"{report_path.stem}_op_log.json")
            op_log_json_report_path.write_text(op_log_json_report)
            exported_run["op_log_json_report_path"] = str(op_log_json_report_path)
        if _active_config.export_raw_records:
            raw_records_path = report_path.with_name(f"{report_path.stem}_raw_records.txt")
            raw_records_path.write_text(_render_raw_records(exported_run, decoded, metadata_dict))
            exported_run["raw_records_path"] = str(raw_records_path)

    return exported_run


def _normalize_device_id(device: Any) -> int:
    if isinstance(device, int):
        return device
    index = getattr(device, "index", None)
    if isinstance(index, int):
        return index
    return 0


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    asdict = getattr(metadata, "_asdict", None)
    if callable(asdict):
        return dict(asdict())
    if hasattr(metadata, "__dict__"):
        return dict(vars(metadata))
    return {}


def _materialize_launch_metadata(launch_metadata: Any) -> Any:
    if launch_metadata is None or isinstance(launch_metadata, dict):
        return launch_metadata
    getter = getattr(launch_metadata, "get", None)
    if callable(getter):
        try:
            return getter()
        except TypeError:
            return launch_metadata
    return launch_metadata


def _normalize_launch_grid(launch_metadata: Any) -> tuple[int, int, int] | None:
    if not isinstance(launch_metadata, Mapping):
        return None
    grid = launch_metadata.get("grid")
    if grid is None:
        return None
    values = tuple(int(dim) for dim in grid)
    if not values:
        return None
    return (
        values[0],
        values[1] if len(values) > 1 else 1,
        values[2] if len(values) > 2 else 1,
    )


def _build_launch_metadata_dict(metadata: Any) -> dict[str, Any]:
    metadata_dict = _metadata_to_dict(metadata)
    target = metadata_dict.get("target")
    target_name = _target_to_name(target)
    backend_name = (metadata_dict.get("debug_backend_name") or os.environ.get("FLAGTREE_BACKEND")
                    or metadata_dict.get("backend_name") or _target_backend(target) or "")
    backend_name = _normalize_backend_name(backend_name)

    launch_dict = dict(metadata_dict)
    launch_dict["debug_enabled"] = True
    launch_dict["debug_protocol_version"] = int(metadata_dict.get("debug_protocol_version", 2))
    launch_dict["debug_record_level"] = int(metadata_dict.get("debug_record_level", _active_config.record_level))
    launch_dict["debug_addr_level"] = _normalize_addr_level(
        metadata_dict.get("debug_addr_level", _active_config.addr_level))
    launch_dict["debug_export_mode"] = _normalize_export_mode(
        metadata_dict.get("debug_export_mode", _active_config.export_mode))
    launch_dict["debug_record_capacity"] = int(
        metadata_dict.get("debug_record_capacity", _active_config.record_capacity))
    launch_dict["debug_record_size"] = int(metadata_dict.get("debug_record_size", 32))
    launch_dict["debug_kernel_id"] = int(metadata_dict.get("debug_kernel_id", _derive_kernel_id(metadata_dict)))
    launch_dict["debug_kernel_name"] = str(metadata_dict.get("debug_kernel_name", metadata_dict.get("name", "")))
    launch_dict["debug_backend_name"] = str(backend_name)
    launch_dict["debug_target_name"] = str(metadata_dict.get("debug_target_name", target_name))
    try:
        from triton.runtime.driver import driver

        launch_dict["debug_device_id"] = _normalize_device_id(driver.active.get_current_device())
    except Exception:
        launch_dict["debug_device_id"] = 0
    return launch_dict


def _default_launch_prepare_hook(metadata: Any, stream: int, launch_metadata: Any,
                                 kernel_args: Sequence[Any]) -> PreparedKernelLaunch:
    launch_metadata = _materialize_launch_metadata(launch_metadata)
    metadata_dict = _build_launch_metadata_dict(metadata)
    runtime_metadata = {}
    if _active_config.runtime_metadata_builder is not None:
        built_runtime_metadata = _active_config.runtime_metadata_builder(metadata, launch_metadata, kernel_args)
        if built_runtime_metadata is not None:
            runtime_metadata = dict(built_runtime_metadata)
    grid = _normalize_launch_grid(launch_metadata)
    if grid is not None:
        runtime_metadata.setdefault("grid", grid)
    records_per_instance = metadata_dict.get("debug_records_per_instance")
    if records_per_instance is not None:
        runtime_metadata.setdefault("records_per_instance", int(records_per_instance))
    if metadata_dict.get("debug_record_layout"):
        runtime_metadata.setdefault("record_layout", metadata_dict["debug_record_layout"])
    if metadata_dict.get("debug_record_plan") is not None:
        runtime_metadata.setdefault("record_plan", metadata_dict["debug_record_plan"])
    if _is_full_dump_run(metadata_dict):
        if _output_dir is None:
            raise RuntimeError("level-2 debugger full dump requires debugger output_dir")
        runtime_metadata.setdefault("full_dump_plan", metadata_dict.get("debug_full_dump_plan", []))

    handle = _load_binding().prepare_launch(metadata_dict, int(stream), runtime_metadata)

    def finalize(error: Optional[BaseException]) -> None:
        if error is not None and not _active_config.export_on_error:
            handle.release()
            return

        exported_run = handle.finish()
        exported_run = _finalize_exported_run(exported_run, metadata_dict)
        _exported_runs.append(exported_run)
        if _active_config.export_handler is not None:
            _active_config.export_handler(exported_run)

    return PreparedKernelLaunch(
        kernel_args=(int(handle.hidden_arg_value), ),
        finalize=finalize,
    )


def prepare_metadata_only_kernel_launch(
    metadata: Any,
    stream: int,
    launch_metadata: Any = None,
    kernel_args: Optional[Sequence[Any]] = None,
) -> Optional[PreparedKernelLaunch]:
    del stream
    enabled = bool(getattr(metadata, "debug_enabled", False)) or _active_config.enabled
    if not enabled:
        return None

    launch_metadata = _materialize_launch_metadata(launch_metadata)
    metadata_dict = _build_launch_metadata_dict(metadata)
    runtime_metadata: dict[str, Any] = {}
    if _active_config.runtime_metadata_builder is not None:
        built_runtime_metadata = _active_config.runtime_metadata_builder(metadata, launch_metadata,
                                                                         tuple(kernel_args or ()))
        if built_runtime_metadata is not None:
            runtime_metadata = dict(built_runtime_metadata)
    grid = _normalize_launch_grid(launch_metadata)
    if grid is not None:
        runtime_metadata.setdefault("grid", grid)
    runtime_metadata.setdefault(
        "records_per_instance",
        int(metadata_dict.get("debug_records_per_instance", 0) or 0),
    )
    if metadata_dict.get("debug_record_layout"):
        runtime_metadata.setdefault("record_layout", metadata_dict["debug_record_layout"])
    if metadata_dict.get("debug_record_plan") is not None:
        runtime_metadata.setdefault("record_plan", metadata_dict["debug_record_plan"])

    def finalize(error: Optional[BaseException]) -> None:
        if error is not None and not _active_config.export_on_error:
            return

        exported_run = _metadata_only_exported_run(metadata_dict, runtime_metadata)
        exported_run = _finalize_exported_run(exported_run, metadata_dict)
        _exported_runs.append(exported_run)
        if _active_config.export_handler is not None:
            _active_config.export_handler(exported_run)

    return PreparedKernelLaunch(kernel_args=(), finalize=finalize)


def register_launch_prepare_hook(hook: Callable[..., Any]) -> None:
    global _launch_prepare_hook
    _launch_prepare_hook = _wrap_launch_prepare_hook(hook)


def clear_launch_prepare_hook() -> None:
    global _launch_prepare_hook
    _launch_prepare_hook = None


def is_active() -> bool:
    return _active_config.enabled


def current_compile_config() -> dict[str, Any]:
    if not _active_config.enabled:
        return {}
    return {
        "debug_enabled": True,
        "debug_protocol_version": 2,
        "debug_record_level": int(_active_config.record_level),
        "debug_addr_level": int(_active_config.addr_level),
        "debug_export_mode": _normalize_export_mode(_active_config.export_mode),
        "debug_record_capacity": int(_active_config.record_capacity),
    }


def activate(
    *,
    level: int | None = None,
    addr_level: int = _DEFAULT_ADDR_LEVEL,
    record_level: int | None = None,
    export_mode: Any = _USE_CURRENT_CONFIG,
    record_capacity: Any = _USE_CURRENT_CONFIG,
    export_on_error: Any = _USE_CURRENT_CONFIG,
    output_dir: Any = _USE_CURRENT_OUTPUT_DIR,
    export_raw_records: Any = _USE_CURRENT_CONFIG,
    runtime_metadata_builder: Optional[Callable[[Any, Any, Sequence[Any]], Any]] = None,
    export_handler: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    global _active_config
    _require_available()
    if output_dir is not _USE_CURRENT_OUTPUT_DIR:
        configure(output_dir=output_dir)

    if level is not None and record_level is not None:
        raise TypeError("use either level or record_level, not both")
    effective_level = record_level if record_level is not None else level
    if effective_level is None:
        effective_level = 1
    effective_addr_level = _normalize_addr_level(addr_level)
    effective_export_mode = (_export_mode
                             if export_mode is _USE_CURRENT_CONFIG else _normalize_export_mode(export_mode))
    effective_record_capacity = (_record_capacity if record_capacity is _USE_CURRENT_CONFIG else int(record_capacity))
    if effective_record_capacity <= 0:
        raise ValueError("debugger record capacity must be positive")
    effective_export_on_error = (_export_on_error if export_on_error is _USE_CURRENT_CONFIG else bool(export_on_error))
    effective_export_raw_records = (_raw_record_export_enabled
                                    if export_raw_records is _USE_CURRENT_CONFIG else bool(export_raw_records))

    _active_config = DebuggerConfig(
        enabled=True,
        record_level=int(effective_level),
        addr_level=effective_addr_level,
        export_mode=effective_export_mode,
        record_capacity=effective_record_capacity,
        export_on_error=effective_export_on_error,
        output_dir=get_output_dir(),
        export_raw_records=effective_export_raw_records,
        runtime_metadata_builder=runtime_metadata_builder,
        export_handler=export_handler,
    )
    register_launch_prepare_hook(_default_launch_prepare_hook)

    try:
        import triton

        triton.knobs.compilation.instrumentation_mode = "debugger"
    except Exception:
        pass


def deactivate() -> None:
    global _active_config
    clear_launch_prepare_hook()
    _active_config = DebuggerConfig()

    try:
        import triton

        if str(triton.knobs.compilation.instrumentation_mode).startswith("debugger"):
            triton.knobs.compilation.instrumentation_mode = ""
    except Exception:
        pass


def clear_exported_runs() -> None:
    _exported_runs.clear()


def peek_exported_runs() -> list[dict[str, Any]]:
    return list(_exported_runs)


def take_exported_runs() -> list[dict[str, Any]]:
    exported_runs = list(_exported_runs)
    _exported_runs.clear()
    return exported_runs


def prepare_kernel_launch(metadata: Any, stream: int, launch_metadata: Any = None,
                          kernel_args: Optional[Sequence[Any]] = None) -> Optional[PreparedKernelLaunch]:
    enabled = bool(getattr(metadata, "debug_enabled", False)) or _active_config.enabled
    if not enabled:
        return None

    if _launch_prepare_hook is None:
        raise RuntimeError("debug-enabled kernel launch requires "
                           "triton.runtime.debugger.register_launch_prepare_hook(...)")

    prepared = _launch_prepare_hook(
        metadata,
        int(stream),
        launch_metadata,
        tuple(kernel_args or ()),
    )
    if prepared is None:
        raise RuntimeError("debugger launch prepare hook returned None for a debug-enabled "
                           "kernel")
    if isinstance(prepared, PreparedKernelLaunch):
        return PreparedKernelLaunch(
            kernel_args=_normalize_kernel_args(prepared.kernel_args),
            finalize=prepared.finalize,
        )
    return PreparedKernelLaunch(kernel_args=_normalize_kernel_args(prepared))


def finalize_prepared_launch(prepared: Optional[PreparedKernelLaunch], error: Optional[BaseException] = None) -> None:
    if prepared is None or prepared.finalize is None:
        return
    prepared.finalize(error)
