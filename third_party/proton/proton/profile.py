import functools
import triton
import os
import json
from pathlib import Path
import pathlib

from triton._C.libproton import proton as libproton
from .hooks import HookManager, InstrumentationHook, LaunchHook
from .flags import set_profiling_off, set_profiling_on, is_command_line
from typing import Optional

DEFAULT_PROFILE_NAME = "proton"
_CANN_TRITON_LEGACY_ENV = "PROTON_CANN_TRITON_HOOK_LEGACY"
_IR_RECORD_BUFFER_MB_ENV = "PROTON_IR_RECORD_BUFFER_MB"
_IR_RECORD_SIZE_BYTES = 64
_DEFAULT_IR_RECORD_BUFFER_MB = 32
_active_sessions: dict[int, dict[str, object]] = {}


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on", "legacy"}


def _is_cann_backend(backend: Optional[str]) -> bool:
    return str(backend or "").lower() in {"cann", "ascend", "npu"}


def _uses_default_ir_triton_hook(backend: Optional[str], hook: Optional[str]) -> bool:
    return (hook == "triton" and _is_cann_backend(backend) and not _env_flag_enabled(_CANN_TRITON_LEGACY_ENV))


def _mode_with_ir_triton_overrides(mode: Optional[str]) -> str:
    tokens = [token for token in str(mode or "").split(":") if token]
    tokens.extend([
        "aclprof_runtime_enabled=false",
        "aclprof_auto_export=false",
        "mstx_enabled=false",
        "aclprof_msproftx_enabled=false",
        "runtime_host_timing_fallback=false",
    ])
    return ":".join(tokens)


def _set_instrumentation_mode(value: str) -> None:
    try:
        from triton.compiler import flagtree_debug

        flagtree_debug.set_instrumentation_mode(value)
    except Exception:
        pass


def _instrumentation_record_capacity() -> int:
    value = os.getenv(_IR_RECORD_BUFFER_MB_ENV, "").strip()
    buffer_mb = _DEFAULT_IR_RECORD_BUFFER_MB if not value else int(value)
    if buffer_mb <= 0:
        raise ValueError(f"{_IR_RECORD_BUFFER_MB_ENV} must be a positive integer")
    return buffer_mb * 1024 * 1024 // _IR_RECORD_SIZE_BYTES


def _activate_instrumentation() -> None:
    from triton.runtime import debugger

    record_capacity = _instrumentation_record_capacity()
    debugger.clear_exported_runs()
    debugger.activate(
        record_level=1,
        addr_level=0,
        record_capacity=record_capacity,
        output_dir=None,
    )
    triton.knobs.compilation.instrumentation_mode = (f"debugger:record_capacity={record_capacity}")
    _set_instrumentation_mode("debugger_auto")


def _deactivate_instrumentation() -> None:
    from triton.runtime import debugger

    debugger.deactivate()
    _set_instrumentation_mode("")


def _take_instrumentation_runs() -> list[dict]:
    from triton.runtime import debugger
    from triton.runtime.debug_collect_runtime import default_debug_collect_runtime

    runs = list(debugger.take_exported_runs())
    runtime_runs = default_debug_collect_runtime.take_exported_runs()
    normalized = []
    for run in runs:
        meta = dict(run.get("meta") or {})
        runtime_metadata = dict(run.get("runtime_metadata") or {})
        decoded = dict(run.get("decoded") or {})
        normalized.append({
            "meta": meta,
            "runtime_metadata": runtime_metadata,
            "decoded": decoded,
            "debug_kernel_name": run.get("debug_kernel_name", ""),
            "debug_tracked_table": list(run.get("debug_tracked_table") or []),
        })
    for run in runtime_runs:
        normalized.append({
            "meta": dict(run.meta or {}),
            "runtime_metadata": dict(run.runtime_metadata or {}),
            "decoded": dict(run.decoded or {}),
            "report": run.report,
        })
    return normalized


def _tracked_op_name(entry: dict) -> str:
    mlir_op = str(entry.get("mlir_op") or entry.get("mlirOpName") or "")
    role = str(entry.get("role") or "")
    if role and role != "<none>":
        return f"{mlir_op} {role}".strip()
    return mlir_op or f"op_{entry.get('op_id', entry.get('opId', 0))}"


def _is_ir_control_flow_op(entry: dict) -> bool:
    mlir_op = str(entry.get("mlir_op") or entry.get("mlirOpName") or "")
    return mlir_op.startswith("scf.") or mlir_op.startswith("cf.")


def _records_by_op_instance(records: list[dict]) -> dict[tuple[int, int], dict[str, float]]:
    summaries: dict[tuple[int, int], dict[str, float]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        kind = record.get("record_kind")
        if kind not in {"SUMMARY_COUNT_BUNDLE_U64", "SUMMARY_VALUE_BUNDLE_F32"}:
            continue
        key = (
            int(record.get("op_id", 0)),
            int(record.get("logical_instance_id", 0)),
        )
        bucket = summaries.setdefault(key, {})
        if kind == "SUMMARY_COUNT_BUNDLE_U64":
            for name in ("nan_count", "inf_count", "zero_count", "element_count"):
                if name in record:
                    bucket[name] = float(record[name])
        else:
            for name in ("mean", "min", "max", "l2_norm"):
                if name in record:
                    bucket[name] = float(record[name])
    return summaries


def _name_matches_kernel(candidate: object, kernel_name: str) -> bool:
    candidate = str(candidate or "")
    return candidate == kernel_name or candidate.startswith(f"{kernel_name} ")


def _kernel_name_matches(event: dict, kernel_name: str) -> bool:
    if not kernel_name:
        return False
    args = event.get("args") or {}
    metrics = args.get("metrics") or {}
    candidates = [
        event.get("name"),
        args.get("name"),
        args.get("cann.op_name"),
        args.get("cann.op_type"),
        args.get("runtime.op_name"),
        metrics.get("cann.op_name"),
        metrics.get("cann.op_type"),
        metrics.get("runtime.op_name"),
    ]
    candidates.extend(args.get("call_stack") or [])
    for candidate in candidates:
        if _name_matches_kernel(candidate, kernel_name):
            return True
    return False


def _find_kernel_event(events: list[dict], kernel_name: str, used: set[int]) -> tuple[int, dict] | tuple[None, None]:
    if not kernel_name:
        return None, None
    for index, event in enumerate(events):
        if index in used:
            continue
        if event.get("ph") != "X":
            continue
        if event.get("cat") == "flagtree.ir_kernel":
            continue
        if _kernel_name_matches(event, kernel_name):
            return index, event
    return None, None


def _internal_timeline_events_for_run(run_index: int, run: dict, kernel_event: dict | None) -> list[dict]:
    records = (run.get("decoded") or {}).get("records") or []
    timeline_records = _timeline_records(run)
    if not timeline_records:
        return []

    tracked = {}
    for entry in run.get("debug_tracked_table") or []:
        if isinstance(entry, dict):
            tracked[int(entry.get("op_id", entry.get("opId", 0)))] = entry
    summaries = _records_by_op_instance(records)

    min_cycle = min(int(record.get("start_cycle", 0)) for record in timeline_records)
    max_cycle = max(int(record.get("end_cycle", 0)) for record in timeline_records)
    cycle_span = max(1, max_cycle - min_cycle)
    if kernel_event is not None:
        base_ts = float(kernel_event.get("ts", 0.0))
        base_dur = max(float(kernel_event.get("dur", 0.0)), 1.0)
        pid = kernel_event.get("pid", 0)
        base_tid = int(kernel_event.get("tid", 0))
    else:
        base_ts = 0.0
        base_dur = float(cycle_span)
        pid = 0
        base_tid = 100000 + run_index * 1000

    events = []
    for record in timeline_records:
        start = int(record.get("start_cycle", 0))
        end = int(record.get("end_cycle", 0))
        duration = int(record.get("duration_cycle", max(0, end - start)))
        if start == 0 and end == 0:
            continue
        if end < start or duration <= 0:
            continue
        op_id = int(record.get("op_id", 0))
        logical_instance_id = int(record.get("logical_instance_id", 0))
        entry = tracked.get(op_id, {})
        if _is_ir_control_flow_op(entry):
            continue
        scaled_ts = base_ts + ((start - min_cycle) / cycle_span) * base_dur
        scaled_dur = max((duration / cycle_span) * base_dur, 0.001 if duration > 0 else 0.0)
        args = {
            "timestamp_unit": "SYS_CNT cycles",
            "op_id": op_id,
            "logical_instance_id": logical_instance_id,
            "mlir_op": entry.get("mlir_op", entry.get("mlirOpName")),
            "source_loc": entry.get("source_loc", entry.get("sourceLoc")),
            "triton_statement": entry.get("triton_statement", entry.get("tritonStatement")),
            "op_category": entry.get("op_category", entry.get("opCategory")),
            "role": entry.get("role"),
            "start_cycle": start,
            "end_cycle": end,
            "duration_cycle": duration,
        }
        bytes_per_instance = _memory_access_bytes_per_instance(entry)
        if bytes_per_instance > 0:
            args["memory_access_bytes"] = bytes_per_instance
            direction = _memory_direction(entry)
            if direction:
                args["memory_direction"] = direction
        summary = summaries.get((op_id, logical_instance_id))
        if summary:
            args["summary"] = summary
        events.append({
            "name": _tracked_op_name(entry) if entry else f"op_{op_id}",
            "cat": "flagtree.kernel_internal",
            "ph": "X",
            "pid": pid,
            "tid": base_tid * 100000 + 10000 + logical_instance_id,
            "ts": scaled_ts,
            "dur": scaled_dur,
            "args": args,
        })
    return events


def _synthetic_kernel_event_for_run(
    run_index: int,
    run: dict,
    trace_events: list[dict],
    base_ts: float,
) -> dict | None:
    summary = _run_ir_summary(run)
    if not summary:
        return None
    kernel_name = str(run.get("debug_kernel_name") or f"instrumentation_run_{run_index}")
    cycle_span = max(summary["duration_cycle"], 1.0)
    tid = 500000 + run_index
    trace_events.append({
        "name": "thread_name",
        "ph": "M",
        "pid": 0,
        "tid": tid,
        "args": {"name": f"flagtree ir kernel {run_index}"},
    })
    metrics = {
        _ir_metric_name("start_cycle"): summary["start_cycle"],
        _ir_metric_name("end_cycle"): summary["end_cycle"],
        _ir_metric_name("duration_cycle"): summary["duration_cycle"],
        _ir_metric_name("op_event_count"): summary["op_event_count"],
        _ir_metric_name("memory_access_bytes"): summary["memory_access_bytes"],
        _ir_metric_name("memory_read_bytes"): summary["memory_read_bytes"],
        _ir_metric_name("memory_write_bytes"): summary["memory_write_bytes"],
    }
    if "estimated_bandwidth_bytes_per_cycle" in summary:
        metrics[_ir_metric_name("estimated_bandwidth_bytes_per_cycle")] = summary["estimated_bandwidth_bytes_per_cycle"]
    event = {
        "name": kernel_name,
        "cat": "flagtree.ir_kernel",
        "ph": "X",
        "pid": 0,
        "tid": tid,
        "ts": base_ts,
        "dur": cycle_span,
        "args": {
            "call_stack": ["ROOT", kernel_name],
            "source": "flagtree_ir",
            "kernel_sequence_index": run_index,
            "timestamp_unit": "SYS_CNT cycles",
            "metrics": metrics,
            **metrics,
        },
    }
    trace_events.append(event)
    return event


def _load_timeline(path: Path) -> dict:
    if path.exists():
        text = path.read_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.loads(text.splitlines()[0])
    return {
        "displayTimeUnit": "us",
        "traceEvents": [{
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "args": {"name": "FlagTree Proton IR"},
        }],
    }


def _augment_timeline(path: Path, runs: list[dict], synthesize_kernel_events: bool = False) -> None:
    timeline = _load_timeline(path)
    trace_events = timeline.setdefault("traceEvents", [])
    used_kernel_events: set[int] = set()
    new_events = []
    metadata_events = set()
    changed = not path.exists()
    synthetic_next_ts = 0.0
    for run_index, run in enumerate(runs):
        kernel_name = str(run.get("debug_kernel_name") or "")
        event_index, kernel_event = _find_kernel_event(trace_events, kernel_name, used_kernel_events)
        if event_index is not None:
            used_kernel_events.add(event_index)
            synthetic_next_ts = max(
                synthetic_next_ts,
                float(kernel_event.get("ts", 0.0)) + float(kernel_event.get("dur", 0.0)),
            )
        elif synthesize_kernel_events:
            kernel_event = _synthetic_kernel_event_for_run(
                run_index,
                run,
                trace_events,
                synthetic_next_ts,
            )
            changed = changed or kernel_event is not None
            if kernel_event is not None:
                kernel_end = float(kernel_event.get("ts", 0.0)) + float(kernel_event.get("dur", 0.0))
                synthetic_next_ts = kernel_end + max(
                    float(kernel_event.get("dur", 0.0)) * 0.02,
                    1.0,
                )
        events = _internal_timeline_events_for_run(run_index, run, kernel_event)
        for event in events:
            tid = event["tid"]
            if tid not in metadata_events:
                metadata_events.add(tid)
                trace_events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": event["pid"],
                    "tid": tid,
                    "args": {"name": f"kernel internal {run_index}:{tid % 10000}"},
                })
        new_events.extend(events)
    if not new_events and not changed:
        return
    trace_events.extend(new_events)
    path.write_text(json.dumps(timeline, separators=(",", ":"), default=str))


def _op_metric_name(name: str) -> str:
    return f"flagtree.internal.{name}"


def _ir_metric_name(name: str) -> str:
    return f"flagtree.ir.{name}"


def _tracked_by_op(run: dict) -> dict[int, dict]:
    tracked = {}
    for entry in run.get("debug_tracked_table") or []:
        if isinstance(entry, dict):
            tracked[int(entry.get("op_id", entry.get("opId", 0)))] = entry
    return tracked


def _numeric(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _entry_field(entry: dict, *names: str) -> object:
    current = entry
    for name in names:
        if isinstance(current, dict) and name in current:
            return current[name]
    return None


def _static_vec_width(entry: dict) -> float:
    result = entry.get("result") or {}
    width = _numeric(result.get("vecWidth", result.get("vec_width")), 0.0)
    if width > 0:
        return width
    shape = result.get("shape")
    if isinstance(shape, str) and shape:
        product = 1
        found = False
        for part in shape.replace("x", ",").replace(";", ",").split(","):
            value = part.strip()
            if value.isdigit():
                found = True
                product *= int(value)
        if found:
            return float(product)
    return 1.0


def _memory_access_bytes_per_instance(entry: dict) -> float:
    semantics = entry.get("memory_semantics") or {}
    access_bytes = _numeric(
        entry.get("accessBytes", entry.get("access_bytes", semantics.get("access_bytes"))),
        0.0,
    )
    if access_bytes <= 0:
        return 0.0
    return access_bytes * _static_vec_width(entry)


def _memory_direction(entry: dict) -> str:
    parts = [
        entry.get("role"),
        entry.get("op_category", entry.get("opCategory")),
        entry.get("accessType", entry.get("access_type")),
        entry.get("mlir_op", entry.get("mlirOpName")),
    ]
    text = " ".join(str(part or "").lower() for part in parts)
    if "store" in text or "write" in text:
        return "write"
    if "load" in text or "read" in text:
        return "read"
    return ""


def _add_memory_metrics(metrics: dict[str, float], entry: dict) -> None:
    count = metrics.get(_op_metric_name("count"), metrics.get(_ir_metric_name("count"), 0.0))
    per_instance_bytes = _memory_access_bytes_per_instance(entry)
    total_bytes = per_instance_bytes * count
    if total_bytes <= 0:
        return
    metrics[_ir_metric_name("memory_access_bytes")] = total_bytes
    metrics[_ir_metric_name("memory_bytes")] = total_bytes
    direction = _memory_direction(entry)
    if direction == "read":
        metrics[_ir_metric_name("memory_read_bytes")] = total_bytes
    elif direction == "write":
        metrics[_ir_metric_name("memory_write_bytes")] = total_bytes
    duration = metrics.get(_op_metric_name("duration_cycle"), 0.0)
    if duration > 0:
        metrics[_ir_metric_name("estimated_bandwidth_bytes_per_cycle")] = (total_bytes / duration)


def _timeline_records(run: dict) -> list[dict]:
    records = (run.get("decoded") or {}).get("records") or []
    return [record for record in records if isinstance(record, dict) and record.get("record_kind") == "TIMELINE"]


def _run_cycle_bounds(run: dict) -> tuple[int, int] | tuple[None, None]:
    timeline_records = _timeline_records(run)
    if not timeline_records:
        return None, None
    starts = [int(record.get("start_cycle", 0)) for record in timeline_records]
    ends = [int(record.get("end_cycle", 0)) for record in timeline_records]
    return min(starts), max(ends)


def _run_ir_summary(run: dict) -> dict[str, float]:
    min_cycle, max_cycle = _run_cycle_bounds(run)
    if min_cycle is None or max_cycle is None:
        return {}
    records = _timeline_records(run)
    tracked = _tracked_by_op(run)
    summary = {
        "start_cycle": float(min_cycle),
        "end_cycle": float(max_cycle),
        "duration_cycle": float(max(0, max_cycle - min_cycle)),
        "op_event_count": float(len(records)),
        "memory_access_bytes": 0.0,
        "memory_read_bytes": 0.0,
        "memory_write_bytes": 0.0,
    }
    for record in records:
        entry = tracked.get(int(record.get("op_id", 0)), {})
        bytes_per_instance = _memory_access_bytes_per_instance(entry)
        if bytes_per_instance <= 0:
            continue
        summary["memory_access_bytes"] += bytes_per_instance
        direction = _memory_direction(entry)
        if direction == "read":
            summary["memory_read_bytes"] += bytes_per_instance
        elif direction == "write":
            summary["memory_write_bytes"] += bytes_per_instance
    if summary["duration_cycle"] > 0 and summary["memory_access_bytes"] > 0:
        summary["estimated_bandwidth_bytes_per_cycle"] = (summary["memory_access_bytes"] / summary["duration_cycle"])
    return summary


def _internal_hatchet_metrics_for_run(run: dict) -> dict[int, dict[str, float]]:
    records = (run.get("decoded") or {}).get("records") or []
    tracked = _tracked_by_op(run)
    by_op: dict[int, dict[str, float]] = {}
    summary_counts: dict[int, dict[str, float]] = {}
    summary_values: dict[int, dict[str, list[float]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        op_id = int(record.get("op_id", 0))
        kind = record.get("record_kind")
        if kind == "TIMELINE":
            duration = float(record.get("duration_cycle", 0))
            bucket = by_op.setdefault(
                op_id, {
                    _op_metric_name("count"): 0.0,
                    _ir_metric_name("count"): 0.0,
                    _op_metric_name("duration_cycle"): 0.0,
                    _ir_metric_name("duration_cycle"): 0.0,
                    _op_metric_name("min_duration_cycle"): duration,
                    _ir_metric_name("min_duration_cycle"): duration,
                    _op_metric_name("max_duration_cycle"): duration,
                    _ir_metric_name("max_duration_cycle"): duration,
                })
            bucket[_op_metric_name("count")] += 1.0
            bucket[_ir_metric_name("count")] += 1.0
            bucket[_op_metric_name("duration_cycle")] += duration
            bucket[_ir_metric_name("duration_cycle")] += duration
            bucket[_op_metric_name("min_duration_cycle")] = min(bucket[_op_metric_name("min_duration_cycle")], duration)
            bucket[_ir_metric_name("min_duration_cycle")] = min(bucket[_ir_metric_name("min_duration_cycle")], duration)
            bucket[_op_metric_name("max_duration_cycle")] = max(bucket[_op_metric_name("max_duration_cycle")], duration)
            bucket[_ir_metric_name("max_duration_cycle")] = max(bucket[_ir_metric_name("max_duration_cycle")], duration)
        elif kind == "SUMMARY_COUNT_BUNDLE_U64":
            bucket = summary_counts.setdefault(op_id, {})
            for name in ("nan_count", "inf_count", "zero_count", "element_count"):
                if name in record:
                    metric = _op_metric_name(name)
                    bucket[metric] = bucket.get(metric, 0.0) + float(record[name])
        elif kind == "SUMMARY_VALUE_BUNDLE_F32":
            bucket = summary_values.setdefault(op_id, {})
            for name in ("mean", "min", "max", "l2_norm"):
                if name in record:
                    bucket.setdefault(_op_metric_name(name), []).append(float(record[name]))
    for op_id, metrics in by_op.items():
        count = metrics.get(_op_metric_name("count"), 0.0)
        if count > 0:
            metrics[_op_metric_name("avg_duration_cycle")] = (metrics[_op_metric_name("duration_cycle")] / count)
            metrics[_ir_metric_name("avg_duration_cycle")] = (metrics[_ir_metric_name("duration_cycle")] / count)
        metrics.update(summary_counts.get(op_id, {}))
        for name, values in summary_values.get(op_id, {}).items():
            if values:
                metrics[name] = sum(values) / len(values)
        _add_memory_metrics(metrics, tracked.get(op_id, {}))
    return by_op


def _find_hatchet_node(node: dict, name: str) -> dict | None:
    node_name = (node.get("frame") or {}).get("name")
    if node_name == name or _name_matches_kernel(node_name, name):
        return node
    for child in node.get("children") or []:
        found = _find_hatchet_node(child, name)
        if found is not None:
            return found
    return None


def _upsert_child(node: dict, name: str) -> dict:
    children = node.setdefault("children", [])
    for child in children:
        if (child.get("frame") or {}).get("name") == name:
            child.setdefault("metrics", {})
            child.setdefault("children", [])
            return child
    child = {
        "frame": {"name": name, "type": "function"},
        "metrics": {},
        "children": [],
    }
    children.append(child)
    return child


def _augment_hatchet(path: Path, runs: list[dict]) -> None:
    if not path.exists():
        return
    database = json.loads(path.read_text())
    if not isinstance(database, list) or not database or not isinstance(database[0], dict):
        return
    root = database[0]
    root_metrics = root.setdefault("metrics", {})
    root_metrics.setdefault("device_id", 0)
    changed = False
    for run_index, run in enumerate(runs):
        metrics_by_op = _internal_hatchet_metrics_for_run(run)
        if not metrics_by_op:
            continue
        tracked = _tracked_by_op(run)
        kernel_name = str(run.get("debug_kernel_name") or "")
        kernel_node = _find_hatchet_node(root, kernel_name) if kernel_name else None
        if kernel_node is None:
            kernel_node = _upsert_child(root, kernel_name or f"instrumentation_run_{run_index}")
        kernel_node.setdefault("metrics", {}).setdefault("device_id", 0)
        run_summary = _run_ir_summary(run)
        if run_summary:
            kernel_metrics = kernel_node.setdefault("metrics", {})
            kernel_metrics.setdefault(
                _ir_metric_name("kernel_elapsed_cycle"),
                run_summary["duration_cycle"],
            )
            root_metrics.setdefault(_ir_metric_name("kernel_elapsed_cycle"), 0)
            kernel_metrics.setdefault(
                _ir_metric_name("op_event_count"),
                run_summary["op_event_count"],
            )
            root_metrics.setdefault(_ir_metric_name("op_event_count"), 0)
            for name in (
                    "memory_access_bytes",
                    "memory_read_bytes",
                    "memory_write_bytes",
                    "estimated_bandwidth_bytes_per_cycle",
            ):
                if name in run_summary:
                    kernel_metrics.setdefault(_ir_metric_name(name), run_summary[name])
                    root_metrics.setdefault(_ir_metric_name(name), 0)
        for op_id in sorted(metrics_by_op):
            entry = tracked.get(op_id, {})
            if _is_ir_control_flow_op(entry):
                continue
            child_name = f"{op_id}:{_tracked_op_name(entry) if entry else f'op_{op_id}'}"
            child = _upsert_child(kernel_node, child_name)
            child_metrics = child.setdefault("metrics", {})
            child_metrics.setdefault("device_id", 0)
            for name, value in metrics_by_op[op_id].items():
                child_metrics[name] = value
                root_metrics.setdefault(name, 0)
        changed = True
    if changed:
        path.write_text("\n" + json.dumps(database, indent=4, default=str) + "\n")


def _load_json_object(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return dict(default)


def _append_unique(items: list, value: object) -> None:
    if value not in items:
        items.append(value)


def _filter_default_ir_cann_reasons(reasons: object) -> list:
    if not isinstance(reasons, list):
        return []
    ignored_fragments = (
        "no cann memory/bandwidth csv columns were imported",
        "no vendor summary csv files were found",
        "no runtime_base events could be imported from aclprof exports",
        "bandwidth requested, but no cann memory/bandwidth csv columns were imported",
    )
    filtered = []
    for reason in reasons:
        text = str(reason)
        lower = text.lower()
        if any(fragment in lower for fragment in ignored_fragments):
            continue
        filtered.append(reason)
    return filtered


def _augment_ir_meta(path: Path, runs: list[dict], ir_default_for_triton: bool) -> None:
    meta = _load_json_object(path, {"schema_version": 1})
    config = meta.setdefault("config", {})
    config["flagtree_ir_enabled"] = "true"
    config["flagtree_ir_runs"] = str(len(runs))
    config["flagtree_ir_kernel_events"] = str(sum(1 for run in runs if _timeline_records(run)))
    config["flagtree_ir_default_for_triton_hook"] = ("true" if ir_default_for_triton else "false")
    if ir_default_for_triton:
        meta["vendor_metrics_enabled"] = []
        config["cann_legacy_profiler_enabled"] = "false"
        config["aclprof_runtime_enabled"] = "false"
        config["aclprof_auto_export"] = "false"
        meta["degrade_reasons"] = _filter_default_ir_cann_reasons(meta.get("degrade_reasons", []))
        reasons = meta.setdefault("degrade_reasons", [])
        _append_unique(
            reasons,
            "CANN legacy aclprof is disabled by default for hook=triton; "
            "CANN-only hardware/runtime metrics are unavailable. Set "
            f"{_CANN_TRITON_LEGACY_ENV}=1 to restore the legacy CANN path.",
        )
    path.write_text(json.dumps(meta, indent=4, default=str) + "\n")


def _ir_associations_for_runs(runs: list[dict]) -> list[dict]:
    associations = []
    for run_index, run in enumerate(runs):
        kernel_name = str(run.get("debug_kernel_name") or f"instrumentation_run_{run_index}")
        summary = _run_ir_summary(run)
        if not summary:
            continue
        kernel_metrics = {
            "kernel_name": kernel_name,
            "start_cycle": summary["start_cycle"],
            "end_cycle": summary["end_cycle"],
            "duration_cycle": summary["duration_cycle"],
            "op_event_count": summary["op_event_count"],
            "memory_access_bytes": summary["memory_access_bytes"],
            "memory_read_bytes": summary["memory_read_bytes"],
            "memory_write_bytes": summary["memory_write_bytes"],
        }
        if "estimated_bandwidth_bytes_per_cycle" in summary:
            kernel_metrics["estimated_bandwidth_bytes_per_cycle"] = summary["estimated_bandwidth_bytes_per_cycle"]
        associations.append({
            "source": "flagtree_ir_kernel",
            "state": "collected",
            "note": "Collected from FlagTree IR instrumentation runtime buffer.",
            "runtime_event": {
                "correlation_id": run_index,
                "device_id": 0,
                "stream_id": 0,
                "task_id": run_index,
                "scope_id": 0,
                "op_name": kernel_name,
                "start_time_ns": 0,
                "end_time_ns": 0,
            },
            "metrics": kernel_metrics,
        })

        metrics_by_op = _internal_hatchet_metrics_for_run(run)
        tracked = _tracked_by_op(run)
        for op_id in sorted(metrics_by_op):
            entry = tracked.get(op_id, {})
            if _is_ir_control_flow_op(entry):
                continue
            op_metrics = {
                "kernel_name": kernel_name,
                "op_id": op_id,
                "op_name": _tracked_op_name(entry) if entry else f"op_{op_id}",
                "mlir_op": entry.get("mlir_op", entry.get("mlirOpName")),
                "source_loc": entry.get("source_loc", entry.get("sourceLoc")),
                "triton_statement": entry.get("triton_statement", entry.get("tritonStatement")),
            }
            for name, value in metrics_by_op[op_id].items():
                if name.startswith("flagtree.ir."):
                    op_metrics[name[len("flagtree.ir."):]] = value
            associations.append({
                "source": "flagtree_ir_op",
                "state": "collected",
                "note": "Collected from FlagTree IR instrumentation runtime buffer.",
                "runtime_event": {
                    "correlation_id": run_index,
                    "device_id": 0,
                    "stream_id": 0,
                    "task_id": run_index,
                    "scope_id": 0,
                    "op_name": kernel_name,
                    "start_time_ns": 0,
                    "end_time_ns": 0,
                },
                "metrics": op_metrics,
            })
    return associations


def _augment_ir_vendor(path: Path, runs: list[dict], ir_default_for_triton: bool) -> None:
    vendor = _load_json_object(
        path, {
            "schema_version": 1,
            "backend": "cann",
            "importer": "flagtree_ir",
            "requested_metrics": [],
            "enabled_metrics": [],
            "raw_inputs": [],
            "degrade_reasons": [],
            "associations": [],
        })
    if ir_default_for_triton:
        vendor["enabled_metrics"] = []
        vendor["degrade_reasons"] = _filter_default_ir_cann_reasons(vendor.get("degrade_reasons", []))
    enabled = vendor.setdefault("enabled_metrics", [])
    for metric in ("ir_timeline", "ir_memory"):
        _append_unique(enabled, metric)
    if ir_default_for_triton:
        reasons = vendor.setdefault("degrade_reasons", [])
        _append_unique(
            reasons,
            "CANN legacy aclprof is disabled by default for hook=triton; "
            "vendor associations in this file include FlagTree IR records. "
            f"Set {_CANN_TRITON_LEGACY_ENV}=1 to collect legacy CANN metrics.",
        )
    associations = vendor.setdefault("associations", [])
    associations.extend(_ir_associations_for_runs(runs))
    vendor["ir_summary"] = {
        "runs": len(runs),
        "kernel_events": sum(1 for run in runs if _timeline_records(run)),
        "source": "flagtree_ir",
        "default_for_triton_hook": ir_default_for_triton,
    }
    path.write_text(json.dumps(vendor, indent=4, default=str) + "\n")


def _augment_instrumentation_artifacts(
    name: str,
    runs: Optional[list[dict]] = None,
    *,
    ir_default_for_triton: bool = False,
) -> None:
    if runs is None:
        runs = _take_instrumentation_runs()
    if not runs:
        return
    base = Path(name)
    _augment_timeline(
        base.with_suffix(".timeline.json"),
        runs,
        synthesize_kernel_events=ir_default_for_triton,
    )
    _augment_hatchet(base.with_suffix(".hatchet"), runs)
    _augment_ir_meta(base.with_suffix(".meta.json"), runs, ir_default_for_triton)
    _augment_ir_vendor(base.with_suffix(".vendor.json"), runs, ir_default_for_triton)


def _drop_session(session: Optional[int]) -> None:
    session_state = _active_sessions.pop(session, None)
    if session_state:
        HookManager.unregister(session)
    if session_state and session_state.get("instrumentation_hook", False):
        still_active = any(state.get("instrumentation_hook", False) for state in _active_sessions.values())
        if not still_active:
            _deactivate_instrumentation()
    if not _active_sessions:
        set_profiling_off()


def _drop_all_sessions() -> None:
    has_instrumentation_session = any(state.get("instrumentation_hook", False) for state in _active_sessions.values())
    _active_sessions.clear()
    set_profiling_off()
    HookManager.unregister()
    if has_instrumentation_session:
        _deactivate_instrumentation()


def _select_backend() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "cuda":
        return "cupti"
    elif backend == "hip":
        return "roctracer"
    elif backend in {"ascend", "npu"}:
        return "cann"
    else:
        raise ValueError("No backend is available for the current target.")


def _check_env(backend: str) -> None:
    if backend == "roctracer":
        hip_device_envs = ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
        for env in hip_device_envs:
            if os.getenv(env, None) is not None:
                raise ValueError(
                    f"Proton does not work when the environment variable {env} is set on AMD GPUs. Please unset it and use `ROCR_VISIBLE_DEVICES` instead"
                )


def _get_backend_default_path(backend: str) -> str:
    if backend != "cupti":
        return ""
    lib_path = triton.knobs.proton.cupti_dir
    if lib_path is not None:
        return lib_path
    return str(pathlib.Path(__file__).parent.parent.absolute() / "backends" / "nvidia" / "lib" / "cupti")


def start(
    name: Optional[str] = None,
    *,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    hook: Optional[str] = None,
):
    """
    Start profiling with the given name and backend.

    Usage:

        ```python
        proton.start("my_profile")
        # do something
        proton.finalize()
        ```

    Args:
        name (str, optional): The name (with path) of the profiling session.
                              If not provided, the default name is "~/proton.hatchet".
        backend (str, optional): The backend to use for profiling.
                     Available options are [None, "cupti", "cupti_pcsampling", "roctracer", "cann"].
                                 Defaults to None, which automatically selects the backend matching the current active runtime.
        context (str, optional): The context to use for profiling.
                                 Available options are ["shadow", "python"].
                                 Defaults to "shadow".
        data (str, optional): The data structure to use for profiling.
                              Available options are ["tree"].
                              Defaults to "tree".
        mode (str, optional): Backend-specific mode string.
                      For "cann", one supported example is
                      "runtime_base:vendor_metrics=aicore,bandwidth".
        hook (str, optional): The hook to use for profiling.
                              Available options are [None, "triton", "instrumentation"].
                              Defaults to None.
    Returns:
        session (int): The session ID of the profiling session.
    """
    if is_command_line():
        # Ignore the start() call if the script is run from the command line.
        return

    if name is None:
        name = DEFAULT_PROFILE_NAME

    if backend is None:
        backend = _select_backend()

    _check_env(backend)

    use_triton_hook = hook == "triton" or (hook == "instrumentation" and _is_cann_backend(backend))
    ir_default_for_triton = _uses_default_ir_triton_hook(backend, hook)
    use_instrumentation_hook = (_is_cann_backend(backend) and (hook == "instrumentation" or ir_default_for_triton))
    use_native_instrumentation_hook = (hook == "instrumentation" and not _is_cann_backend(backend))
    effective_mode = (_mode_with_ir_triton_overrides(mode) if ir_default_for_triton else (mode or ""))
    set_profiling_on()
    instrumentation_was_active = any(state.get("instrumentation_hook", False) for state in _active_sessions.values())
    instrumentation_activated = use_instrumentation_hook and not instrumentation_was_active
    if instrumentation_activated:
        _activate_instrumentation()
    session = None
    try:
        session = libproton.start(
            name,
            context,
            data,
            backend,
            effective_mode,
            _get_backend_default_path(backend),
            hook or "",
        )
        if use_triton_hook:
            HookManager.register(LaunchHook(), session)
        if use_native_instrumentation_hook:
            HookManager.register(InstrumentationHook(mode), session)
    except Exception:
        if session is not None:
            HookManager.unregister(session)
        if instrumentation_activated:
            _deactivate_instrumentation()
        if not _active_sessions:
            set_profiling_off()
        raise
    if session in _active_sessions:
        if use_triton_hook:
            _active_sessions[session]["triton_hook"] = True
        if use_instrumentation_hook:
            _active_sessions[session]["instrumentation_hook"] = True
            _active_sessions[session]["name"] = name
            _active_sessions[session]["ir_default_for_triton"] = (_active_sessions[session].get(
                "ir_default_for_triton", False) or ir_default_for_triton)
        return session
    _active_sessions[session] = {
        "triton_hook": use_triton_hook,
        "instrumentation_hook": use_instrumentation_hook,
        "native_instrumentation_hook": use_native_instrumentation_hook,
        "ir_default_for_triton": ir_default_for_triton,
        "name": name,
    }
    return session


def activate(session: Optional[int] = 0) -> None:
    """
    Activate the specified session.
    The profiling session will be active and data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0 (the first session started.)

    Returns:
        None
    """
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be activated when running from the command line.")
    HookManager.activate(session)
    libproton.activate_all() if session is None else libproton.activate(session)


def deactivate(session: Optional[int] = 0) -> None:
    """
    Stop the specified session.
    The profiling session's data will still be in the memory, but no more data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0 (the first session started.)

    Returns:
        None
    """
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be deactivated when running from the command line.")
    HookManager.deactivate(session)
    libproton.deactivate_all() if session is None else libproton.deactivate(session)


def finalize(session: Optional[int] = None, output_format: str = "hatchet") -> None:
    """
    Finalizes a profiling session.
    Flush and write the profiling data to the file specified by the session name.

    Args:
        session (int, optional): The session ID to finalize. If None, all sessions are finalized. Defaults to None.
        output_format (str, optional): The output format for the profiling results.
                                       Aavailable options are ["hatchet"].

    Returns:
        None
    """
    if session is None:
        session_states = list(_active_sessions.values())
        try:
            libproton.finalize_all(output_format)
            instrumentation_states = [state for state in session_states if state.get("instrumentation_hook", False)]
            if instrumentation_states:
                runs = _take_instrumentation_runs()
                for state in instrumentation_states:
                    _augment_instrumentation_artifacts(
                        str(state.get("name") or DEFAULT_PROFILE_NAME),
                        runs,
                        ir_default_for_triton=bool(state.get("ir_default_for_triton", False)),
                    )
        finally:
            _drop_all_sessions()
    else:
        if is_command_line() and session != 0:
            raise ValueError("Only one session can be finalized when running from the command line.")
        session_state = _active_sessions.get(session, {})
        try:
            libproton.finalize(session, output_format)
            if session_state.get("instrumentation_hook", False):
                _augment_instrumentation_artifacts(
                    str(session_state.get("name") or DEFAULT_PROFILE_NAME),
                    ir_default_for_triton=bool(session_state.get("ir_default_for_triton", False)),
                )
        finally:
            _drop_session(session)


def _profiling(
    func,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    hook: Optional[str] = None,
):
    """
    Context manager for profiling. Internally use only.

    Args:
        See start() for the arguments.

    Returns:
        wrapper (function): The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        session = start(name, context=context, data=data, backend=backend, mode=mode, hook=hook)
        ret = func(*args, **kwargs)
        deactivate(session)
        return ret

    return wrapper


def profile(
    func=None,
    *,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    hook: Optional[str] = None,
):
    """
    Decorator for profiling.

    Usage:

    ```python
    @proton.profile
    def foo():
        pass
    ```

    Args:
        See start() for the arguments.

    Returns:
        decorator (function): The decorator function.
    """
    if func is None:
        # It's being used with parentheses, so return a decorator
        def decorator(f):
            return _profiling(f, name=name, context=context, data=data, backend=backend, mode=mode, hook=hook)

        return decorator
    else:
        # It's being used without parentheses, so apply the decorator directly
        return _profiling(func, name=name, context=context, data=data, backend=backend, mode=mode, hook=hook)
