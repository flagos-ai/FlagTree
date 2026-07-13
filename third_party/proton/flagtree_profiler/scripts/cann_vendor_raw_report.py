#!/usr/bin/env python3
"""Print important raw CANN metrics from profile.vendor.json."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

IMPORTANT_METRICS = [
    "task_type",
    "task_duration_us",
    "aicore_time_us",
    "aic_total_cycles",
    "aiv_time_us",
    "aiv_total_cycles",
    "memory_access_bytes",
    "memory_read_bytes",
    "memory_write_bytes",
    "aic_read_main_memory_datas_kb",
    "aic_write_main_memory_datas_kb",
    "aic_gm_to_l1_datas_kb",
    "aic_l0c_to_gm_datas_kb",
    "aiv_read_main_memory_datas_kb",
    "aiv_write_main_memory_datas_kb",
    "bandwidth_gb_s",
    "memory_read_bandwidth_gb_s",
    "memory_write_bandwidth_gb_s",
    "block_dim",
    "mix_block_dim",
    "op_state",
    "hf32_eligible",
    "input_shapes",
    "output_shapes",
    "input_data_types",
    "output_data_types",
]


def _metric_value(metrics: dict[str, Any], name: str) -> Any:
    return metrics.get(name)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _kernel_name(association: dict[str, Any]) -> str:
    metrics = association.get("metrics", {})
    event = association.get("runtime_event", {})
    return (metrics.get("op_name") or metrics.get("op_type") or metrics.get("message") or event.get("op_name")
            or "<unknown>")


def _canonical_name(name: str) -> str:
    for suffix in (" mix", " aiv", " aic"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _source_prefix(source: str) -> str:
    if source == "aclprof_op_summary":
        return "op_summary"
    if source == "msprof_mstx":
        return "mstx"
    return source


def _aggregate_rows(associations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for association in associations:
        source = association.get("source", "")
        if source not in {"aclprof_op_summary", "msprof_mstx"}:
            continue
        grouped[_canonical_name(_kernel_name(association))].append(association)

    rows = []
    for name, items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "kernel": name,
            "sources": ",".join(sorted({str(item.get("source", ""))
                                        for item in items})),
            "rows": len(items),
        }
        for source in sorted({str(item.get("source", "")) for item in items}):
            source_items = [item for item in items if item.get("source") == source]
            source_key = _source_prefix(source)
            row[source_key + ".rows"] = len(source_items)
            _aggregate_metrics(source_items, row, source_key)
        rows.append(row)
    return rows


def _aggregate_metrics(items: list[dict[str, Any]], row: dict[str, Any], prefix: str):
    for metric in IMPORTANT_METRICS:
        values = [
            _metric_value(item.get("metrics", {}), metric)
            for item in items
            if _metric_value(item.get("metrics", {}), metric) is not None
        ]
        numeric = [_as_float(value) for value in values]
        numeric = [value for value in numeric if value is not None]
        column = prefix + "." + metric
        if numeric:
            if (metric.endswith("_us") or metric.endswith("_cycles") or metric.endswith("_bytes")
                    or metric.endswith("_kb")):
                row[column + ".sum"] = sum(numeric)
                row[column + ".avg"] = sum(numeric) / len(numeric)
            elif metric.endswith("_gb_s"):
                row[column + ".avg"] = sum(numeric) / len(numeric)
                row[column + ".max"] = max(numeric)
            else:
                row[column] = values[-1]
        elif values:
            unique = []
            for value in values:
                if value not in unique:
                    unique.append(value)
            row[column] = "; ".join(str(value) for value in unique[:3])


def _print_table(rows: list[dict[str, Any]], columns: list[str]):
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(_format_value(row.get(column))))

    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(_format_value(row.get(column)).ljust(widths[column]) for column in columns))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor_json", type=Path, help="Path to profile.vendor.json")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print all aggregated important columns instead of the compact default.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print aggregated rows as JSON.",
    )
    args = parser.parse_args()

    artifact = json.loads(args.vendor_json.read_text())
    rows = _aggregate_rows(artifact.get("associations", []))
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    compact_columns = [
        "kernel",
        "sources",
        "rows",
        "mstx.rows",
        "mstx.task_duration_us.sum",
        "mstx.task_duration_us.avg",
        "op_summary.rows",
        "op_summary.task_duration_us.sum",
        "op_summary.task_duration_us.avg",
        "op_summary.aicore_time_us.sum",
        "op_summary.aic_total_cycles.sum",
        "op_summary.memory_access_bytes.sum",
        "op_summary.memory_read_bytes.sum",
        "op_summary.memory_write_bytes.sum",
        "op_summary.bandwidth_gb_s.avg",
        "op_summary.bandwidth_gb_s.max",
        "op_summary.input_shapes",
        "op_summary.output_shapes",
    ]
    if args.full:
        columns = sorted({key for row in rows for key in row.keys()})
        columns = ["kernel", "sources", "rows"
                   ] + [column for column in columns if column not in {"kernel", "sources", "rows"}]
    else:
        columns = compact_columns
    _print_table(rows, columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
