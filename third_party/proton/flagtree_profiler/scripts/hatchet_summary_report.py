#!/usr/bin/env python3
"""Print a compact performance table from a Proton Hatchet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _format_number(value: float | None, width: int, precision: int = 3) -> str:
    if value is None:
        return "nan".rjust(width)
    return f"{value:.{precision}f}".rjust(width)


def _display_name(name: str) -> str:
    for suffix in (" mix", " aiv", " aic"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _has_numeric_metrics(node: dict[str, Any]) -> bool:
    return any(isinstance(v, (int, float)) and v != 0 for v in node.get("metrics", {}).values())


def _node_values(node: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    metrics = node.get("metrics", {})
    bandwidth_gb_s = _metric(
        metrics,
        "cann.bandwidth_gb_s",
        "cann.estimated_bandwidth_gb_s",
        "bandwidth_gb_s",
    )
    tbps = bandwidth_gb_s / 1000.0 if bandwidth_gb_s is not None else None

    gpu_us = _metric(metrics, "cann.task_duration_us", "runtime.duration_us")
    gpu_ms = gpu_us / 1000.0 if gpu_us is not None else None

    cpu_ns = _metric(metrics, "time (ns)")
    cpu_ms = cpu_ns / 1.0e6 if cpu_ns is not None else None
    return tbps, gpu_ms, cpu_ms


def _fallback_values(
    node: dict[str, Any],
    fallback: dict[str, tuple[float | None, float | None, float | None]],
    occurrence_counts: dict[str, int],
) -> tuple[float | None, float | None, float | None]:
    name = _display_name(node.get("frame", {}).get("name", ""))
    values = fallback.get(name)
    if values is None:
        return None, None, None
    count = max(1, occurrence_counts.get(name, 1))
    tbps, gpu_ms, cpu_ms = values
    return (
        tbps,
        gpu_ms / count if gpu_ms is not None else None,
        cpu_ms / count if cpu_ms is not None else None,
    )


def _aggregate_children(
    node: dict[str, Any],
    fallback: dict[str, tuple[float | None, float | None, float | None]],
    occurrence_counts: dict[str, int],
) -> tuple[float | None, float | None, float | None]:
    tbps_num = 0.0
    tbps_den_ms = 0.0
    gpu_ms_total = 0.0
    cpu_ms_total = 0.0
    has_gpu = False
    has_cpu = False
    for child in node.get("children", []):
        tbps, gpu_ms, cpu_ms = _aggregate_children(child, fallback, occurrence_counts)
        if tbps is not None and gpu_ms is not None:
            tbps_num += tbps * gpu_ms
            tbps_den_ms += gpu_ms
        if gpu_ms is not None:
            gpu_ms_total += gpu_ms
            has_gpu = True
        if cpu_ms is not None:
            cpu_ms_total += cpu_ms
            has_cpu = True

    own_tbps, own_gpu_ms, own_cpu_ms = _node_values(node)
    if not node.get("children"):
        if not _has_numeric_metrics(node) or (own_tbps is None and own_gpu_ms is None and own_cpu_ms is None):
            return _fallback_values(node, fallback, occurrence_counts)
        return own_tbps, own_gpu_ms, own_cpu_ms

    tbps = tbps_num / tbps_den_ms if tbps_den_ms > 0 else own_tbps
    gpu_ms = gpu_ms_total if has_gpu else own_gpu_ms
    cpu_ms = cpu_ms_total if has_cpu else own_cpu_ms
    return tbps, gpu_ms, cpu_ms


def _print_node(
    node: dict[str, Any],
    *,
    prefix: str,
    is_last: bool,
    root: bool,
    fallback: dict[str, tuple[float | None, float | None, float | None]],
    occurrence_counts: dict[str, int],
):
    frame = node.get("frame", {})
    name = _display_name(frame.get("name", ""))
    tbps, gpu_ms, cpu_ms = _aggregate_children(node, fallback, occurrence_counts)
    if root:
        line_prefix = ""
        label = name or "ROOT"
        child_prefix = ""
    else:
        connector = "+- " if is_last else "|- "
        line_prefix = prefix + connector
        label = name
        child_prefix = prefix + ("   " if is_last else "|  ")

    print(f"{line_prefix}{label:<42}"
          f"{_format_number(tbps, 14)}"
          f"{_format_number(gpu_ms, 12)}"
          f"{_format_number(cpu_ms, 12)}")

    children = node.get("children", [])
    for index, child in enumerate(children):
        _print_node(
            child,
            prefix=child_prefix,
            is_last=index == len(children) - 1,
            root=False,
            fallback=fallback,
            occurrence_counts=occurrence_counts,
        )


def _contains_named_scope(node: dict[str, Any], scope_name: str) -> bool:
    if node.get("frame", {}).get("name") == scope_name:
        return True
    return any(_contains_named_scope(child, scope_name) for child in node.get("children", []))


def _count_scoped_leaves(node: dict[str, Any], counts: dict[str, int]):
    children = node.get("children", [])
    if not children:
        name = _display_name(node.get("frame", {}).get("name", ""))
        if name:
            counts[name] = counts.get(name, 0) + 1
        return
    for child in children:
        _count_scoped_leaves(child, counts)


def _build_fallback(root: dict[str, Any]) -> dict[str, tuple[float | None, float | None, float | None]]:
    fallback = {}
    for child in root.get("children", []):
        if child.get("children"):
            continue
        name = _display_name(child.get("frame", {}).get("name", ""))
        if name:
            fallback[name] = _node_values(child)
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hatchet", type=Path, help="Path to profile.hatchet")
    parser.add_argument("--title", default="", help="Optional title line")
    args = parser.parse_args()

    data = json.loads(args.hatchet.read_text())
    root = data[0]
    has_prefill_scope = _contains_named_scope(root, "prefill")
    fallback = _build_fallback(root) if has_prefill_scope else {}
    occurrence_counts: dict[str, int] = {}
    if has_prefill_scope:
        for child in root.get("children", []):
            if child.get("children"):
                _count_scoped_leaves(child, occurrence_counts)
        root = {
            **root,
            "children": [child for child in root.get("children", []) if child.get("children")],
        }
    if args.title:
        print(args.title)
    print(f"{'phase/node':<45}{'TBPS(tbyte/s)':>14}{'GPU(ms)':>12}{'CPU(ms)':>12}")
    _print_node(
        root,
        prefix="",
        is_last=True,
        root=True,
        fallback=fallback,
        occurrence_counts=occurrence_counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
