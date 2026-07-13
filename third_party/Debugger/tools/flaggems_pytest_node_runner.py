#!/usr/bin/env python3
"""Collect and run FlagGems pytest nodes under the FlagTree debugger.

This module is shared by the direct-case and saved-sample regression runners.
It keeps each pytest node isolated and records progress after every invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover - environment check
    raise SystemExit(f"PyYAML is required for --collect-marks parsing: {exc}") from exc

DEFAULT_FLAGGEMS_ROOT = Path(".cache/flaggems_debugger_batch/front_two_classes_full_final/"
                             "worktrees/FlagGems_direct_instrumented_20260626_150558")
DEFAULT_SOURCE_SUMMARY = Path(".cache/flaggems_debugger_batch/front_two_classes_full_final/"
                              "direct_runs/20260626_150558/summary.json")
DEFAULT_WORKSPACE = Path(".cache/flaggems_debugger_batch/pytest_node_runs")

BUILTIN_TIMEOUT_OPS = {
    "acos",
    "conv1d",
    "conv2d",
    "group_norm",
    "i0",
    "layer_norm",
    "log_softmax",
    "max_pool2d_with_indices",
}


@dataclass
class NodeInfo:
    nodeid: str
    file: str
    test_case: str
    function: str
    cls: str | None
    marks: list[str]
    selected_ops: list[str]
    timeout_class: str
    first_report_timeout_sec: int
    report_timeout_sec: int
    node_total_timeout_sec: int


@dataclass
class NodeRunResult:
    nodeid: str
    selected_ops: list[str]
    status: str
    duration_sec: float
    exit_code: int | None
    debug_txt_count: int
    debug_json_count: int
    debug_report_dir: str
    stdout_log: str
    stderr_log: str
    pytest_result: str
    first_error: str
    timeout_reason: str


def now_stamp() -> str:
    import datetime as _dt

    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def selected_ops_from_summary(path: Path, include_status: set[str]) -> list[str]:
    rows = read_json(path)
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        op = str(row.get("op", "")).strip()
        status = str(row.get("status", "")).strip()
        if op and status in include_status and op not in seen:
            result.append(op)
            seen.add(op)
    return result


def normalize_name(name: str) -> str:
    name = name.strip()
    name = name.replace(".", "_")
    name = name.replace("-", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    return name


def op_aliases(op: str) -> set[str]:
    aliases = {op, normalize_name(op)}
    if op.endswith(".out"):
        aliases.add(normalize_name(op[:-4] + "_out"))
    if op.endswith("_out"):
        aliases.add(op[:-4] + ".out")
    if op.startswith("native_"):
        aliases.add(op[len("native_"):])
    return {alias for alias in aliases if alias}


def nodeid_from_collect_item(item: dict[str, Any]) -> str:
    file_name = str(item.get("file", "")).strip()
    test_case = str(item.get("test_case", "")).strip()
    cls = item.get("class")
    if cls:
        return f"{file_name}::{cls}::{test_case}"
    return f"{file_name}::{test_case}"


def collect_marks(
    flaggems_root: Path,
    python: Path,
    output_dir: Path,
    extra_pytest_args: list[str],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "collect_stdout.log"
    stderr_log = output_dir / "collect_stderr.log"
    entry_py = output_dir / "collect_entry.py"
    entry_py.write_text("""
import platform
import sys

platform.python_implementation = lambda: "CPython"
platform.python_version = lambda: "3.11.15"
platform.python_version_tuple = lambda: ("3", "11", "15")

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
""".lstrip())
    cmd = [
        str(python),
        str(entry_py),
        "tests",
        "--collect-marks",
        "-q",
        *extra_pytest_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=flaggems_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    stdout_log.write_text(proc.stdout)
    stderr_log.write_text(proc.stderr)
    if proc.returncode not in (0, 5):
        raise RuntimeError(f"pytest collection failed with exit {proc.returncode}; "
                           f"see {stdout_log} and {stderr_log}")
    yaml_text = proc.stdout.split("\n================", 1)[0].strip()
    docs = list(yaml.safe_load_all(yaml_text))
    items: list[dict[str, Any]] = []
    for doc in docs:
        if isinstance(doc, list):
            items.extend(x for x in doc if isinstance(x, dict))
    if not items:
        raise RuntimeError(f"pytest collection produced no parseable items; see {stdout_log}")
    return items


def choose_timeouts(selected_ops: list[str], args: argparse.Namespace) -> tuple[str, int, int, int]:
    if any(op in BUILTIN_TIMEOUT_OPS for op in selected_ops):
        return (
            "known_heavy",
            args.heavy_first_report_timeout,
            args.heavy_report_timeout,
            args.heavy_node_total_timeout,
        )
    if any(token in op
           for op in selected_ops
           for token in ("conv", "norm", "softmax", "topk", "sort", "pool", "bmm", "mm")):
        return (
            "maybe_heavy",
            args.medium_first_report_timeout,
            args.medium_report_timeout,
            args.medium_node_total_timeout,
        )
    return (
        "default",
        args.first_report_timeout,
        args.report_timeout,
        args.node_total_timeout,
    )


def build_node_plan(
    items: list[dict[str, Any]],
    selected_ops: list[str],
    args: argparse.Namespace,
) -> list[NodeInfo]:
    alias_to_op: dict[str, set[str]] = {}
    for op in selected_ops:
        for alias in op_aliases(op):
            alias_to_op.setdefault(alias, set()).add(op)

    nodes: list[NodeInfo] = []
    seen: set[str] = set()
    for item in items:
        marks = [str(mark) for mark in item.get("marks", []) if str(mark)]
        matched: set[str] = set()
        for mark in marks:
            for key in op_aliases(mark):
                matched.update(alias_to_op.get(key, set()))
        if not matched:
            continue
        nodeid = nodeid_from_collect_item(item)
        if nodeid in seen:
            continue
        seen.add(nodeid)
        ops = sorted(matched)
        klass, first_timeout, report_timeout, total_timeout = choose_timeouts(ops, args)
        nodes.append(
            NodeInfo(
                nodeid=nodeid,
                file=str(item.get("file", "")),
                test_case=str(item.get("test_case", "")),
                function=str(item.get("function", "")),
                cls=item.get("class"),
                marks=marks,
                selected_ops=ops,
                timeout_class=klass,
                first_report_timeout_sec=first_timeout,
                report_timeout_sec=report_timeout,
                node_total_timeout_sec=total_timeout,
            ))
    class_rank = {"default": 0, "maybe_heavy": 1, "known_heavy": 2}
    nodes.sort(key=lambda x: (class_rank.get(x.timeout_class, 99), x.file, x.function, x.test_case))
    if args.one_node_per_op:
        covered: set[str] = set()
        representative: list[NodeInfo] = []
        for node in nodes:
            if any(op not in covered for op in node.selected_ops):
                representative.append(node)
                covered.update(node.selected_ops)
        nodes = representative
    if args.max_nodes_per_op:
        counts: dict[str, int] = {}
        limited: list[NodeInfo] = []
        for node in nodes:
            if all(counts.get(op, 0) >= args.max_nodes_per_op for op in node.selected_ops):
                continue
            limited.append(node)
            for op in node.selected_ops:
                counts[op] = counts.get(op, 0) + 1
        nodes = limited
    if args.max_nodes:
        nodes = nodes[:args.max_nodes]
    return nodes


def report_counts(debug_dir: Path) -> tuple[int, int]:
    if not debug_dir.exists():
        return 0, 0
    return (
        sum(1 for _ in debug_dir.glob("*.txt")),
        sum(1 for _ in debug_dir.glob("*.json")),
    )


def write_node_entry(path: Path) -> None:
    path.write_text("""
import os
import platform
import sys

platform.python_implementation = lambda: "CPython"
platform.python_version = lambda: "3.11.15"
platform.python_version_tuple = lambda: ("3", "11", "15")

from triton.runtime import debugger
import triton
import pytest

debugger.configure(
    output_dir=os.environ["FLAGTREE_DEBUGGER_NODE_OUTPUT_DIR"],
    record_capacity=int(os.environ.get("FLAGTREE_DEBUGGER_NODE_RECORD_CAPACITY", "4096")),
    export_raw_records=os.environ.get("FLAGTREE_DEBUGGER_NODE_EXPORT_RAW", "0") == "1",
)
triton.enable_debug(
    level=int(os.environ.get("FLAGTREE_DEBUGGER_NODE_LEVEL", "1")),
    addr_level=int(os.environ.get("FLAGTREE_DEBUGGER_NODE_ADDR_LEVEL", "1")),
)

nodeid = os.environ["FLAGTREE_DEBUGGER_NODEID"]
pytest_args = [
    "-q",
    nodeid,
    "--quick",
    "--ref=cpu",
    "--record=json",
    "--output",
    os.environ["FLAGTREE_DEBUGGER_NODE_PYTEST_JSON"],
    "-s",
]
raise SystemExit(pytest.main(pytest_args))
""".lstrip())


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def run_node(
    node: NodeInfo,
    flaggems_root: Path,
    python: Path,
    run_dir: Path,
    entry_py: Path,
    args: argparse.Namespace,
) -> NodeRunResult:
    safe_name = normalize_name(node.nodeid)[:180]
    node_dir = run_dir / "nodes" / safe_name
    debug_dir = node_dir / "debug_reports"
    stdout_log = node_dir / "stdout.log"
    stderr_log = node_dir / "stderr.log"
    pytest_result = node_dir / "pytest_result.json"
    node_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "FLAGTREE_DEBUGGER_NODEID": node.nodeid,
        "FLAGTREE_DEBUGGER_NODE_OUTPUT_DIR": str(debug_dir),
        "FLAGTREE_DEBUGGER_NODE_PYTEST_JSON": str(pytest_result),
        "FLAGTREE_DEBUGGER_NODE_RECORD_CAPACITY": str(args.record_capacity),
        "FLAGTREE_DEBUGGER_NODE_LEVEL": str(args.level),
        "FLAGTREE_DEBUGGER_NODE_ADDR_LEVEL": str(args.addr_level),
        "PYTHONUNBUFFERED": "1",
    })
    cmd = [str(python), "-u", str(entry_py)]
    start = time.time()
    last_report_time = start
    first_report_seen = False
    last_txt_count = 0
    last_json_count = 0
    timeout_reason = ""
    with stdout_log.open("w") as out, stderr_log.open("w") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=flaggems_root,
            env=env,
            text=True,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        while proc.poll() is None:
            time.sleep(args.poll_interval)
            now = time.time()
            txt_count, json_count = report_counts(debug_dir)
            if txt_count > last_txt_count or json_count > last_json_count:
                if not first_report_seen:
                    first_report_seen = True
                last_report_time = now
                last_txt_count = txt_count
                last_json_count = json_count
            if not first_report_seen and now - start > node.first_report_timeout_sec:
                timeout_reason = "first_report_timeout"
                terminate_process(proc)
                break
            if first_report_seen and now - last_report_time > node.report_timeout_sec:
                timeout_reason = "report_timeout"
                terminate_process(proc)
                break
            if now - start > node.node_total_timeout_sec:
                timeout_reason = "node_total_timeout"
                terminate_process(proc)
                break
    duration = time.time() - start
    exit_code = proc.poll()
    txt_count, json_count = report_counts(debug_dir)

    first_error = ""
    if timeout_reason:
        status = "partial_timeout" if txt_count or json_count else "timeout"
        first_error = timeout_reason
    elif exit_code == 0 and txt_count > 0 and json_count > 0:
        status = "passed"
    elif exit_code == 0:
        status = "missing_debug_report"
        first_error = "pytest node passed but debugger report is missing"
    else:
        status = "pytest_error"
        first_error = f"pytest exit code {exit_code}"

    return NodeRunResult(
        nodeid=node.nodeid,
        selected_ops=node.selected_ops,
        status=status,
        duration_sec=duration,
        exit_code=exit_code,
        debug_txt_count=txt_count,
        debug_json_count=json_count,
        debug_report_dir=str(debug_dir),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        pytest_result=str(pytest_result),
        first_error=first_error,
        timeout_reason=timeout_reason,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flaggems-root", type=Path, default=DEFAULT_FLAGGEMS_ROOT)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--include-status", default="no_direct_case")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index into the selected node plan, after filtering.",
    )
    parser.add_argument(
        "--one-node-per-op",
        action="store_true",
        help="Run only the first representative pytest node for each selected op.",
    )
    parser.add_argument(
        "--max-nodes-per-op",
        type=int,
        help="Cap the number of pytest nodes selected for each op.",
    )
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--addr-level", type=int, default=1)
    parser.add_argument("--record-capacity", type=int, default=4096)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--first-report-timeout", type=int, default=180)
    parser.add_argument("--report-timeout", type=int, default=120)
    parser.add_argument("--node-total-timeout", type=int, default=600)
    parser.add_argument("--medium-first-report-timeout", type=int, default=300)
    parser.add_argument("--medium-report-timeout", type=int, default=180)
    parser.add_argument("--medium-node-total-timeout", type=int, default=1200)
    parser.add_argument("--heavy-first-report-timeout", type=int, default=480)
    parser.add_argument("--heavy-report-timeout", type=int, default=240)
    parser.add_argument("--heavy-node-total-timeout", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    include_status = {x.strip() for x in args.include_status.split(",") if x.strip()}
    stamp = now_stamp()
    run_dir = args.workspace_root.resolve() / "pytest_node_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    selected_ops = selected_ops_from_summary(args.source_summary, include_status)
    write_json(run_dir / "selected_ops.json", selected_ops)
    print(f"[INFO] selected ops: {len(selected_ops)}")
    print(f"[INFO] run dir: {run_dir}")

    items = collect_marks(args.flaggems_root, args.python, run_dir / "collect", [])
    write_json(run_dir / "collect_items.json", items)
    nodes = build_node_plan(items, selected_ops, args)
    write_json(run_dir / "collected_nodes.json", [asdict(node) for node in nodes])

    by_class: dict[str, int] = {}
    by_op: dict[str, int] = {}
    for node in nodes:
        by_class[node.timeout_class] = by_class.get(node.timeout_class, 0) + 1
        for op in node.selected_ops:
            by_op[op] = by_op.get(op, 0) + 1
    estimation = {
        "selected_ops": len(selected_ops),
        "collected_pytest_items": len(items),
        "matched_nodes": len(nodes),
        "timeout_class_counts": dict(sorted(by_class.items())),
        "node_count_by_op": dict(sorted(by_op.items())),
        "timeout_policy": {
            "default": {
                "first_report": args.first_report_timeout,
                "report": args.report_timeout,
                "total": args.node_total_timeout,
            },
            "maybe_heavy": {
                "first_report": args.medium_first_report_timeout,
                "report": args.medium_report_timeout,
                "total": args.medium_node_total_timeout,
            },
            "known_heavy": {
                "first_report": args.heavy_first_report_timeout,
                "report": args.heavy_report_timeout,
                "total": args.heavy_node_total_timeout,
            },
        },
    }
    write_json(run_dir / "estimation.json", estimation)
    print(json.dumps(estimation, indent=2, sort_keys=True))

    if args.collect_only:
        return 0

    entry_py = run_dir / "node_entry.py"
    write_node_entry(entry_py)
    results: list[NodeRunResult] = []
    status_counts: dict[str, int] = {}
    if args.start_index < 1:
        raise ValueError("--start-index must be >= 1")
    nodes_to_run = nodes[args.start_index - 1:]

    for offset, node in enumerate(nodes_to_run, start=args.start_index):
        print(f"[RUN] {offset}/{len(nodes)} {node.nodeid} ops={node.selected_ops}")
        result = run_node(node, args.flaggems_root, args.python, run_dir, entry_py, args)
        results.append(result)
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        write_json(run_dir / "summary.json", [asdict(x) for x in results])
        write_json(run_dir / "status_counts.json", dict(sorted(status_counts.items())))
        print(f"[RESULT] {result.status} exit={result.exit_code} "
              f"txt={result.debug_txt_count} json={result.debug_json_count} "
              f"sec={result.duration_sec:.1f}")
    print("[DONE]", json.dumps(dict(sorted(status_counts.items())), sort_keys=True))
    return 0 if all(x.status == "passed" for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
