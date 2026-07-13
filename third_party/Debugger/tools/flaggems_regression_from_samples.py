#!/usr/bin/env python3
"""Replay previously successful FlagGems debugger samples and compare results.

It uses the existing FlagGems copy/instrument workflow and treats the module's
`samples/index.json` as the historical success baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flaggems_debug_batch import (  # noqa: E402
    DEFAULT_FLAGGEMS_ROOT, DEFAULT_PYTHON, InstrumentationStats, build_env, classify_status, copy_flaggems_source,
    create_bootstrap, first_error_from_logs, instrument_flaggems_tree, load_operator_inventory, no_cpu_ops, now_stamp,
    run_phase_for_op, shell_command, write_text,
)
from flaggems_direct_op_cases import (  # noqa: E402
    complete_report_count, debug_report_snapshot, kill_process_group,
)

DEFAULT_WORKSPACE_ROOT = Path(".cache/flaggems_debugger_batch/regression_runs")
DEFAULT_SAMPLES_ROOT = Path(__file__).resolve().parents[1] / "samples"


@dataclass
class BaselineItem:
    index: int
    folder: str
    op: str
    kind: str
    sample_dir: str
    baseline_status: str
    baseline_case_id: str | None
    baseline_nodeid: str | None
    baseline_description: str | None
    baseline_debug_txt_count: int
    baseline_debug_json_count: int
    baseline_duration_sec: float | None
    baseline_reports: int
    baseline_summary_file: str


@dataclass
class RegressionResult:
    index: int
    folder: str
    op: str
    kind: str
    status: str
    exit_code: int | None
    duration_sec: float
    command: str
    script: str | None
    stdout_log: str
    stderr_log: str
    debug_report_dir: str
    debug_txt_count: int
    debug_json_count: int
    first_error: str
    baseline_case_id: str | None
    baseline_nodeid: str | None
    baseline_debug_txt_count: int
    baseline_debug_json_count: int
    baseline_duration_sec: float | None
    comparison_status: str
    warnings: list[str]
    timeout_reason: str = ""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def normalize_name(text: str) -> str:
    keep: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def load_baseline(samples_root: Path, index_name: str) -> list[BaselineItem]:
    index_path = samples_root / index_name
    rows = read_json(index_path)
    if not isinstance(rows, list):
        raise TypeError(f"{index_path} must contain a list")

    baseline: list[BaselineItem] = []
    for idx, row in enumerate(rows, start=1):
        folder = str(row["folder"])
        sample_dir = samples_root / folder
        run_info_path = sample_dir / "run_info.json"
        run_info = read_json(run_info_path)
        source_dir = sample_dir / "source"
        if (source_dir / "case.py").exists():
            kind = "direct_case"
        elif (source_dir / "runner.py").exists():
            kind = "pytest_node"
        else:
            kind = "legacy_marker"
        baseline.append(
            BaselineItem(
                index=idx,
                folder=folder,
                op=str(row["op"]),
                kind=kind,
                sample_dir=str(sample_dir),
                baseline_status=str(run_info.get("status", "passed")),
                baseline_case_id=run_info.get("case_id"),
                baseline_nodeid=run_info.get("nodeid"),
                baseline_description=run_info.get("description"),
                baseline_debug_txt_count=int(run_info.get("debug_txt_count") or 0),
                baseline_debug_json_count=int(run_info.get("debug_json_count") or 0),
                baseline_duration_sec=(float(run_info["duration_sec"])
                                       if run_info.get("duration_sec") is not None else None),
                baseline_reports=int(row.get("reports") or 0),
                baseline_summary_file=str(row.get("summary_file") or ""),
            ))
    return baseline


def select_items(items: list[BaselineItem], args: argparse.Namespace) -> list[BaselineItem]:
    selected = items
    if args.kinds:
        kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
        selected = [item for item in selected if item.kind in kinds]
    if args.ops:
        ops = {x.strip() for x in args.ops.split(",") if x.strip()}
        selected = [item for item in selected if item.op in ops]
    if args.start_index:
        selected = [item for item in selected if item.index >= args.start_index]
    if args.max_items:
        selected = selected[:args.max_items]
    return selected


def count_complete_reports(debug_dir: Path) -> int:
    return complete_report_count(debug_dir)


def wait_for_process_with_reports(
    proc: subprocess.Popen[Any],
    debug_dir: Path,
    *,
    total_timeout: int | None,
    first_report_timeout: int | None,
    report_timeout: int | None,
    poll_interval: float,
) -> tuple[int | None, bool, str]:
    start = time.time()
    last_report_time: float | None = None
    last_complete_reports = 0
    while True:
        exit_code = proc.poll()
        now = time.time()
        if exit_code is not None:
            return exit_code, False, ""

        complete_reports = count_complete_reports(debug_dir)
        if complete_reports > last_complete_reports:
            last_complete_reports = complete_reports
            last_report_time = now

        elapsed = now - start
        if total_timeout is not None and elapsed > total_timeout:
            kill_process_group(proc)
            return None, True, "total_timeout"
        if last_complete_reports == 0:
            if first_report_timeout is not None and elapsed > first_report_timeout:
                kill_process_group(proc)
                return None, True, "first_report_timeout"
        elif (report_timeout is not None and last_report_time is not None and now - last_report_time > report_timeout):
            kill_process_group(proc)
            return None, True, "report_timeout"
        time.sleep(poll_interval)


def classify_comparison(item: BaselineItem, result: RegressionResult) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if result.status != "passed":
        return "regressed", warnings
    if result.debug_txt_count <= 0 or result.debug_json_count <= 0:
        return "regressed", warnings
    if (item.baseline_debug_txt_count > 0 and result.debug_txt_count < item.baseline_debug_txt_count):
        warnings.append(f"txt reports decreased {item.baseline_debug_txt_count}->{result.debug_txt_count}")
    if (item.baseline_debug_json_count > 0 and result.debug_json_count < item.baseline_debug_json_count):
        warnings.append(f"json reports decreased {item.baseline_debug_json_count}->{result.debug_json_count}")
    if item.baseline_duration_sec and result.duration_sec > 2 * item.baseline_duration_sec:
        warnings.append(f"duration >2x baseline {item.baseline_duration_sec:.1f}s->{result.duration_sec:.1f}s")
    if result.duration_sec > 300:
        warnings.append(f"duration >300s ({result.duration_sec:.1f}s)")
    return ("passed_with_warning" if warnings else "passed"), warnings


def make_base_result(
    item: BaselineItem,
    *,
    status: str,
    exit_code: int | None,
    duration_sec: float,
    command: str,
    script: Path | None,
    stdout_log: Path,
    stderr_log: Path,
    debug_dir: Path,
    first_error: str,
    timeout_reason: str = "",
) -> RegressionResult:
    txt_count, json_count, _ = debug_report_snapshot(debug_dir)
    result = RegressionResult(
        index=item.index,
        folder=item.folder,
        op=item.op,
        kind=item.kind,
        status=status,
        exit_code=exit_code,
        duration_sec=duration_sec,
        command=command,
        script=str(script) if script else None,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        debug_report_dir=str(debug_dir),
        debug_txt_count=txt_count,
        debug_json_count=json_count,
        first_error=first_error,
        baseline_case_id=item.baseline_case_id,
        baseline_nodeid=item.baseline_nodeid,
        baseline_debug_txt_count=item.baseline_debug_txt_count,
        baseline_debug_json_count=item.baseline_debug_json_count,
        baseline_duration_sec=item.baseline_duration_sec,
        comparison_status="",
        warnings=[],
        timeout_reason=timeout_reason,
    )
    comparison_status, warnings = classify_comparison(item, result)
    result.comparison_status = comparison_status
    result.warnings = warnings
    return result


def run_saved_script_case(
    item: BaselineItem,
    source_script: Path,
    worktree: Path,
    run_dir: Path,
    bootstrap_dir: Path,
    args: argparse.Namespace,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> RegressionResult:
    case_dir = run_dir / "items" / f"{item.index:03d}_{normalize_name(item.op)}"
    debug_dir = case_dir / "debug_reports"
    case_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    script = case_dir / source_script.name
    shutil.copy2(source_script, script)
    stdout_log = case_dir / "stdout.log"
    stderr_log = case_dir / "stderr.log"

    env = build_env(os.environ, worktree, bootstrap_dir, debug_dir, args)
    if env_extra:
        env.update(env_extra)
    command = shell_command(cwd or worktree, [str(args.python), "-u", str(script)])

    start = time.time()
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        exit_code, timed_out, timeout_reason = wait_for_process_with_reports(
            proc,
            debug_dir,
            total_timeout=args.case_total_timeout,
            first_report_timeout=args.first_report_timeout,
            report_timeout=args.report_timeout,
            poll_interval=args.poll_interval,
        )
    duration = time.time() - start
    txt_count, json_count, _ = debug_report_snapshot(debug_dir)
    status = classify_status(
        exit_code,
        timed_out,
        txt_count,
        json_count,
        stdout_log,
        stderr_log,
    )
    if timed_out and min(txt_count, json_count) > 0:
        status = "partial_timeout"
    first_error = first_error_from_logs(stdout_log, stderr_log)
    if status == "missing_debug_report":
        first_error = ("case exited successfully but debugger report is missing or incomplete: "
                       f"txt={txt_count}, json={json_count}, dir={debug_dir}")
    elif status in {"timeout", "partial_timeout"}:
        first_error = timeout_reason or first_error
    result = make_base_result(
        item,
        status=status,
        exit_code=exit_code,
        duration_sec=duration,
        command=command,
        script=script,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        debug_dir=debug_dir,
        first_error=first_error,
        timeout_reason=timeout_reason,
    )
    write_json(case_dir / "status.json", asdict(result))
    return result


def run_direct_item(
    item: BaselineItem,
    worktree: Path,
    run_dir: Path,
    bootstrap_dir: Path,
    args: argparse.Namespace,
) -> RegressionResult:
    source_script = Path(item.sample_dir) / "source" / "case.py"
    return run_saved_script_case(
        item,
        source_script,
        worktree,
        run_dir,
        bootstrap_dir,
        args,
        cwd=worktree,
    )


def run_pytest_node_item(
    item: BaselineItem,
    worktree: Path,
    run_dir: Path,
    bootstrap_dir: Path,
    args: argparse.Namespace,
) -> RegressionResult:
    source_script = Path(item.sample_dir) / "source" / "runner.py"
    case_dir = run_dir / "items" / f"{item.index:03d}_{normalize_name(item.op)}"
    debug_dir = case_dir / "debug_reports"
    pytest_json = case_dir / "pytest_result.json"
    env_extra = {
        "FLAGTREE_DEBUGGER_NODEID": item.baseline_nodeid or "",
        "FLAGTREE_DEBUGGER_NODE_OUTPUT_DIR": str(debug_dir),
        "FLAGTREE_DEBUGGER_NODE_PYTEST_JSON": str(pytest_json),
        "FLAGTREE_DEBUGGER_NODE_RECORD_CAPACITY": str(args.record_capacity),
        "FLAGTREE_DEBUGGER_NODE_LEVEL": str(args.level),
        "FLAGTREE_DEBUGGER_NODE_ADDR_LEVEL": str(args.addr_level),
        "FLAGTREE_DEBUGGER_NODE_EXPORT_RAW": "1" if args.export_raw_records else "0",
    }
    return run_saved_script_case(
        item,
        source_script,
        worktree,
        run_dir,
        bootstrap_dir,
        args,
        env_extra=env_extra,
        cwd=worktree,
    )


def run_legacy_marker_item(
    item: BaselineItem,
    worktree: Path,
    run_dir: Path,
    bootstrap_dir: Path,
    no_cpu: set[str],
    args: argparse.Namespace,
) -> RegressionResult:
    legacy_dir = run_dir / "items" / f"{item.index:03d}_{normalize_name(item.op)}"
    phase_run_dir = legacy_dir / "legacy_marker"
    status = run_phase_for_op(
        item.op,
        "accuracy",
        worktree,
        phase_run_dir,
        bootstrap_dir,
        no_cpu,
        args,
    )
    debug_dir = Path(status.debug_report_dir)
    stdout_log = Path(status.stdout_log)
    stderr_log = Path(status.stderr_log)
    result = make_base_result(
        item,
        status=status.status,
        exit_code=status.exit_code,
        duration_sec=status.duration_sec,
        command=status.command,
        script=None,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        debug_dir=debug_dir,
        first_error=status.first_error,
        timeout_reason="timeout" if status.status in {"timeout", "partial_timeout"} else "",
    )
    write_json(legacy_dir / "status.json", asdict(result))
    return result


def write_markdown_report(run_dir: Path, baseline: list[BaselineItem], results: list[RegressionResult]) -> None:
    by_status: dict[str, int] = {}
    by_comparison: dict[str, int] = {}
    for result in results:
        by_status[result.status] = by_status.get(result.status, 0) + 1
        by_comparison[result.comparison_status] = by_comparison.get(result.comparison_status, 0) + 1

    lines = [
        "# FlagGems Debugger Regression",
        "",
        f"- baseline_items: {len(baseline)}",
        f"- executed_items: {len(results)}",
        f"- runtime_status: {json.dumps(dict(sorted(by_status.items())), sort_keys=True)}",
        f"- comparison_status: {json.dumps(dict(sorted(by_comparison.items())), sort_keys=True)}",
        "",
        "## Regressions",
        "",
    ]
    regressions = [r for r in results if r.comparison_status == "regressed"]
    if not regressions:
        lines.append("None.")
    else:
        for result in regressions:
            lines.append(f"- {result.index:03d} {result.op} [{result.kind}] "
                         f"status={result.status} exit={result.exit_code} "
                         f"txt={result.debug_txt_count} json={result.debug_json_count} "
                         f"error={result.first_error}")
            lines.append(f"  - stdout: `{result.stdout_log}`")
            lines.append(f"  - stderr: `{result.stderr_log}`")
            lines.append(f"  - reports: `{result.debug_report_dir}`")
    lines.extend(["", "## Warnings", ""])
    warnings = [r for r in results if r.comparison_status == "passed_with_warning"]
    if not warnings:
        lines.append("None.")
    else:
        for result in warnings:
            lines.append(f"- {result.index:03d} {result.op} [{result.kind}] "
                         f"{'; '.join(result.warnings)}")
    write_text(run_dir / "comparison.md", "\n".join(lines) + "\n")


def write_progress(run_dir: Path, baseline: list[BaselineItem], results: list[RegressionResult]) -> None:
    write_json(run_dir / "comparison.json", [asdict(result) for result in results])
    status_counts: dict[str, int] = {}
    comparison_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        comparison_counts[result.comparison_status] = (comparison_counts.get(result.comparison_status, 0) + 1)
    write_json(run_dir / "status_counts.json", status_counts)
    write_json(run_dir / "comparison_counts.json", comparison_counts)
    write_json(
        run_dir / "failed_cases.json",
        [asdict(r) for r in results if r.comparison_status == "regressed"],
    )
    write_json(
        run_dir / "warning_cases.json",
        [asdict(r) for r in results if r.comparison_status == "passed_with_warning"],
    )
    write_markdown_report(run_dir, baseline, results)


def write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    worktree: Path,
    selected: list[BaselineItem],
    stats: InstrumentationStats,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "samples_regression",
        "samples_root": str(args.samples_root),
        "flaggems_root": str(args.flaggems_root),
        "worktree": str(worktree),
        "selected_items": len(selected),
        "level": args.level,
        "addr_level": args.addr_level,
        "record_capacity": args.record_capacity,
        "case_total_timeout": args.case_total_timeout,
        "first_report_timeout": args.first_report_timeout,
        "report_timeout": args.report_timeout,
        "instrument_pointwise_generated": args.instrument_pointwise_generated,
        "normalize_ext_launch_ids": args.normalize_ext_launch_ids,
        "prune_item_caches": args.prune_item_caches,
        "shared_compiler_cache": args.shared_compiler_cache,
        "shared_cache_dir": (str(args.shared_cache_dir) if args.shared_cache_dir else None),
        "instrumentation": asdict(stats),
    }
    write_json(run_dir / "manifest.json", manifest)


def prune_transient_item_caches(result: RegressionResult) -> list[str]:
    """Remove rebuildable compiler caches while retaining reports and logs."""
    debug_dir = Path(result.debug_report_dir)
    removed: list[str] = []
    for name in ("triton_cache", "flaggems_cache"):
        cache_dir = debug_dir.parent / name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            removed.append(str(cache_dir))
    return removed


def parse_args(argv: list[str]) -> argparse.Namespace:

    def add_bool_argument(name: str, *, default: bool, help_text: str = "") -> None:
        dest = name.replace("-", "_")
        parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
        parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
        parser.set_defaults(**{dest: default})

    parser = argparse.ArgumentParser(
        description="Replay FlagGems debugger samples and compare against samples/index.json.")
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument(
        "--sample-index",
        default="stable_index.json",
        help=("Index file under --samples-root. Use index.json to replay stable "
              "and coverage samples together."),
    )
    parser.add_argument("--flaggems-root", type=Path, default=DEFAULT_FLAGGEMS_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
    )
    parser.add_argument("--ops", help="comma-separated op ids to replay")
    parser.add_argument(
        "--kinds",
        help="comma-separated baseline kinds: direct_case,pytest_node,legacy_marker",
    )
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--addr-level", type=int, default=1)
    parser.add_argument("--record-capacity", type=int, default=4096)
    parser.add_argument("--case-total-timeout", type=int, default=1800)
    parser.add_argument("--first-report-timeout", type=int, default=480)
    parser.add_argument("--report-timeout", type=int, default=240)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=1800, help="Legacy marker pytest timeout.")
    add_bool_argument("quick", default=True)
    parser.add_argument("--export-raw-records", action="store_true")
    add_bool_argument("instrument-pointwise-generated", default=True)
    add_bool_argument("normalize-ext-launch-ids", default=False)
    parser.add_argument(
        "--prune-item-caches",
        action="store_true",
        help=("Delete each completed item's rebuildable Triton and FlagGems caches "
              "after its status, logs, and debugger reports are persisted."),
    )
    parser.add_argument(
        "--shared-compiler-cache",
        action="store_true",
        help=("Reuse a run-local Triton/FlagGems compiler cache across sequential "
              "sample processes while keeping reports and process state isolated."),
    )
    parser.add_argument("--dry-run", action="store_true")
    add_bool_argument("keep-worktree", default=True)
    add_bool_argument(
        "stop-on-device-error",
        default=True,
        help_text="Stop the replay after an NPU/device error so later cases are not run on a poisoned device context.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    runs_root = workspace_root / "runs"
    worktrees_root = workspace_root / "worktrees"
    bootstrap_root = workspace_root / "bootstrap"
    for directory in (runs_root, worktrees_root, bootstrap_root):
        directory.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(args.samples_root.resolve(), args.sample_index)
    selected = select_items(baseline, args)
    stamp = now_stamp()
    run_dir = runs_root / stamp
    worktree = worktrees_root / f"FlagGems_regression_instrumented_{stamp}"
    bootstrap_dir = bootstrap_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    args.shared_cache_dir = (run_dir / "shared_compiler_cache" if args.shared_compiler_cache else None)
    create_bootstrap(bootstrap_dir)

    write_json(run_dir / "baseline_all.json", [asdict(item) for item in baseline])
    write_json(run_dir / "baseline_selected.json", [asdict(item) for item in selected])
    print(f"[INFO] baseline items: {len(baseline)}")
    print(f"[INFO] selected items: {len(selected)}")
    print(f"[INFO] run dir: {run_dir}")
    print(f"[INFO] worktree: {worktree}")

    if args.dry_run:
        write_manifest(run_dir, args, worktree, selected, InstrumentationStats())
        write_progress(run_dir, baseline, [])
        return 0

    flaggems_root = args.flaggems_root.resolve()
    if not flaggems_root.exists():
        raise FileNotFoundError(f"FlagGems root not found: {flaggems_root}")
    print(f"[INFO] copying FlagGems to {worktree}")
    copy_flaggems_source(flaggems_root, worktree)
    warnings_path = run_dir / "instrumentation_warnings.json"
    classifications_path = run_dir / "jit_function_classifications.json"
    print("[INFO] instrumenting Triton JIT functions")
    stats = instrument_flaggems_tree(
        worktree,
        args.level,
        args.addr_level,
        warnings_path,
        classifications_path,
        args.instrument_pointwise_generated,
        args.normalize_ext_launch_ids,
    )
    print("[INFO] instrumentation: "
          f"{stats.functions_instrumented} functions in {stats.files_changed} files")
    write_manifest(run_dir, args, worktree, selected, stats)

    inventory = load_operator_inventory(worktree)
    no_cpu = no_cpu_ops(inventory)
    results: list[RegressionResult] = []
    for position, item in enumerate(selected, start=1):
        print(f"[RUN] {position}/{len(selected)} {item.op} [{item.kind}]")
        try:
            if item.kind == "direct_case":
                result = run_direct_item(item, worktree, run_dir, bootstrap_dir, args)
            elif item.kind == "pytest_node":
                result = run_pytest_node_item(item, worktree, run_dir, bootstrap_dir, args)
            elif item.kind == "legacy_marker":
                result = run_legacy_marker_item(item, worktree, run_dir, bootstrap_dir, no_cpu, args)
            else:
                raise ValueError(f"unknown baseline kind: {item.kind}")
        except Exception as exc:
            error_dir = run_dir / "items" / f"{item.index:03d}_{normalize_name(item.op)}"
            error_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = error_dir / "stdout.log"
            stderr_log = error_dir / "stderr.log"
            write_text(stdout_log, "")
            write_text(stderr_log, f"{type(exc).__name__}: {exc}\n")
            debug_dir = error_dir / "debug_reports"
            debug_dir.mkdir(exist_ok=True)
            result = make_base_result(
                item,
                status="harness_error",
                exit_code=None,
                duration_sec=0.0,
                command="",
                script=None,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                debug_dir=debug_dir,
                first_error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        write_progress(run_dir, baseline, results)
        if args.prune_item_caches:
            removed = prune_transient_item_caches(result)
            if removed:
                print(f"[CLEAN] removed {len(removed)} transient item cache(s)")
        print(f"[RESULT] {item.op}: {result.comparison_status} "
              f"status={result.status} txt={result.debug_txt_count} "
              f"json={result.debug_json_count} sec={result.duration_sec:.1f}")
        if args.stop_on_device_error and result.status == "device_error":
            print("[STOP] device_error encountered; stopping before later cases "
                  "reuse a possibly unhealthy NPU context")
            break

    if not args.keep_worktree and worktree.exists():
        shutil.rmtree(worktree)
    regressions = [result for result in results if result.comparison_status == "regressed"]
    print(f"[DONE] regressions={len(regressions)} comparison={run_dir / 'comparison.md'}")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
