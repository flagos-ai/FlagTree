#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_gems_record.py - Flatten a FlagGems CI run into one analysable record

Takes the artifacts of a single FlagGems run (summary.json, and the
compare_report.json produced by compare_test_result.py) plus the CI context
that only exists in the environment (backend, runner, commit, PR number), and
writes:

    <out-dir>/record.json          one flat object, ready to push to a table
    <ledger>                       the same object appended as JSONL
    <op-ledger>/<record_id>.jsonl  one line per operator (optional)

The ledger is the durable store: it is written unconditionally so that history
can be re-ingested later if the reporting schema changes.

Several runners of one backend share the ledger directory through a host
mount, so run-level records are appended with a single atomic write while
operator rows, being far too large for that, get a file per run.

Usage:
    python collect_gems_record.py --summary PR/1400/summary.json \\
        --ledger /root/flaggems_ci/Ledger/runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Context is taken from CTX_* environment variables rather than flags, so that
# secrets and multi-line CI values never land in a process listing.
CTX_PREFIX = "CTX_"

# Operator-level status values that mean "this operator produced no result".
_NO_RESULT_STATUSES = {"NotFound", "Not Found", "NOT_FOUND"}

# ============================================================================
# Helpers
# ============================================================================


def read_baseline_ref(baseline_dir: Optional[Path]) -> str:
    """Read the baseline's self-identification from <baseline_dir>/commit_id.txt.

    Written by the baseline workflow as "<date> : <commit>". Without it the
    delta fields cannot be attributed to a specific baseline, so it is read
    from the directory directly rather than relying on the CI passing a label
    through to compare_test_result.py.
    """
    if baseline_dir is None:
        return ""
    path = baseline_dir / "commit_id.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        print(f"[INFO] No baseline ref at {path}")
        return ""


def read_json(path: Optional[Path], what: str, required: bool) -> Optional[Dict]:
    """Load a JSON file, or return None when it is absent/unreadable."""
    if path is None:
        return None
    if not path.exists():
        msg = f"{what} not found: {path}"
        if required:
            print(f"[ERROR] {msg}")
            sys.exit(1)
        print(f"[INFO] {msg} (continuing without it)")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        msg = f"Failed to read {what} {path}: {e}"
        if required:
            print(f"[ERROR] {msg}")
            sys.exit(1)
        print(f"[WARN] {msg} (continuing without it)")
        return None


def ctx(name: str, default: str = "") -> str:
    """Read a CTX_<NAME> environment variable."""
    return os.environ.get(f"{CTX_PREFIX}{name}", default).strip()


def ctx_int(name: str) -> Optional[int]:
    """Read a CTX_<NAME> variable as int, or None when unset/not a number."""
    raw = ctx(name)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def ctx_bool(name: str) -> Optional[bool]:
    """Read a CTX_<NAME> variable as bool, or None when unset."""
    raw = ctx(name).lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    return None


def ratio(numerator: int, denominator: int) -> Optional[float]:
    """Safe division, rounded, or None when the denominator is zero."""
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def join_names(names: List[str], limit: int = 50) -> str:
    """Join names into one comma-separated cell, truncating politely."""
    if not names:
        return ""
    head = names[:limit]
    text = ",".join(head)
    if len(names) > limit:
        text += f",... (+{len(names) - limit} more)"
    return text


def append_jsonl(path: Optional[Path], rows: List[Dict]) -> bool:
    """Append rows to a shared JSONL file. Never raises.

    Several runners of the same backend run in separate containers that share
    one host directory, so two jobs can append here at the same time. The
    payload is therefore serialised first and handed to a single unbuffered
    write(): under O_APPEND the kernel makes one write atomic, so lines never
    interleave.

    This holds only while the payload stays small. Anything that could exceed
    a pipe buffer's worth of data must go to a per-run file instead -- see
    run_scoped_path().
    """
    if path is None or not rows:
        return True

    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # os.open/os.write rather than open(..., "a"): buffered writes would be
        # split into several syscalls and lose the atomicity above.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(fd, payload)
        finally:
            os.close(fd)
        if written != len(payload):
            print(f"[WARN] Short write to {path}: {written}/{len(payload)} bytes")
            return False
        print(f"[INFO] Appended {len(rows)} row(s) to {path}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to append to {path}: {e}")
        return False


def write_jsonl(path: Optional[Path], rows: List[Dict]) -> bool:
    """Write rows to their own JSONL file, atomically. Never raises.

    For payloads too large to append atomically. Written to a temporary name
    and then renamed, so a reader globbing the directory never sees a partial
    file.
    """
    if path is None or not rows:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        tmp.replace(path)
        print(f"[INFO] Wrote {len(rows)} row(s) to {path}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to write {path}: {e}")
        return False


def run_scoped_path(base: Path, record_id: str) -> Path:
    """Per-run file under `base`, named after the record it belongs to.

    Accepts either a directory or a .jsonl path for convenience: passing
    `ledger/operators.jsonl` yields `ledger/operators/<record_id>.jsonl`.
    """
    directory = base.parent / base.stem if base.suffix == ".jsonl" else base
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", record_id) or "unknown"
    return directory / f"{safe_id}.jsonl"


def write_json(path: Path, payload: Dict) -> bool:
    """Write a JSON file atomically. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
        print(f"[INFO] Saved {path}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to write {path}: {e}")
        return False


# ============================================================================
# Record building
# ============================================================================

SHORT_SHA_LEN = 12


def build_record_id() -> str:
    """Dedup key identifying what was measured, not which CI run measured it.

    Reporting updates the row whose key matches, so the key decides what
    collapses into one row:

      pr-<number>-<backend>            a PR keeps only its latest result,
                                       since earlier commits in a PR are
                                       usually work in progress
      baseline-<short sha>-<backend>   one row per main commit, which is the
                                       series worth analysing

    Reruns therefore overwrite rather than add a row. The run that produced
    the current values stays visible through run_id/run_attempt/run_url, and
    the superseded attempts remain in the ledger.

    Falls back to the run id when neither a PR number nor a commit is known,
    so a manual invocation still produces something unique.
    """
    backend = ctx("BACKEND") or "unknown"
    kind = ctx("KIND", "test")
    pr_number = ctx_int("PR_NUMBER")
    sha = ctx("FLAGTREE_SHA")

    if kind == "test" and pr_number:
        return f"pr-{pr_number}-{backend}"
    if sha:
        return f"{kind}-{sha[:SHORT_SHA_LEN]}-{backend}"
    return f"{ctx('RUN_ID') or 'local'}-{ctx_int('RUN_ATTEMPT') or 1}-{backend}"


def job_status(compare: Optional[Dict]) -> str:
    """Derive a single outcome for the whole job.

    Ordered by how early the pipeline broke, so the most upstream failure wins
    and is never masked by a missing later stage.
    """
    if ctx("BUILD_OUTCOME") not in ("", "success"):
        return "build_failed"
    if ctx("TEST_OUTCOME") not in ("", "success"):
        return "test_failed"
    if compare is None:
        return "no_baseline"
    if compare.get("summary", {}).get("verdict") == "REJECT":
        return "regression"
    return "success"


def build_record(summary: Dict, compare: Optional[Dict], baseline_ref: str = "") -> Dict:
    """Flatten summary + compare report + CI context into one row."""
    totals = summary.get("totals", {}) or {}
    passed = totals.get("passed", 0)
    failed = totals.get("failed", 0)
    errors = totals.get("errors", 0)
    skipped = totals.get("skipped", 0)
    total_cases = totals.get("total_cases", 0)
    skip_ops = (summary.get("skip_lists", {}) or {}).get("skip_operators", []) or []

    record = {
        # --- identity ---
        "record_id": build_record_id(),
        "kind": ctx("KIND", "test"),
        "backend": ctx("BACKEND"),
        "runner": ctx("RUNNER"),
        "flaggems_tag": ctx("FLAGGEMS_TAG"),
        "flagtree_sha": ctx("FLAGTREE_SHA"),
        "flagtree_branch": ctx("FLAGTREE_BRANCH"),
        "pr_number": ctx_int("PR_NUMBER"),
        "pr_url": ctx("PR_URL"),
        "actor": ctx("ACTOR"),
        "run_id": ctx("RUN_ID"),
        "run_attempt": ctx_int("RUN_ATTEMPT") or 1,
        "run_url": ctx("RUN_URL"),
        "job_status": job_status(compare),
        # Baseline runs only: whether this run became the new best baseline
        "best_updated": ctx_bool("BEST_UPDATED"),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # --- this run, absolute ---
        "tested_at": summary.get("timestamp", ""),
        "duration_s": round(summary.get("duration_seconds", 0) or 0, 2),
        "marks_collection_ok": summary.get("marks_collection_ok"),
        "total_ops": totals.get("total_ops", 0),
        "total_cases": total_cases,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "not_found": totals.get("not_found", 0),
        # Two rates on purpose. pass_rate_all matches what the comparison
        # report prints. pass_rate_run excludes skipped cases, which is the
        # only rate comparable ACROSS backends, since each backend skips a
        # different set of operators via its own skip_list.yaml.
        "pass_rate_all": ratio(passed, total_cases),
        "pass_rate_run": ratio(passed, passed + failed + errors),
        # --- comparability metadata ---
        "skip_operators_cnt": len(skip_ops),
        "skip_operators": join_names(sorted(skip_ops)),
    }

    record.update(compare_fields(compare, baseline_ref))
    return record


def compare_fields(compare: Optional[Dict], baseline_ref: str = "") -> Dict:
    """Extract the delta-vs-baseline half of a record.

    All keys are always present so that every ledger row has the same shape;
    they stay None when no comparison was made (e.g. the very first baseline,
    where there is no previous best to compare against).
    """
    fields: Dict = {
        "compared": compare is not None,
        # Known from the baseline dir even when no comparison was made
        "baseline_ref": baseline_ref or None,
        "verdict": None,
        "regressions_cnt": None,
        "improvements_cnt": None,
        "warnings_cnt": None,
        "info_changes_cnt": None,
        "new_operators_cnt": None,
        "removed_operators_cnt": None,
        "delta_passed": None,
        "delta_failed": None,
        "delta_skipped": None,
        "delta_errors": None,
        "delta_cases": None,
        "regressed_ops": None,
        "improved_ops": None,
    }
    if compare is None:
        return fields

    stats = compare.get("statistics", {}) or {}
    totals = compare.get("totals", {}) or {}
    summary = compare.get("summary", {}) or {}

    def delta(key: str) -> Optional[int]:
        return (stats.get(key) or {}).get("delta")

    # Fall back to the comparison's own label when the baseline dir carried no
    # commit_id.txt (e.g. a hand-run comparison between two arbitrary dirs).
    label = compare.get("baseline_label") or ""
    fields.update({
        "compared": True,
        "baseline_ref": baseline_ref or (label if label != "Baseline" else None),
        # Plain token from compare_test_result.py, free of emoji so it can
        # be filtered and counted directly.
        "verdict": summary.get("verdict"),
        "regressions_cnt": totals.get("regressions"),
        "improvements_cnt": totals.get("improvements"),
        "warnings_cnt": totals.get("warnings"),
        "info_changes_cnt": totals.get("info_changes"),
        "new_operators_cnt": totals.get("new_operators"),
        "removed_operators_cnt": totals.get("removed_operators"),
        "delta_passed": delta("passed"),
        "delta_failed": delta("failed"),
        "delta_skipped": delta("skipped"),
        "delta_errors": delta("errors"),
        "delta_cases": delta("test_cases"),
        "regressed_ops": join_names(sorted({c.get("operator", "")
                                            for c in compare.get("regressions", [])})),
        "improved_ops": join_names(sorted({c.get("operator", "")
                                           for c in compare.get("improvements", [])})),
    })
    return fields


def build_operator_rows(record: Dict, summary: Dict, compare: Optional[Dict]) -> List[Dict]:
    """One row per operator, for per-operator trend analysis.

    Kept out of the run-level table: ~420 operators x 10 backends per day is
    too many rows for a spreadsheet, but cheap as local JSONL.
    """
    # operator -> worst change seen, so a regression is never hidden behind an
    # unchanged case of the same operator.
    change_by_op: Dict[str, str] = {}
    if compare:
        for bucket in ("regressions", "warnings", "improvements", "info_changes"):
            for change in compare.get(bucket, []) or []:
                change_by_op.setdefault(change.get("operator", ""), change.get("change_type", ""))

    rows = []
    for name, op in (summary.get("operators", {}) or {}).items():
        status = op.get("status", "Unknown")
        rows.append({
            "record_id": record["record_id"],
            "backend": record["backend"],
            "runner": record["runner"],
            "kind": record["kind"],
            "flagtree_sha": record["flagtree_sha"],
            "flaggems_tag": record["flaggems_tag"],
            "pr_number": record["pr_number"],
            "tested_at": record["tested_at"],
            "operator": name,
            "status": status,
            "has_result": status not in _NO_RESULT_STATUSES,
            "total": op.get("total", 0),
            "passed": op.get("passed", 0),
            "failed": op.get("failed", 0),
            "skipped": op.get("skipped", 0),
            "errors": op.get("errors", 0),
            "change_type": change_by_op.get(name, ""),
            "failed_cases": join_names(op.get("failed_cases", []) or [], limit=20),
            "error_cases": join_names(op.get("error_cases", []) or [], limit=20),
        })
    return rows


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Context is read from CTX_* environment variables, all optional:

  CTX_KIND            test | baseline            (default: test)
  CTX_BACKEND         amd, nvidia, ...
  CTX_RUNNER          amd3.6, ...
  CTX_FLAGGEMS_TAG    v5.3.5
  CTX_FLAGTREE_SHA    commit under test
  CTX_FLAGTREE_BRANCH main
  CTX_PR_NUMBER       1400
  CTX_PR_URL          https://github.com/...
  CTX_ACTOR           github login
  CTX_RUN_ID          GitHub run id
  CTX_RUN_ATTEMPT     GitHub run attempt    (default: 1)
  CTX_RUN_URL         link to the run
  CTX_BUILD_OUTCOME   success | failure | ...
  CTX_TEST_OUTCOME    success | failure | ...
  CTX_BEST_UPDATED    true | false              (baseline runs only)

Examples:
  # PR test run, with comparison, appending to both ledgers
  CTX_BACKEND=amd CTX_RUNNER=amd3.6 CTX_PR_NUMBER=1400 \\
  python collect_gems_record.py \\
      --summary /root/flaggems_ci/PR/1400/summary.json \\
      --ledger /root/flaggems_ci/Ledger/runs.jsonl \\
      --op-ledger /root/flaggems_ci/Ledger/operators.jsonl

  # First baseline on a fresh runner: no compare report exists yet
  CTX_KIND=baseline CTX_BACKEND=amd \\
  python collect_gems_record.py \\
      --summary /root/flaggems_ci/Baseline/20261020_080513_<sha>/summary.json
        """,
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Path to the candidate summary.json produced by ci_run_tests.py",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help=("Path to compare_report.json (default: compare_report.json next to "
              "--summary). Missing is tolerated: delta fields stay empty."),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help=("Baseline directory, read for commit_id.txt to record which "
              "baseline the deltas are relative to"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write record.json (default: alongside --summary)",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="JSONL file to append the run-level record to",
    )
    parser.add_argument(
        "--op-ledger",
        type=Path,
        default=None,
        help=("Directory for per-operator rows. Each run writes its own "
              "<record_id>.jsonl there, so concurrent runners never share a "
              "file. Passing 'ledger/operators.jsonl' is accepted and means "
              "the 'ledger/operators' directory."),
    )
    parser.add_argument(
        "--no-record-file",
        action="store_true",
        help="Skip writing record.json (ledgers only)",
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print the record to stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = read_json(args.summary, "Summary", required=True)
    compare_path = args.compare or args.summary.parent / "compare_report.json"
    compare = read_json(compare_path, "Compare report", required=False)

    record = build_record(summary, compare, read_baseline_ref(args.baseline_dir))

    if args.do_print:
        print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"[INFO] {record['record_id']} "
              f"kind={record['kind']} status={record['job_status']} "
              f"passed={record['passed']}/{record['total_cases']} "
              f"pass_rate_run={record['pass_rate_run']} "
              f"verdict={record['verdict']}")

    if not args.no_record_file:
        out_dir = args.out_dir or args.summary.parent
        write_json(out_dir / "record.json", record)

    append_jsonl(args.ledger, [record])
    # Operator rows go to their own file per run: ~420 lines is far too much to
    # append atomically while other runners share this directory.
    if args.op_ledger:
        write_jsonl(
            run_scoped_path(args.op_ledger, record["record_id"]),
            build_operator_rows(record, summary, compare),
        )

    # Collection is observability, not a gate: always succeed so a CI step
    # using this script can never fail a PR.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
