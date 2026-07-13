#!/usr/bin/env python3
"""Unified test entry for FlagTree Profiler CANN suites.

This is the user-facing test runner. It dispatches to the focused internal
suite drivers for custom Triton kernels, Liger-Kernel, and FlagGems, then
writes one top-level summary.

Example:

    python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
      --out /tmp/proton_cann_tests --clean
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

SUITE_SCRIPTS = {
    "custom": SCRIPT_DIR / "cann_operator_profile_suite.py",
    "liger": SCRIPT_DIR / "cann_liger_profile_suite.py",
    "flaggems": SCRIPT_DIR / "cann_flaggems_profile_suite.py",
}


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/proton_cann_profile_tests")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--with-liger",
        action="store_true",
        help="Also run the Liger-Kernel public Triton operator suite.",
    )
    parser.add_argument(
        "--with-flaggems",
        action="store_true",
        help="Also run the FlagGems public Triton operator suite.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Return non-zero when any selected suite reports a failure.",
    )

    parser.add_argument(
        "--custom-operator",
        action="append",
        dest="custom_operators",
        help="Custom Triton operator name to run. May be repeated.",
    )

    parser.add_argument(
        "--liger-source",
        help="Existing Liger-Kernel checkout. Omit to let the Liger suite clone into its output directory.",
    )
    parser.add_argument(
        "--liger-case",
        action="append",
        dest="liger_cases",
        help="Liger case name to run. May be repeated.",
    )

    parser.add_argument(
        "--flaggems-source",
        help="Existing FlagGems checkout. Omit to let the FlagGems suite clone into its output directory.",
    )
    parser.add_argument("--flaggems-all", action="store_true", help="Run all FlagGems op-level cases.")
    parser.add_argument(
        "--flaggems-op",
        action="append",
        dest="flaggems_ops",
        help="FlagGems op marker/name to run. May be repeated.",
    )
    parser.add_argument("--flaggems-max-ops", type=int)
    parser.add_argument(
        "--list-flaggems-ops",
        action="store_true",
        help="Only list discovered FlagGems op-level cases.",
    )
    parser.add_argument("--pytest-timeout", type=float, default=300.0)
    return parser


def _selected_suites(args: argparse.Namespace) -> list[str]:
    selected = ["custom"]
    if args.with_liger:
        selected.append("liger")
    if args.with_flaggems:
        selected.append("flaggems")
    return selected


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _repo_root() -> pathlib.Path:
    current = SCRIPT_DIR
    for parent in [current, *current.parents]:
        if (parent / "python").exists() and (parent / "third_party").exists():
            return parent
    return pathlib.Path.cwd()


def _flagtree_python_build_path(repo_root: pathlib.Path) -> pathlib.Path | None:
    build_root = repo_root / "python" / "build"
    if not build_root.exists():
        return None
    candidates = sorted(path for path in build_root.glob("lib.*") if path.is_dir())
    return candidates[-1] if candidates else None


def _child_env(repo_root: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    build_path = _flagtree_python_build_path(repo_root)
    if build_path is None:
        return env
    build_path_str = str(build_path)
    current_pythonpath = env.get("PYTHONPATH")
    if current_pythonpath:
        paths = current_pythonpath.split(os.pathsep)
        paths = [path for path in paths if path != build_path_str]
        env["PYTHONPATH"] = os.pathsep.join([build_path_str, *paths])
    else:
        env["PYTHONPATH"] = build_path_str
    return env


def _run(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, env=env)


def _suite_command(suite: str, suite_out: pathlib.Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(SUITE_SCRIPTS[suite]),
        "--out",
        str(suite_out),
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--clean",
    ]

    if suite == "custom":
        for operator in args.custom_operators or []:
            cmd.extend(["--operator", operator])
        if not args.require_all:
            cmd.append("--allow-op-failures")
        return cmd

    if suite == "liger":
        if args.liger_source:
            cmd.extend(["--liger-source", args.liger_source])
        for case in args.liger_cases or []:
            cmd.extend(["--case", case])
        if not args.require_all:
            cmd.append("--allow-case-failures")
        return cmd

    if suite == "flaggems":
        cmd.extend(["--op-level", "--pytest-timeout", str(args.pytest_timeout)])
        if args.flaggems_source:
            cmd.extend(["--flaggems-source", args.flaggems_source])
        if args.flaggems_all:
            cmd.append("--all")
        if args.list_flaggems_ops:
            cmd.append("--list-ops")
        for op in args.flaggems_ops or []:
            cmd.extend(["--op", op])
        if args.flaggems_max_ops is not None:
            cmd.extend(["--max-ops", str(args.flaggems_max_ops)])
        if args.require_all:
            cmd.append("--require-all")
        return cmd

    raise AssertionError(f"unknown suite: {suite}")


def main() -> int:
    args = _make_arg_parser().parse_args()
    repo_root = _repo_root()
    child_env = _child_env(repo_root)
    out = pathlib.Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    start = time.perf_counter()
    for suite in _selected_suites(args):
        suite_out = out / suite
        cmd = _suite_command(suite, suite_out, args)
        suite_start = time.perf_counter()
        completed = _run(cmd, repo_root, child_env)
        elapsed_s = time.perf_counter() - suite_start
        summary_path = suite_out / "summary.json"
        summary = _load_json(summary_path)
        result = {
            "suite": suite,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "elapsed_s": elapsed_s,
            "out": str(suite_out),
            "summary_json": str(summary_path),
            "summary": summary,
        }
        results.append(result)
        if completed.returncode != 0:
            failures.append({
                "suite": suite,
                "returncode": completed.returncode,
                "summary_json": str(summary_path),
            })
            if args.require_all:
                break

    summary = {
        "backend": "cann",
        "suite": "FlagTree Profiler CANN unified test suite",
        "selected_suites": [result["suite"] for result in results],
        "suite_count": len(results),
        "ok_count": sum(1 for result in results if result["status"] == "ok"),
        "failed_count": len(failures),
        "failures": failures,
        "elapsed_s": time.perf_counter() - start,
        "repo_root": str(repo_root),
        "flagtree_python_build": str(_flagtree_python_build_path(repo_root) or ""),
        "results": results,
        "summary_json": str(out / "summary.json"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures and args.require_all:
        return 4
    if summary["ok_count"] == 0:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
