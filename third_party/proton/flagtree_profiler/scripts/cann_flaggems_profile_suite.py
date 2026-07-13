#!/usr/bin/env python3
"""Profile FlagGems benchmark operators with Proton CANN.

FlagGems is a public Triton operator library with an Ascend backend. This
driver runs selected FlagGems benchmark files twice: once as a clean baseline
and once inside ``proton.start(..., backend="cann", hook="triton")``. It keeps
the FlagGems checkout untouched and imports it from source.

Example:

    python third_party/proton/flagtree_profiler/scripts/cann_flaggems_profile_suite.py \
      --out /tmp/proton_cann_flaggems_full \
      --clean
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from typing import Any

DEFAULT_CASES = (
    "test_add.py",
    "test_addmm.py",
    "test_bmm.py",
    "test_mm.py",
    "test_softmax.py",
    "test_log_softmax.py",
    "test_amax.py",
    "test_argmax.py",
    "test_cumsum.py",
    "test_where_self_out.py",
    "test_arange.py",
    "test_zeros.py",
)


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/proton_cann_flaggems_full")
    parser.add_argument("--flaggems-source", help="Path to a FlagGems checkout.")
    parser.add_argument(
        "--clone-flaggems",
        action="store_true",
        help=
        "Clone FlagGems into --flaggems-source, or <out>/FlagGems when omitted. This is the default when --flaggems-source is not set.",
    )
    parser.add_argument(
        "--no-clone-flaggems",
        action="store_true",
        help="Do not auto-clone FlagGems when --flaggems-source is omitted.",
    )
    parser.add_argument(
        "--flaggems-repo",
        default="https://github.com/flagos-ai/FlagGems.git",
        help="Repository used by --clone-flaggems.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pytest-timeout", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--level", choices=("core", "comprehensive"), default="core")
    parser.add_argument(
        "--dtypes",
        action="append",
        default=["float32"],
        help="FlagGems benchmark dtype. Repeat to pass multiple --dtypes values.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Benchmark file name, stem, or path relative to FlagGems. May be repeated.",
    )
    parser.add_argument("--max-cases", type=int, help="Run only the first N selected cases.")
    parser.add_argument("--all", action="store_true", help="Run all benchmark/test_*.py files.")
    parser.add_argument(
        "--op-level",
        action="store_true",
        help="Run benchmark operators one by one using pytest markers instead of whole files.",
    )
    parser.add_argument(
        "--op",
        action="append",
        dest="ops",
        help="Operator marker/name to run in --op-level mode. May be repeated.",
    )
    parser.add_argument("--max-ops", type=int, help="Run only the first N selected operator cases.")
    parser.add_argument(
        "--list-ops",
        action="store_true",
        help="Discover FlagGems benchmark operators and write summary.json without running them.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Return non-zero when any selected FlagGems case fails.",
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-name", help=argparse.SUPPRESS)
    parser.add_argument("--case-path", help=argparse.SUPPRESS)
    parser.add_argument("--pytest-marker", help=argparse.SUPPRESS)
    parser.add_argument("--pytest-function", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("baseline", "profiled"), help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    parser.add_argument("--pytest-json", help=argparse.SUPPRESS)
    parser.add_argument("--profile-base", help=argparse.SUPPRESS)
    parser.add_argument("--msprof-output", help=argparse.SUPPRESS)
    return parser


def _run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _prepare_flaggems(args: argparse.Namespace, out: pathlib.Path) -> pathlib.Path:
    source = pathlib.Path(args.flaggems_source) if args.flaggems_source else None
    should_clone = args.clone_flaggems or source is None
    if args.no_clone_flaggems:
        should_clone = False
    if should_clone:
        if source is None:
            source = out / "FlagGems"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--depth", "1", args.flaggems_repo, str(source)])
    if source is None:
        raise RuntimeError("Pass --flaggems-source /path/to/FlagGems or allow auto-clone.")
    if not (source / "benchmark").exists() or not (source / "src" / "flag_gems").exists():
        raise RuntimeError(f"FlagGems checkout is incomplete: {source}")
    return source


def _discover_cases(source: pathlib.Path) -> list[dict[str, str]]:
    cases = []
    for path in sorted((source / "benchmark").glob("test_*.py")):
        cases.append({
            "name": path.stem,
            "path": str(path),
            "relative_path": path.relative_to(source).as_posix(),
            "stem": path.stem,
        })
    return cases


def _pytest_mark_name(decorator: ast.AST) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(target, ast.Attribute):
        return None
    mark_name = target.attr
    value = target.value
    if not isinstance(value, ast.Attribute) or value.attr != "mark":
        return None
    if not isinstance(value.value, ast.Name) or value.value.id != "pytest":
        return None
    if mark_name in {
            "parametrize",
            "skip",
            "skipif",
            "xfail",
            "usefixtures",
            "filterwarnings",
            "timeout",
            "tryfirst",
            "trylast",
    }:
        return None
    return mark_name


def _discover_ops(source: pathlib.Path) -> list[dict[str, str]]:
    ops: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in sorted((source / "benchmark").glob("test_*.py")):
        try:
            module = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            for decorator in node.decorator_list:
                marker = _pytest_mark_name(decorator)
                if marker is None:
                    continue
                key = (path.as_posix(), node.name, marker)
                if key in seen:
                    continue
                seen.add(key)
                rel = path.relative_to(source).as_posix()
                ops.append({
                    "name": f"{path.stem}::{node.name}::{marker}",
                    "op": marker,
                    "marker": marker,
                    "test_function": node.name,
                    "path": str(path),
                    "relative_path": rel,
                    "stem": path.stem,
                })
    return ops


def _select_cases(cases: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.all:
        selected = cases
    elif args.cases:
        requested = set(args.cases)
        selected = [
            case for case in cases if case["name"] in requested or case["stem"] in requested
            or case["relative_path"] in requested or pathlib.Path(case["relative_path"]).name in requested
        ]
        found = {case["name"] for case in selected}
        found |= {case["stem"] for case in selected}
        found |= {case["relative_path"] for case in selected}
        found |= {pathlib.Path(case["relative_path"]).name for case in selected}
        unknown = sorted(requested - found)
        if unknown:
            raise RuntimeError(f"Unknown FlagGems case(s): {', '.join(unknown)}")
    else:
        default = set(DEFAULT_CASES)
        selected = [case for case in cases if pathlib.Path(case["relative_path"]).name in default]

    if args.max_cases is not None:
        selected = selected[:max(0, args.max_cases)]
    return selected


def _select_ops(ops: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.ops:
        requested = set(args.ops)
        selected = [
            op for op in ops if op["name"] in requested or op["op"] in requested or op["marker"] in requested
            or f"{op['relative_path']}::{op['marker']}" in requested or f"{op['stem']}::{op['marker']}" in requested
        ]
        found = {op["name"] for op in selected}
        found |= {op["op"] for op in selected}
        found |= {op["marker"] for op in selected}
        found |= {f"{op['relative_path']}::{op['marker']}" for op in selected}
        found |= {f"{op['stem']}::{op['marker']}" for op in selected}
        unknown = sorted(requested - found)
        if unknown:
            raise RuntimeError(f"Unknown FlagGems op(s): {', '.join(unknown)}")
    elif args.all:
        selected = ops
    else:
        default = set(DEFAULT_CASES)
        selected = [op for op in ops if pathlib.Path(op["relative_path"]).name in default]

    if args.max_ops is not None:
        selected = selected[:max(0, args.max_ops)]
    elif args.max_cases is not None:
        selected = selected[:max(0, args.max_cases)]
    return selected


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _flagtree_python_build_path() -> pathlib.Path | None:
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    build_root = repo_root / "python" / "build"
    if not build_root.exists():
        return None
    candidates = sorted(build_root.glob("lib.*"))
    return candidates[0] if candidates else None


def _prepend_python_path(path: pathlib.Path | None) -> None:
    if path is None or not path.exists():
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _prepare_worker_imports(case_path: pathlib.Path, device_index: int) -> None:
    _prepend_python_path(_flagtree_python_build_path())
    source = case_path.parents[1]
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source))

    import torch
    import torch_npu  # noqa: F401

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu is unavailable or torch.npu.is_available() is false.")
    torch.npu.set_device(device_index)
    # Initialize before FlagGems benchmark imports set torch.backends.cuda flags.
    torch.empty((1, ), device=f"npu:{device_index}")


def _run_pytest(args: argparse.Namespace) -> tuple[int, float]:
    import pytest

    test_target = str(pathlib.Path(args.case_path).relative_to(pathlib.Path(args.case_path).parents[1]))
    if args.pytest_function:
        test_target = f"{test_target}::{args.pytest_function}"
    pytest_args = [
        test_target,
        "--level",
        args.level,
        "--warmup",
        str(args.warmup),
        "--iter",
        str(args.iters),
        "--record",
        "json",
        "--output",
        args.pytest_json,
        "-q",
    ]
    if args.pytest_marker:
        pytest_args.extend(["-m", args.pytest_marker])
    for dtype in args.dtypes:
        pytest_args.extend(["--dtypes", dtype])

    start = time.perf_counter()
    rc = pytest.main(pytest_args)
    return int(rc), time.perf_counter() - start


def _worker_main(args: argparse.Namespace) -> int:
    result_path = pathlib.Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "name": args.case_name,
        "phase": args.phase,
        "case_path": args.case_path,
        "pytest_marker": args.pytest_marker,
        "pytest_function": args.pytest_function,
        "status": "failed",
    }
    session_id = None
    try:
        case_path = pathlib.Path(args.case_path)
        _prepare_worker_imports(case_path, args.device)

        if args.phase == "profiled":
            import triton.profiler as proton
            from triton._C.libproton import proton as libproton

            mode = ("runtime_base:"
                    "vendor_metrics=aicore,bandwidth:"
                    f"aclprof_output_path={args.msprof_output}:"
                    "aclprof_runtime_enabled=true:"
                    "aclprof_auto_export=true:"
                    "mstx_enabled=true:"
                    "mstx_domain=proton")
            session_id = proton.start(
                name=args.profile_base,
                context="shadow",
                data="tree",
                backend="cann",
                hook="triton",
                mode=mode,
            )
            scope_name = f"proton_cann_flaggems::{args.case_name}"
            scope_id = libproton.record_scope()
            libproton.enter_op(scope_id, scope_name)
            try:
                pytest_rc, elapsed_s = _run_pytest(args)
            finally:
                libproton.exit_op(scope_id, scope_name)
            result["scope"] = scope_name
            result["profile_base"] = args.profile_base
        else:
            pytest_rc, elapsed_s = _run_pytest(args)

        result.update({
            "pytest_returncode": pytest_rc,
            "elapsed_s": elapsed_s,
            "status": "ok" if pytest_rc == 0 else "failed",
            "pytest_json": args.pytest_json,
            "iters": args.iters,
            "warmup": args.warmup,
        })
        if pathlib.Path(args.pytest_json).exists():
            result["pytest_result"] = json.loads(pathlib.Path(args.pytest_json).read_text())
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if session_id is not None:
            try:
                proton.finalize(session_id)
            except Exception as exc:
                result["finalize_error"] = repr(exc)
                if result.get("status") == "ok":
                    result["status"] = "failed"
        result_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0 if result.get("status") == "ok" else 3


def _run_worker(args: argparse.Namespace, case: dict[str, str], phase: str, out: pathlib.Path) -> dict:
    case_dir = out / "cases" / _safe_name(case["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result_json = case_dir / f"{phase}.json"
    pytest_json = case_dir / f"{phase}.pytest.json"
    profile_base = case_dir / "profile"
    msprof_output = case_dir / "msprof"
    cmd = [
        sys.executable,
        pathlib.Path(__file__).as_posix(),
        "--worker",
        "--case-name",
        case["name"],
        "--case-path",
        case["path"],
        "--phase",
        phase,
        "--result-json",
        str(result_json),
        "--pytest-json",
        str(pytest_json),
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--level",
        args.level,
        "--profile-base",
        str(profile_base),
        "--msprof-output",
        str(msprof_output),
    ]
    if case.get("marker"):
        cmd.extend(["--pytest-marker", case["marker"]])
    if case.get("test_function"):
        cmd.extend(["--pytest-function", case["test_function"]])
    for dtype in args.dtypes:
        cmd.extend(["--dtypes", dtype])

    env = os.environ.copy()
    build_path = _flagtree_python_build_path()
    if build_path is not None and build_path.exists():
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (str(build_path) if not current_pythonpath else str(build_path) + os.pathsep +
                             current_pythonpath)

    try:
        completed = subprocess.run(
            cmd,
            cwd=pathlib.Path(case["path"]).parents[1],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.pytest_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "name": case["name"],
            "phase": phase,
            "status": "failed",
            "error": f"timeout after {args.pytest_timeout}s",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        result_json.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return result

    if result_json.exists():
        result = json.loads(result_json.read_text())
    else:
        result = {
            "name": case["name"],
            "phase": phase,
            "status": "failed",
            "error": "worker exited before writing result_json",
        }
    result["returncode"] = completed.returncode
    if completed.stdout:
        result["stdout"] = completed.stdout[-4000:]
    if completed.stderr:
        result["stderr"] = completed.stderr[-4000:]
    result_json.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return result


def _timing_fields(baseline: dict, profiled: dict) -> dict:
    baseline_elapsed = baseline.get("elapsed_s")
    profiled_elapsed = profiled.get("elapsed_s")
    if baseline_elapsed is None or profiled_elapsed is None:
        return {}
    overhead_s = profiled_elapsed - baseline_elapsed
    overhead_ratio = overhead_s / baseline_elapsed if baseline_elapsed > 0 else None
    return {
        "baseline_elapsed_s": baseline_elapsed,
        "profiled_elapsed_s": profiled_elapsed,
        "elapsed_s": profiled_elapsed,
        "overhead_s": overhead_s,
        "overhead_ratio": overhead_ratio,
        "overhead_percent": overhead_ratio * 100.0 if overhead_ratio is not None else None,
    }


def _summarize_timing(results: list[dict]) -> dict:
    timed = [
        result for result in results if result.get("status") == "ok" and result.get("baseline_elapsed_s") is not None
        and result.get("profiled_elapsed_s") is not None
    ]
    ratios = [result["overhead_ratio"] for result in timed if result.get("overhead_ratio") is not None]
    baseline_total = sum(result["baseline_elapsed_s"] for result in timed)
    profiled_total = sum(result["profiled_elapsed_s"] for result in timed)
    weighted_ratio = (profiled_total - baseline_total) / baseline_total if baseline_total > 0 else None
    return {
        "timed_case_count": len(timed),
        "baseline_total_s": baseline_total,
        "profiled_total_s": profiled_total,
        "overhead_total_s": profiled_total - baseline_total,
        "average_overhead_ratio": sum(ratios) / len(ratios) if ratios else None,
        "average_overhead_percent": (sum(ratios) / len(ratios) * 100.0) if ratios else None,
        "weighted_overhead_ratio": weighted_ratio,
        "weighted_overhead_percent": weighted_ratio * 100.0 if weighted_ratio is not None else None,
    }


def _summarize_profile_artifacts(results: list[dict]) -> dict:
    source_counts: Counter[str] = Counter()
    op_types: Counter[str] = Counter()
    mstx_messages: set[str] = set()
    raw_input_count = 0
    association_count = 0
    bandwidth_count = 0
    timeline_event_count = 0
    degrade_reasons: list[str] = []

    for result in results:
        if result.get("status") != "ok" or not result.get("profile_base"):
            continue
        base = pathlib.Path(result["profile_base"])
        meta_path = base.with_suffix(".meta.json")
        vendor_path = base.with_suffix(".vendor.json")
        timeline_path = base.with_suffix(".timeline.json")
        if not meta_path.exists() or not vendor_path.exists() or not timeline_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            vendor = json.loads(vendor_path.read_text())
            timeline = json.loads(timeline_path.read_text().splitlines()[0])
        except Exception:
            continue

        degrade_reasons.extend(meta.get("degrade_reasons", []))
        raw_input_count += len(vendor.get("raw_inputs", []))
        associations = vendor.get("associations", [])
        association_count += len(associations)
        timeline_event_count += len(timeline.get("traceEvents", []))
        for assoc in associations:
            source = assoc.get("source", "")
            if source:
                source_counts[source] += 1
            metrics = assoc.get("metrics", {})
            op_type = metrics.get("op_type")
            if op_type:
                op_types[op_type] += 1
            if "bandwidth_gb_s" in metrics:
                bandwidth_count += 1
            if source == "msprof_mstx" and metrics.get("message"):
                mstx_messages.add(metrics["message"])

    return {
        "raw_input_count": raw_input_count,
        "association_count": association_count,
        "association_sources": dict(source_counts),
        "bandwidth_association_count": bandwidth_count,
        "mstx_range_count": len(mstx_messages),
        "mstx_ranges": sorted(mstx_messages)[:200],
        "top_op_types": dict(op_types.most_common(80)),
        "timeline_event_count": timeline_event_count,
        "degrade_reasons": sorted(set(degrade_reasons)),
    }


def _driver_main(args: argparse.Namespace) -> int:
    out = pathlib.Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)

    source = _prepare_flaggems(args, out)
    all_cases = _discover_ops(source) if args.op_level or args.list_ops else _discover_cases(source)
    selected = _select_ops(all_cases, args) if args.op_level or args.list_ops else _select_cases(all_cases, args)
    if not selected:
        raise RuntimeError("No FlagGems benchmark cases selected.")

    if args.list_ops:
        op_counts: Counter[str] = Counter(case.get("op", case["name"]) for case in all_cases)
        summary = {
            "backend": "cann",
            "suite": "FlagGems benchmark op-level",
            "flaggems_source": str(source),
            "op_case_count": len(all_cases),
            "unique_op_count": len(op_counts),
            "selected_count": len(selected),
            "ops": selected,
            "unique_ops": sorted(op_counts),
            "duplicate_op_markers": {name: count
                                     for name, count in op_counts.items()
                                     if count > 1},
            "summary_json": str(out / "summary.json"),
        }
        summary = _jsonable(summary)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    results = []
    failures = []
    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case['name']} baseline", flush=True)
        baseline = _run_worker(args, case, "baseline", out)
        if baseline.get("status") != "ok":
            failures.append({"name": case["name"], "phase": "baseline", "error": baseline.get("error", "failed")})
            results.append({**case, "status": "failed", "phase": "baseline", "baseline": baseline})
            continue

        print(f"[{index}/{len(selected)}] {case['name']} profiled", flush=True)
        profiled = _run_worker(args, case, "profiled", out)
        if profiled.get("status") != "ok":
            failures.append({"name": case["name"], "phase": "profiled", "error": profiled.get("error", "failed")})
            results.append(
                {**case, "status": "failed", "phase": "profiled", "baseline": baseline, "profiled": profiled})
            continue

        result = {
            **case,
            "status": "ok",
            "baseline": baseline.get("pytest_result", {}),
            "profiled": profiled.get("pytest_result", {}),
            "profile_base": profiled.get("profile_base"),
            "scope": profiled.get("scope"),
        }
        result.update(_timing_fields(baseline, profiled))
        results.append(result)

    summary = {
        "backend": "cann",
        "suite": "FlagGems benchmark op-level" if args.op_level else "FlagGems benchmark",
        "flaggems_source": str(source),
        "case_count": len(selected),
        "op_level": bool(args.op_level),
        "unique_op_count": len({case.get("op", case["name"])
                                for case in selected}),
        "ok_count": sum(1 for result in results if result.get("status") == "ok"),
        "failed_count": len(failures),
        "failures": failures,
        "results": results,
        "timing": _summarize_timing(results),
        "overhead_method": "separate_process_no_profiler_baseline",
        "summary_json": str(out / "summary.json"),
    }
    summary.update(_summarize_profile_artifacts(results))
    summary = _jsonable(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))

    if failures and args.require_all:
        return 4
    if summary["ok_count"] == 0:
        return 5
    return 0


def main() -> int:
    args = _make_arg_parser().parse_args()
    if args.worker:
        return _worker_main(args)
    return _driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
