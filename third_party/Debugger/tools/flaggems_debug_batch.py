#!/usr/bin/env python3
"""Batch-instrument and run FlagGems operators with FlagTree debugger enabled.

It copies FlagGems to an ignored worktree, instruments the copy, runs each op
in an isolated subprocess, and writes all experiment output under one run dir.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FLAGGEMS_ROOT = Path(os.environ.get("FLAGGEMS_ROOT", REPO_ROOT.parent / "FlagGems"))
DEFAULT_WORKSPACE_ROOT = Path(".cache/flaggems_debugger_batch")
DEFAULT_PYTHON = Path(os.environ.get("PYTHON", sys.executable))
ASCEND_SET_ENV = Path(os.environ.get("ASCEND_SET_ENV", "/usr/local/Ascend/ascend-toolkit/set_env.sh"))


@dataclass
class InstrumentationStats:
    files_scanned: int = 0
    files_changed: int = 0
    files_normalized: int = 0
    ext_launch_id_calls_normalized: int = 0
    launch_jit_functions_detected: int = 0
    functions_instrumented: int = 0
    functions_skipped_existing_debug: int = 0
    functions_skipped_jit_helper: int = 0
    pointwise_dynamic_codegen_patched: bool = False
    pointwise_dynamic_helpers_skipped: int = 0
    parse_errors: int = 0


@dataclass
class OpStatus:
    op: str
    phase: str
    status: str
    exit_code: int | None
    duration_sec: float
    command: str
    stdout_log: str
    stderr_log: str
    result_json: str | None
    debug_report_dir: str
    debug_txt_count: int
    debug_json_count: int
    first_error: str


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_operator_inventory(flaggems_root: Path) -> list[dict[str, Any]]:
    inventory_path = flaggems_root / "conf" / "operators.yaml"
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(read_text(inventory_path))
        return list(data.get("ops", []))
    except Exception:
        # Minimal fallback for environments without PyYAML. It is enough for
        # selecting op ids, but stage/label filtering will be unavailable.
        ops: list[dict[str, Any]] = []
        for line in read_text(inventory_path).splitlines():
            match = re.match(r"\s*-\s+id:\s*['\"]?([^'\"]+)['\"]?\s*$", line)
            if match:
                ops.append({"id": match.group(1)})
        return ops


def select_ops(args: argparse.Namespace, inventory: list[dict[str, Any]]) -> list[str]:
    if args.ops:
        return [op.strip().lstrip("_") for op in args.ops.split(",") if op.strip()]

    if args.op_list_file:
        selected = []
        for line in read_text(Path(args.op_list_file)).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                selected.append(stripped.lstrip("_"))
        return selected

    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if "all" in stages:
        stages = {"alpha", "beta", "stable"}
    if not stages:
        stages = {"stable"}

    selected = []
    for item in inventory:
        op_id = str(item.get("id", "")).strip()
        if not op_id:
            continue
        item_stages = item.get("stages") or []
        stage = None
        if item_stages:
            last = item_stages[-1]
            if isinstance(last, dict) and last:
                stage = next(iter(last.keys()))
        if stage not in stages:
            continue
        if args.start and op_id < args.start:
            continue
        selected.append(op_id)

    if args.max_ops:
        selected = selected[:args.max_ops]
    return selected


def no_cpu_ops(inventory: list[dict[str, Any]]) -> set[str]:
    result = set()
    for item in inventory:
        labels = item.get("labels") or []
        if "NoCPU" in labels and item.get("id"):
            result.add(str(item["id"]))
    return result


def inventory_by_id(inventory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in inventory if item.get("id")}


def op_source_candidates(op: str) -> list[str]:
    base = op.split(".")[0]
    candidates = [op, base, op.rstrip("_"), base.rstrip("_")]
    result = []
    for candidate in candidates:
        if candidate and candidate not in result:
            result.append(candidate)
    return result


def op_uses_pointwise_dynamic(
    worktree: Path,
    op: str,
    inventory_map: dict[str, dict[str, Any]],
) -> bool:
    item = inventory_map.get(op) or {}
    labels = item.get("labels") or []
    if "pointwise" in labels:
        return True

    ops_dir = worktree / "src" / "flag_gems" / "ops"
    for candidate in op_source_candidates(op):
        path = ops_dir / f"{candidate}.py"
        if path.exists() and "pointwise_dynamic" in read_text(path):
            return True
    return False


def copy_flaggems_source(src: Path, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"worktree already exists: {dest}")

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".pytest_cache",
            "__pycache__",
            "build",
            "dist",
            "htmlcov",
        }
        return {
            name
            for name in names
            if name in ignored or name.endswith(".pyc") or name.endswith(".pyo")
            or re.match(r"accuracy_.*\.json$", name) or re.match(r"benchmark_.*\.json$", name)
        }

    shutil.copytree(src, dest, symlinks=True, ignore=ignore)


def is_tl_import_present(module: ast.Module) -> bool:
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton.language" and alias.asname == "tl":
                    return True
        if isinstance(node, ast.ImportFrom) and node.module == "triton":
            for alias in node.names:
                if alias.name == "language" and alias.asname == "tl":
                    return True
    return False


def insert_tl_import(module: ast.Module) -> None:
    import_node = ast.Import(names=[ast.alias(name="triton.language", asname="tl")])
    index = 0
    if (module.body and isinstance(module.body[0], ast.Expr) and isinstance(module.body[0].value, ast.Constant)
            and isinstance(module.body[0].value.value, str)):
        index = 1

    while (index < len(module.body) and isinstance(module.body[index], ast.ImportFrom)
           and module.body[index].module == "__future__"):
        index += 1

    while index < len(module.body) and isinstance(module.body[index], (ast.Import, ast.ImportFrom)):
        index += 1

    module.body.insert(index, import_node)


class ExtLaunchIdNormalizer(ast.NodeTransformer):

    def __init__(self):
        self.rewrite_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if not isinstance(node, ast.Call):
            return node
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in {"program_id", "num_programs"}
                and isinstance(func.value, ast.Name) and func.value.id == "ext"):
            return node

        self.rewrite_count += 1
        native_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr=func.attr,
                ctx=ast.Load(),
            ),
            args=node.args,
            keywords=node.keywords,
        )
        return ast.Call(
            func=ast.Attribute(value=native_call, attr="to", ctx=ast.Load()),
            args=[ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="int64",
                ctx=ast.Load(),
            )],
            keywords=[],
        )


def normalize_ext_launch_ids_in_file(path: Path) -> tuple[int, int, str]:
    source = read_text(path)
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return 0, 1, str(exc)

    normalizer = ExtLaunchIdNormalizer()
    module = normalizer.visit(module)  # type: ignore[assignment]
    if normalizer.rewrite_count == 0:
        return 0, 0, ""

    if not is_tl_import_present(module):
        insert_tl_import(module)
    ast.fix_missing_locations(module)
    write_text(path, ast.unparse(module) + "\n")
    return normalizer.rewrite_count, 0, ""


def decorator_is_triton_jit(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if (isinstance(target, ast.Attribute) and target.attr == "jit" and isinstance(target.value, ast.Name)
            and target.value.id == "triton"):
        return True
    try:
        return "triton.jit" in ast.unparse(decorator)
    except Exception:
        return False


def decorator_mentions(decorator: ast.expr, text: str) -> bool:
    try:
        return text in ast.unparse(decorator)
    except Exception:
        return False


def function_is_pointwise_dynamic(fn: ast.FunctionDef | ast.AsyncFunctionDef, ) -> bool:
    return any(decorator_mentions(d, "pointwise_dynamic") for d in fn.decorator_list)


def is_debug_collect_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr in {"debug_collect_start", "debug_collect_end"}
            and isinstance(func.value, ast.Name) and func.value.id == "tl")


def function_has_debug_collect(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(fn):
        if is_debug_collect_call(node):
            return True
    return False


def local_jit_function_names(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(decorator_is_triton_jit(d) for d in node.decorator_list):
            names.add(node.name)
    return names


def launch_target_name(call: ast.Call) -> str | None:
    # Triton launches use kernel[grid](...), represented as a Call whose func is
    # a Subscript.  Restrict automatic instrumentation to local names so module
    # attributes such as ext.program_id are not mistaken for launch kernels.
    if not isinstance(call.func, ast.Subscript):
        return None
    target = call.func.value
    if isinstance(target, ast.Name):
        return target.id
    return None


def launched_local_jit_functions(module: ast.Module) -> set[str]:
    jit_names = local_jit_function_names(module)
    launched: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        name = launch_target_name(node)
        if name in jit_names:
            launched.add(name)
    return launched


def make_debug_start(level: int, addr_level: int) -> ast.Expr:
    return ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="tl", ctx=ast.Load()),
            attr="debug_collect_start",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[
            ast.keyword(arg="level", value=ast.Constant(value=level)),
            ast.keyword(arg="addr_level", value=ast.Constant(value=addr_level)),
        ],
    ))


def make_debug_end() -> ast.Expr:
    return ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="tl", ctx=ast.Load()),
            attr="debug_collect_end",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    ))


def names_in_function(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def unique_temp_name(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    used = names_in_function(fn)
    base = "__flagtree_debug_result"
    name = base
    suffix = 0
    while name in used:
        suffix += 1
        name = f"{base}_{suffix}"
    return name


class ReturnRewriter(ast.NodeTransformer):

    def __init__(self, temp_name: str):
        self.temp_name = temp_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node

    def visit_Return(self, node: ast.Return) -> list[ast.stmt]:
        if node.value is None:
            return [make_debug_end(), node]

        assign = ast.Assign(
            targets=[ast.Name(id=self.temp_name, ctx=ast.Store())],
            value=node.value,
        )
        new_return = ast.Return(value=ast.Name(id=self.temp_name, ctx=ast.Load()))
        return [assign, make_debug_end(), new_return]


def function_body_insert_index(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if (fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        index = 1
    else:
        index = 0

    # Do not start the collect region around launch-index helper calls such as
    # ext.program_id(). Those helpers are @triton.jit functions in FlagGems and
    # appear as tt.call in TTIR; dynamic debugger instrumentation is not yet
    # call-graph aware. Prefer the first real tensor/memory tl.* operation.
    tensor_region_index = first_tensor_region_index(fn, index)
    if tensor_region_index is not None:
        return tensor_region_index

    while index < len(fn.body) and is_launch_index_prologue(fn.body[index]):
        index += 1
    return index


def first_tensor_region_index(fn: ast.FunctionDef | ast.AsyncFunctionDef, start_index: int) -> int | None:
    for index in range(start_index, len(fn.body)):
        if contains_tl_region_entry_call(fn.body[index]):
            return index
    return None


def contains_tl_region_entry_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not isinstance(func, ast.Attribute):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "tl"):
            continue
        if func.attr in {
                "program_id",
                "num_programs",
                "debug_collect_start",
                "debug_collect_end",
        }:
            continue
        return True
    return False


def is_launch_index_prologue(stmt: ast.stmt) -> bool:
    value: ast.AST | None = None
    if isinstance(stmt, ast.Assign):
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        value = stmt.value
    elif isinstance(stmt, ast.AugAssign):
        value = stmt.value

    if value is None:
        return False
    return contains_launch_index_call(value)


def contains_launch_index_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"program_id", "num_programs"}:
            continue
        if isinstance(func.value, ast.Name) and func.value.id in {"ext", "tl"}:
            return True
    return False


def instrument_function(fn: ast.FunctionDef | ast.AsyncFunctionDef, level: int, addr_level: int) -> bool:
    if not any(decorator_is_triton_jit(d) for d in fn.decorator_list):
        return False
    if function_is_pointwise_dynamic(fn):
        return False
    if function_has_debug_collect(fn):
        return False

    temp_name = unique_temp_name(fn)
    rewriter = ReturnRewriter(temp_name)
    insert_index = function_body_insert_index(fn)
    prefix = fn.body[:insert_index]
    suffix = fn.body[insert_index:]

    rewritten_suffix: list[ast.stmt] = []
    for stmt in suffix:
        rewritten = rewriter.visit(stmt)
        if isinstance(rewritten, list):
            rewritten_suffix.extend(rewritten)
        else:
            rewritten_suffix.append(rewritten)  # type: ignore[arg-type]

    fn.body = prefix + [make_debug_start(level, addr_level)] + rewritten_suffix
    fn.body.append(make_debug_end())
    return True


def instrument_file(
    path: Path,
    level: int,
    addr_level: int,
    classifications: list[dict[str, Any]],
) -> tuple[int, int, int, int, int, int, str]:
    source = read_text(path)
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return 0, 0, 0, 0, 0, 1, str(exc)

    changed = 0
    skipped_existing_debug = 0
    skipped_jit_helper = 0
    pointwise_skipped = 0
    launch_targets = launched_local_jit_functions(module)
    local_path = str(path)
    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(decorator_is_triton_jit(d) for d in node.decorator_list):
            continue
        record = {
            "path": local_path,
            "function": node.name,
            "classification": "",
            "instrumented": False,
        }
        if function_is_pointwise_dynamic(node):
            pointwise_skipped += 1
            record["classification"] = "pointwise_dynamic_helper"
            classifications.append(record)
            continue
        if function_has_debug_collect(node):
            skipped_existing_debug += 1
            record["classification"] = "skipped_existing_debug"
            classifications.append(record)
            continue
        if node.name not in launch_targets:
            skipped_jit_helper += 1
            record["classification"] = "jit_helper"
            classifications.append(record)
            continue
        if instrument_function(node, level, addr_level):
            changed += 1
            record["classification"] = "launch_kernel"
            record["instrumented"] = True
            classifications.append(record)

    if changed > 0 and not is_tl_import_present(module):
        insert_tl_import(module)

    if changed > 0:
        ast.fix_missing_locations(module)
        write_text(path, ast.unparse(module) + "\n")
    return (
        changed,
        len(launch_targets),
        skipped_existing_debug,
        skipped_jit_helper,
        pointwise_skipped,
        0,
        "",
    )


def patch_pointwise_dynamic_codegen(root: Path, level: int, addr_level: int) -> bool:
    """Patch copied FlagGems pointwise generator to emit collect markers.

    PointwiseDynamic scalar helpers are @triton.jit functions called from a
    generated wrapper kernel. Inserting collect markers into the scalar helper
    breaks tt.call operand matching on Ascend, so the generated wrapper kernel
    gets the collect region instead.
    """
    path = root / "src" / "flag_gems" / "utils" / "pointwise_dynamic.py"
    if not path.exists():
        return False
    text = read_text(path)
    if "tl.debug_collect_start(level=" in text:
        return False

    start_line = (f'code.writeline("tl.debug_collect_start(level={level}, '
                  f'addr_level={addr_level})")')
    end_line = 'code.writeline("tl.debug_collect_end()")'
    replacements = [
        (
            '        code.writeline("# loads")\n'
            '        for i in range(schema.num_input_tensors()):\n',
            f'        {start_line}\n'
            '        code.writeline("# loads")\n'
            '        for i in range(schema.num_input_tensors()):\n',
            1,
        ),
        (
            '        code.newline()\n'
            '        return code\n'
            '\n'
            '    # nd tile 1d grid kernel with block pointer\n',
            f'        {end_line}\n'
            '        code.newline()\n'
            '        return code\n'
            '\n'
            '    # nd tile 1d grid kernel with block pointer\n',
            1,
        ),
        (
            '            code.writeline("pid = ext.program_id(0)")\n'
            '            self.gen_num_tiles(code)\n',
            '            code.writeline("pid = ext.program_id(0)")\n'
            f'            {start_line}\n'
            '            self.gen_num_tiles(code)\n',
            2,
        ),
        (
            '                self.gen_body_gsl_with_bptr(code)\n'
            '        code.newline()\n',
            '                self.gen_body_gsl_with_bptr(code)\n'
            f'            {end_line}\n'
            '        code.newline()\n',
            1,
        ),
        (
            '                self.gen_body_gsl_without_bptr(code)\n'
            '        code.newline()\n',
            '                self.gen_body_gsl_without_bptr(code)\n'
            f'            {end_line}\n'
            '        code.newline()\n',
            1,
        ),
        (
            '            code.writeline("pid = ext.program_id(0)")\n'
            '            # code.writeline("num_ctas = te.num_programs(0)")\n',
            '            code.writeline("pid = ext.program_id(0)")\n'
            f'            {start_line}\n'
            '            # code.writeline("num_ctas = te.num_programs(0)")\n',
            1,
        ),
        (
            '                self.gen_body_gsl_1d_tile(code)\n'
            '        code.newline()\n',
            '                self.gen_body_gsl_1d_tile(code)\n'
            f'            {end_line}\n'
            '        code.newline()\n',
            1,
        ),
    ]

    patched = text
    applied = 0
    expected = 0
    for old, new, count in replacements:
        expected += count
        found = patched.count(old)
        if found < count:
            continue
        patched = patched.replace(old, new, count)
        applied += count

    if applied != expected:
        raise RuntimeError("failed to patch pointwise_dynamic.py: "
                           f"applied {applied}/{expected} replacements")
    write_text(path, patched)
    return True


def instrument_flaggems_tree(
    root: Path,
    level: int,
    addr_level: int,
    warnings_path: Path,
    classifications_path: Path,
    instrument_pointwise_generated: bool,
    normalize_ext_launch_ids: bool,
) -> InstrumentationStats:
    stats = InstrumentationStats()
    if instrument_pointwise_generated:
        stats.pointwise_dynamic_codegen_patched = patch_pointwise_dynamic_codegen(root, level, addr_level)
    warnings: list[dict[str, str]] = []
    classifications: list[dict[str, Any]] = []
    scan_roots = [root / "src" / "flag_gems", root / "triton_src"]
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*.py"):
            stats.files_scanned += 1
            if normalize_ext_launch_ids:
                normalized, parse_error, warning = normalize_ext_launch_ids_in_file(path)
                if parse_error:
                    stats.parse_errors += 1
                    warnings.append({"path": str(path), "error": warning})
                    continue
                if normalized:
                    stats.files_normalized += 1
                    stats.ext_launch_id_calls_normalized += normalized

            (
                changed,
                launch_count,
                skipped_existing_debug,
                skipped_jit_helper,
                pointwise_skipped,
                parse_error,
                warning,
            ) = instrument_file(
                path,
                level,
                addr_level,
                classifications,
            )
            stats.launch_jit_functions_detected += launch_count
            stats.functions_skipped_existing_debug += skipped_existing_debug
            stats.functions_skipped_jit_helper += skipped_jit_helper
            stats.pointwise_dynamic_helpers_skipped += pointwise_skipped
            if parse_error:
                stats.parse_errors += 1
                warnings.append({"path": str(path), "error": warning})
                continue
            if changed > 0:
                stats.files_changed += 1
                stats.functions_instrumented += changed
            else:
                pass

    write_text(warnings_path, json.dumps(warnings, indent=2, sort_keys=True))
    write_text(
        classifications_path,
        json.dumps(classifications, indent=2, sort_keys=True),
    )
    return stats


def create_bootstrap(bootstrap_dir: Path) -> Path:
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = bootstrap_dir / "sitecustomize.py"
    write_text(
        sitecustomize,
        """
import os
import platform
import sys

platform.python_implementation = lambda: "CPython"
platform.python_version = lambda: "3.11.15"
platform.python_version_tuple = lambda: ("3", "11", "15")

try:
    import triton
    from triton.runtime import debugger

    output_dir = os.environ.get("FLAGTREE_DEBUGGER_BATCH_OUTPUT_DIR")
    if output_dir:
        debugger.configure(
            output_dir=output_dir,
            record_capacity=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_RECORD_CAPACITY", "4096")),
            export_raw_records=os.environ.get("FLAGTREE_DEBUGGER_BATCH_EXPORT_RAW", "0") == "1",
        )
        triton.enable_debug(
            level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_LEVEL", "1")),
            addr_level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_ADDR_LEVEL", "1")),
        )
except Exception as exc:
    print(f"[FlagTreeDebuggerBatch] debugger bootstrap failed: {exc}", file=sys.stderr)
    if os.environ.get("FLAGTREE_DEBUGGER_BATCH_BOOTSTRAP_STRICT", "0") == "1":
        raise
""".lstrip(),
    )
    return sitecustomize


def shell_command(cwd: Path, argv: list[str]) -> str:
    prefix = ""
    if ASCEND_SET_ENV.exists():
        prefix = f"source {shlex.quote(str(ASCEND_SET_ENV))} >/dev/null 2>&1; "
    return f"{prefix}cd {shlex.quote(str(cwd))}; exec {shlex.join(argv)}"


def build_env(
    base_env: dict[str, str],
    worktree: Path,
    bootstrap_dir: Path,
    debug_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    env = dict(base_env)
    old_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [
        str(bootstrap_dir),
        str(worktree / "src"),
        str(worktree),
    ]
    if old_pythonpath:
        pythonpath_parts.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("FLAGTREE_BACKEND", "ascend")
    runtime_home = env.get("HOME") or str(Path.home())
    env["HOME"] = runtime_home
    env.setdefault("TRITON_HOME", runtime_home)
    env["FLAGTREE_DEBUGGER_BATCH_OUTPUT_DIR"] = str(debug_dir)
    env["FLAGTREE_DEBUGGER_BATCH_LEVEL"] = str(args.level)
    env["FLAGTREE_DEBUGGER_BATCH_ADDR_LEVEL"] = str(args.addr_level)
    env["FLAGTREE_DEBUGGER_BATCH_RECORD_CAPACITY"] = str(args.record_capacity)
    env["FLAGTREE_DEBUGGER_BATCH_EXPORT_RAW"] = "1" if args.export_raw_records else "0"
    shared_cache_dir = getattr(args, "shared_cache_dir", None)
    if shared_cache_dir:
        cache_root = Path(shared_cache_dir)
        env["TRITON_CACHE_DIR"] = str(cache_root / "triton_cache")
        env["FLAGGEMS_CACHE_DIR"] = str(cache_root / "flaggems_cache")
    else:
        env["TRITON_CACHE_DIR"] = str(debug_dir.parent / "triton_cache")
        env["FLAGGEMS_CACHE_DIR"] = str(debug_dir.parent / "flaggems_cache")
    return env


def first_error_from_logs(stdout_log: Path, stderr_log: Path) -> str:
    patterns = (
        "Traceback",
        "RuntimeError",
        "CompilationError",
        "AssertionError",
        "NPU function error",
        "vector core exception",
        "acl",
        "FAILED",
        "ERROR",
    )
    combined = ""
    for path in (stderr_log, stdout_log):
        if path.exists():
            text = read_text(path)
            combined += text + "\n"
            for line in text.splitlines():
                if any(pattern in line for pattern in patterns):
                    return line.strip()[:500]
    for line in combined.splitlines():
        if line.strip():
            return line.strip()[:500]
    return ""


def classify_status(
    exit_code: int | None,
    timed_out: bool,
    debug_txt_count: int,
    debug_json_count: int,
    stdout_log: Path,
    stderr_log: Path,
) -> str:
    if timed_out:
        return "timeout"
    text = ""
    for path in (stderr_log, stdout_log):
        if path.exists():
            text += read_text(path) + "\n"

    if exit_code == 5 or "no tests ran" in text.lower():
        return "no_test_found"
    if exit_code == 0:
        return ("passed" if debug_txt_count > 0 and debug_json_count > 0 else "missing_debug_report")
    lowered = text.lower()
    if "compilationerror" in lowered or "compile" in lowered and "error" in lowered:
        return "compile_error"
    if ("npu function error" in lowered or "vector core exception" in lowered or "aclrt" in lowered):
        return "device_error"
    if "runtimeerror" in lowered or "traceback" in lowered or "importerror" in lowered:
        return "runtime_error"
    return "failed"


def run_phase_for_op(
    op: str,
    phase: str,
    worktree: Path,
    run_dir: Path,
    bootstrap_dir: Path,
    no_cpu: set[str],
    args: argparse.Namespace,
) -> OpStatus:
    op_dir = run_dir / op
    phase_dir = op_dir / phase
    debug_dir = phase_dir / "debug_reports"
    phase_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = phase_dir / f"{phase}_stdout.log"
    stderr_log = phase_dir / f"{phase}_stderr.log"
    result_name = f"{phase}_{op}.json"
    result_src = (worktree / ("tests" if phase == "accuracy" else "benchmark")) / result_name
    result_dst = phase_dir / f"{phase}_result.json"
    if result_src.exists():
        result_src.unlink()

    if phase == "accuracy":
        cwd = worktree / "tests"
        argv = [
            str(args.python),
            "-m",
            "pytest",
            "-m",
            op,
            "--record",
            "json",
            "--output",
            result_name,
            "-vs",
        ]
        if args.quick:
            argv.append("--quick")
        if op not in no_cpu:
            argv.extend(["--ref", "cpu"])
    else:
        cwd = worktree / "benchmark"
        argv = [
            str(args.python),
            "-m",
            "pytest",
            "-m",
            op,
            "--level",
            "core",
            "--record",
            "json",
            "--output",
            result_name,
        ]

    command = shell_command(cwd, argv)
    env = build_env(os.environ, worktree, bootstrap_dir, debug_dir, args)
    start = time.time()
    timed_out = False
    exit_code: int | None = None
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        try:
            exit_code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            exit_code = None

    duration = time.time() - start
    result_json = None
    if result_src.exists():
        shutil.copy2(result_src, result_dst)
        result_json = str(result_dst)

    debug_txt_count = len(list(debug_dir.glob("*.txt")))
    debug_json_count = len(list(debug_dir.glob("*.json")))
    status = classify_status(
        exit_code,
        timed_out,
        debug_txt_count,
        debug_json_count,
        stdout_log,
        stderr_log,
    )
    first_error = first_error_from_logs(stdout_log, stderr_log)
    if status == "missing_debug_report":
        first_error = ("test exited successfully but debugger report is missing or "
                       f"incomplete: txt={debug_txt_count}, json={debug_json_count}, "
                       f"dir={debug_dir}")
    op_status = OpStatus(
        op=op,
        phase=phase,
        status=status,
        exit_code=exit_code,
        duration_sec=duration,
        command=command,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        result_json=result_json,
        debug_report_dir=str(debug_dir),
        debug_txt_count=debug_txt_count,
        debug_json_count=debug_json_count,
        first_error=first_error,
    )
    write_text(phase_dir / "status.json", json.dumps(asdict(op_status), indent=2))
    return op_status


def write_unsupported_pointwise_status(
    op: str,
    phase: str,
    run_dir: Path,
) -> OpStatus:
    phase_dir = run_dir / op / phase
    debug_dir = phase_dir / "debug_reports"
    phase_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    reason = ("skipped: current Ascend debugger mode does not support "
              "FlagGems pointwise_dynamic generated kernels with tt.call")
    op_status = OpStatus(
        op=op,
        phase=phase,
        status="unsupported_pointwise_dynamic",
        exit_code=None,
        duration_sec=0.0,
        command="",
        stdout_log=str(phase_dir / f"{phase}_stdout.log"),
        stderr_log=str(phase_dir / f"{phase}_stderr.log"),
        result_json=None,
        debug_report_dir=str(debug_dir),
        debug_txt_count=0,
        debug_json_count=0,
        first_error=reason,
    )
    write_text(phase_dir / f"{phase}_stdout.log", reason + "\n")
    write_text(phase_dir / f"{phase}_stderr.log", "")
    write_text(phase_dir / "status.json", json.dumps(asdict(op_status), indent=2))
    return op_status


def write_summary(run_dir: Path, statuses: list[OpStatus]) -> None:
    rows = [asdict(status) for status in statuses]
    write_text(run_dir / "summary.json", json.dumps(rows, indent=2, sort_keys=True))
    if not rows:
        write_text(run_dir / "summary.csv", "")
        return
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    worktree: Path,
    ops: list[str],
    stats: InstrumentationStats,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "flaggems_root": str(args.flaggems_root),
        "worktree": str(worktree),
        "ops": ops,
        "level": args.level,
        "addr_level": args.addr_level,
        "record_capacity": args.record_capacity,
        "quick": args.quick,
        "include_benchmark": args.include_benchmark,
        "instrument_pointwise_generated": args.instrument_pointwise_generated,
        "skip_pointwise_dynamic": args.skip_pointwise_dynamic,
        "normalize_ext_launch_ids": args.normalize_ext_launch_ids,
        "instrumentation": asdict(stats),
    }
    write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run FlagGems ops with FlagTree debugger instrumentation.")
    parser.add_argument("--flaggems-root", type=Path, default=DEFAULT_FLAGGEMS_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--python", type=Path,
                        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--ops", help="comma-separated op ids")
    parser.add_argument("--op-list-file")
    parser.add_argument("--stages", default="stable")
    parser.add_argument("--start")
    parser.add_argument("--max-ops", type=int)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--addr-level", type=int, default=1)
    parser.add_argument("--record-capacity", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--include-benchmark", action="store_true")
    add_bool_argument(
        parser,
        "skip-pointwise-dynamic",
        default=False,
        help_text=("Skip FlagGems pointwise_dynamic ops instead of patching the copied "
                   "pointwise code generator. Off by default so generated pointwise "
                   "kernels are covered by the debugger sample sweep."),
    )
    parser.add_argument(
        "--normalize-ext-launch-ids",
        dest="normalize_ext_launch_ids",
        action="store_true",
        help=("In the copied FlagGems tree, rewrite ext.program_id/"
              "ext.num_programs helper calls to native tl.program_id/"
              "tl.num_programs casts. Off by default so the copied source keeps "
              "FlagGems launch helper calls unchanged."),
    )
    parser.add_argument("--no-normalize-ext-launch-ids", dest="normalize_ext_launch_ids", action="store_false")
    parser.set_defaults(normalize_ext_launch_ids=False)
    add_bool_argument(
        parser,
        "instrument-pointwise-generated",
        default=True,
        help_text=("Patch the copied FlagGems pointwise_dynamic code generator so "
                   "generated wrapper kernels contain tl.debug_collect_start/end."),
    )
    parser.add_argument("--export-raw-records", action="store_true")
    add_bool_argument(parser, "quick", default=True, help_text="Pass --quick to FlagGems tests.")
    parser.add_argument("--dry-run", action="store_true")
    add_bool_argument(parser, "keep-worktree", default=True, help_text="Keep copied worktree.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    scripts_dir = workspace_root / "scripts"
    runs_root = workspace_root / "runs"
    worktrees_root = workspace_root / "worktrees"
    bootstrap_dir = workspace_root / "bootstrap"
    for directory in (scripts_dir, runs_root, worktrees_root, bootstrap_dir):
        directory.mkdir(parents=True, exist_ok=True)

    flaggems_root = args.flaggems_root.resolve()
    if not flaggems_root.exists():
        raise FileNotFoundError(f"FlagGems root not found: {flaggems_root}")

    inventory = load_operator_inventory(flaggems_root)
    inventory_map = inventory_by_id(inventory)
    ops = select_ops(args, inventory)
    no_cpu = no_cpu_ops(inventory)
    stamp = now_stamp()
    run_dir = runs_root / stamp
    worktree = worktrees_root / f"FlagGems_instrumented_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    create_bootstrap(bootstrap_dir)

    print(f"[INFO] FlagGems root: {flaggems_root}")
    print(f"[INFO] workspace: {workspace_root}")
    print(f"[INFO] run dir: {run_dir}")
    print(f"[INFO] selected ops: {len(ops)}")

    if args.dry_run:
        stats = InstrumentationStats()
        write_manifest(run_dir, args, worktree, ops, stats)
        write_summary(run_dir, [])
        print("[INFO] dry run complete")
        return 0

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
    write_manifest(run_dir, args, worktree, ops, stats)
    print("[INFO] instrumentation: "
          f"{stats.functions_instrumented} functions in {stats.files_changed} files")

    statuses: list[OpStatus] = []
    phases = ["accuracy"] + (["benchmark"] if args.include_benchmark else [])
    for index, op in enumerate(ops, start=1):
        print(f"[INFO] [{index}/{len(ops)}] running {op}")
        if args.skip_pointwise_dynamic and op_uses_pointwise_dynamic(worktree, op, inventory_map):
            for phase in phases:
                status = write_unsupported_pointwise_status(op, phase, run_dir)
                statuses.append(status)
                write_summary(run_dir, statuses)
                print(f"[INFO] {op}/{phase}: {status.status} "
                      f"exit={status.exit_code} reports={status.debug_txt_count}")
            continue
        for phase in phases:
            status = run_phase_for_op(op, phase, worktree, run_dir, bootstrap_dir, no_cpu, args)
            statuses.append(status)
            write_summary(run_dir, statuses)
            print(f"[INFO] {op}/{phase}: {status.status} "
                  f"exit={status.exit_code} reports={status.debug_txt_count}")

    write_summary(run_dir, statuses)
    failing = [
        status for status in statuses if status.status not in {
            "passed",
            "no_test_found",
            "unsupported_pointwise_dynamic",
        }
    ]
    if failing:
        print(f"[WARN] {len(failing)} phase(s) did not pass. See {run_dir / 'summary.json'}")
        return 1
    print(f"[INFO] complete. See {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
