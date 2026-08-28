#!/usr/bin/env python3
"""
CI benchmark / accuracy runner — operator-level parallel dispatch across cards.
Migrated from scripts/ci/; lives alongside gen_bench_summary.py, compare_benchmark_logs.py,
and test_flag_gems_ci_benchmark.py in the benchmark/ directory.

Design:

  ┌─────────────────────────────────────────────────────────────┐
  │  run_benchmark.py                                           │
  │                                                             │
  │  parse_args() → resolve_devices() → resolve_operators()     │
  │       → run_operator_queue()  (worker threads per card)     │
  │       → {bench,acc}_summary.log                             │
  └─────────────────────────────────────────────────────────────┘

  Each operator is a separate pytest invocation dispatched via a work queue.
  Fast ops finish early, slow ops don't block others.

  Two modes:
    benchmark (default): performance benchmark, reads all_perf_tasks from JSON,
                         runs from flaggems/benchmark/ with --warmup/--iter/--metrics.
    accuracy:            accuracy test, reads all_tasks from JSON, runs from
                         flaggems/tests/ with --ref cpu [--mode quick].
"""

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging

logger = logging.getLogger("benchmark")

# Patterns that indicate a card has died (unrecoverable device error).
# Both patterns appear in pytest stdout/stderr merged into the per-op log.
DEAD_CARD_PATTERNS = [
    r"HPGR: device error",
    r"TXDA error: \(txGetDevice",
]

# Shared state for dead-card tracking across worker threads.
_dead_cards: set = set()          # card indices that have died
_dead_cards_lock = threading.Lock()
_dead_card_records: list = []     # (card, op_name, log_file) for final summary

# Global shutdown signal — set by SIGINT/SIGTERM handler,
# checked by workers and queue loop to stop dispatching new tasks
# while letting already-running pytest processes finish.
_global_shutdown = threading.Event()


def _signal_handler(signum, _frame):
    logger.info(f"\n*** Signal {signal.Signals(signum).name} received, "
                f"stopping dispatch (running tasks will finish)... ***\n")
    _global_shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
logger.setLevel(logging.DEBUG)

# Console handler -- INFO and above to stdout
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_console)

# Stderr handler -- ERROR and above to stderr (WARNING already goes to stdout)
_err = logging.StreamHandler(sys.stderr)
_err.setLevel(logging.ERROR)
_err.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_err)

# Prevent propagation to root logger (avoids duplicate WARNING+ output)
logger.propagate = False


SCRIPT_DIR = Path(__file__).resolve().parent
CI_DIR = SCRIPT_DIR.parent  # benchmark/ -> ci/


def _find_workspace_root(start_dir):
    """Walk up from *start_dir* until both tx8_deps/ and flaggems/ are found.

    Returns the first ancestor directory that contains both markers.
    Raises RuntimeError if the workspace root cannot be located.
    """
    current = start_dir.resolve()
    root = current.root
    while current != root:
        if (current / "tx8_deps").is_dir() and (current / "flaggems").is_dir():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not auto-detect workspace root. "
        "Expected both 'tx8_deps/' and 'flaggems/' directories at the "
        "workspace level. Ensure the script is running from within a "
        "triton workspace tree."
    )


def _find_triton_dir(workspace, script_dir):
    """Walk up from *script_dir* to find the immediate child of *workspace*
    whose name is 'triton' or 'triton-tsingmicro-backend'.

    Returns the triton project directory Path.
    Raises RuntimeError if neither is found.
    """
    current = script_dir.resolve()
    while current != workspace:
        if current.parent == workspace and current.name in (
            "triton",
            "triton-tsingmicro-backend",
        ):
            return current
        current = current.parent
    raise RuntimeError(
        f"Could not auto-detect triton project directory under "
        f"workspace {workspace}. Expected 'triton/' or "
        f"'triton-tsingmicro-backend/' as an immediate child of the "
        f"workspace."
    )


WORKSPACE = _find_workspace_root(SCRIPT_DIR)
TRITON_DIR = _find_triton_dir(WORKSPACE, SCRIPT_DIR)

FLAGGEMS_ROOT = WORKSPACE / "flaggems"
OPS_JSON = FLAGGEMS_ROOT / "tests/flag_gems_ci_ops.json"
BENCH_DIR = FLAGGEMS_ROOT / "benchmark"
TESTS_DIR = FLAGGEMS_ROOT / "tests"
CI_COMMON = CI_DIR / "ci_common.sh"


def apply_flaggems_path(path: str):
    """Update global flaggems paths to use a custom root directory."""
    global FLAGGEMS_ROOT, OPS_JSON, BENCH_DIR, TESTS_DIR
    FLAGGEMS_ROOT = Path(path)
    OPS_JSON = FLAGGEMS_ROOT / "tests/flag_gems_ci_ops.json"
    BENCH_DIR = FLAGGEMS_ROOT / "benchmark"
    TESTS_DIR = FLAGGEMS_ROOT / "tests"
    logger.info(f"Flaggems root: {FLAGGEMS_ROOT}")


# ===========================================================================
# 1. Device / card utilities
# ===========================================================================

def expand_range(s: str) -> List[int]:
    """Parse comma-separated device list with ~ or - range support.

    Examples:
        "0,1,2"      -> [0, 1, 2]
        "8~15"       -> [8, 9, ..., 15]
        "8-15"       -> [8, 9, ..., 15]
        "0,8~12,15"  -> [0, 8, 9, 10, 11, 12, 15]
    """
    result: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "~" in part:
            start_str, end_str = part.split("~", 1)
            result.extend(range(int(start_str), int(end_str) + 1))
        elif "-" in part:
            start_str, end_str = part.split("-", 1)
            result.extend(range(int(start_str), int(end_str) + 1))
        else:
            result.append(int(part))
    return result


def check_card_status(card: int) -> str:
    """Run tsm_smi -i <card>, return 'free', 'busy', or 'missing'."""
    try:
        output = subprocess.run(
            ["tsm_smi", "-i", str(card)],
            capture_output=True, text=True, timeout=30
        )
        if output.returncode != 0:
            if "Invalid device index" in output.stderr or "Invalid device index" in output.stdout:
                return "missing"
            return "missing"
        stdout = output.stdout
        if "No running processes found" in stdout:
            return "free"
        return "busy"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "missing"


def resolve_devices(args: argparse.Namespace) -> List[int]:
    """Resolve --devices / --cards / default into a list of free card indices.

    --devices and --cards are mutually exclusive (sys.exit on error).
    """
    explicit = args.devices
    num_cards = args.cards
    skip_raw = args.skip

    if explicit is not None and num_cards is not None:
        logger.error("ERROR: --devices and --cards are mutually exclusive. Use one or the other.")
        sys.exit(1)

    if explicit is not None:
        # --devices mode
        devices = expand_range(explicit)
        missing = [c for c in devices if check_card_status(c) == "missing"]
        busy = [c for c in devices if check_card_status(c) == "busy"]

        # Multi-card (>3): be lenient — warn about unavailable cards and
        # continue with the free ones. Some machines always have stuck cards.
        if len(devices) > 3:
            for c in missing:
                logger.warning(f"  skip card {c} (not exist)")
            for c in busy:
                logger.warning(f"  skip card {c} (busy)")
            free = [c for c in devices if c not in missing and c not in busy]
            if not free:
                logger.error("ERROR: no free cards available "
                             f"(missing={missing}, busy={busy})")
                sys.exit(1)
            logger.info(f"Free cards ({len(free)}/{len(devices)}): {free}")
            return free

        # Few cards (<=3): strict — fail fast on any unavailable card
        if missing:
            logger.error(f"ERROR: cards not exist: {missing}")
            sys.exit(1)
        if busy:
            logger.error(f"ERROR: cards busy: {busy}")
            sys.exit(1)
        logger.info(f"All specified cards free: {devices}")
        return devices

    if num_cards is not None:
        # --cards N --skip IDs: scan 0..N-1, remove skipped, filter to free
        all_cards = list(range(num_cards))
        if skip_raw:
            skip_set = set(expand_range(skip_raw))
        else:
            skip_set = set()
        candidates = [c for c in all_cards if c not in skip_set]
        free_cards = []
        skipped_busy = []
        skipped_missing = []
        for c in candidates:
            s = check_card_status(c)
            if s == "free":
                free_cards.append(c)
            elif s == "busy":
                skipped_busy.append(c)
            else:
                skipped_missing.append(c)
        for c in skipped_missing:
            logger.info(f"  skip card {c} (not exist)")
        for c in skipped_busy:
            logger.info(f"  skip card {c} (busy)")
        if not free_cards:
            logger.error("ERROR: no free cards available")
            sys.exit(1)
        logger.info(f"Free cards ({len(free_cards)}): {free_cards}")
        return free_cards

    # Default: single card 0
    status = check_card_status(0)
    if status == "missing":
        logger.error("ERROR: card 0 not exist")
        sys.exit(1)
    if status == "busy":
        logger.error("ERROR: card 0 is busy")
        sys.exit(1)
    return [0]


# ===========================================================================
# 2. CI config parsing
# ===========================================================================

def _read_ci_config() -> Dict[str, str]:
    """Parse ci_common.sh to extract config variables."""
    config: Dict[str, str] = {}
    if not CI_COMMON.exists():
        logger.warning(f"WARNING: {CI_COMMON} not found, using empty config")
        return config

    content = CI_COMMON.read_text()
    var_names = [
        "precision_mode", "tx8_depends_name", "torch_txda_name",
        "txops_name", "txda_skip_ops", "txda_fallback_cpu_ops",
    ]
    for var in var_names:
        m = re.search(rf'^(?:export\s+)?{var}=(.+)$', content, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            val = val.strip('"').strip("'")
            config[var] = val
    return config


# ===========================================================================
# 3. Environment setup
# ===========================================================================

def setup_base_env():
    """Set base environment variables needed for Triton + Flaggems."""
    config = _read_ci_config()

    tx8_deps_root = str(WORKSPACE / "tx8_deps")
    llvm_syspath = str(WORKSPACE / "llvm-a66376b0-ubuntu-x64")
    llvm_binary_dir = str(Path(llvm_syspath) / "bin")

    os.environ.setdefault("TX8_DEPS_ROOT", tx8_deps_root)
    os.environ.setdefault("LLVM_SYSPATH", llvm_syspath)
    os.environ.setdefault("LLVM_BINARY_DIR", llvm_binary_dir)

    # PYTHONPATH: prepend mlir_core
    mlir_path = str(Path(llvm_syspath) / "python_packages" / "mlir_core")
    existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{mlir_path}:{existing_pp}" if existing_pp else mlir_path

    # LD_LIBRARY_PATH: prepend tx8_deps/lib
    tx8_lib = str(Path(tx8_deps_root) / "rcs1fw-rtt" / "lib")
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{tx8_lib}:{existing_ld}" if existing_ld else tx8_lib

    # Backend selection
    os.environ.setdefault("FLAG_GEMS_CUSTOM_OPS", "1")
    os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"
    os.environ["TX_LAUNCH_LOG_LEVEL"] = "info"

    # Skip / fallback ops
    txda_skip = config.get("txda_skip_ops", "")
    txda_fallback = config.get("txda_fallback_cpu_ops", "")
    if txda_skip:
        os.environ["TXDA_SKIP_OPS"] = txda_skip
    if txda_fallback:
        os.environ["TXDA_FALLBACK_CPU_OPS"] = txda_fallback

    # Test config
    os.environ["JSON_FILE_PATH"] = str(FLAGGEMS_ROOT / "tests")

    logger.info("=== base env ===")
    logger.info(f"TX8_DEPS_ROOT={os.environ.get('TX8_DEPS_ROOT', '')}")
    logger.info(f"LLVM_SYSPATH={os.environ.get('LLVM_SYSPATH', '')}")
    logger.info(f"LLVM_BINARY_DIR={os.environ.get('LLVM_BINARY_DIR', '')}")
    logger.info(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
    logger.info(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
    logger.info(f"FLAG_GEMS_CUSTOM_OPS={os.environ.get('FLAG_GEMS_CUSTOM_OPS', '')}")
    logger.info(f"TRITON_ALLOW_NON_CONSTEXPR_GLOBALS={os.environ.get('TRITON_ALLOW_NON_CONSTEXPR_GLOBALS', '')}")
    logger.info(f"TX_LAUNCH_LOG_LEVEL={os.environ.get('TX_LAUNCH_LOG_LEVEL', '')}")
    logger.info(f"TXDA_SKIP_OPS={os.environ.get('TXDA_SKIP_OPS', '')}")
    logger.info(f"TXDA_FALLBACK_CPU_OPS={os.environ.get('TXDA_FALLBACK_CPU_OPS', '')}")
    logger.info(f"JSON_FILE_PATH={os.environ.get('JSON_FILE_PATH', '')}")


def _setup_common_precision():
    """Set common precision/quick-mode env vars for both precision and profiler modes."""
    config = _read_ci_config()
    precision_mode = config.get("precision_mode", "2")
    os.environ["PRECISION_MODE"] = precision_mode
    os.environ["TRITON_QUICK_MODE"] = "1"
    return precision_mode


def setup_precision_env():
    """Set precision-related environment variables."""
    precision_mode = _setup_common_precision()
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    logger.info("=== precision env ===")
    logger.info(f"PRECISION_MODE={precision_mode}")
    logger.info(f"TRITON_ALWAYS_COMPILE=1")
    logger.info(f"TRITON_QUICK_MODE=1")
    logger.info(f"TRITON_PRINT_AUTOTUNING=1")


def setup_profiler_env():
    """Set profiler-related environment variables."""
    precision_mode = _setup_common_precision()
    profiler_lib = "/usr/local/kuiper/tsm8-profiler/lib"
    os.environ["TSM_PROFILER_EN"] = "1"

    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{profiler_lib}:{existing_ld}" if existing_ld else profiler_lib

    os.environ["LD_PRELOAD"] = (
        f"{profiler_lib}/libtsmprofiler-register.so:"
        f"{profiler_lib}/libtsmprofiler-sdk.so"
    )
    os.environ["ROCP_TOOL_LIBRARIES"] = f"{profiler_lib}/libtsm-api-log-tracing.so"

    logger.info("=== profiler env ===")
    logger.info(f"TSM_PROFILER_EN={os.environ.get('TSM_PROFILER_EN', '')}")
    logger.info(f"LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}")
    logger.info(f"ROCP_TOOL_LIBRARIES={os.environ.get('ROCP_TOOL_LIBRARIES', '')}")
    logger.info(f"PRECISION_MODE={precision_mode}")
    logger.info(f"TRITON_QUICK_MODE=1")


def _run_bash_script(script_body: str, description: str) -> int:
    """Run an inline bash script and return the return code."""
    logger.info(f"\n=== {description} ===\n")
    result = subprocess.run(
        ["bash", "-c", script_body],
        cwd=str(TRITON_DIR),
    )
    return result.returncode


# ===========================================================================
# 4. Install / Build / Venv
# ===========================================================================

def run_install():
    """Source ci_common.sh and call download_deps + install_deps."""
    script = f'''
export SCRIPT_PATH="{CI_DIR}/run_benchmark.sh"
source "{CI_COMMON}"
download_deps
install_deps
'''
    rc = _run_bash_script(script, "download + install deps")
    if rc != 0:
        logger.error("ERROR: install failed")
        sys.exit(rc)

    # Activate venv by prepending to PATH
    venv_bin = TRITON_DIR / ".venv" / "bin"
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        logger.info(f"Activated venv: {venv_bin.parent}")


def run_build():
    """Source ci_common.sh and call build_triton."""
    script = f'''
export SCRIPT_PATH="{CI_DIR}/run_benchmark.sh"
source "{CI_COMMON}"
build_triton
'''
    rc = _run_bash_script(script, "build triton")
    if rc != 0:
        logger.error("ERROR: build failed")
        sys.exit(rc)


def activate_venv():
    """Activate triton venv (for when --install was skipped but venv exists)."""
    venv_bin = TRITON_DIR / ".venv" / "bin"
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        logger.info(f"Activated venv: {venv_bin.parent}")
    else:
        logger.warning(f"WARNING: venv not found at {venv_bin}. Continuing with system python.")


# ===========================================================================
# 5. Operator resolution
# ===========================================================================

def resolve_operators(test_set: str, accuracy: bool = False) -> List[Tuple[str, List[str]]]:
    """Resolve test_set to a list of (op_name, node_ids) tuples.

    All data (groups + test IDs) comes from flag_gems_ci_ops.json.
    Groups are top-level keys containing operator name lists.
    In benchmark mode, test IDs come from the all_perf_tasks key
    (e.g. "test_blas_perf.py::test_blas_benchmark[addmm-addmm-addmm_input_fn]").
    In accuracy mode, test IDs come from the all_tasks key
    (e.g. "test_binary_pointwise_ops.py::test_accuracy_add").

    For "all" test_set, returns an empty list (pytest discover mode).
    """
    if test_set == "all":
        return []

    if not OPS_JSON.exists():
        logger.error(f"ERROR: {OPS_JSON} not found")
        sys.exit(1)

    with open(OPS_JSON) as f:
        data = json.load(f)

    # Resolve operator names from group
    if test_set not in data:
        logger.error(f"ERROR: test_set '{test_set}' not found in {OPS_JSON}")
        available = [k for k in data if not k.startswith("_") and isinstance(data[k], list)]
        logger.error(f"  Available groups: {', '.join(available)}")
        sys.exit(1)

    op_names = [o for o in data[test_set] if not o.startswith("#")]

    tasks_key = "all_tasks" if accuracy else "all_perf_tasks"
    tasks = data.get(tasks_key, {})
    results: List[Tuple[str, List[str]]] = []
    missing_ops: List[str] = []
    for name in op_names:
        if name in tasks:
            results.append((name, tasks[name]))
        else:
            missing_ops.append(name)

    if missing_ops:
        logger.warning(f"WARNING: ops not in {tasks_key}: {missing_ops}")

    total_ids = sum(len(tasks[name]) for name in op_names if name in tasks)
    logger.info(f"  → {len(results)} operators, {total_ids} test IDs from {OPS_JSON.name} ({tasks_key})")

    return results


# ===========================================================================
# 6. Worker / Queue system
# ===========================================================================

class WorkerResult:
    """Result of a single operator run on a card."""
    __slots__ = ("op_name", "card", "rc", "start_time", "end_time", "log_file")

    def __init__(self, op_name: str, card: int, rc: int,
                 start_time: float, end_time: float, log_file: str):
        self.op_name = op_name
        self.card = card
        self.rc = rc
        self.start_time = start_time
        self.end_time = end_time
        self.log_file = log_file


def _check_log_for_dead_card(log_file: Path) -> bool:
    """Return True if the log contains any dead-card pattern."""
    if not log_file.exists():
        return False
    try:
        content = log_file.read_text()
        for pattern in DEAD_CARD_PATTERNS:
            if re.search(pattern, content):
                return True
    except (OSError, IOError):
        pass
    return False


def _rename_dead_card_log(log_file: Path, card: int, op_name: str) -> Path:
    """Rename log to <op_name>_card<N>_dead.log, return new path."""
    new_path = log_file.parent / f"{op_name}_card{card}_dead.log"
    try:
        shutil.move(str(log_file), str(new_path))
    except OSError:
        pass
    return new_path


def worker(card: int, task_queue: queue.Queue, results: List[WorkerResult],
           stop_event: threading.Event, ops_dir: Path, pytest_base_args: List[str],
           test_dir: str, devices: List[int], active: Dict[int, str] = None):
    """Worker thread: pull ops from queue, run pytest, write per-op log.

    Each worker passes TXDA_VISIBLE_DEVICES via a per-subprocess env dict
    to avoid thread-safety issues with os.environ.
    Results are appended to the shared results list (thread-safe in CPython).
    test_dir is the base directory for resolving pytest node paths
    (flaggems/benchmark for perf, flaggems/tests for accuracy).
    active dict maps card -> op_name for progress monitoring.
    """
    if active is None:
        active = {}

    # Build a base env with TXDA_VISIBLE_DEVICES set for this card.
    # We pass this to each subprocess.run() to avoid racing on os.environ.
    proc_env = os.environ.copy()
    proc_env["TXDA_VISIBLE_DEVICES"] = str(card)

    # Re-check card status right before use (TOCTOU safety)
    status = check_card_status(card)
    if status == "missing":
        logger.info(f"  [SKIP] card {card} not available (missing)")
        return
    if status == "busy":
        logger.info(f"  [SKIP] card {card} not available (busy)")
        return

    while not stop_event.is_set() and not _global_shutdown.is_set():
        try:
            op_name, node_ids, attempt, failed_cards = task_queue.get(block=True, timeout=1.0)
        except queue.Empty:
            # No more tasks
            break

        # If this card died while waiting, re-enqueue and exit
        with _dead_cards_lock:
            if card in _dead_cards:
                task_queue.put((op_name, node_ids, attempt, failed_cards))
                task_queue.task_done()
                break

        active[card] = op_name
        rc = -1
        start_time = time.time()
        log_file = ops_dir / f"{op_name}.log"
        try:
            # Build command with absolute node ID paths (test_dir + node_id)
            node_paths = [os.path.join(str(test_dir), nid) for nid in node_ids]
            cmd = [
                "python3", "-m", "pytest",
            ] + node_paths + pytest_base_args

            # Write log header
            with open(log_file, "w") as f:
                f.write(f"=== card {card} op {op_name} ===\n")
                f.write(f"=== command: {' '.join(cmd)} ===\n")
                f.write(f"[{datetime.now().strftime('%m%d %H:%M:%S')}] pytest start\n")

            # Launch pytest (separate file handle in append mode so we can
            # monitor log file size independently for timeout detection)
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, "a"),
                stderr=subprocess.STDOUT,
                env=proc_env,
                start_new_session=True,
            )

            # Monitor log file size for timeout (same mechanism as test_flag_gems_ci.py).
            # If log size is unchanged for _threshold consecutive _interval-second checks,
            # the pytest is considered stuck and gets terminated.
            _interval = 60
            _threshold = 5
            _counter = 0
            _prev_size = log_file.stat().st_size
            while True:
                time.sleep(_interval)
                try:
                    _cur_size = log_file.stat().st_size
                except FileNotFoundError:
                    break
                if _cur_size == _prev_size:
                    _counter += 1
                    if _counter >= _threshold:
                        proc.send_signal(signal.SIGINT)
                        try:
                            proc.wait(timeout=10)
                            _kill_method = "SIGINT"
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                            _kill_method = "SIGKILL (SIGINT timed out after 10s)"
                        logger.warning(
                            f"  [TIMEOUT] card {card} {op_name} "
                            f"(log unchanged {_threshold}x{_interval}s), "
                            f"killed with {_kill_method}"
                        )
                        with open(log_file, "a") as _lf:
                            _lf.write(
                                f"\n[{datetime.now().strftime('%m%d %H:%M:%S')}] "
                                f"[TIMEOUT] log unchanged for {_threshold}x{_interval}s, "
                                f"killed process with {_kill_method}\n"
                            )
                        break
                else:
                    _counter = 0
                    _prev_size = _cur_size
                if proc.poll() is not None:
                    break

            rc = proc.wait()

            # Classify hang type from backtrace (if SIGINT triggered a backtrace dump)
            _hang_type = ""
            if rc != 0:
                try:
                    _log_tail = log_file.read_text()
                    if "itxStreamSynchronize" in _log_tail or "txStreamSynchronize" in _log_tail:
                        _hang_type = "HPGR-STREAM-HANG (device stream never completed)"
                    elif "awaitCompletion" in _log_tail:
                        _hang_type = "HPGR-EVENT-HANG (device event never completed)"
                    elif "<current backtrace>" in _log_tail:
                        _hang_type = "HANG (backtrace captured, see log)"
                except (OSError, IOError):
                    pass
            if _hang_type:
                with open(log_file, "a") as _lf:
                    _lf.write(f"[HANG-TYPE] {_hang_type}\n")

            # Write end marker
            with open(log_file, "a") as f:
                f.write(f"\n[{datetime.now().strftime('%m%d %H:%M:%S')}] pytest end (rc={rc})\n")
        except Exception as exc:
            end_time = time.time()
            # Write error to log file
            try:
                with open(log_file, "a") as f:
                    f.write(f"\n=== WORKER EXCEPTION: {exc} ===\n")
            except OSError:
                pass
            logger.error(f"  [ERROR] card {card} {op_name} worker exception: {exc}")
        finally:
            active.pop(card, None)
            end_time = time.time()

            # Check for dead card or hung card if pytest failed
            if rc != 0 and _check_log_for_dead_card(log_file):
                # Rename log to preserve it
                renamed_log = _rename_dead_card_log(log_file, card, op_name)

                # Mark card as dead
                with _dead_cards_lock:
                    _dead_cards.add(card)
                    _dead_card_records.append((card, op_name, str(renamed_log)))

                logger.warning(
                    f"  [DEAD] card {card} has died on {op_name} "
                    f"(attempt {attempt}/2), log: {renamed_log.name}"
                )

                if attempt < 2:
                    # Retry on another card
                    logger.info(
                        f"  [RETRY] re-enqueuing {op_name} "
                        f"(attempt {attempt+1}/2, failed_cards: {failed_cards + [card]})"
                    )
                    task_queue.put((op_name, node_ids, attempt + 1, failed_cards + [card]))
                else:
                    # Two cards died on this operator — likely operator-induced
                    logger.error(
                        f"  [FATAL] {op_name} killed 2 cards "
                        f"({failed_cards + [card]}), giving up on this operator"
                    )

                # Check if all cards are dead → signal full stop
                with _dead_cards_lock:
                    if len(_dead_cards) >= len(devices):
                        logger.error(
                            f"  [ALL-DEAD] all {len(devices)} cards have died, "
                            f"stopping all workers"
                        )
                        stop_event.set()
                        _global_shutdown.set()

                task_queue.task_done()
                break  # exit worker loop — this card is dead

            else:
                # Normal path: record result (includes timeout/hang as regular failure)
                results.append(WorkerResult(
                    op_name=op_name, card=card, rc=rc,
                    start_time=start_time, end_time=end_time,
                    log_file=str(log_file),
                ))
                elapsed = end_time - start_time
                status = "PASSED" if rc == 0 else "FAILED"
                logger.info(f"  [{status}] card {card} {op_name} ({elapsed:.1f}s)")
                task_queue.task_done()


def monitor_progress(results: List[WorkerResult], stop_event: threading.Event,
                     total_ops: int, active: Dict[int, str], devices: List[int],
                     ops_dir: Path):
    """Every 60 seconds, print progress summary with per-card detail."""
    while not stop_event.wait(60):
        done = len(results)
        passed = sum(1 for r in results if r.rc == 0)
        failed = sum(1 for r in results if r.rc != 0)
        ts = datetime.now().strftime("%m%d %H:%M:%S")
        logger.info(f"  [progress {ts}] {done}/{total_ops}  pass={passed} fail={failed}")

        # Per-card detail — only show cards with activity
        active_cards = sorted(set(
            [r.card for r in results] + [c for c, op in active.items() if op]
        ))
        for card in active_cards:
            card_results = [r for r in results if r.card == card]
            running = active.get(card)
            if running:
                # Show last log line + file mtime to detect stuck ops
                log_file = ops_dir / f"{running}.log"
                last_line = ""
                log_mtime = ""
                if log_file.exists():
                    try:
                        log_mtime = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%m%d %H:%M:%S")
                        with open(log_file) as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()[:90]
                    except (OSError, IOError):
                        pass
                logger.info(f"    card {card}: {running}  log={log_mtime}  {last_line}")
            elif card_results:
                last = card_results[-1]
                status = "PASS" if last.rc == 0 else f"FAIL({last.rc})"
                elapsed = last.end_time - last.start_time
                logger.info(f"    card {card}: {last.op_name} ({status}, {elapsed:.0f}s)")

        # Report dead cards
        with _dead_cards_lock:
            if _dead_cards:
                dead_str = ", ".join(str(c) for c in sorted(_dead_cards))
                logger.warning(f"    [DEAD CARDS] {dead_str}")


def stop_file_watcher(stop_file_path: str, stop_event: threading.Event):
    """Every 2 seconds, check if stop_file exists. If yes, set stop_event."""
    while not stop_event.is_set():
        if os.path.exists(stop_file_path):
            logger.info(f"\n*** Stop file detected: {stop_file_path}. Signaling stop... ***\n")
            stop_event.set()
            break
        time.sleep(2)


def run_operator_queue(operators: List[Tuple[str, List[str]]], devices: List[int],
                       ops_dir: Path, pytest_base_args: List[str],
                       test_dir: Path,
                       stop_file: Optional[str] = None
                       ) -> Tuple[List[WorkerResult], int]:
    """Fill work queue and dispatch via worker threads (one per device).

    Args:
        operators: List of (op_name, node_ids) tuples.
        devices: List of device indices to use.
        ops_dir: Directory for per-op log files.
        pytest_base_args: Base args for pytest (mode-specific flags).
        test_dir: Base directory for resolving pytest node paths
                  (BENCH_DIR for perf, TESTS_DIR for accuracy).
        stop_file: Optional path to a stop-file. If it appears, workers stop.

    Returns:
        (results_list, remaining_count)
    """
    task_queue: queue.Queue = queue.Queue()
    for op_name, node_ids in operators:
        task_queue.put((op_name, node_ids, 1, []))

    results: List[WorkerResult] = []
    stop_event = threading.Event()

    # Shared dict for progress monitor: card -> current op_name
    active: Dict[int, str] = {}

    # Start worker threads
    threads = []
    test_dir_str = str(test_dir)
    for card in devices:
        t = threading.Thread(
            target=worker,
            args=(card, task_queue, results, stop_event, ops_dir, pytest_base_args, test_dir_str, devices, active),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Start progress monitor
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(results, stop_event, len(operators), active, devices, ops_dir),
        daemon=True,
    )
    monitor_thread.start()

    # Start stop-file watcher if configured
    watcher_thread = None
    if stop_file:
        watcher_thread = threading.Thread(
            target=stop_file_watcher,
            args=(stop_file, stop_event),
            daemon=True,
        )
        watcher_thread.start()

    # Wait for queue to drain (poll unfinished_tasks — join() blocks forever on errors)
    while not stop_event.is_set() and not _global_shutdown.is_set():
        time.sleep(2.0)
        if task_queue.unfinished_tasks == 0:
            break
        # If all cards are dead, stop waiting
        with _dead_cards_lock:
            if len(_dead_cards) >= len(devices):
                logger.error("All cards dead — stopping queue wait.")
                stop_event.set()
                break
        # Fallback: if all workers died from exceptions, exit even if unfinished_tasks > 0
        if task_queue.empty() and all(not t.is_alive() for t in threads):
            logger.info("  All workers exited, stopping queue wait.")
            break

    # If global shutdown was requested, also signal the local stop_event
    # so workers check both and stop taking new tasks.
    if _global_shutdown.is_set():
        stop_event.set()

    # If stop_event was set, we have remaining tasks
    # Note: qsize() is approximate; unfinished_tasks accounts for in-flight work
    remaining = task_queue.qsize() + task_queue.unfinished_tasks

    # Signal workers to stop
    stop_event.set()

    # Wait for workers to finish their current task
    for t in threads:
        t.join(timeout=30)

    return results, remaining


# ===========================================================================
# 7. CLI argument parsing
# ===========================================================================

def add_common_args(parser):
    """Add args shared by benchmark / accuracy / ci-baseline / ts-compare."""
    parser.add_argument("-t", "--test", default="all",
                        help="Test set: all / ci_ops / ts_opt_ops / race_ops / custom group")
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device list (supports ranges: 8~15 or 8-15)")
    parser.add_argument("--cards", default=None, type=int,
                        help="N cards mode: scan 0..N-1 for free cards")
    parser.add_argument("--skip", default=None,
                        help="Cards to skip in --cards mode (comma-separated, supports ranges)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Smoke-test: dispatch only first 2 operators then stop")
    parser.add_argument("--flaggems_path", default=None,
                        help="Flaggems root path. Defaults to $WORKSPACE/flaggems.")


def add_perf_args(parser):
    """Add benchmark-performance-specific args (warmup/iter/level/metrics)."""
    parser.add_argument("--level", default="core",
                        help="Benchmark level: core / comprehensive")
    parser.add_argument("-w", "--warmup", type=int, default=10,
                        help="Single shape warmup count (default: 10)")
    parser.add_argument("-r", "--rep", "--iter", dest="rep", type=int, default=20,
                        help="Single shape benchmark repetitions (default: 20)")
    parser.add_argument("--metrics", default="latency",
                        help="Benchmark metrics: latency (default), latency_base, speedup, gbps, tflops")


def add_exec_args(parser):
    """Add install/build/no-run/stop-file args (used by benchmark and accuracy)."""
    parser.add_argument("--install", action="store_true", help="Reinstall dependencies")
    parser.add_argument("--build", action="store_true", help="Rebuild triton")
    parser.add_argument("--no-run", action="store_true", help="Skip execution (install/build only)")
    parser.add_argument("--stop-file", default=None,
                        help="Path to stop sentinel file")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CI benchmark / accuracy runner — operator-level parallel dispatch across cards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  benchmark    Run performance benchmark (default when no subcommand given)
  accuracy     Run accuracy tests (--ref cpu)
  ci-baseline  Run benchmark, optionally compare to a baseline directory
  ts-compare   TS optimization on/off comparison (CUSTOM_OPS=0 vs =1)
  compare      Compare two existing log directories

Common options (benchmark / accuracy / ci-baseline / ts-compare):
  -t, --test TEST       Test set: all | ts_opt_ops | ci_ops | race_ops | <custom group>
                        ts_opt_ops: ~84 ops with custom TS backend
                        race_ops: 129 ops for regression testing
                        ci_ops: ~150 ops defined in flag_gems_ci_ops.json
                        all: run pytest discover on entire test/benchmark dir
  --devices IDS         Explicit card list, comma-separated, supports ranges: 8~15, 8-15, 0,8~12,15
  --cards N             Dynamic mode: scan cards 0..N-1, pick free ones
  --skip IDS            Skip specified cards (used with --cards), supports ranges
  --test-mode           Smoke-test: run only the first 2 operators, then stop
  --flaggems_path PATH  Flaggems root path (default: $WORKSPACE/flaggems)

benchmark options:
  --level LVL           Benchmark level: core (default) | comprehensive
  -w, --warmup N        Single-shape warmup count (default: 10)
  -r, --rep, --iter N   Single-shape benchmark repetitions (default: 20)
  --metrics M1 M2 ...   Benchmark metrics: latency (default) | latency_base | speedup | gbps | tflops
  -k, --op FILTER       pytest -k filter expression (manual override)
  --install             Reinstall dependencies before running
  --build               Rebuild triton before running
  --no-run              Skip execution (install/build only)
  --stop-file PATH      Sentinel file: touch it to gracefully stop after current op

accuracy options:
  --quick               Pass --mode quick to pytest (fewer test cases)
  --install             Reinstall dependencies before running
  --build               Rebuild triton before running
  --no-run              Skip execution (install/build only)
  --stop-file PATH      Sentinel file: touch it to gracefully stop after current op

ci-baseline options:
  --level LVL, -w N, -r N, --metrics M   (same as benchmark)
  --baseline DIR        Baseline directory to compare against
  --threshold N         Degradation threshold in %% (default: 20)
  --install, --build

ts-compare options:
  --level LVL, -w N, -r N, --metrics M   (same as benchmark)
  --threshold N         Degradation threshold in %% (default: 20)
  --install, --build

compare options:
  DIR1 DIR2             Baseline and target log directories
  --threshold N         Degradation threshold in %% (default: 20)
  --install, --build

Examples:
  # Performance benchmark
  python run_benchmark.py benchmark -t ts_opt_ops --devices 0,1
  python run_benchmark.py benchmark -t ts_opt_ops --devices 0~7 --install --build
  python run_benchmark.py benchmark -t ts_opt_ops --devices 3~7 --test-mode
  python run_benchmark.py ci-baseline -t ci_ops --baseline ci_log/old --devices 0~7
  python run_benchmark.py ts-compare --devices 0,1 --threshold 5

  # Accuracy tests
  python run_benchmark.py accuracy -t ci_ops --devices 0,1
  python run_benchmark.py accuracy --quick -t ci_ops --devices 0~7 --install --build
  python run_benchmark.py accuracy -t test_op --devices 0 --test-mode

  # Compare (no subcommand needed, defaults to benchmark)
  python run_benchmark.py -t ts_opt_ops --devices 0,1
  python run_benchmark.py compare ci_log_s/run_A ci_log_s/run_B --threshold 20
""")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # --- benchmark ---
    bench_p = subparsers.add_parser("benchmark",
        help="Run performance benchmark (default subcommand)")
    add_common_args(bench_p)
    add_perf_args(bench_p)
    add_exec_args(bench_p)
    bench_p.add_argument("-k", "--op", dest="op_filter", default="",
                         help="pytest -k filter expression")
    bench_p.set_defaults(accuracy=False, quick=False)

    # --- accuracy ---
    acc_p = subparsers.add_parser("accuracy",
        help="Run accuracy tests (--ref cpu, uses all_tasks)")
    add_common_args(acc_p)
    add_exec_args(acc_p)
    acc_p.add_argument("--quick", action="store_true",
                       help="Pass --mode quick to pytest (fewer test cases)")
    acc_p.set_defaults(accuracy=True, op_filter="")

    # --- ci-baseline ---
    cib_p = subparsers.add_parser("ci-baseline",
        help="Run benchmark, optionally compare to a baseline directory")
    add_common_args(cib_p)
    add_perf_args(cib_p)
    cib_p.add_argument("--baseline", default=None,
                       help="Baseline log directory for comparison")
    cib_p.add_argument("--threshold", type=int, default=20,
                       help="Degradation threshold %% (default: 20)")
    cib_p.add_argument("--install", action="store_true", help="Reinstall dependencies")
    cib_p.add_argument("--build", action="store_true", help="Rebuild triton")
    cib_p.set_defaults(test="ci_ops", accuracy=False, quick=False,
                       op_filter="", no_run=False, stop_file=None)

    # --- ts-compare ---
    tsc_p = subparsers.add_parser("ts-compare",
        help="TS optimization comparison: CUSTOM_OPS=0 vs CUSTOM_OPS=1")
    add_common_args(tsc_p)
    add_perf_args(tsc_p)
    tsc_p.add_argument("--threshold", type=int, default=20,
                       help="Degradation threshold %% (default: 20)")
    tsc_p.add_argument("--install", action="store_true", help="Reinstall dependencies")
    tsc_p.add_argument("--build", action="store_true", help="Rebuild triton")
    tsc_p.set_defaults(test="ts_opt_ops", accuracy=False, quick=False,
                       op_filter="", no_run=False, stop_file=None)

    # --- compare ---
    cmp_p = subparsers.add_parser("compare", help="Compare two existing log directories")
    cmp_p.add_argument("dir1", help="Baseline log directory")
    cmp_p.add_argument("dir2", help="Target log directory")
    cmp_p.add_argument("--threshold", type=int, default=20,
                       help="Degradation threshold %% (default: 20)")
    cmp_p.add_argument("--install", action="store_true", help="Reinstall dependencies")
    cmp_p.add_argument("--build", action="store_true", help="Rebuild triton")

    return parser


# ===========================================================================
# 8. Cleanup
# ===========================================================================

def cleanup():
    """Remove temporary/cache directories older than 3 days."""
    now = time.time()
    max_age = 3 * 24 * 3600  # 3 days in seconds

    dirs_to_check = [
        Path.home() / ".triton",
        Path.home() / ".flaggems",
        TRITON_DIR / "dump",
    ]
    for d in dirs_to_check:
        if d.exists():
            _remove_if_old(d, now, max_age)

    # /tmp patterns (skip triton_log which holds per-run isolated caches)
    for pattern in ["triton_*", "flaggems_*"]:
        for p in Path("/tmp").glob(pattern):
            if "triton_log" in str(p):
                continue
            _remove_if_old(p, now, max_age)

    # Files
    for f in ["result.json", "tsingmicro_launch.log"]:
        p = Path(f)
        if p.exists():
            _remove_if_old(p, now, max_age)


def _remove_if_old(path: Path, now: float, max_age: float):
    """Remove *path* if its mtime is older than *max_age* seconds."""
    try:
        mtime = path.stat().st_mtime
        age = now - mtime
        if age > max_age:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            logger.info(f"  Removed {path} (age: {age/3600:.1f}h)")
        else:
            logger.info(f"  Skip {path} (age: {age/3600:.1f}h, < 3 days)")
    except (OSError, FileNotFoundError):
        pass


# ===========================================================================
# 9. Argument validation
# ===========================================================================

VALID_LEVELS = {"core", "comprehensive"}
VALID_METRICS = {"latency", "latency_base", "speedup", "gbps", "tflops"}


def _abort(errors: List[str]):
    """Print all errors and exit."""
    for e in errors:
        logger.error(f"ERROR: {e}")
    logger.error("Aborting due to invalid arguments. Use --help for usage.")
    sys.exit(1)


def validate_bench_args(args):
    """Validate args for run / ci-baseline / ts-compare subcommands."""
    errors = []
    accuracy = getattr(args, 'accuracy', False)
    if not accuracy:
        if args.warmup < 0:
            errors.append(f"--warmup must be >= 0, got {args.warmup}")
        if args.rep < 1:
            errors.append(f"--rep/--iter must be >= 1, got {args.rep}")
        if args.level not in VALID_LEVELS:
            errors.append(f"--level: invalid value '{args.level}', "
                           f"must be one of {sorted(VALID_LEVELS)}")
        for m in args.metrics.split():
            if m not in VALID_METRICS:
                errors.append(f"--metrics: invalid value '{m}', "
                               f"must be one of {sorted(VALID_METRICS)}")
    if args.flaggems_path and not Path(args.flaggems_path).is_dir():
        errors.append(f"--flaggems_path: directory not found: {args.flaggems_path}")
    if errors:
        _abort(errors)


def validate_compare_args(args):
    """Validate args for compare subcommand."""
    errors = []
    if not Path(args.dir1).is_dir():
        errors.append(f"dir1: directory not found: {args.dir1}")
    if not Path(args.dir2).is_dir():
        errors.append(f"dir2: directory not found: {args.dir2}")
    if args.threshold < 0:
        errors.append(f"--threshold must be >= 0, got {args.threshold}")
    if errors:
        _abort(errors)


# ===========================================================================
# 10. cmd_run — execute a single run (benchmark or accuracy)
# ===========================================================================

def cmd_run(args, log_dir_override: Path = None) -> Optional[Path]:
    """Execute a single run (benchmark or accuracy). Returns log_dir Path on success, None on failure.

    If log_dir_override is given, use it directly (skips auto-creation).
    Used by ts-compare to nest baseline/ and target/ under a parent dir.
    """
    # Some attrs are only defined in the 'run' subparser; provide safe defaults
    op_filter = getattr(args, 'op_filter', '')
    no_run = getattr(args, 'no_run', False)
    stop_file = getattr(args, 'stop_file', None)
    test_mode = getattr(args, 'test_mode', False)
    accuracy = getattr(args, 'accuracy', False)
    quick = getattr(args, 'quick', False)

    validate_bench_args(args)

    devices = resolve_devices(args)

    run_mode = "accuracy" if accuracy else "benchmark"

    # Print config summary
    logger.info("=" * 60)
    logger.info(f" mode         : {run_mode}")
    logger.info(f" install      : {args.install}")
    logger.info(f" build        : {args.build}")
    logger.info(f" no_run       : {no_run}")
    logger.info(f" test_set     : {args.test}")
    if accuracy:
        logger.info(f" quick        : {quick}")
    else:
        logger.info(f" op_filter    : {op_filter}")
        logger.info(f" level        : {args.level}")
        logger.info(f" warmup       : {args.warmup}")
        logger.info(f" rep          : {args.rep}")
    logger.info(f" devices      : {devices} ({len(devices)} card(s))")
    logger.info(f" stop_file    : {stop_file}")
    logger.info(f" flaggems     : {args.flaggems_path or FLAGGEMS_ROOT}")
    logger.info("=" * 60)

    # Phase 1: Install
    if args.install:
        run_install()

    # Phase 2: Build
    if args.build:
        run_build()

    if no_run:
        logger.info(f"--no-run specified, exiting before {run_mode} execution.")
        return None

    # Apply custom flaggems path if specified
    if args.flaggems_path:
        apply_flaggems_path(args.flaggems_path)

    # Setup env
    setup_base_env()
    if accuracy:
        setup_precision_env()
    else:
        setup_profiler_env()

    # Activate venv if install was skipped
    if not args.install:
        activate_venv()

    # Cleanup temporary files older than 3 days
    logger.info("\n=== Cleanup ===")
    cleanup()

    # Log directory — override (ts-compare nested) or auto-create
    tag = "_test" if test_mode else ""
    dir_prefix = "acc" if accuracy else "bench"
    if log_dir_override is not None:
        log_dir = log_dir_override
        ops_dir = log_dir / "ops"
        ops_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        for base_name in ("ci_log", "ci_log_s"):
            log_dir = WORKSPACE / base_name / f"{dir_prefix}_{args.test}{tag}_{timestamp}"
            ops_dir = log_dir / "ops"
            try:
                ops_dir.mkdir(parents=True, exist_ok=True)
                if base_name == "ci_log_s":
                    logger.info(f"Note: ci_log is not writable, using ci_log_s instead")
                break
            except PermissionError:
                continue
        else:
            log_dir = Path.cwd() / f"{dir_prefix}_{args.test}{tag}_{timestamp}"
            ops_dir = log_dir / "ops"
            ops_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Note: neither ci_log nor ci_log_s writable, using {log_dir}")

    logger.info(f"\nLog dir: {log_dir}")

    # Isolate triton/flaggems cache per run to avoid PID-based filename races
    # when concurrent benchmark runs share /tmp and ~/.cache.
    _cache_base = Path("/tmp/triton_log") / log_dir.name
    os.environ["TRITON_CACHE_DIR"] = str(_cache_base / "triton_cache")
    os.environ["FLAGGEMS_CACHE_DIR"] = str(_cache_base / "flaggems_cache")
    logger.info(f"TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")
    logger.info(f"FLAGGEMS_CACHE_DIR={os.environ['FLAGGEMS_CACHE_DIR']}")

    # Tee all subsequent output to log_dir/full_output.log
    _fh = logging.FileHandler(str(log_dir / "full_output.log"), mode="a")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_fh)

    def _cleanup_fh():
        logger.removeHandler(_fh)

    # Determine test directory for pytest node paths
    test_dir = TESTS_DIR if accuracy else BENCH_DIR

    # Resolve operators
    if op_filter and not accuracy:
        # Manual -k filter: wrap with -k flag for pytest (benchmark only)
        operators: List[Tuple[str, List[str]]] = [("__filter__", ["-k", op_filter])]
    else:
        operators = resolve_operators(args.test, accuracy=accuracy)
        if args.test != "all" and not operators:
            logger.error("ERROR: no operators resolved")
            _cleanup_fh()
            return None
        logger.info(f"Resolved {len(operators)} operators")

    # --- Test mode: only run first 2 operators ---
    if test_mode and len(operators) > 2:
        operators = operators[:2]
        logger.info(f"  [TEST-MODE] truncated to first {len(operators)} operators: "
                    f"{', '.join(op for op, _ in operators)}")

    # Build pytest base args
    if accuracy:
        pytest_base_args = ["-v", "-s", "--ref", "cpu"]
        if quick:
            pytest_base_args.extend(["--mode", "quick"])
    else:
        pytest_base_args = [
            "-v", "-s",
            f"--warmup={args.warmup}",
            f"--iter={args.rep}",
            f"--level={args.level}",
        ]
        for m in args.metrics.split():
            pytest_base_args.extend(["--metrics", m])

    # Write run_info.json
    card_mode = "dynamic" if args.cards is not None else "exact"
    run_info = {
        "log_dir": str(log_dir),
        "mode": run_mode,
        "test_set": args.test,
        "devices": " ".join(str(d) for d in devices),
        "card_count": len(devices),
        "card_mode": card_mode,
        "cards": args.cards,
        "skip": args.skip,
        "test_mode": test_mode,
        "precision_mode": os.environ.get("PRECISION_MODE", ""),
        "flag_gems_custom_ops": os.environ.get("FLAG_GEMS_CUSTOM_OPS", ""),
        "tsm_profiler_en": os.environ.get("TSM_PROFILER_EN", ""),
        "flaggems_path": args.flaggems_path or str(FLAGGEMS_ROOT),
        "triton_workspace": str(WORKSPACE),
        "llvm_binary_dir": os.environ.get("LLVM_BINARY_DIR", ""),
        "tx8_deps_root": os.environ.get("TX8_DEPS_ROOT", ""),
    }
    if accuracy:
        run_info["quick"] = quick
    else:
        run_info.update({
            "op_filter": op_filter,
            "level": args.level,
            "warmup": args.warmup,
            "rep": args.rep,
            "bench_mode": "default",
        })
    with open(log_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    logger.info("  run_info.json written")

    # Special case: "all" test_set (no op_filter) — run pytest on entire directory
    if args.test == "all" and not operators and (accuracy or not op_filter):
        mode_label = "accuracy tests" if accuracy else "benchmark"
        logger.info(f"\n=== Running full {mode_label} on {len(devices)} card(s) ===\n")
        failed = 0
        t_all_start = time.time()
        for card in devices:
            card_log = ops_dir / f"card_{card}.log"
            proc_env = os.environ.copy()
            proc_env["TXDA_VISIBLE_DEVICES"] = str(card)
            cmd = ["python3", "-m", "pytest", str(test_dir)] + pytest_base_args
            logger.info(f"  card {card} start -> {card_log}")
            with open(card_log, "w") as f:
                f.write(f"=== card {card} ===\n")
                f.write(f"=== command: {' '.join(cmd)} ===\n")
                f.write(f"[{datetime.now().strftime('%m%d %H:%M:%S')}] pytest start\n")
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=proc_env)
                f.write(f"[{datetime.now().strftime('%m%d %H:%M:%S')}] pytest end (rc={result.returncode})\n")
            status = "PASSED" if result.returncode == 0 else "FAILED"
            logger.info(f"  [{status}] card {card} (rc={result.returncode})")
            if result.returncode != 0:
                failed = 1

        logger.info(f"\n{'=' * 60}")
        ts = datetime.now().strftime("%m%d %H:%M:%S")
        logger.info(f"  Full {mode_label} complete [{ts}]")
        logger.info(f"  cards: {devices}")
        logger.info(f"  log dir: {log_dir}")
        logger.info(f"{'=' * 60}")

        # Write summary log for "all" mode
        summary_name = "acc_summary.log" if accuracy else "bench_summary.log"
        ts = datetime.now().strftime("%m%d %H:%M:%S")
        all_log = "\n".join([
            f"Full {mode_label} complete [{ts}]",
            f"test_set: {args.test}  cards: {' '.join(str(d) for d in devices)}",
            f"mode: all (pytest discover)",
            f"elapsed: {round(time.time() - t_all_start, 1)}s",
            f"log dir: {log_dir}",
        ])
        with open(log_dir / summary_name, "w") as f:
            f.write(all_log + "\n")

        # Generate CSV summary only for benchmark mode
        if not accuracy:
            _gen_csv = SCRIPT_DIR / "gen_bench_summary.py"
            subprocess.run([sys.executable, str(_gen_csv), str(log_dir)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if failed:
            _cleanup_fh()
            return None
        _cleanup_fh()
        return log_dir

    # Normal path: operator queue
    mode_label = "accuracy tests" if accuracy else "benchmark"
    logger.info(f"\n=== Running {len(operators)} {mode_label} operators on {len(devices)} card(s) ===\n")
    t0 = time.time()
    results, remaining = run_operator_queue(
        operators, devices, ops_dir, pytest_base_args,
        test_dir=test_dir,
        stop_file=stop_file,
    )
    elapsed_total = time.time() - t0

    # Summary
    passed = sum(1 for r in results if r.rc == 0)
    failed_count = sum(1 for r in results if r.rc != 0)

    # Build summary log
    ts = datetime.now().strftime("%m%d %H:%M:%S")
    failed_ops = [r for r in results if r.rc != 0]

    # Collect error info for failed ops
    failed_details = []
    for r in failed_ops:
        err_info = ""
        try:
            with open(r.log_file) as lf:
                for line in lf:
                    if "Error:" in line or "FAILED" in line or "FAILURES" in line or "TIMEOUT" in line or "HANG-TYPE" in line:
                        err_info = line.strip()[:120]
                        break
        except Exception:
            pass
        failed_details.append((r.op_name, r.card, r.rc, err_info))

    # Collect dead card summary
    dead_card_summary: List[str] = []
    op_killed_cards: List[str] = []
    with _dead_cards_lock:
        for card, op_name, log_path in _dead_card_records:
            dead_card_summary.append(f"  card {card}: died on {op_name}, log: {log_path}")
        # Count how many cards each operator killed
        from collections import Counter
        op_death_count = Counter(r[1] for r in _dead_card_records)
        for op_name, count in op_death_count.items():
            if count >= 2:
                op_killed_cards.append(f"  {op_name}: killed {count} cards (likely operator-induced)")

    summary_title = "Accuracy test complete" if accuracy else "Benchmark complete"
    summary_file = "acc_summary.log" if accuracy else "bench_summary.log"
    log_lines = [
        f"{summary_title} [{ts}]",
        f"test_set: {args.test}  cards: {' '.join(str(d) for d in devices)}",
        f"total operators: {len(operators)}",
        f"completed: {len(results)}  passed: {passed}  failed: {failed_count}",
        f"elapsed: {elapsed_total:.1f}s",
        f"log dir: {log_dir}",
    ]
    if failed_details:
        log_lines.append(f"--- {len(failed_details)} FAILED ---")
        for op, card, rc, err in failed_details:
            log_lines.append(f"  {op} card={card} rc={rc}: {err}")
    if dead_card_summary:
        log_lines.append(f"--- {len(_dead_cards)} DEAD CARD(S) ---")
        log_lines.extend(dead_card_summary)
    if op_killed_cards:
        log_lines.append(f"--- OPERATOR-INDUCED DEAD CARDS ---")
        log_lines.extend(op_killed_cards)
    log_text = "\n".join(log_lines)

    with open(log_dir / summary_file, "w") as f:
        f.write(log_text + "\n")

    # Console output
    logger.info(f"\n{'=' * 60}")
    for line in log_lines:
        logger.info(f"  {line}")
    logger.info(f"{'=' * 60}")

    # Generate per-operator CSV only for benchmark mode
    if not accuracy:
        _gen_csv = SCRIPT_DIR / "gen_bench_summary.py"
        subprocess.run([sys.executable, str(_gen_csv), str(log_dir)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if failed_count > 0:
        logger.warning(f"WARNING: {failed_count} operator(s) failed")
    _cleanup_fh()
    return log_dir


# ===========================================================================
# 10. Subcommand implementations
# ===========================================================================

def _run_on_off_compare(args, label: str):
    """Run CUSTOM_OPS=0 as baseline, CUSTOM_OPS=1 as target, then compare.

    Shared by ts-compare (CUSTOM_OPS=0 vs CUSTOM_OPS=1 comparison).
    Creates a parent directory with baseline/ and target/ subdirs.
    """
    # Create parent dir for the comparison
    tag = "_test" if getattr(args, 'test_mode', False) else ""
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    parent_dir = WORKSPACE / "ci_log" / f"ts-compare{tag}_{timestamp}"
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        parent_dir = WORKSPACE / "ci_log_s" / f"ts-compare{tag}_{timestamp}"
        parent_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = parent_dir / "baseline"
    target_dir = parent_dir / "target"

    # Run baseline (CUSTOM_OPS=0)
    logger.info("\n" + "=" * 70)
    logger.info(f"  {label} — Baseline (FLAG_GEMS_CUSTOM_OPS=0)")
    logger.info("=" * 70)
    os.environ["FLAG_GEMS_CUSTOM_OPS"] = "0"
    result = cmd_run(args, log_dir_override=baseline_dir)
    if result is None:
        logger.error("ERROR: baseline run failed")
        sys.exit(1)

    # Run target (CUSTOM_OPS=1) — skip install/build on second run
    logger.info("\n" + "=" * 70)
    logger.info(f"  {label} — Target (FLAG_GEMS_CUSTOM_OPS=1)")
    logger.info("=" * 70)
    os.environ["FLAG_GEMS_CUSTOM_OPS"] = "1"
    args.install = False
    args.build = False
    result = cmd_run(args, log_dir_override=target_dir)
    if result is None:
        logger.error("ERROR: target run failed")
        sys.exit(1)

    # Compare
    logger.info("\n" + "=" * 70)
    logger.info(f"  Baseline : {baseline_dir}")
    logger.info(f"  Target   : {target_dir}")
    logger.info("=" * 70)
    compare_script = SCRIPT_DIR / "compare_benchmark_logs.py"
    cmd = ["python3", str(compare_script), str(baseline_dir), str(target_dir),
           "--threshold", str(args.threshold), "-o", str(parent_dir)]
    logger.info(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("ERROR: comparison found degradations exceeding threshold")
        sys.exit(1)
    ts = datetime.now().strftime("%m%d %H:%M:%S")
    logger.info(f"\n  {label} complete [{ts}].")
    logger.info(f"  Directory: {parent_dir}")
    logger.info(f"    baseline/  (CUSTOM_OPS=0)")
    logger.info(f"    target/    (CUSTOM_OPS=1)")
    logger.info(f"    comparison.csv / comparison_report.md")


def cmd_ts_compare(args):
    """Run CUSTOM_OPS=0 vs CUSTOM_OPS=1 full comparison (all ts_ops)."""
    _run_on_off_compare(args, "ts-compare")


def cmd_ci_baseline(args):
    """Run benchmark once. If --baseline specified, compare to it."""
    logger.info("\n" + "=" * 70)
    logger.info(f"  CI benchmark — {args.test}")
    if args.baseline:
        logger.info(f"  Baseline: {args.baseline}")
    logger.info("=" * 70)

    target_dir = cmd_run(args)
    if target_dir is None:
        logger.error("ERROR: benchmark run failed")
        sys.exit(1)

    if not args.baseline:
        logger.info(f"\n  Baseline saved: {target_dir}")
        logger.info(f"  Use as future baseline: python run_benchmark.py ci-baseline --baseline {target_dir}")
        return

    # Compare to baseline
    if not Path(args.baseline).is_dir():
        logger.error(f"ERROR: baseline directory not found: {args.baseline}")
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info(f"  Baseline : {args.baseline}")
    logger.info(f"  Target   : {target_dir}")
    logger.info("=" * 70)
    compare_script = SCRIPT_DIR / "compare_benchmark_logs.py"
    cmd = ["python3", str(compare_script), args.baseline, str(target_dir),
           "--threshold", str(args.threshold), "--fail-on-threshold"]
    logger.info(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("\nERROR: degradation exceeds threshold!")
        sys.exit(1)
    ts = datetime.now().strftime("%m%d %H:%M:%S")
    logger.info(f"\n  ci-baseline complete [{ts}] — no degradation beyond {args.threshold}% threshold.")


def cmd_compare(args):
    """Call compare_benchmark_logs.py on two directories."""
    validate_compare_args(args)

    if args.install:
        run_install()
    if args.build:
        run_build()
    if not args.install:
        activate_venv()

    compare_script = SCRIPT_DIR / "compare_benchmark_logs.py"
    cmd = ["python3", str(compare_script), args.dir1, args.dir2,
           "--threshold", str(args.threshold)]
    logger.info(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ===========================================================================
# 11. Main entry point
# ===========================================================================

def main():
    # Known subcommand names
    SUBCOMMANDS = {"benchmark", "accuracy", "ci-baseline", "ts-compare", "compare",
                   "-h", "--help"}

    # No args → show help
    if len(sys.argv) == 1:
        parser = make_parser()
        parser.print_help()
        return

    # Backward compat: if first arg is not a known subcommand, insert "benchmark"
    if sys.argv[1] not in SUBCOMMANDS:
        sys.argv.insert(1, "benchmark")

    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    if args.command in ("benchmark", "accuracy"):
        log_dir = cmd_run(args)
        if log_dir is None:
            if not getattr(args, 'no_run', False):
                sys.exit(1)
            return
    elif args.command == "ci-baseline":
        cmd_ci_baseline(args)
    elif args.command == "ts-compare":
        cmd_ts_compare(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
