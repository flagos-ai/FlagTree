#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
generate_skip_list.py - FlagGems CI skip list generator.

Identifies operators that cause GPU state corruption (cascade failures) by
running ci_run_tests.py repeatedly, using two detection phases.  Optionally
clones and builds FlagTree and the operator library in a preliminary Phase 0.

Phase 0 - clone + build (optional)
  Triggered by --flagtree-ref and/or --flaggems-ref.
  - --flagtree-ref: clone flagos-ai/FlagTree at the given branch/tag into
    --work-dir/FlagTree, uninstall the current flagtree package, then build and
    install from the clone.  Omit to skip; FlagTree is assumed already installed.
  - --flaggems-ref: clone the operator library (URL given by --flaggems-url) at
    the given tag into --work-dir/<repo-basename>.  Mutually exclusive with
    --flaggems-path.
  - --flaggems-path: use an already-cloned operator library at this path.
    Mutually exclusive with --flaggems-ref.

Phase 1 - Sliding-window cascade detection
  After each run, scan the case-level execution sequence in pytest_stdout.log.
  For every failing case, check whether the next W cases have a fail rate above
  threshold T.  If so, that file is a cascade source: add its operators to the
  skip list and start the next round.  Repeat until no trigger is found.

  Window adaptation for the tail: if the last W cases have >50% fail rate, cases
  in those tail positions use a smaller window (W//2, floor --window-shrink).

  Killed-process handling: if pytest_stdout.log is incomplete (missing [100%]
  line or summary line), the last test file in the log is likely the culprit.
  Its operators are added to the skip list via collected_marks.yaml, which is
  written before pytest starts and is always complete.

Phase 2 - Flaky rolling detection
  Seed the reference result map from the final Phase 1 round.  Run one round,
  compare each test case against the reference: any non-fail → fail transition
  means that case's file is a flaky corruption source; add its operators to the
  skip list and reset the consecutive-stable counter.  When the counter reaches
  --flaky-rounds consecutive rounds with no new additions, the skip list is
  stable and the tool exits.  No Phase 3 needed: --flaky-rounds consecutive
  stable rounds in Phase 2 ARE the stability proof.

Usage
-----
  # Full run from scratch (clone FlagTree + FlagGems, build, detect)
  python3 generate_skip_list.py \\
      --backend xpu \\
      --flagtree-ref main \\
      --flaggems-ref v5.5.0.dev1 \\
      --build-command "source ~/env-3.6.sh && MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v" \\
      --pre-command "source ~/env.sh && source ~/env-3.6.sh" \\
      --gpus 0 --first-parameter-only
  # (--flaggems-url defaults to https://github.com/flagos-ai/FlagGems.git)
  # (--work-dir defaults to ~/flaggems_ci/skip_list/<repo-kind>_skip_list_<YYYYMMDD_HHMMSS>)
  # (--output-dir defaults to <work-dir>/output)

  # FlagGems-vllm variant (only URL differs)
  python3 generate_skip_list.py \\
      --backend mthreads \\
      --flagtree-ref main \\
      --flaggems-ref v1.2.0 \\
      --flaggems-url https://github.com/flagos-ai/FlagGems-vllm.git \\
      --build-command "source ~/env-3.6.sh && MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v" \\
      --gpus 0

  # FlagTree already installed, clone FlagGems only
  python3 generate_skip_list.py \\
      --backend xpu \\
      --flaggems-ref v5.5.0.dev1 \\
      --gpus 0 --first-parameter-only

  # Everything already ready, detect only
  python3 generate_skip_list.py \\
      --backend xpu \\
      --flaggems-path /root/flaggems_ci/FlagGems/test_xxx \\
      --gpus 0

  # Resume any of the above
  python3 generate_skip_list.py ... --resume

Output
------
  <output-dir>/
    state.json               - resumable state (overwritten each round)
    phase0/                  - clone/build artifacts
    phase1/round_N/          - phase 1 round output dirs
    phase2/round_N/          - phase 2 round output dirs
    <backend>.yaml           - final skip list in FlagGems YAML format
"""

import argparse
import atexit
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[93m"
CYAN = "\033[36m"
BOLD = "\033[1m"
NC = "\033[0m"


def info(msg: str) -> None:
    print(f"{GREEN}[INFO]{NC} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC} {msg}", flush=True)


def error(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", flush=True)


def section(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*70}{NC}", flush=True)
    print(f"{BOLD}{CYAN}  {msg}{NC}", flush=True)
    print(f"{BOLD}{CYAN}{'='*70}{NC}\n", flush=True)


# ---------------------------------------------------------------------------
# Logging: duplicate stdout/stderr to <output-dir>/test_output.log
# ---------------------------------------------------------------------------

# Strips ANSI color codes before writing to the log file so it stays readable.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

_LOG_FILE_HANDLE = None


class _Tee:
    """File-like object writing to several streams at once.

    ANSI color codes are kept on the terminal but stripped for the log file.
    """

    def __init__(self, *streams, strip_ansi_for=None):
        self._streams = streams
        self._strip_ansi_for = strip_ansi_for or set()

    def write(self, data):
        for s in self._streams:
            if s in self._strip_ansi_for:
                s.write(_ANSI_ESCAPE_RE.sub("", data))
            else:
                s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        # Report False so subprocesses (pip, git, pytest) emit plain output,
        # which keeps the log file free of progress-bar control sequences.
        return False


def _close_log_file() -> None:
    global _LOG_FILE_HANDLE
    if _LOG_FILE_HANDLE is not None:
        try:
            _LOG_FILE_HANDLE.flush()
            _LOG_FILE_HANDLE.close()
        except Exception:
            pass
        _LOG_FILE_HANDLE = None


def setup_logging(output_dir: Path, filename: str = "test_output.log") -> None:
    """Duplicate stdout/stderr to ``output_dir/<filename>``.

    The directory must already exist.  Safe to call more than once; only the
    first call takes effect.
    """
    global _LOG_FILE_HANDLE

    if _LOG_FILE_HANDLE is not None:
        return

    log_path = output_dir / filename
    # Line-buffered so partial output survives a crash.
    _LOG_FILE_HANDLE = open(log_path, "w", buffering=1, encoding="utf-8")

    sys.stdout = _Tee(sys.__stdout__, _LOG_FILE_HANDLE,
                      strip_ansi_for={_LOG_FILE_HANDLE})
    sys.stderr = _Tee(sys.__stderr__, _LOG_FILE_HANDLE,
                      strip_ansi_for={_LOG_FILE_HANDLE})

    atexit.register(_close_log_file)

    info(f"Log file: {log_path}")


# ---------------------------------------------------------------------------
# Skip list YAML I/O
# ---------------------------------------------------------------------------

_YAML_HEADER = """\
# Skip operators that cause GPU state corruption
skip_operators:
{ops}
# Skip specific test functions
skip_test_patterns:
  #- test_example_pattern

# Skip entire test files
skip_test_files:
  #- test_example.py
"""


def write_skip_yaml(skip_operators: List[str], path: Path) -> None:
    """Write a skip list YAML in the canonical FlagTree format."""
    if skip_operators:
        ops_lines = "\n".join(f"  - {op}" for op in sorted(skip_operators))
    else:
        ops_lines = "  #- example_operator"
    path.write_text(_YAML_HEADER.format(ops=ops_lines), encoding="utf-8")
    info(f"Wrote skip list ({len(skip_operators)} ops) → {path}")


def read_skip_yaml(path: Path) -> List[str]:
    """Read skip_operators from a YAML skip list file."""
    try:
        import yaml
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return list(data.get("skip_operators") or [])
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# State persistence (resumability)
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            warn(f"Could not parse state file {state_path}; starting fresh.")
    return {}


def save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)


# ---------------------------------------------------------------------------
# Phase 0: clone + build + install helpers
# ---------------------------------------------------------------------------

FLAGTREE_URL = "https://github.com/flagos-ai/FlagTree.git"
DEFAULT_FLAGGEMS_URL = "https://github.com/flagos-ai/FlagGems.git"
DEFAULT_FLAGGEMS_VLLM_URL = "https://github.com/flagos-ai/FlagGems-vllm.git"
FLAGTREE_PIP_NAME = "flagtree"

# Shell used for every user-supplied command.  /bin/sh is dash on Debian/Ubuntu
# and lacks bash builtins such as `source`, which --pre-command relies on.
BASH = "/bin/bash"


def _repo_basename(url: str) -> str:
    """Derive a filesystem-safe directory name from a git URL.

    Examples:
      https://github.com/org/FlagGems.git  → FlagGems
      https://github.com/org/FlagGems-vllm → FlagGems-vllm
    """
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    return name.split("/")[-1]


# Accepted --repo-kind values, mapped to the canonical form passed through to
# ci_run_tests.py.  Matched case-insensitively ("FlagGems" is accepted).
_REPO_KIND_ALIASES = {
    "flaggems": "flaggems",
    "flaggems-vllm": "flaggems-vllm",
}


def parse_repo_kind(value: str) -> str:
    """Normalise a user-supplied --repo-kind value.

    Accepts "flaggems" and "flaggems-vllm" in any capitalisation.  Anything
    else is rejected with a message listing the valid values.
    """
    canonical = _REPO_KIND_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"invalid repo-kind '{value}'. "
            f"Valid values: {', '.join(sorted(_REPO_KIND_ALIASES))} "
            "(case-insensitive)."
        )
    return canonical


def _capture_env(pre_command: str, base_env: Optional[dict] = None) -> dict:
    """Run `pre_command` in bash and return the resulting environment.

    The command is followed by `env`, separated by an end-of-options marker, so
    the command's own stdout (pip notices, `cd` output, ...) does not get parsed
    as environment variables.

    bash is used because --pre-command is documented as sourcing files such as
    ~/env.sh, and `source` does not exist in dash (/bin/sh on Debian/Ubuntu).
    """
    env = dict(base_env if base_env is not None else os.environ)
    if not pre_command:
        return env

    marker = "__SKIP_LIST_ENV_MARKER__"
    script = f"{pre_command}\nprintf '%s\\n' '{marker}'\nenv"

    result = subprocess.run(
        script,
        shell=True, executable=BASH,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        warn(f"pre-command returned {result.returncode}:\n"
             f"{result.stderr.strip()[:500]}")

    stdout = result.stdout or ""
    if marker in stdout:
        command_output, _, env_dump = stdout.partition(marker)
        # Show what the pre-command printed; it is often pip output the caller
        # wants to see, and it must not be mistaken for the environment.
        for line in command_output.splitlines():
            if line.strip():
                info(f"  pre-command: {line}")
        for line in env_dump.splitlines():
            k, sep, v = line.partition("=")
            if sep and k.isidentifier() and not k.startswith("_"):
                env[k] = v
    else:
        warn("pre-command produced no environment marker; "
             "its variables will not be applied.")
    return env


def _run(cmd: str, cwd=None, env=None, check: bool = True, label: str = ""):
    """Run a shell command, streaming its output through our stdout tee.

    Output is piped (rather than inherited) so it also lands in
    <output-dir>/test_output.log; it is echoed line by line as it arrives.

    Uses bash rather than /bin/sh: the commands users pass here routinely use
    bash-only builtins such as `source`, which dash rejects with exit 127.
    """
    tag = f"[{label}] " if label else ""
    info(f"{tag}running: {cmd}")
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable=BASH,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
            proc.stdout.close()
        returncode = proc.wait()
    except Exception as exc:
        error(f"{tag}command failed to execute: {exc}")
        if check:
            raise
        return None
    if check and returncode != 0:
        raise RuntimeError(f"{tag}command exited {returncode}: {cmd}")
    return subprocess.CompletedProcess(cmd, returncode)


def _current_flagtree_version() -> Optional[str]:
    """Return the installed flagtree package version, or None."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", FLAGTREE_PIP_NAME],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _clone_repo(url: str, ref: str, dest: Path, label: str,
                max_retries: int = 5, env: Optional[dict] = None) -> None:
    """Clone `url` at `ref` into `dest`, retrying on network failures.

    If `dest` already is a git repo at the requested ref, skip the clone.

    `env` must be the environment prepared by --pre-command. The proxy
    variables it exports exist only in that dict, not in os.environ, so
    subprocesses that omit `env=` would hit the network without a proxy and
    hang.
    """
    if dest.exists() and (dest / ".git").exists():
        try:
            cur = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=dest, capture_output=True, text=True, env=env,
            ).stdout.strip()
            cur_ref = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=dest, capture_output=True, text=True, env=env,
            ).stdout.strip()
            wanted = subprocess.run(
                ["git", "rev-parse", ref],
                cwd=dest, capture_output=True, text=True, env=env,
            ).stdout.strip()
            if cur == ref or cur_ref == wanted:
                info(f"{label}: already at ref '{ref}' ({cur_ref[:8]}), skip clone.")
                return
        except Exception:
            pass
        warn(f"{label}: existing clone is not at ref '{ref}'; re-cloning.")
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        info(f"{label}: clone attempt {attempt}/{max_retries}")
        try:
            # --branch accepts a tag or a branch name and fetches only that
            # ref's history.  Note that the previous approach -- a shallow
            # clone followed by `git fetch --depth 1 origin <tag>` and
            # `git checkout <tag>` -- always failed for tags: the fetch only
            # writes FETCH_HEAD and never creates a local tag ref, so the
            # checkout could not resolve the name.
            res = subprocess.run(
                f"git clone --depth 1 --branch {ref} {url} {dest}",
                shell=True, executable=BASH,
                capture_output=True, text=True,
                env=env,
            )
            if res.returncode != 0:
                # --branch cannot resolve a raw commit SHA; fall back to a
                # full clone, which can check out any revision.
                warn(f"{label}: shallow clone of '{ref}' failed "
                     f"({res.stderr.strip().splitlines()[-1:] or 'no detail'}); "
                     "doing full clone.")
                shutil.rmtree(dest, ignore_errors=True)
                _run(f"git clone {url} {dest} 2>&1",
                     label=label, check=False, env=env)
                _run(f"git checkout {ref}", cwd=dest, label=label, env=env)
            info(f"{label}: checked out '{ref}'.")
            return
        except Exception as exc:
            last_err = exc
            warn(f"{label}: attempt {attempt} failed: {exc}")
            shutil.rmtree(dest, ignore_errors=True)
            time.sleep(5 * attempt)
    raise RuntimeError(
        f"{label}: clone failed after {max_retries} attempts: {last_err}"
    )


def _uninstall_flagtree() -> None:
    """Remove the installed flagtree package and leftover triton directory."""
    info("Uninstalling existing flagtree pip package…")
    _run(
        f"{sys.executable} -m pip uninstall -y {FLAGTREE_PIP_NAME} "
        f"--break-system-packages 2>&1 || true",
        label="uninstall",
        check=False,
    )
    try:
        site_packages = subprocess.run(
            [sys.executable, "-c",
             "import sysconfig; print(sysconfig.get_path('purelib'))"],
            capture_output=True, text=True,
        ).stdout.strip()
        if site_packages:
            triton_dir = Path(site_packages) / "triton"
            if triton_dir.exists():
                shutil.rmtree(triton_dir, ignore_errors=True)
                info(f"Removed leftover {triton_dir}")
    except Exception as exc:
        warn(f"Could not locate site-packages for triton cleanup: {exc}")


def _build_flagtree(flagtree_dir: Path, backend: str,
                    build_command: str, env: dict) -> str:
    """Build and install FlagTree; return the installed version string.

    `env` is the environment the caller already prepared (--pre-command
    sourced and captured once at the start of the run); the build inherits it
    with FLAGTREE_BACKEND forced to `backend`.
    """
    info(f"Building FlagTree in {flagtree_dir} (FLAGTREE_BACKEND={backend})")
    env = dict(env)
    env["FLAGTREE_BACKEND"] = backend

    if not build_command:
        build_command = (
            f"{sys.executable} -m pip install . --no-build-isolation -v"
        )
        warn("No --build-command given; using default pip install.")

    _run(build_command, cwd=flagtree_dir, env=env, label="build")

    version = _current_flagtree_version()
    if version is None:
        raise RuntimeError(
            "FlagTree build finished but `pip show flagtree` found no package."
        )
    info(f"FlagTree installed: {version}")
    return version


def phase0_setup(
    work_dir: Path,
    backend: str,
    flagtree_ref: Optional[str],
    flaggems_ref: Optional[str],
    flaggems_url: str,
    flaggems_path_arg: Optional[Path],
    repo_kind: str,
    flaggems_vllm_ref: Optional[str],
    flaggems_vllm_url: str,
    flaggems_vllm_path_arg: Optional[Path],
    build_command: str,
    env: dict,
    skip_clone: bool,
    state: dict,
    state_path: Path,
) -> Tuple[Path, Optional[Path]]:
    """
    Clone repos and/or build FlagTree as needed.

    Returns (under_test_path, dependency_path).

    under_test_path is the checkout whose tests/ and conf/ drive the run — it
    is what ci_run_tests.py receives as --flaggems-path. dependency_path is the
    flag_gems checkout that flaggems_vllm imports at runtime; it is None for
    --repo-kind flaggems, where the library under test IS flag_gems.
    """
    section("Phase 0: environment setup")

    p0 = state.setdefault("phase0", {
        "flagtree_ref": flagtree_ref,
        "flaggems_ref": flaggems_ref,
        "flaggems_url": flaggems_url,
        # What FlagTree build this run ended up with. Informational: it records
        # the version actually installed, not whether Phase 0 succeeded.
        "installed_version": None,
        # Outcome of Phase 0:
        #   "pending"  - not started / interrupted before finishing
        #   "installed" - FlagTree was cloned, built and installed this run
        #   "reused"   - a matching install already existed, nothing rebuilt
        #   "skipped"  - --flagtree-ref not given, existing install used as-is
        "status": "pending",
    })

    # --- FlagTree clone + build ------------------------------------------
    if flagtree_ref:
        flagtree_dir = work_dir / "FlagTree"
        if not skip_clone:
            _clone_repo(FLAGTREE_URL, flagtree_ref, flagtree_dir, "FlagTree",
                        env=env)
        else:
            info(f"Skipping FlagTree clone (--skip-clone). Using {flagtree_dir}")

        need_rebuild = True
        installed = _current_flagtree_version()
        if (p0.get("status") in ("installed", "reused")
                and p0.get("installed_version") == installed
                and installed):
            info(f"FlagTree already built ({installed}); skipping rebuild.")
            need_rebuild = False
            p0["status"] = "reused"
            save_state(state_path, state)

        if need_rebuild:
            _uninstall_flagtree()
            installed = _build_flagtree(
                flagtree_dir, backend, build_command, env,
            )
            p0["installed_version"] = installed
            p0["status"] = "installed"
            save_state(state_path, state)
    else:
        info("--flagtree-ref not specified; using already-installed FlagTree.")
        p0["status"] = "skipped"
        p0["installed_version"] = _current_flagtree_version()
        save_state(state_path, state)

    # --- FlagGems checkout (dependency for vllm, library under test otherwise)
    if flaggems_ref:
        basename = _repo_basename(flaggems_url)
        flaggems_dir = work_dir / basename
        if not skip_clone:
            _clone_repo(flaggems_url, flaggems_ref, flaggems_dir, basename,
                        env=env)
        else:
            info(f"Skipping {basename} clone (--skip-clone). Using {flaggems_dir}")
        flaggems_path = flaggems_dir
    else:
        # --flaggems-path was provided (enforced by argparse mutual exclusion)
        flaggems_path = flaggems_path_arg

    if not (flaggems_path / "src").is_dir():
        raise RuntimeError(
            f"FlagGems checkout at {flaggems_path} has no src/ directory; "
            "cannot set PYTHONPATH."
        )

    # --- FlagGems-vllm checkout (the library under test in vllm mode) -----
    if repo_kind == "flaggems-vllm":
        if flaggems_vllm_ref:
            vllm_base = _repo_basename(flaggems_vllm_url)
            vllm_dir = work_dir / vllm_base
            if not skip_clone:
                _clone_repo(flaggems_vllm_url, flaggems_vllm_ref, vllm_dir,
                            vllm_base, env=env)
            else:
                info(f"Skipping {vllm_base} clone (--skip-clone). "
                     f"Using {vllm_dir}")
            under_test = vllm_dir
        else:
            under_test = flaggems_vllm_path_arg

        if under_test is None:
            raise RuntimeError(
                "--repo-kind flaggems-vllm requires --flaggems-vllm-ref or "
                "--flaggems-vllm-path."
            )
        if not (under_test / "src").is_dir():
            raise RuntimeError(
                f"FlagGems-vllm checkout at {under_test} has no src/ directory; "
                "cannot set PYTHONPATH."
            )
        return under_test, flaggems_path

    return flaggems_path, None


# ---------------------------------------------------------------------------
# Running ci_run_tests.py
# ---------------------------------------------------------------------------


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC} {msg}", flush=True)


def error(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", flush=True)


def section(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*70}{NC}", flush=True)
    print(f"{BOLD}{CYAN}  {msg}{NC}", flush=True)
    print(f"{BOLD}{CYAN}{'='*70}{NC}\n", flush=True)


# ---------------------------------------------------------------------------
# Skip list YAML I/O
# ---------------------------------------------------------------------------

_YAML_HEADER = """\
# Skip operators that cause GPU state corruption
skip_operators:
{ops}
# Skip specific test functions
skip_test_patterns:
  #- test_example_pattern

# Skip entire test files
skip_test_files:
  #- test_example.py
"""


def write_skip_yaml(skip_operators: List[str], path: Path) -> None:
    """Write a skip list YAML in the canonical FlagTree format."""
    if skip_operators:
        ops_lines = "\n".join(f"  - {op}" for op in sorted(skip_operators))
    else:
        ops_lines = "  #- example_operator"
    path.write_text(_YAML_HEADER.format(ops=ops_lines), encoding="utf-8")
    info(f"Wrote skip list ({len(skip_operators)} ops) → {path}")


def read_skip_yaml(path: Path) -> List[str]:
    """Read skip_operators from a YAML skip list file."""
    try:
        import yaml
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return list(data.get("skip_operators") or [])
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# State persistence (resumability)
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            warn(f"Could not parse state file {state_path}; starting fresh.")
    return {}


def save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)


# ---------------------------------------------------------------------------
# Running ci_run_tests.py
# ---------------------------------------------------------------------------

def _build_cmd(
    runner_script: Path,
    flaggems_path: Path,
    gpus: str,
    output_dir: Path,
    skip_file: Optional[Path],
    args,
) -> List[str]:
    cmd = [
        sys.executable,
        str(runner_script),
        "--gpus", gpus,
        "--output-dir", str(output_dir),
        "--flaggems-path", str(flaggems_path),
    ]
    if skip_file is not None:
        cmd += ["--skip-file", str(skip_file)]
    if args.first_parameter_only:
        cmd.append("--first-parameter-only")
    # repo-kind defaults to "flaggems"; the operator library identity is determined
    # by what is installed under PYTHONPATH, not by this flag.  Pass it explicitly
    # only if the caller sets an override attribute.
    # --repo-kind is validated by argparse (case-insensitive) and forwarded to
    # ci_run_tests.py verbatim.
    cmd += ["--repo-kind", args.repo_kind]
    return cmd


def run_ci(
    runner_script: Path,
    flaggems_path: Path,
    gpus: str,
    output_dir: Path,
    skip_operators: List[str],
    args,
    max_retries: int = 2,
    label: str = "",
) -> Optional[dict]:
    """
    Invoke ci_run_tests.py and return the parsed summary.json, or None if the
    run failed or produced no usable output.

    A temporary skip list file is created from skip_operators for each call so
    the caller never has to manage files directly.

    Unlike the old LOO implementation, we do NOT require marks_collection_ok
    here: phase functions read pytest_stdout.log and accuracy_all.json directly
    and can tolerate a partial marks map.  We still check that summary.json
    was produced so callers can detect hard failures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_file = output_dir / "skip_list.yaml"
    write_skip_yaml(skip_operators, skip_file)

    # Base env is prepared once at startup (--pre-command already sourced).
    # --per-round-command, if given, is applied on every round on top of it.
    env = getattr(args, "base_env", None) or os.environ.copy()
    per_round_command = getattr(args, "per_round_command", None)
    if per_round_command:
        env = _capture_env(per_round_command, base_env=env)

    for attempt in range(1, max_retries + 1):
        attempt_dir = output_dir if attempt == 1 else output_dir / f"retry_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_cmd(runner_script, flaggems_path, gpus, attempt_dir, skip_file, args)
        tag = f"{label} " if label else ""
        info(f"{tag}Running ci_run_tests.py (attempt {attempt}/{max_retries})…")
        info(f"  cmd: {' '.join(cmd)}")

        stdout_log = attempt_dir / "runner_stdout.log"
        stderr_log = attempt_dir / "runner_stderr.log"

        try:
            with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
                subprocess.run(cmd, env=env, stdout=fout, stderr=ferr)
        except Exception as exc:
            warn(f"Subprocess failed: {exc}")
            if attempt < max_retries:
                time.sleep(5)
            continue

        summary_path = attempt_dir / "summary.json"
        if not summary_path.exists():
            warn(f"No summary.json produced; see {attempt_dir}/runner_stderr.log")
            if attempt < max_retries:
                info("Retrying…")
                time.sleep(5)
            continue

        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            warn(f"Could not parse summary.json: {exc}")
            if attempt < max_retries:
                time.sleep(5)
            continue

        if attempt > 1:
            (output_dir / "summary.json").write_text(summary_path.read_text())

        return summary

    error(f"All {max_retries} attempts failed for {output_dir}.")
    return None


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------

def log_is_complete(stdout_log: Path) -> bool:
    """Return True if pytest_stdout.log contains [100%] and a summary line."""
    if not stdout_log.exists():
        return False
    has_100 = False
    has_summary = False
    with open(stdout_log, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[100%]" in line:
                has_100 = True
            if re.search(r"=\s*\d+ (failed|passed)", line):
                has_summary = True
    return has_100 and has_summary


def last_file_in_log(stdout_log: Path) -> Optional[str]:
    """Return the basename of the last test file seen in the progress output."""
    last = None
    with open(stdout_log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(test_\S+\.py)\s+", line)
            if m:
                last = m.group(1)
    return last


def parse_stdout_cases(stdout_log: Path) -> List[Tuple[str, bool]]:
    """
    Return [(filename, is_fail), ...] in test execution order.

    Each character in the pytest progress marker string becomes one entry.
    Only 'F' is treated as a failure; '.', 's', 'x', 'X', 'p' are non-fail.
    """
    cases: List[Tuple[str, bool]] = []
    with open(stdout_log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(test_\S+\.py)\s+([.FsxXp]+)", line)
            if m:
                for ch in m.group(2):
                    cases.append((m.group(1), ch == "F"))
    return cases


def get_ops_for_file(accuracy_json: dict, filename: str) -> Set[str]:
    """
    Extract operator names for all test cases in the given file.

    accuracy_json keys look like 'tests/test_foo.py::test_bar[...]'.
    filename is the basename (e.g. 'test_foo.py').
    """
    ops: Set[str] = set()
    for key, val in accuracy_json.items():
        if key.split("::")[0].split("/")[-1] == filename:
            for op in (val.get("opname") or []):
                ops.add(op)
    return ops


def get_ops_from_marks(marks_yaml: list, filename: str) -> Set[str]:
    """
    Extract operator names for a file from collected_marks.yaml.

    Used when the test process was killed and accuracy_all.json is unavailable.
    marks_yaml is a list of {file, function, marks, ...} dicts.
    """
    ops: Set[str] = set()
    for item in marks_yaml:
        if item.get("file", "").split("/")[-1] == filename:
            for op in (item.get("marks") or []):
                ops.add(op)
    return ops


def load_accuracy_json(run_dir: Path) -> dict:
    """Load accuracy_all.json from a run output directory, or {} on failure."""
    path = run_dir / "accuracy_all.json"
    if not path.exists():
        warn(f"accuracy_all.json not found in {run_dir}")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        warn(f"Could not parse accuracy_all.json: {exc}")
        return {}


def load_marks_yaml(run_dir: Path) -> list:
    """Load collected_marks.yaml from a run output directory, or [] on failure."""
    path = run_dir / "collected_marks.yaml"
    if not path.exists():
        return []
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or []
    except Exception as exc:
        warn(f"Could not parse collected_marks.yaml: {exc}")
        return []


# ---------------------------------------------------------------------------
# Cascade detection (shared by both phases)
# ---------------------------------------------------------------------------

def scan_cascade(
    cases: List[Tuple[str, bool]],
    accuracy: dict,
    skip_list: Set[str],
    window_size: int,
    window_shrink: int,
    threshold: float,
) -> Optional[str]:
    """Return the file that starts a cascade, or None if there is none.

    A cascade source is a failing case whose next `window_size` successors fail
    at a rate above `threshold`: everything downstream of it is collateral
    damage from corrupted GPU state rather than a genuine per-operator failure.

    Tail adaptation: when the last `window_size` cases fail more than half the
    time, positions near the end use a smaller window (`window_size // 2`, floored
    at `window_shrink`), so a trigger sitting close to the end of the suite is
    still detectable.

    `skip_list` is unused for the decision; it is accepted so callers can pass
    the current state without special-casing.
    """
    if not cases:
        return None

    if len(cases) >= window_size:
        tail_fail_rate = sum(1 for _, f in cases[-window_size:] if f) / window_size
        W_tail = (max(window_size // 2, window_shrink)
                  if tail_fail_rate > 0.5 else window_size)
    else:
        W_tail = window_shrink
    if W_tail != window_size:
        info(f"  Tail ({window_size} cases) fail rate={tail_fail_rate:.0%} >50%; "
             f"using W_tail={W_tail} for tail positions.")

    for i, (fname, is_fail) in enumerate(cases):
        if not is_fail:
            continue
        remaining = len(cases) - (i + 1)
        # Tail positions (remaining < window_size) use the adapted W_tail.
        effective_W = W_tail if remaining < window_size else window_size
        if remaining < effective_W:
            continue
        window = cases[i + 1: i + 1 + effective_W]
        if sum(1 for _, f in window if f) / effective_W > threshold:
            return fname
    return None


# ---------------------------------------------------------------------------
# Phase 1: sliding-window cascade detection
# ---------------------------------------------------------------------------

def phase1_sliding_window(
    runner_script: Path,
    flaggems_path: Path,
    gpus: str,
    base_dir: Path,
    args,
    skip_list: List[str],
    state: dict,
    state_path: Path,
    total_rounds_so_far: int,
) -> Tuple[List[str], int, Optional[dict]]:
    """
    Run the sliding-window cascade detection loop.

    Each round runs ci_run_tests.py once and scans pytest_stdout.log for the
    first failing case whose next W successors have a fail rate above threshold.
    If found, that file's operators are added to the skip list and the next
    round starts.  When no trigger is found, the loop exits.

    Returns
    -------
    skip_list        : updated skip list (may be extended)
    total_rounds     : updated round counter (Phase 1 + Phase 2 share the budget)
    last_accuracy    : accuracy_all.json dict from the final round (seed for Phase 2)
    """
    section("Phase 1: sliding-window cascade detection")

    p1 = state.setdefault("phase1", {
        "rounds": [],
        "skip_list": list(skip_list),
        "done": False,
        "last_accuracy_dir": None,
        # Directory of the most recent round that had NO cascade.  Phase 2
        # seeds its reference from here: a cascade round's failures are
        # collateral damage, and using them as the baseline blinds the
        # per-case comparison for the rest of the run.
        "last_clean_accuracy_dir": None,
    })

    if p1.get("done"):
        info("Phase 1 already complete (resuming).")
        skip_list = list(p1["skip_list"])
        last_acc = {}
        clean_dir = p1.get("last_clean_accuracy_dir")
        if clean_dir:
            last_acc = load_accuracy_json(Path(clean_dir))
        elif p1.get("last_accuracy_dir"):
            # No clean round was ever seen; do not seed Phase 2 with cascade
            # damage.  An empty seed bootstraps from Phase 2's first clean round.
            warn("Phase 1 recorded no cascade-free round; Phase 2 will start "
                 "with an empty reference.")
        return skip_list, total_rounds_so_far, last_acc

    W = args.window_size
    W_shrink = args.window_shrink
    T = args.window_threshold
    max_p1 = args.max_phase1_rounds
    max_total = args.max_detection_rounds

    p1_rounds_this_iter = len(p1["rounds"])
    skip_list = list(p1["skip_list"])
    last_accuracy: dict = {}
    consecutive_hard_failures = 0

    while (p1_rounds_this_iter < max_p1
           and total_rounds_so_far < max_total):

        round_num = total_rounds_so_far + 1
        round_dir = base_dir / f"round_{round_num}"
        label = f"Phase1 round {round_num}"

        info(f"Phase 1 round {round_num} "
             f"(skip_list={len(skip_list)} ops, "
             f"p1_iter_round={p1_rounds_this_iter + 1}/{max_p1})")

        summary = run_ci(
            runner_script, flaggems_path, gpus, round_dir,
            skip_operators=skip_list,
            args=args,
            label=label,
        )

        total_rounds_so_far += 1
        p1_rounds_this_iter += 1

        # summary may be None (hard failure) - record and retry next round.
        stdout_log = round_dir / "pytest_stdout.log"

        if summary is None:
            consecutive_hard_failures += 1
            warn(f"Round {round_num}: ci_run_tests produced no summary "
                 f"(consecutive hard failures: {consecutive_hard_failures}).")
            if consecutive_hard_failures >= args.max_consecutive_failures:
                raise RuntimeError(
                    f"Aborting: {consecutive_hard_failures} consecutive rounds "
                    "produced no summary. Check the environment (FlagTree "
                    "install, FlagGems checkout, GPU visibility) and rerun "
                    "with --resume."
                )
            p1["rounds"].append({
                "round_num": round_num,
                "trigger_file": None,
                "ops_added": [],
                # "no_summary": ci_run_tests.py exited without producing
                # summary.json — an environment/hardware problem, not a result.
                "status": "no_summary",
            })
            save_state(state_path, state)
            continue

        # A round that produced a summary is a healthy round, regardless of
        # whether the tests passed: reset the hard-failure streak.
        consecutive_hard_failures = 0

        accuracy = load_accuracy_json(round_dir)
        last_accuracy = accuracy

        # --- completeness check -------------------------------------------
        if not log_is_complete(stdout_log):
            warn(f"Round {round_num}: pytest_stdout.log appears incomplete "
                 "(process may have been killed).")
            last_file = last_file_in_log(stdout_log)
            new_ops: Set[str] = set()
            if last_file:
                marks = load_marks_yaml(round_dir)
                new_ops = get_ops_from_marks(marks, last_file) - set(skip_list)
                if new_ops:
                    info(f"  Adding ops from killed-run last file "
                         f"'{last_file}': {sorted(new_ops)}")
                    skip_list.extend(sorted(new_ops))
                    p1["skip_list"] = list(skip_list)
            p1["rounds"].append({
                "round_num": round_num,
                "trigger_file": last_file,
                "ops_added": sorted(new_ops),
                # "killed": summary.json exists but pytest_stdout.log is
                # truncated (missing the [100%] or summary line), so the run
                # died mid-suite. The last file seen is the likely culprit.
                "status": "killed",
            })
            p1["last_accuracy_dir"] = str(round_dir)
            save_state(state_path, state)
            continue

        # --- parse case sequence -------------------------------------------
        cases = parse_stdout_cases(stdout_log)
        if not cases:
            warn(f"Round {round_num}: no cases parsed from pytest_stdout.log.")
            p1["rounds"].append({
                "round_num": round_num,
                "trigger_file": None,
                "ops_added": [],
                "status": "no_cases",
            })
            p1["last_accuracy_dir"] = str(round_dir)
            save_state(state_path, state)
            break

        # --- cascade scan ---------------------------------------------------
        trigger_file = scan_cascade(
            cases, accuracy, set(skip_list),
            W, W_shrink, T,
        )
        new_ops_set: Set[str] = set()
        if trigger_file:
            new_ops_set = get_ops_for_file(accuracy, trigger_file) - set(skip_list)

        p1["rounds"].append({
            "round_num": round_num,
            "trigger_file": trigger_file,
            "ops_added": sorted(new_ops_set),
            # "ok": the run completed and was analysed.
            "status": "ok",
        })
        p1["last_accuracy_dir"] = str(round_dir)

        if trigger_file:
            info(f"  Trigger: '{trigger_file}' → adding {sorted(new_ops_set)}")
            skip_list.extend(sorted(new_ops_set))
            p1["skip_list"] = list(skip_list)
            save_state(state_path, state)
        else:
            # A cascade-free round: this is the only kind of result Phase 2 may
            # use as its flakiness baseline.
            p1["last_clean_accuracy_dir"] = str(round_dir)
            info(f"  No sliding-window trigger found. Phase 1 complete.")
            p1["skip_list"] = list(skip_list)
            p1["done"] = True
            save_state(state_path, state)
            break

    else:
        # Exited by round limit.
        warn(f"Phase 1 hit its round limit after {p1_rounds_this_iter} rounds "
             f"without converging: a cascade was still detectable on the last "
             f"round. Detection is INCOMPLETE — the skip list may be missing "
             f"cascade sources. Raise --max-phase1-rounds and rerun with "
             f"--resume, or inspect the phase1/round_* logs.")
        p1["done"] = True
        p1["skip_list"] = list(skip_list)
        save_state(state_path, state)

    return skip_list, total_rounds_so_far, last_accuracy


# ---------------------------------------------------------------------------
# Phase 2: flaky rolling detection
# ---------------------------------------------------------------------------

def phase2_flaky(
    runner_script: Path,
    flaggems_path: Path,
    gpus: str,
    base_dir: Path,
    args,
    skip_list: List[str],
    seed_accuracy: dict,
    state: dict,
    state_path: Path,
    total_rounds_so_far: int,
) -> Tuple[List[str], int]:
    """
    Rolling flaky detection.

    Compare each round's accuracy_all.json against the previous round's results.
    Any test case that transitions from non-fail to fail triggers adding its
    file's operators to the skip list and resets the consecutive-stable counter.
    When the counter reaches --flaky-rounds, the skip list is stable and we exit.

    Returns
    -------
    skip_list    : final (stable) skip list
    total_rounds : updated round counter
    """
    section("Phase 2: flaky rolling detection")

    p2 = state.setdefault("phase2", {
        "rounds": [],
        "skip_list": list(skip_list),
        "consecutive_stable": 0,
        "reference_results": {k: v.get("result") for k, v in seed_accuracy.items()},
        "done": False,
    })

    if p2.get("done"):
        info("Phase 2 already complete (resuming).")
        return list(p2["skip_list"]), total_rounds_so_far

    flaky_rounds_target = args.flaky_rounds
    max_total = args.max_detection_rounds

    skip_list = list(p2["skip_list"])
    consecutive_stable: int = p2["consecutive_stable"]
    reference_results: Dict[str, str] = p2["reference_results"]
    consecutive_hard_failures = 0

    while (consecutive_stable < flaky_rounds_target
           and total_rounds_so_far < max_total):

        round_num = total_rounds_so_far + 1
        round_dir = base_dir / f"round_{round_num}"
        label = f"Phase2 round {round_num} (stable={consecutive_stable}/{flaky_rounds_target})"

        info(f"Phase 2 round {round_num}: "
             f"consecutive_stable={consecutive_stable}/{flaky_rounds_target}, "
             f"skip_list={len(skip_list)} ops")

        summary = run_ci(
            runner_script, flaggems_path, gpus, round_dir,
            skip_operators=skip_list,
            args=args,
            label=label,
        )

        total_rounds_so_far += 1
        stdout_log = round_dir / "pytest_stdout.log"

        if summary is None:
            consecutive_hard_failures += 1
            warn(f"Round {round_num}: no summary produced; not counting as stable "
                 f"(consecutive hard failures: {consecutive_hard_failures}).")
            if consecutive_hard_failures >= args.max_consecutive_failures:
                raise RuntimeError(
                    f"Aborting: {consecutive_hard_failures} consecutive rounds "
                    "produced no summary. Check the environment (FlagTree "
                    "install, FlagGems checkout, GPU visibility) and rerun "
                    "with --resume."
                )
            p2["rounds"].append({
                "round_num": round_num,
                "new_ops": [],
                "consecutive_stable": consecutive_stable,
                "status": "no_summary",
            })
            save_state(state_path, state)
            continue

        consecutive_hard_failures = 0

        accuracy = load_accuracy_json(round_dir)

        # --- completeness check -------------------------------------------
        if not log_is_complete(stdout_log):
            warn(f"Round {round_num}: log incomplete (process killed).")
            last_file = last_file_in_log(stdout_log)
            new_ops_set: Set[str] = set()
            if last_file:
                marks = load_marks_yaml(round_dir)
                new_ops_set = get_ops_from_marks(marks, last_file) - set(skip_list)
            if new_ops_set:
                info(f"  Killed-run: adding {sorted(new_ops_set)} from '{last_file}'")
                skip_list.extend(sorted(new_ops_set))
                consecutive_stable = 0
                reference_results = {}
            p2["skip_list"] = list(skip_list)
            p2["consecutive_stable"] = consecutive_stable
            p2["reference_results"] = reference_results
            p2["rounds"].append({
                "round_num": round_num,
                "new_ops": sorted(new_ops_set),
                "consecutive_stable": consecutive_stable,
                "status": "killed",
            })
            save_state(state_path, state)
            continue

        # --- cascade check --------------------------------------------------
        # A random corruption source can fire at any time, not just during
        # Phase 1.  Run the same scan here first: without it, one cascade round
        # would be misread by the per-case comparison below as hundreds of
        # independently-flaky operators.
        cases = parse_stdout_cases(stdout_log)
        trigger_file = scan_cascade(
            cases, accuracy, set(skip_list),
            args.window_size, args.window_shrink, args.window_threshold,
        )

        if trigger_file:
            cascade_ops = get_ops_for_file(accuracy, trigger_file) - set(skip_list)
            info(f"  Cascade detected at '{trigger_file}' → "
                 f"adding {sorted(cascade_ops)}")
            skip_list.extend(sorted(cascade_ops))
            consecutive_stable = 0
            # reference_results is deliberately left untouched: this round's
            # failures are cascade damage, not flakiness.  Storing them would
            # make the next round blind to genuine failures of the same cases.
            p2["skip_list"] = list(skip_list)
            p2["consecutive_stable"] = consecutive_stable
            p2["rounds"].append({
                "round_num": round_num,
                "new_ops": sorted(cascade_ops),
                "consecutive_stable": consecutive_stable,
                "status": "cascade",
            })
            save_state(state_path, state)
            continue

        if not cases:
            warn(f"Round {round_num}: no cases parsed from pytest_stdout.log; "
                 "round ignored (counter and reference left unchanged).")
            p2["rounds"].append({
                "round_num": round_num,
                "new_ops": [],
                "consecutive_stable": consecutive_stable,
                "status": "no_cases",
            })
            save_state(state_path, state)
            continue

        # --- compare against reference ------------------------------------
        current_results: Dict[str, str] = {
            k: v.get("result", "") for k, v in accuracy.items()
        }

        new_flaky_ops: Set[str] = set()
        for case, prev_result in reference_results.items():
            cur_result = current_results.get(case)
            # Any non-fail → fail transition is a flaky corruption signal.
            if prev_result != "failed" and cur_result == "failed":
                fname = case.split("::")[0].split("/")[-1]
                new_flaky_ops |= get_ops_for_file(accuracy, fname) - set(skip_list)

        if new_flaky_ops:
            info(f"  Flaky ops detected: {sorted(new_flaky_ops)}")
            skip_list.extend(sorted(new_flaky_ops))
            consecutive_stable = 0
        else:
            consecutive_stable += 1
            info(f"  No new flaky ops. consecutive_stable → {consecutive_stable}")

        # Slide reference forward regardless of whether ops were added.
        reference_results = current_results

        p2["skip_list"] = list(skip_list)
        p2["consecutive_stable"] = consecutive_stable
        p2["reference_results"] = reference_results
        p2["rounds"].append({
            "round_num": round_num,
            "new_ops": sorted(new_flaky_ops),
            "consecutive_stable": consecutive_stable,
            "status": "ok",
        })
        save_state(state_path, state)

    if consecutive_stable >= flaky_rounds_target:
        info(f"Phase 2 complete: {consecutive_stable} consecutive stable rounds.")
    else:
        warn(f"Phase 2 hit max-detection-rounds ({max_total}); "
             "skip list may not be fully stable.")

    p2["done"] = True
    p2["skip_list"] = list(skip_list)
    save_state(state_path, state)
    return skip_list, total_rounds_so_far


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a FlagGems CI skip list by detecting cascade-failure "
                    "and flaky operators. Optionally clones and builds FlagTree and "
                    "the operator library first.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run from scratch
  python generate_skip_list.py --backend xpu \\
      --flagtree-ref main --flaggems-ref v5.5.0.dev1 \\
      --build-command "source ~/env-3.6.sh && MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v" \\
      --pre-command "source ~/env.sh" --gpus 0 --first-parameter-only

  # Detect only (FlagTree installed, FlagGems already cloned)
  python generate_skip_list.py --backend xpu \\
      --flaggems-path /root/flaggems_ci/FlagGems/test_xxx --gpus 0

  # Resume an interrupted run
  python generate_skip_list.py --backend xpu --flaggems-path /path/to/FlagGems --resume
        """,
    )

    # --- Phase 0: clone / build args ------------------------------------
    parser.add_argument(
        "--flagtree-ref",
        help="Branch or tag of flagos-ai/FlagTree to clone and build. "
             "If omitted, FlagTree is assumed already installed.",
    )
    flaggems_src = parser.add_mutually_exclusive_group(required=True)
    flaggems_src.add_argument(
        "--flaggems-ref",
        help="Tag or branch of FlagGems to clone. The URL to clone from is "
             "--flaggems-url.\n"
             "When --repo-kind is flaggems-vllm this is the flag_gems "
             "DEPENDENCY (flaggems_vllm imports it); the library actually "
             "under test is selected by --flaggems-vllm-ref/--flaggems-vllm-path.",
    )
    flaggems_src.add_argument(
        "--flaggems-path", type=Path,
        help="Path to an already-cloned FlagGems checkout. "
             "Mutually exclusive with --flaggems-ref.",
    )
    parser.add_argument(
        "--flaggems-url",
        default=DEFAULT_FLAGGEMS_URL,
        help=f"Git URL of FlagGems to clone when --flaggems-ref is used. "
             f"Default: {DEFAULT_FLAGGEMS_URL}",
    )

    # FlagGems-vllm: only meaningful with --repo-kind flaggems-vllm. Required
    # there, ignored otherwise. Same ref/path symmetry as FlagGems above.
    flaggems_vllm_src = parser.add_mutually_exclusive_group(required=False)
    flaggems_vllm_src.add_argument(
        "--flaggems-vllm-ref",
        help="Tag or branch of FlagGems-vllm to clone. Required when "
             "--repo-kind is flaggems-vllm. The URL is --flaggems-vllm-url.",
    )
    flaggems_vllm_src.add_argument(
        "--flaggems-vllm-path", type=Path,
        help="Path to an already-cloned FlagGems-vllm checkout. "
             "Mutually exclusive with --flaggems-vllm-ref.",
    )
    parser.add_argument(
        "--flaggems-vllm-url",
        default=DEFAULT_FLAGGEMS_VLLM_URL,
        help=f"Git URL of FlagGems-vllm to clone when --flaggems-vllm-ref is "
             f"used. Default: {DEFAULT_FLAGGEMS_VLLM_URL}",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Directory for cloned repos and this run's output. "
             "Default: ~/flaggems_ci/skip_list/<repo-kind>_skip_list_"
             "<YYYYMMDD_HHMMSS>, stamped once at startup so concurrent or "
             "successive runs never share a directory. Required when --resume "
             "is used, so the interrupted run's directory is reused instead of "
             "a new timestamp being minted.",
    )
    parser.add_argument(
        "--build-command",
        help="Shell command to build and install FlagTree (run inside the clone "
             "with FLAGTREE_BACKEND=<backend> exported). "
             "Required when --flagtree-ref is used.",
    )
    parser.add_argument(
        "--skip-clone", action="store_true",
        help="Skip git clone steps; use existing checkouts in --work-dir.",
    )

    # --- detection args -------------------------------------------------
    parser.add_argument(
        "--backend", required=True,
        help="Backend name (e.g. xpu). Sets FLAGTREE_BACKEND and names the "
             "output YAML (<backend>.yaml).",
    )
    parser.add_argument(
        "--gpus", default="0",
        help='GPU IDs, comma-separated or "all". Default: "0".',
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for all test outputs and state. "
             "Default: <work-dir>/output",
    )
    parser.add_argument(
        "--window-size", type=int, default=10,
        help="W: initial look-ahead window size for cascade detection. Default: 10.",
    )
    parser.add_argument(
        "--window-shrink", type=int, default=2,
        help="Minimum W when tail-shrinking applies (last W cases fail>50%%). Default: 2.",
    )
    parser.add_argument(
        "--window-threshold", type=float, default=0.8,
        help="T: fail rate in look-ahead window to trigger a skip. Default: 0.8.",
    )
    parser.add_argument(
        "--flaky-rounds", type=int, default=20,
        help="Consecutive clean rounds with no new findings required to declare "
             "the skip list done. A cascade or a new flaky operator resets the "
             "counter. Default: 20.",
    )
    parser.add_argument(
        "--max-phase1-rounds", type=int, default=20,
        help="Maximum rounds in a single Phase 1 iteration. Default: 20.",
    )
    parser.add_argument(
        "--max-detection-rounds", type=int, default=60,
        help="Total round budget shared by Phase 1 and Phase 2. Default: 60.",
    )
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="Abort when this many consecutive rounds produce no summary.json "
             "(environment problem rather than a test result). Default: 3.",
    )
    parser.add_argument(
        "--repo-kind", type=parse_repo_kind, default="flaggems",
        help="Operator library under test, passed to ci_run_tests.py to select "
             "the Python package it imports. One of: flaggems, flaggems-vllm "
             "(case-insensitive). Default: flaggems.",
    )
    parser.add_argument(
        "--first-parameter-only", action="store_true",
        help="Pass --first-parameter-only to ci_run_tests.py for faster runs.",
    )
    parser.add_argument(
        "--pre-command",
        help="Shell command executed once at startup, before Phase 0 (clone/"
             "build) and before any test round. Its exported environment is "
             "captured and propagated to the build and to every round "
             "(e.g. 'source ~/env.sh && source ~/env-3.6.sh').",
    )
    parser.add_argument(
        "--per-round-command", default=None,
        help="Shell command executed before every test round, on top of the "
             "environment prepared by --pre-command. Its exported variables "
             "apply to that round only. Default: empty (nothing extra runs).",
    )
    parser.add_argument(
        "--runner-script", type=Path, default=None,
        help="Path to ci_run_tests.py. Defaults to the copy next to this script.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a previous interrupted run using saved state.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved configuration and exit without running anything.",
    )

    args = parser.parse_args()

    # --- Resolve work-dir / output-dir ------------------------------------
    # A fresh timestamped directory per run keeps concurrent and successive
    # runs apart.  Runs are grouped under ~/flaggems_ci/skip_list/ so repeated
    # generations do not litter ~/flaggems_ci itself; the prefix is the
    # repo-kind (lowercase), so flaggems and flaggems-vllm runs stay apart.
    # --resume must name the interrupted run's directory, because a newly
    # minted timestamp would point at an empty tree with no state.json.
    if args.work_dir is None:
        if args.resume:
            error("--resume requires --work-dir to name the interrupted run's "
                  "directory (e.g. "
                  "~/flaggems_ci/skip_list/flaggems_skip_list_20261118_133721).")
            sys.exit(1)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.work_dir = Path(
            f"~/flaggems_ci/skip_list/{args.repo_kind}_skip_list_{stamp}"
        ).expanduser()
    else:
        args.work_dir = args.work_dir.expanduser()

    if args.output_dir is None:
        args.output_dir = args.work_dir / "output"
    else:
        args.output_dir = args.output_dir.expanduser()

    if args.flaggems_path:
        args.flaggems_path = args.flaggems_path.expanduser()

    if args.runner_script is None:
        args.runner_script = Path(__file__).parent / "ci_run_tests.py"
    if not args.runner_script.exists():
        error(f"ci_run_tests.py not found at {args.runner_script}")
        sys.exit(1)

    if args.flagtree_ref and not args.build_command:
        error("--build-command is required when --flagtree-ref is specified.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        info("Dry-run mode: printing resolved configuration and exiting.")
        info(f"  backend:              {args.backend}")
        info(f"  flagtree-ref:         {args.flagtree_ref or '(not set, using installed)'}")
        info(f"  flaggems-ref:         {args.flaggems_ref or '(not set)'}")
        info(f"  flaggems-url:         {args.flaggems_url}")
        info(f"  flaggems-path:        {args.flaggems_path or '(not set)'}")
        info(f"  flaggems-vllm-ref:    {args.flaggems_vllm_ref or '(not set)'}")
        info(f"  flaggems-vllm-url:    {args.flaggems_vllm_url}")
        info(f"  flaggems-vllm-path:   {args.flaggems_vllm_path or '(not set)'}")
        info(f"  work-dir:             {args.work_dir}")
        info(f"  output-dir:           {args.output_dir}")
        info(f"  gpus:                 {args.gpus}")
        info(f"  window:               W={args.window_size}, shrink={args.window_shrink}, "
             f"threshold={args.window_threshold}")
        info(f"  flaky-rounds:         {args.flaky_rounds}")
        info(f"  max-phase1-rounds:    {args.max_phase1_rounds}")
        info(f"  max-detection-rounds: {args.max_detection_rounds}")
        info(f"  repo-kind:            {args.repo_kind}")
        info(f"  first-param-only:     {args.first_parameter_only}")
        info(f"  runner-script:        {args.runner_script}")
        sys.exit(0)

    # From here on, duplicate everything to <output-dir>/test_output.log.
    # Must happen after output_dir exists and after the dry-run early exit.
    setup_logging(args.output_dir)

    state_path = args.output_dir / "state.json"
    if args.resume and state_path.exists():
        info(f"Resuming from {state_path}")
        state = load_state(state_path)
    else:
        if args.resume:
            warn("--resume given but no state file found; starting fresh.")
        state = {}
        save_state(state_path, state)

    section("FlagGems skip list generator")
    info(f"Backend:              {args.backend}")
    info(f"Window:               W={args.window_size}, shrink={args.window_shrink}, "
         f"threshold={args.window_threshold}")
    info(f"Flaky rounds:         {args.flaky_rounds}")
    info(f"Max phase1 rounds:    {args.max_phase1_rounds}")
    info(f"Max detection rounds: {args.max_detection_rounds}")
    info(f"Output dir:           {args.output_dir}")

    # --- Pre-command (once, before anything else) -------------------------
    # Everything downstream (build, every test round) inherits this
    # environment. --per-round-command, if any, is layered on top per round.
    section("Pre-command")
    base_env = _capture_env(args.pre_command)
    args.base_env = base_env
    info("Environment prepared"
         + (" (pre-command applied)." if args.pre_command else " (no pre-command)."))
    if args.per_round_command:
        info(f"Per-round command: {args.per_round_command}")

    # --- Phase 0 ----------------------------------------------------------
    under_test_path, dependency_path = phase0_setup(
        work_dir=args.work_dir,
        backend=args.backend,
        flagtree_ref=args.flagtree_ref,
        flaggems_ref=args.flaggems_ref,
        flaggems_url=args.flaggems_url,
        flaggems_path_arg=args.flaggems_path,
        repo_kind=args.repo_kind,
        flaggems_vllm_ref=args.flaggems_vllm_ref,
        flaggems_vllm_url=args.flaggems_vllm_url,
        flaggems_vllm_path_arg=args.flaggems_vllm_path,
        build_command=args.build_command or "",
        env=base_env,
        skip_clone=args.skip_clone,
        state=state,
        state_path=state_path,
    )

    # Set PYTHONPATH so ci_run_tests.py can import the operator library without
    # installing it.  This must be written into BOTH os.environ and base_env:
    # rounds run with base_env (captured before Phase 0), so updating only
    # os.environ would leave the test subprocesses unable to import flag_gems.
    #
    # Both checkouts go on PYTHONPATH in vllm mode: flaggems_vllm imports
    # flag_gems at module level, so the dependency has to be importable too.
    # The library under test goes first so its own package wins if both ever
    # contain the same name.
    src_dirs = [under_test_path / "src"]
    if dependency_path is not None:
        src_dirs.append(dependency_path / "src")
    new_pp = os.pathsep.join(str(d) for d in src_dirs)

    existing_pp = base_env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if existing_pp:
        new_pp = f"{new_pp}{os.pathsep}{existing_pp}"
    base_env["PYTHONPATH"] = new_pp
    os.environ["PYTHONPATH"] = new_pp
    args.base_env = base_env
    info(f"PYTHONPATH includes {', '.join(str(d) for d in src_dirs)}")

    # ci_run_tests.py takes the checkout whose tests/ and conf/ drive the run.
    args.flaggems_path_resolved = under_test_path

    info(f"repo-kind for ci_run_tests.py: {args.repo_kind}")

    skip_list: List[str] = list(state.get("skip_list", []))
    total_rounds: int = state.get("total_rounds", 0)

    # --- Phase 1 ----------------------------------------------------------
    skip_list, total_rounds, last_accuracy = phase1_sliding_window(
        runner_script=args.runner_script,
        flaggems_path=under_test_path,
        gpus=args.gpus,
        base_dir=args.output_dir / "phase1",
        args=args,
        skip_list=skip_list,
        state=state,
        state_path=state_path,
        total_rounds_so_far=total_rounds,
    )
    state["skip_list"] = list(skip_list)
    state["total_rounds"] = total_rounds
    save_state(state_path, state)

    # --- Phase 2 ----------------------------------------------------------
    skip_list, total_rounds = phase2_flaky(
        runner_script=args.runner_script,
        flaggems_path=under_test_path,
        gpus=args.gpus,
        base_dir=args.output_dir / "phase2",
        args=args,
        skip_list=skip_list,
        seed_accuracy=last_accuracy,
        state=state,
        state_path=state_path,
        total_rounds_so_far=total_rounds,
    )
    state["skip_list"] = list(skip_list)
    state["total_rounds"] = total_rounds
    state["final_skip_list"] = sorted(skip_list)
    save_state(state_path, state)

    final_path = args.output_dir / f"{args.backend}.yaml"
    write_skip_yaml(skip_list, final_path)

    section("Done")
    info(f"Final skip list ({len(skip_list)} ops): {sorted(skip_list)}")
    info(f"Written to: {final_path}")
    info(f"Total rounds used: {total_rounds}")
    info(f"Copy to .github/workflows/gems/<tag>/skip_list/{args.backend}.yaml "
         "and commit.")


if __name__ == "__main__":
    main()
