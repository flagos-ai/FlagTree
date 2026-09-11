import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class DmaResult:
    oob_count: int
    bad_magic_count: int

    @property
    def passed(self) -> bool:
        return self.oob_count == 0 and self.bad_magic_count == 0

    @property
    def detected(self) -> bool:
        return self.oob_count > 0


def _workspace():
    return os.environ.get("WORKSPACE", os.path.expanduser("~/workspace/triton"))


def _parse_kcore_logs(test_name: str) -> DmaResult:
    """Find the latest kcore_log/<id>_<test_name>/ directory and grep t* for DMA errors.

    'fatal error: ddr memory OOB' → oob_count
    'error: ... bad magic'       → bad_magic_count
    """
    import glob

    kcore_dir = os.path.join(_workspace(), "kcore_log")
    target_dir = os.path.join(kcore_dir, test_name)
    if not os.path.isdir(target_dir):
        return DmaResult(oob_count=-1, bad_magic_count=-1)
    oob = 0
    bad_magic = 0
    for logfile in sorted(glob.glob(os.path.join(target_dir, "t*"))):
        try:
            with open(logfile) as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        oob += len(re.findall(r"fatal error: ddr memory OOB", content))
        bad_magic += len(re.findall(r"error:.*bad magic", content))
    return DmaResult(oob_count=oob, bad_magic_count=bad_magic)


@pytest.fixture
def dma_env():
    """Enable DMA checking and logging."""
    old_checking = os.environ.get("TRITON_DMA_CHECKING")
    old_logging = os.environ.get("TRITON_DMA_LOGGING")
    os.environ["TRITON_DMA_CHECKING"] = "1"
    os.environ["TRITON_DMA_LOGGING"] = "0"
    yield
    if old_checking is not None:
        os.environ["TRITON_DMA_CHECKING"] = old_checking
    else:
        del os.environ["TRITON_DMA_CHECKING"]
    if old_logging is not None:
        os.environ["TRITON_DMA_LOGGING"] = old_logging
    else:
        del os.environ["TRITON_DMA_LOGGING"]


def run_kernel_script(script_content: str, tmp_path: Path, caller: str = "") -> DmaResult:
    """Write script to temp file, run with python, extract kcore logs, parse DMA result."""
    workspace = _workspace()
    script_file = tmp_path / "test_kernel.py"
    script_file.write_text(script_content)

    # 0. Check device health before running the kernel
    smi = subprocess.run(["tsm_smi"], capture_output=True, text=True, timeout=30)
    if smi.returncode != 0:
        RED = "\033[31m"
        RST = "\033[0m"
        pytest.exit(f"{RED}tsm_smi failed (rc={smi.returncode}). Device may be in bad state. Aborting all tests.{RST}\n"
                    f"{RED}stdout: {smi.stdout}{RST}\n{RED}stderr: {smi.stderr}{RST}")

    # 1. Run the kernel
    proc = subprocess.run(
        [sys.executable, str(script_file)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env=os.environ.copy(),
    )
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)

    # 2. Extract with test name suffix, then parse
    subprocess.run(['sleep', '3'])
    extract_sh = os.path.join(workspace, "third_party/tsingmicro/scripts/extract_kcore.sh")
    proc = subprocess.run(["bash", extract_sh, "all", "-s", caller], capture_output=True, text=True, timeout=60,
                          cwd=workspace)

    # if proc.stdout:
    #     print(proc.stdout, flush=True)
    # if proc.stderr:
    #     print(proc.stderr, flush=True)

    return _parse_kcore_logs(caller)
