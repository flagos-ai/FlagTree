import subprocess
from pathlib import Path

import triton

from python.build_helpers import get_cmake_dir


def test_atomic_scratch_reuse():
    # Use the HCU pass (including AsyncWait preprocessing and membarFilter).
    triton_opt = get_cmake_dir() / "third_party/hcu/bin/triton-opt"
    test = Path(__file__).resolve().parents[5] / "test/Analysis/atomic-scratch-reuse.mlir"
    output = subprocess.check_output([triton_opt, test, "--test-tritonhcugpu-membar", "-verify-each"])
    filecheck = Path(triton.__file__).parent / "FileCheck"
    subprocess.run([filecheck, "--enable-var-scope", test], input=output, check=True)
