import subprocess
from pathlib import Path

import triton
import pytest

from python.build_helpers import get_cmake_dir


@pytest.mark.parametrize("input_file", [
    "test/Analysis/atomic-scratch-reuse.mlir",
    "third_party/hcu/test/Analysis/atomic-call-scratch.mlir",
    "third_party/hcu/test/Analysis/atomic-shared-memory.mlir",
    "third_party/hcu/test/Analysis/atomic-loop-scratch.mlir",
    "third_party/hcu/test/Analysis/atomic-scratch-types.mlir",
])
def test_atomic_scratch_reuse(input_file):
    # Use the HCU pass (including AsyncWait preprocessing and membarFilter).
    triton_opt = get_cmake_dir() / "third_party/hcu/bin/triton-opt"
    test = Path(__file__).resolve().parents[5] / input_file
    output = subprocess.check_output([triton_opt, test, "--test-tritonhcugpu-membar", "-verify-each"])
    filecheck = Path(triton.__file__).parent / "FileCheck"
    subprocess.run([filecheck, "--enable-var-scope", test], input=output, check=True)
