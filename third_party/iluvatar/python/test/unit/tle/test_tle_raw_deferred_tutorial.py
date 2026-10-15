"""Run the trunk deferred tle_raw tutorial on corex.

The autotuned matmul in python/tutorials/tle/raw/cuda/03-matrix-multiplication-smem-defered.py
is the trunk's end-to-end check that a deferred stub survives tracing and gets
filled at make_llir. It is the case deferred exists for -- autotune traces every
config, and only the deferred path compiles the .cu once per source instead of
once per trace -- so iluvatar drives the trunk file directly rather than keeping
a second copy of the same kernel.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

FLAGTREE_ROOT = Path(__file__).resolve().parents[6]
TUTORIAL = FLAGTREE_ROOT / "python/tutorials/tle/raw/cuda/03-matrix-multiplication-smem-defered.py"

SIZE = 512


@pytest.fixture(scope="module")
def tutorial():
    if not TUTORIAL.is_file():
        pytest.skip(f"trunk tutorial not found at {TUTORIAL}")
    spec = importlib.util.spec_from_file_location("tle_raw_deferred_tutorial", TUTORIAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("deferred", [False, True])
def test_tutorial_matmul(tutorial, deferred):
    torch.manual_seed(0)
    a = torch.rand((SIZE, SIZE), device=tutorial.DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((SIZE, SIZE), device=tutorial.DEVICE, dtype=torch.float16) - 0.5
    tutorial.run_case(deferred=deferred, a=a, b=b)
