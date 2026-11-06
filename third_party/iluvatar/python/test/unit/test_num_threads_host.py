"""The native COREX helper must use the target's physical warp size."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize("warp_size", [32, 64])
def test_num_threads_uses_target_warp_size(num_warps, warp_size):
    source = Path(os.environ.get("FLAGTREE_NATIVE_THREADS_SOURCE", str(
        Path(__file__).resolve().parents[3] / "language/corex/utils.py")))
    spec = importlib.util.spec_from_file_location("native_corex_thread_helpers", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    semantic = SimpleNamespace(builder=SimpleNamespace(options=SimpleNamespace(
        num_warps=num_warps, warp_size=warp_size)))
    assert module.num_threads(_semantic=semantic).value == num_warps * warp_size
