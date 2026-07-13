# SPDX-License-Identifier: MIT
"""§7 Phase 1 smoke：带 collect 的 kernel 编译并在有 CUDA 时运行一行。

无 CUDA / torch 时跳过。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_unit = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_module_a_doc", _unit / "_module_a_doc.py")
_mad = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mad)
__doc__ = _mad.extend_doc(__doc__)

torch = pytest.importorskip("torch")
pytest.importorskip("triton.backends.nvidia.driver")

import triton
import triton.language as tl
from triton.runtime import debugger


@pytest.mark.module_a
@pytest.mark.module_a_smoke
def test_module_a_phase1_collect_kernel_compiles_and_runs(fresh_triton_cache):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    _ = fresh_triton_cache

    def hook(metadata, stream, launch_metadata, kernel_args):
        del metadata, stream, launch_metadata, kernel_args
        return debugger.PreparedKernelLaunch(kernel_args=(0, ), finalize=lambda error: None)

    debugger.clear_launch_prepare_hook()
    debugger.register_launch_prepare_hook(hook)

    try:

        @triton.jit
        def kernel(out_ptr, n: tl.constexpr):
            tl.debug_collect_start(level=1)
            tl.store(out_ptr + tl.arange(0, n), tl.full([n], 1.0, dtype=tl.float32))
            tl.debug_collect_end()

        n = 16
        out = torch.empty(n, device="cuda", dtype=torch.float32)
        grid = (1, )
        kernel[grid](out, n)
        assert torch.allclose(out, torch.ones_like(out))
    finally:
        debugger.clear_launch_prepare_hook()
