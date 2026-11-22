"""Device check for integer narrowing followed by cross-warp shared reads."""
from pathlib import Path

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.flagmega.runtime.prepared import ResourceContract, prepare_compiled_kernel


def test_cast_wrap_cross_warp_shared(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Requires an SM120 device")
    source = (Path(__file__).resolve().parents[4]
              / "third_party/tle/test/GPU/membar-index-cast-wrap.mlir")
    path = tmp_path / "counterexample.ttgir"
    path.write_text(source.read_text())
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "fresh-cache"))
    compiled = triton.compile(str(path), target=GPUTarget("cuda", 120, 32),
                              options=dict(num_warps=4, num_ctas=1, ptx_options="--minnctapersm=1"))
    values = torch.empty(128, dtype=torch.int32, device="cuda")
    guarded = torch.empty(144, dtype=torch.int32, device="cuda")
    output = guarded[8:136]
    prepared = prepare_compiled_kernel(compiled, (values, output), (), grid=(1,),
                                      contract=ResourceContract(4, 1, shared_memory_capacity_bytes=102400))
    for repetition in range(64):
        oracle = torch.arange(128, dtype=torch.int32) * 7919 + repetition * 104729
        values.copy_(oracle)
        guarded.fill_(-123456789)
        prepared.launch()
        torch.cuda.synchronize()
        assert torch.equal(output.cpu(), oracle)
        assert torch.all(guarded[:8] == -123456789)
        assert torch.all(guarded[136:] == -123456789)
