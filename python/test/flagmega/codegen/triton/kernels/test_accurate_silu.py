# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An explicit accurate SiLU keeps native exp/div arithmetic through fusion."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("fused", [False, True])
def test_accurate_silu_matches_independent_native_formula(tmp_path, fused):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def reference(X, Y, FUSED: tl.constexpr):
        i = tl.program_id(0) * 128 + tl.arange(0, 128)
        x = tl.load(X + i)
        y = x / (1.0 + libdevice.exp(-x))
        if FUSED:
            y = y * x
        tl.store(Y + i, y)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("float32", (32, 128)), id="x")
            y = fm.F.math.silu(x, math_mode="accurate")
            if fused:
                y = fm.F.math.mul(y, x)
            self.function("main", (x,), (y,))

    source = Graph(dialect="high_level", stage="imported", entry="main").build()
    assert fm.load_module(fm.emit_module(source, tmp_path / "input.py")).semantic_hash == source.semantic_hash
    compiled = Compiler().compile(source).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    x = torch.linspace(-8, 8, 4096, device="cuda").reshape(32, 128)
    actual = torch.empty_like(x)
    expected = torch.empty_like(x)
    reference[(32,)](x, expected, fused, num_warps=4)
    try:
        runtime.prepare(x, output=actual)
        runtime.run_into(actual, x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reports = runtime.resource_report
        reports = reports if isinstance(reports, tuple) else (reports,)
        assert all(r["spill_bytes"] == 0 for r in reports)
    finally:
        runtime.close()
