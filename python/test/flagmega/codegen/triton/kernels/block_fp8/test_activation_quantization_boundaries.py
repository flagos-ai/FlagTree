# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Approximate division must not change the FP8 quantization tie decision."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.selection import apply_plan, override_plan


@pytest.mark.parametrize("family", ("linear", "glu"))
@pytest.mark.parametrize("variant", ("simt", "mma"))
@pytest.mark.parametrize("sign", (-1, 1))
def test_bf16_activation_at_fp8_round_to_even_boundary(tmp_path, family, variant, sign):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")

    class Graph(fm.Module):
        def forward(self):
            x = self.input("value", fm.tensor_type("bfloat16", (1, 128)), id="value")
            w = self.input("weight", fm.tensor_type("float8_e4m3fn", (32, 128)), id="weight")
            s = self.input("scale", fm.tensor_type("float32", (2, 1)), id="scale")
            if family == "linear":
                y = fm.F.math.block_scaled_matmul(x, w, s, weight_block_n=16, weight_block_k=128, name="projection")
            else:
                y = fm.F.nn.matmul_glu(x, w, w, s, s, activation="silu", weight_block_n=16,
                                      weight_block_k=128, name="projection")
            self.function("main", (x, w, s), (y,))

    compiler = Compiler()
    proposed = compiler.compile(Graph(dialect="ntt", stage="frozen_constants", entry="main").build(),
                                stop_after="propose-tir").module
    candidate = f"tir.{'block_fp8' if family == 'linear' else 'matmul_glu'}.{variant}"
    choices = [(point.id, candidate) for point in proposed.selection_points
               if any(value.id == candidate for value in point.candidates)]
    assert len(choices) == 1
    selected = apply_plan(proposed, override_plan(proposed, choices))
    path = write_artifact(compiler.compile(selected).module, tmp_path / "tie",
                          target="nvidia-sm90", emit_executable=True)
    runtime = load(path, device="cuda:0")
    try:
        x = torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda")
        x[0, 0], x[0, 1] = 1.203125, sign * .81640625
        weights = torch.zeros((32, 128), dtype=torch.float32, device="cuda")
        weights[:, 1] = 1
        weights = weights.to(torch.float8_e4m3fn)
        scales = torch.ones((2, 1), device="cuda")
        # scale = 1.203125 / 448 = 11/4096 exactly; x[1]/scale is exactly
        # +/-304, halfway between FP8 +/-288 and +/-320. Ties round to 320.
        projected = torch.full((1, 32), sign * .859375, dtype=torch.bfloat16, device="cuda")
        expected = projected if family == "linear" else torch.nn.functional.silu(projected) * projected
        output = torch.empty_like(expected)
        runtime.prepare(x, weights, scales, output=output)
        runtime.run_into(output, x, weights, scales)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
