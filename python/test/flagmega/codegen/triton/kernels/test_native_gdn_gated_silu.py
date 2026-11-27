# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep the CUDA SiLU division at a native BF16 rounding boundary."""

import json
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load


def _module():
    class Graph(fm.Module):
        def forward(self):
            core = self.input("core", fm.tensor_type("bfloat16", (1, 128)))
            z = self.input("z", fm.tensor_type("float32", (1, 128)))
            weight = self.input("weight", fm.tensor_type("float32", (128,)))
            zero = self.input("zero", weight.type)
            stats = fm.F.nn.norm_stats(core, axis=-1, use_mean=False)
            normalized = fm.F.nn.norm_apply(core, stats, weight, zero, axis=-1,
                epsilon=1e-6, use_mean=False, input_dtype="float32", output_dtype="float32")
            gate = fm.F.math.silu(z, math_mode="accurate")
            result = fm.F.tensors.cast(fm.F.math.mul(normalized, gate), dtype="bfloat16")
            self.function("main", (core, z, weight, zero), (result,))
    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_native_gdn_gate_keeps_division_rounding(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    fixture = json.loads((Path(__file__).resolve().parents[3] / "data" /
                          "native_gdn_gated_silu_rounding.json").read_text())
    inputs = {"core": torch.tensor([fixture["core"]], device="cuda", dtype=torch.bfloat16),
              "z": torch.tensor([fixture["z"]], device="cuda"),
              "weight": torch.tensor(fixture["weight"], device="cuda")}
    inputs["zero"] = torch.zeros_like(inputs["weight"])
    expected = torch.tensor([fixture["output"]], device="cuda", dtype=torch.bfloat16)
    module = _module()
    if backend == "eager":
        actual = TorchEvaluator(DictWeightResolver({})).run(module, inputs)[0]
    else:
        compiled = Compiler().compile(module).module
        runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90",
                                     emit_executable=True), device="cuda:0")
        try:
            runtime.prepare(*inputs.values())
            actual = runtime.run(*inputs.values())
            assert runtime.resource_report["spill_bytes"] == 0
        finally:
            runtime.close()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
