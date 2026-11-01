# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native GDN prefill uses eight-column sums and an accurate reciprocal root."""

import json
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load


def _module(*, legacy=False):
    class Norm(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 128)))
            z = self.input("z", x.type)
            weight = self.input("weight", fm.tensor_type("float32", (128,)))
            zero = self.input("zero", weight.type)
            stats = fm.F.nn.norm_stats(x, axis=-1, use_mean=False,
                **({} if legacy else {"reduction_group_size": 8, "reduction_order": "tree"}))
            normalized = fm.F.nn.norm_apply(x, stats, weight, zero, axis=-1,
                use_mean=False, epsilon=1e-6, input_dtype="float32", output_dtype="float32",
                round_before_scale=False, math_mode="fast" if legacy else "accurate")
            gate = fm.F.math.silu(fm.F.tensors.cast(z, dtype="float32"), math_mode="accurate")
            result = fm.F.tensors.cast(fm.F.math.mul(normalized, gate), dtype="bfloat16")
            self.function("main", (x, z, weight, zero), (result, stats))
    return Norm(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_native_gdn_prefill_norm(tmp_path, backend):
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_gdn_prefill_norm.json").read_text())
    x = torch.tensor([fixture["input"]], dtype=torch.bfloat16, device="cuda")
    z = torch.tensor([fixture["z"]], dtype=torch.bfloat16, device="cuda")
    weight = torch.tensor(fixture["weight"], device="cuda")
    zero = torch.zeros_like(weight)
    expected = torch.tensor([fixture["output"]], dtype=torch.bfloat16, device="cuda")
    expected_stats = torch.tensor([[[fixture["statistics"]]]], device="cuda")
    module = _module()
    if backend == "eager":
        evaluator = TorchEvaluator(DictWeightResolver({}))
        values = {"x": x, "z": z, "weight": weight, "zero": zero}
        old, old_stats = evaluator.run(_module(legacy=True), values)
        assert not torch.equal(old, expected)
        assert not torch.equal(old_stats, expected_stats)
        actual, stats = evaluator.run(module, values)
    else:
        compiled = Compiler().compile(module).module
        runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90",
                                     emit_executable=True), device="cuda:0")
        try:
            actual, stats = torch.empty_like(expected), torch.empty_like(expected_stats)
            binding = runtime.buffer_plan.function_map[compiled.entry]
            values = {names[0]: value for (_, names), value in
                      zip(binding.parameters, (x, z, weight, zero), strict=True)}
            values.update({names[0]: value for (_, names), value in
                           zip(binding.outputs, (actual, stats), strict=True)})
            arguments = tuple(values[a["buffer"]] for a in runtime.external_arguments)
            runtime.prepare(*arguments)
            runtime.run_into(*arguments)
            reports = runtime.resource_report
            reports = reports if isinstance(reports, tuple) else (reports,)
            assert all(report["spill_bytes"] == 0 for report in reports)
        finally:
            runtime.close()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
