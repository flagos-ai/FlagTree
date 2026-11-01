# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native tiled input RMS reduces warp groups before the outer tree."""

import json
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load


def _module():
    class Norm(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("float32", (1, 2048)))
            weight = self.input("weight", fm.tensor_type("float32", (2048,)))
            zero = self.input("zero", weight.type)
            stats = fm.F.nn.norm_stats(x, axis=-1, use_mean=False,
                reduction_order="tile_fma", reduction_group_size=8, reduction_block_size=1024,
                reduction_block_traversal="reverse", reduction_tree_group_size=32)
            result = fm.F.nn.norm_apply(x, stats, weight, zero, axis=-1, use_mean=False,
                epsilon=1e-6, output_dtype="bfloat16", math_mode="accurate")
            self.function("main", (x, weight, zero), (result, stats))
    return Norm(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_native_input_rms_hierarchical_tree(tmp_path, backend):
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_input_norm_hierarchical_tree.json").read_text())
    x = torch.tensor([fixture["input"]], device="cuda")
    weight = torch.tensor(fixture["weight"], device="cuda")
    zero = torch.zeros_like(weight)
    expected = torch.tensor([fixture["output"]], dtype=torch.bfloat16, device="cuda")
    expected_stats = torch.tensor([[[fixture["statistics"]]]], device="cuda")
    module = _module()
    if backend == "eager":
        actual, stats = TorchEvaluator(DictWeightResolver({})).run(module, {
            "x": x, "weight": weight, "zero": zero})
    else:
        compiled = Compiler().compile(module).module
        runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90",
                                     emit_executable=True), device="cuda:0")
        try:
            actual, stats = torch.empty_like(expected), torch.empty_like(expected_stats)
            binding = runtime.buffer_plan.function_map[compiled.entry]
            values = {names[0]: value for (_, names), value in
                      zip(binding.parameters, (x, weight, zero), strict=True)}
            values.update({names[0]: value for (_, names), value in
                           zip(binding.outputs, (actual, stats), strict=True)})
            arguments = tuple(values[a["buffer"]] for a in runtime.external_arguments)
            runtime.prepare(*arguments)
            runtime.run_into(*arguments)
            reports = runtime.resource_report
            reports = reports if isinstance(reports, tuple) else (reports,)
            assert all(r["spill_bytes"] == 0 for r in reports)
        finally:
            runtime.close()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
