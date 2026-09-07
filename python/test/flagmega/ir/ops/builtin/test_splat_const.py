# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.constants import ConstnessAnalysis, freeze_constant_islands


class _SplatModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="nn", stage="imported", entry="main")

    def forward(self):
        value = fm.F.builtin.splat_const(
            fm.tensor_type("bfloat16", [2, 8]), 0.25, name="value")
        self.function("main", (), (value,))


def test_splat_const_is_editable_compressed_constant_and_freezes_to_asset(tmp_path):
    module = _SplatModule().build()
    assert ConstnessAnalysis.analyze(module).constants == {"value"}
    assert module.node_map["value"].attrs == {"value": 0.25}

    actual = TorchEvaluator(DictWeightResolver({})).run(module, {})[0]
    torch.testing.assert_close(
        actual,
        torch.full((2, 8), 0.25, dtype=torch.bfloat16),
    )

    checkpoint = fm.emit_module(module, tmp_path / "splat.py")
    source = checkpoint.read_text()
    assert "F.builtin.splat_const" in source
    assert "[0.25" not in source

    frozen = freeze_constant_islands(module)
    assert frozen.node_map["value"].op == "builtin.const_asset"
    assert frozen.constant_recipes[0].nodes[0].op == "builtin.splat_const"
