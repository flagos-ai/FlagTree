# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Scalar TensorType payloads must preserve their raw element bytes."""

import json

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager, DumpScope
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


@pytest.mark.parametrize("dtype", [dtype.value for dtype in fm.DType])
def test_rank_zero_tensor_dump_keeps_shape_and_exact_bytes(tmp_path, dtype):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type(dtype, ()), id="value")
            output = fm.F.tensors.cast(value, dtype=dtype, name="output")
            self.function("main", (value,), (output,))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    value = torch.tensor(1, dtype=getattr(torch, dtype))
    with DumpScope(DumpManager(tmp_path, DumpFlags.EVALUATOR).root):
        output, = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value})
    directory = tmp_path / "Evaluate" / "Run0000" / "0000_output"
    result = json.loads((directory / "result.json").read_text())["result"]
    assert result["shape"] == []
    assert result["nbytes"] == value.element_size()
    assert (directory / result["payload"]).read_bytes() == output.reshape(-1).view(torch.uint8).numpy().tobytes()
