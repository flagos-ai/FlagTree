# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.target_independent import decompose_complex_ops


@pytest.mark.parametrize("kind", ["round_trip", "matmul"])
def test_strict_contract_preserves_observable_bf16_rounding(kind):
    class Graph(fm.Module):
        def forward(self):
            dtype = "float32" if kind == "round_trip" else "bfloat16"
            x = self.input("x", fm.tensor_type(dtype, (1, 16)))
            if kind == "round_trip":
                rounded = fm.F.tensors.cast(x, "bfloat16")
                params = (x,)
            else:
                w = self.input("w", fm.tensor_type(dtype, (1, 16)))
                rounded = fm.F.math.matmul(x, w, transpose_b=True)
                params = (x, w)
            result = fm.F.tensors.cast(rounded, "float32")
            self.function("main", params, (result,))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    module = replace(module, metadata={**module.metadata, "floating_point_contract": "strict"})
    inputs = {"x": torch.full((1, 16), 1.001 if kind == "round_trip" else 1.0078125)}
    if kind == "matmul":
        inputs = {"x": inputs["x"].bfloat16(), "w": inputs["x"].bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected = evaluator.run(module, inputs)[0]
    optimized = decompose_complex_ops(module)
    actual = evaluator.run(optimized, inputs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_strict_contract_still_folds_lossless_widening_round_trip():
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (16,)), id="x")
            result = fm.F.tensors.cast(fm.F.tensors.cast(x, "float32"), "bfloat16")
            self.function("main", (x,), (result,))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    module = replace(module, metadata={**module.metadata, "floating_point_contract": "strict"})
    assert decompose_complex_ops(module).functions[0].outputs == ("x",)


@pytest.mark.parametrize("source,intermediate", [("float32", "bfloat16"), ("float32", "float8_e4m3fn")])
def test_explicit_cast_store_preserves_rounding_in_relaxed_module(source, intermediate, tmp_path):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(source, (16,)), id="x")
            rounded = fm.F.tensors.cast(x, intermediate, rounding_contract="preserve")
            result = fm.F.tensors.cast(rounded, source)
            self.function("main", (x,), (result,))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    restored = fm.load_module(fm.emit_module(module, tmp_path / "cast.py"))
    assert restored.semantic_hash == module.semantic_hash
    evaluator = TorchEvaluator(DictWeightResolver({}))
    values = {"x": torch.linspace(.991, 1.009, 16)}
    expected = values["x"].to(getattr(torch, intermediate)).float()
    actual = evaluator.run(decompose_complex_ops(restored), values)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_explicit_lossless_cast_round_trip_still_folds():
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (16,)), id="x")
            wide = fm.F.tensors.cast(x, "float32", rounding_contract="preserve")
            result = fm.F.tensors.cast(wide, "bfloat16")
            self.function("main", (x,), (result,))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    assert decompose_complex_ops(module).functions[0].outputs == ("x",)
