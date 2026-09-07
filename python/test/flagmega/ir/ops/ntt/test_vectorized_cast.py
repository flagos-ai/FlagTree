# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def test_vectorized_cast_changes_outer_extent_to_preserve_scalar_shape():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type(fm.vector_type("bfloat16", (8,)), (3,)),
                id="value",
            )
            result = fm.F.ntt.vectorized_cast(
                value,
                fm.vector_type("float32", (2,)),
                (0,),
                name="result",
            )
            self.function("main", (value,), (result,))

    module = Graph().build()
    assert module.node_map["result"].type == fm.tensor_type(
        fm.vector_type("float32", (2,)), (12,)
    )

    value = torch.randn(3, 8, dtype=torch.bfloat16)
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value})[0]
    assert actual.shape == (12, 2)
    torch.testing.assert_close(actual.reshape(24), value.reshape(24).float())


def test_vectorized_cast_python_dump_round_trips_static_ntt_api():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2,)), id="value"
            )
            result = fm.F.ntt.vectorized_cast(
                value, fm.vector_type("float32", (4,)), (0,), name="result"
            )
            self.function("main", (value,), (result,))

    module = Graph().build()
    source = fm.module_source(module)
    assert "F.ntt.vectorized_cast(" in source
