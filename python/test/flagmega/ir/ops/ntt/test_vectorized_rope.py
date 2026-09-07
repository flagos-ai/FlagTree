# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


class _VectorizedRoPEGraph(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="vectorized", entry="main")

    def forward(self):
        value = self.input(
            "value",
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 2, 2)),
            id="value",
        )
        cos = self.input(
            "cos",
            fm.tensor_type(fm.vector_type("float32", (2, 8)), (1, 1, 1)),
            id="cos",
        )
        sin = self.input(
            "sin",
            fm.tensor_type(fm.vector_type("float32", (2, 8)), (1, 1, 1)),
            id="sin",
        )
        result = fm.F.ntt.vectorized_rope(value, cos, sin, name="result")
        self.function("main", (value, cos, sin), (result,))


def test_vectorized_rope_has_static_functional_pattern_and_python_dump_api():
    module = _VectorizedRoPEGraph().build()

    assert module.node_map["result"].type == module.node_map["value"].type
    assert pm.try_match_root(
        module.node_map["result"],
        pm.F.ntt.is_vectorized_rope(call_name="rope"),
        module,
    ) is not None
    assert "F.ntt.vectorized_rope(" in fm.module_source(module)


def test_vectorized_rope_evaluator_preserves_packed_physical_layout():
    module = _VectorizedRoPEGraph().build()
    torch.manual_seed(17)
    value = torch.randn((1, 2, 2, 8), dtype=torch.bfloat16)
    cos = torch.randn((1, 1, 1, 2, 8), dtype=torch.float32)
    sin = torch.randn((1, 1, 1, 2, 8), dtype=torch.float32)

    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value, "cos": cos, "sin": sin}
    )[0]

    scalar_value = value.reshape(1, 2, 16)
    scalar_cos = cos.reshape(1, 1, 16).to(torch.bfloat16)
    scalar_sin = sin.reshape(1, 1, 16).to(torch.bfloat16)
    half = scalar_value.shape[-1] // 2
    rotated = torch.cat(
        (-scalar_value[..., half:], scalar_value[..., :half]), dim=-1
    )
    expected = (
        scalar_value * scalar_cos + rotated * scalar_sin
    ).reshape(1, 2, 2, 8)
    torch.testing.assert_close(actual, expected)


def test_vectorized_rope_rejects_a_rotary_table_without_pair_and_lane_groups():
    value = _typed(
        "value", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 2, 2))
    )
    malformed = _typed(
        "table", fm.tensor_type(fm.vector_type("float32", (8,)), (1, 1, 2))
    )

    with pytest.raises(IRSchemaError, match="pair and lane"):
        fm.get_definition("ntt.vectorized_rope").infer_type(
            (value, malformed, malformed), {}
        )


def _typed(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})
