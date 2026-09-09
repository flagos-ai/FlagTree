# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.rope import RoPE
from triton.flagmega.ir.ops.ntt.vectorized_rope import VectorizedRoPE
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("head_dim,rotary_dim,lane", [(24, 16, 8), (256, 64, 8), (32, 32, 4)])
def test_vector_partial_rope_evaluation_and_scalar_units(head_dim, rotary_dim, lane, tmp_path):
    value_type = fm.tensor_type(fm.vector_type("bfloat16", (lane, )), (2, 3, head_dim // lane))
    table_type = fm.tensor_type(fm.vector_type("float32", (2, lane)), (2, 1, rotary_dim // (2 * lane)))
    module = primitive_module(VectorizedRoPE, (value_type, table_type, table_type), rotary_dim=rotary_dim)
    generator = torch.Generator().manual_seed(38)
    value = torch.randn((2, 3, head_dim), generator=generator).bfloat16()
    cos = torch.randn((2, 1, rotary_dim), generator=generator)
    sin = torch.randn((2, 1, rotary_dim), generator=generator)
    arguments = {
        "input": value.reshape(2, 3, head_dim // lane, lane), "cos":
        cos.reshape(2, 1, rotary_dim // (2 * lane), 2, lane), "sin": sin.reshape(2, 1, rotary_dim // (2 * lane), 2,
                                                                                 lane)
    }
    result = TorchEvaluator(DictWeightResolver({})).run(module, arguments)[0]
    expected = evaluate(RoPE, (value, cos, sin), rotary_dim=rotary_dim)
    torch.testing.assert_close(result.reshape_as(expected), expected, rtol=0, atol=0)
    output = module.node_map["output"]
    assert output.type == value_type
    assert output.attrs["rotary_dim"] == rotary_dim
    assert VectorizedRoPE.cost(output).flops == 2 * 3 * rotary_dim * 3
    assert pm.try_match_root(output, pm.F.ntt.is_vectorized_rope(rotary_dim=rotary_dim), module) is not None
    assert fm.load_module(fm.emit_module(module, tmp_path / "vector.py")).semantic_hash == module.semantic_hash


def test_vector_partial_rope_rejects_scalar_unit_mismatch():
    value = fm.tensor_type(fm.vector_type("bfloat16", (8, )), (1, 2, 32))
    table = fm.tensor_type(fm.vector_type("float32", (2, 8)), (1, 1, 4))
    with pytest.raises(IRSchemaError, match="scalar rotary dimension"):
        primitive_module(VectorizedRoPE, (value, table, table), rotary_dim=8)
