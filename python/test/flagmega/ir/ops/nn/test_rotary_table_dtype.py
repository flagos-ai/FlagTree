# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.ir.ops.nn.rotary_embedding import RotaryEmbedding
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("lanes", [(), (8, ), (2, 4)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_rotary_table_dtype_is_the_final_store_boundary(dtype, lanes, tmp_path):
    config = PagedAttentionStateConfig(1, 2, 32)
    module = primitive_module(RotaryEmbedding, (fm.tensor_type("bfloat16", (3, 16)), config.ref_type), head_dim=32,
                              theta=10000., attention_scaling=1.3, output_lanes=lanes, output_dtype=dtype)
    output = module.node_map["output"]
    assert output.type.fields[0].dtype == (fm.vector_type(dtype, lanes) if lanes else fm.DType(dtype))
    assert ("output_dtype" in output.attrs) == (dtype != "float32")
    assert fm.load_module(fm.emit_module(module, tmp_path / "tables.py")).semantic_hash == module.semantic_hash
    state = create_paged_attention_state(config)
    state.seq_lens.fill_(7)
    result, = TorchEvaluator(DictWeightResolver({})).run(module,
                                                         {"reference": torch.zeros(3, 16).bfloat16(), "state": state})
    angles = torch.outer(torch.arange(7, 10).float(), 10000.**(-torch.arange(0, 32, 2).float() / 32))
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    for actual, expected in zip(result, ((angles.cos() * 1.3).to(getattr(torch, dtype)),
                                         (angles.sin() * 1.3).to(getattr(torch, dtype)))):
        torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["int32", "float8_e4m3fn", "unknown", None, True])
def test_rotary_rejects_unsupported_output_dtype(dtype):
    with pytest.raises(IRSchemaError, match="output_dtype"):
        RotaryEmbedding.normalize_attrs({"head_dim": 32, "theta": 10000., "output_dtype": dtype})
