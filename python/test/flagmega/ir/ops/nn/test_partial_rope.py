# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Partial RoPE is one typed operation, including its untouched tail."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.rope import RoPE
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim,rotary_dim", [(24, 16), (256, 64), (16, 16), (7, 2)])
def test_partial_rope_matches_expanded_reference(dtype, head_dim, rotary_dim):
    generator = torch.Generator().manual_seed(47)
    value = torch.randn((2, 3, head_dim), generator=generator).to(dtype)
    cosine = torch.randn((2, 1, rotary_dim), generator=generator)
    sine = torch.randn((2, 1, rotary_dim), generator=generator)
    prefix = value[..., :rotary_dim].float()
    half = rotary_dim // 2
    rotated = torch.cat((-prefix[..., half:], prefix[..., :half]), dim=-1)
    expected = torch.cat(((prefix * cosine.float() + rotated * sine.float()).to(dtype), value[..., rotary_dim:]), dim=-1)
    result = evaluate(RoPE, (value, cosine, sine), rotary_dim=rotary_dim)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rotary_dim", [0, -2, 3, True, 4.0, 18])
def test_partial_rope_rejects_invalid_rotary_dim(rotary_dim):
    with pytest.raises(IRSchemaError, match="rotary_dim"):
        primitive_module(RoPE,
                         (fm.tensor_type("float32",
                                         (1, 2, 16)), fm.tensor_type("float32",
                                                                     (1, 1, 4)), fm.tensor_type("float32", (1, 1, 4))),
                         rotary_dim=rotary_dim)


def test_partial_rope_checks_table_extent():
    with pytest.raises(IRSchemaError, match="rotary"):
        primitive_module(RoPE,
                         (fm.tensor_type("float32",
                                         (1, 2, 16)), fm.tensor_type("float32",
                                                                     (1, 1, 16)), fm.tensor_type("float32",
                                                                                                 (1, 1, 16))),
                         rotary_dim=8)


def test_full_rope_default_keeps_old_serialized_contract():
    types = (fm.tensor_type("float32", (1, 1, 16)), ) * 3
    assert primitive_module(RoPE, types).node_map["output"].attrs == {}
    assert primitive_module(RoPE, types, rotary_dim=None).node_map["output"].attrs == {}


def test_partial_rope_dynamic_bounds_and_cost():
    tokens = fm.dim("tokens", minimum=1, maximum=32)
    head = fm.dim("head", minimum=16, maximum=64)
    types = (fm.tensor_type("float32", (tokens, 2, head)), fm.tensor_type("float32", (tokens, 1, 8)),
             fm.tensor_type("float32", (tokens, 1, 8)))
    module = primitive_module(RoPE, types, rotary_dim=8)
    fm.verify_module(module)
    assert module.node_map["output"].type == types[0]
    assert RoPE.cost(module.node_map["output"]).flops is None
    with pytest.raises(IRSchemaError, match="minimum input head"):
        primitive_module(RoPE, types, rotary_dim=32)
    static_types = (fm.tensor_type("float32", (2, 3, 24)), ) + (fm.tensor_type("float32", (2, 1, 16)), ) * 2
    static = primitive_module(RoPE, static_types, rotary_dim=16)
    assert RoPE.cost(static.node_map["output"]).flops == 2 * 3 * 16 * 3


def test_partial_rope_python_and_pattern_keep_rotary_dim(tmp_path):
    types = (fm.tensor_type("float32", (1, 2, 24)), ) + (fm.tensor_type("float32", (1, 1, 16)), ) * 2
    module = primitive_module(RoPE, types, rotary_dim=16)
    source = fm.module_source(module)
    assert "rotary_dim=16" in source
    restored = fm.load_module(fm.emit_module(module, tmp_path / "partial.py"))
    assert restored.semantic_hash == module.semantic_hash
    node = module.node_map["output"]
    assert pm.try_match_root(node, pm.F.nn.is_rope(rotary_dim=16), module) is not None
    assert pm.try_match_root(node, pm.F.nn.is_rope(rotary_dim=8), module) is None


def test_partial_rope_preserves_two_dimensional_head_ownership():
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    value = fm.DistributedType(fm.tensor_type("float32", (1, 4, 24)), (b, fm.SBP.split_contiguous((0, 1)), b),
                               placement)
    table = fm.DistributedType(fm.tensor_type("float32", (1, 1, 16)), (b, b, b), placement)
    module = primitive_module(RoPE, (value, table, table), rotary_dim=16)
    assert module.node_map["output"].type == value
