# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.delta_rule_coefficients import DeltaRuleCoefficients
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("block", [8, 16, 32, 64])
@pytest.mark.parametrize("tokens", [0, 1, 7, 32, 65])
def test_coefficient_extent_counts_partial_blocks(block, tokens):
    module = primitive_module(DeltaRuleCoefficients,
                              (fm.tensor_type("bfloat16", (tokens, 2, 16)), fm.tensor_type("float32", (tokens, 6))),
                              block_size=block)
    output = fm.verify_module(module).node_map["output"]
    assert output.type == fm.tensor_type("bfloat16", ((tokens + block - 1) // block, 6, block, block))
    assert DeltaRuleCoefficients.cost(output).bytes_written == ((tokens + block - 1) // block) * 6 * block * block * 2


def test_symbolic_tokens_preserve_block_expression():
    tokens = fm.DimVar("tokens")
    module = primitive_module(DeltaRuleCoefficients,
                              (fm.tensor_type("bfloat16", (tokens, 2, 16)), fm.tensor_type("float32", (tokens, 4))))
    assert module.node_map["output"].type.shape[0] == (tokens + 63) // 64


@pytest.mark.parametrize("block", [True, False, 0, 7, 128, 8.0, "8"])
def test_reject_unsupported_inverse_rounding_tree(block):
    with pytest.raises(IRSchemaError, match="block_size"):
        primitive_module(DeltaRuleCoefficients,
                         (fm.tensor_type("bfloat16", (3, 2, 16)), fm.tensor_type("float32", (3, 4))), block_size=block)


@pytest.mark.parametrize("key,beta", [
    (fm.tensor_type("float32", (3, 2, 16)), fm.tensor_type("float32", (3, 4))),
    (fm.tensor_type("bfloat16", (3, 2, 16)), fm.tensor_type("bfloat16", (3, 4))),
    (fm.tensor_type("bfloat16", (3, 2, 16)), fm.tensor_type("float32", (4, 4))),
    (fm.tensor_type("bfloat16", (3, 2, 16)), fm.tensor_type("float32", (3, 3))),
    (fm.tensor_type("bfloat16", (3, 2, 0)), fm.tensor_type("float32", (3, 4))),
])
def test_reject_incompatible_operands(key, beta):
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleCoefficients, (key, beta))


@pytest.mark.parametrize("axes", [(), (0, ), (1, ), (0, 1)])
def test_grouped_contiguous_head_partition(axes):
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    head = fm.SBP.split_contiguous(axes) if axes else b
    key = fm.DistributedType(fm.tensor_type("bfloat16", (65, 4, 16)), (b, head, b), placement)
    beta = fm.DistributedType(fm.tensor_type("float32", (65, 8)), (b, head), placement)
    module = fm.verify_module(primitive_module(DeltaRuleCoefficients, (key, beta)))
    assert module.node_map["output"].type == fm.DistributedType(fm.tensor_type("bfloat16", (2, 8, 64, 64)),
                                                                (b, head, b, b), placement)


@pytest.mark.parametrize("bad", ["tokens", "key", "heads", "cyclic", "partial", "mixed"])
def test_reject_nonlocal_or_mismatched_head_contract(bad):
    placement = fm.Placement((2, ), "x", "b")
    b, s = fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ))
    policies = {
        "tokens": (s, b, b), "key": (b, b, s), "heads": (b, s, b), "cyclic": (b, fm.SBP.split_block_cyclic((0, ), 1), b)
    }.get(bad, (b, b, b))
    key = fm.DistributedType(fm.tensor_type("bfloat16", (4, 2, 16)), policies, placement, partial=fm.SBPPartial(
        (0, )) if bad == "partial" else None)
    beta = fm.tensor_type("float32", (4, 4))
    if bad != "mixed":
        beta = fm.DistributedType(beta, (b, policies[1] if bad == "cyclic" else b), placement)
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleCoefficients, (key, beta))
