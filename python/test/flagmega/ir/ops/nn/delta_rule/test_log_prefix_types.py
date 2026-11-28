# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.delta_rule_log_prefix import DeltaRuleLogPrefix
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("tokens", [0, 1, 31, 32, 33, 64, 65, 129])
@pytest.mark.parametrize("block,group", [(8, 4), (32, 32), (64, 32), (128, 16)])
def test_prefix_block_type_and_cost(tokens, block, group):
    module = fm.verify_module(
        primitive_module(DeltaRuleLogPrefix, (fm.tensor_type("float32", (tokens, 4)), ), block_size=block,
                         scan_group_size=group))
    node = module.node_map["output"]
    assert node.type == fm.tensor_type("float32", ((tokens + block - 1) // block, 4, block))
    assert DeltaRuleLogPrefix.cost(node).bytes_written == ((tokens + block - 1) // block) * 4 * block * 4


def test_prefix_symbolic_tokens_and_heads():
    tokens, heads = fm.DimVar("tokens"), fm.DimVar("heads", 1, 16)
    module = primitive_module(DeltaRuleLogPrefix, (fm.tensor_type("float32", (tokens, heads)), ))
    assert module.node_map["output"].type.shape == ((tokens + 63) // 64, heads, fm.dim(64))


@pytest.mark.parametrize("attrs",
                         [{"block_size": True}, {"block_size": 0}, {"block_size": 3}, {"scan_group_size": 0},
                          {"scan_group_size": 33}, {"scan_group_size": 128}, {"epsilon": -1}, {"epsilon": float("inf")},
                          {"epsilon": float("nan")}, {"epsilon": True}, {"log2_mode": "arbitrary"}])
def test_prefix_rejects_undefined_rounding_topology(attrs):
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleLogPrefix, (fm.tensor_type("float32", (65, 4)), ), **attrs)


@pytest.mark.parametrize(
    "value_type",
    [fm.tensor_type("bfloat16", (3, 4)),
     fm.tensor_type("float32", (3, )),
     fm.tensor_type("float32", (3, 0))])
def test_prefix_rejects_invalid_input(value_type):
    with pytest.raises(IRSchemaError):
        primitive_module(DeltaRuleLogPrefix, (value_type, ))


@pytest.mark.parametrize(
    "head",
    [fm.SBP.broadcast(), fm.SBP.split_contiguous(
        (0, 1)), fm.SBP.split_block_cyclic((0, ), 2)])
def test_prefix_distributes_only_heads(head):
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    source = fm.DistributedType(fm.tensor_type("float32", (65, 8)), (b, head), placement)
    module = primitive_module(DeltaRuleLogPrefix, (source, ))
    assert module.node_map["output"].type == fm.DistributedType(fm.tensor_type("float32", (2, 8, 64)), (b, head, b),
                                                                placement)


@pytest.mark.parametrize("partial", [False, True])
def test_prefix_rejects_distributed_recurrence(partial):
    placement = fm.Placement((2, ), "x", "b")
    b, s = fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ))
    value = fm.DistributedType(fm.tensor_type("float32", (64, 8)), (b if partial else s, b), placement,
                               partial=fm.SBPPartial((0, )) if partial else None)
    with pytest.raises(IRSchemaError, match="materialized token axis"):
        primitive_module(DeltaRuleLogPrefix, (value, ))
