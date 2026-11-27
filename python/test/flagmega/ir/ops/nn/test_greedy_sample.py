# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.greedy_sample import GreedySample


def infer(value_type):
    value = fm.Node("logits", "builtin.var", (), value_type, attrs={"name": "logits"})
    return GreedySample.infer_type((value,), {})


@pytest.mark.parametrize("vocab_policy", [
    fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 8),
    fm.SBP.split_block_cyclic((1,), 3),
])
def test_argmax_materializes_vocab_reduction_and_preserves_batch_split(vocab_policy):
    placement = fm.Placement((2, 4), "yx", "bb")
    batch_policy = fm.SBP.split_contiguous((0,), 2)
    value = fm.DistributedType(fm.tensor_type("float32", [4, 32]),
                               (batch_policy, vocab_policy), placement)
    result = infer(value)
    assert result == fm.DistributedType(fm.tensor_type("int32", [4]),
                                        (batch_policy,), placement)


def test_argmax_does_not_commute_with_additive_partial():
    value = fm.DistributedType(fm.tensor_type("float32", [1, 32]),
                               (fm.SBP.broadcast(), fm.SBP.broadcast()),
                               fm.Placement((2, 4), "yx", "bb"),
                               partial=fm.SBP.partial((0,)))
    with pytest.raises(IRSchemaError, match="materialized"):
        infer(value)


def test_empty_vocabulary_is_invalid():
    with pytest.raises(IRSchemaError, match="non-empty"):
        infer(fm.tensor_type("float32", [1, 0]))
