# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit type transitions must not receive elementwise broadcast lifting."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.builtin.call import Call as FunctionCall
from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.ir.ops.distributed.force_boxing import ForceBoxing
from triton.flagmega.ir.ops.tir.call import Call as TirCall
from triton.flagmega.ir.ops.tir.kernel import Kernel


@pytest.mark.parametrize("definition", [Boxing, ForceBoxing, FunctionCall, TirCall, Kernel])
@pytest.mark.parametrize("tuple_result", [False, True])
def test_explicit_plain_tensor_result_is_not_broadcast_lifted(definition, tuple_result):
    plain = fm.tensor_type("float32", (17,))
    distributed = fm.DistributedType(plain, (fm.SBP.broadcast(),), fm.Placement((2, 4), "yx", "bb"))
    result = fm.TupleType((plain, plain)) if tuple_result else plain
    source_type = fm.TupleType((distributed, distributed)) if tuple_result else distributed
    source = fm.IRBuilder(dialect="distributed", stage="distributed").var("source", source_type)
    key = "new_type" if definition in (Boxing, ForceBoxing) else "result_type"
    assert definition.infer_call_type((source,), {key: result}) == result


def test_ordinary_elementwise_broadcast_lifting_is_preserved():
    from triton.flagmega.ir.ops.math.silu import Silu

    plain = fm.tensor_type("float32", (17,))
    distributed = fm.DistributedType(plain, (fm.SBP.broadcast(),), fm.Placement((2, 4), "yx", "bb"))
    source = fm.IRBuilder(dialect="distributed", stage="distributed").var("source", distributed)
    assert Silu.infer_call_type((source,), {}) == distributed
