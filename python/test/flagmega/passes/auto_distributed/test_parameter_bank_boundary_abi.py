# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A bank row feeding a call must agree with the callee's distributed ABI."""

import pytest

from triton.flagmega.ir import Placement, RefType, TupleType, tensor_type
from triton.flagmega.ir.distributed_inference import broadcast_ir_type


def _bank_ref():
    """A parameter bank whose element is a tuple of weight tensors."""
    element = TupleType((
        tensor_type("bfloat16", [2048]),
        tensor_type("bfloat16", [128, 256]),
    ))
    return RefType("parameter_bank", (("element", element),))


@pytest.mark.xfail(
    reason="AutoDistributed gives the bank handle a distributed element via "
           "search candidates, not via broadcast_ir_type, so this helper is "
           "still logical-only.  The round trip is fixed far enough that the "
           "call ABI survives; a For capture check fails next.",
    strict=True,
)
def test_broadcast_reaches_tensors_behind_a_parameter_bank():
    placement = Placement((8, 16), "yx", "bb")
    broadcast = broadcast_ir_type(_bank_ref(), placement)
    element = dict(broadcast.fields)["element"]
    # A bank row is destructured: `control.bank_get` yields this tuple and a
    # `get_item` per field, so every field becomes its own call argument and
    # meets a distributed callee parameter.  A state handle is different --
    # it is passed whole, so both sides stay references and agree.  Leaving
    # these fields logical makes the arguments disagree with the callee, and
    # codegen then emits sharded addressing against an unsharded weight.
    kinds = {type(field).__name__ for field in element.fields}
    assert kinds == {"DistributedType"}, (
        "a destructured reference's leaves must be broadcast like any other "
        f"call argument; got {kinds}"
    )
    for field in element.fields:
        assert field.placement == placement


def test_broadcast_keeps_the_reference_shape():
    placement = Placement((8, 16), "yx", "bb")
    broadcast = broadcast_ir_type(_bank_ref(), placement)
    # Only the leaves change: the handle stays a reference with the same
    # field names, or every consumer that resolves the bank by name breaks.
    assert isinstance(broadcast, RefType)
    assert broadcast.name == "parameter_bank"
    assert [name for name, _ in broadcast.fields] == ["element"]


def test_state_reference_leaves_are_untouched():
    placement = Placement((8, 16), "yx", "bb")
    state = RefType("paged_attention_kv_cache", (
        ("kv_caches", tensor_type("bfloat16", [16, 1, 2, 256, 8, 16])),
        ("seq_lens", tensor_type("int32", [1])),
    ))
    broadcast = broadcast_ir_type(state, placement)
    # A state handle crosses a call boundary whole, so its parameter is a
    # reference too and the two already agree.  Distributing its leaves would
    # change that ABI for every state consumer.
    kinds = {type(field).__name__ for _, field in broadcast.fields}
    assert kinds == {"TensorType"}
