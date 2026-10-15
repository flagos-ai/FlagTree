# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from itertools import product

import pytest

from triton.flagmega.ir import DistributedType, Placement, SBP, SBPBroadCast, tensor_type
from triton.flagmega.ir.ops.nn.dsa_indexer import _dsa_broadcast_choices


class CountedChoices:
    def __init__(self, values):
        self.values = values
        self.visits = 0

    def __iter__(self):
        for value in self.values:
            self.visits += 1
            yield value


def _layouts():
    plain = tensor_type("float32", (16, 16))
    placement = Placement((2, 2), "cx", "cb")
    broadcast = DistributedType(plain, (SBPBroadCast(), SBPBroadCast()), placement)
    split = DistributedType(plain, (SBP.split_contiguous((0,)), SBPBroadCast()), placement)
    return plain, broadcast, split


def test_rejected_prefixes_do_not_enumerate_remaining_operands():
    plain, broadcast, split = _layouts()
    tail = CountedChoices((plain,))
    choices = ((plain, broadcast, split),) * 7 + (tail,)
    result = tuple(_dsa_broadcast_choices(choices, tuple(range(7))))
    assert len(result) == 2
    assert tail.visits == 2


@pytest.mark.parametrize("tensor_indices", [(0, 1, 3), (), (0,), (1, 3)])
def test_valid_tuple_order_matches_exhaustive_filter(tensor_indices):
    plain, broadcast, split = _layouts()
    other = replace(broadcast, placement=Placement((2, 2), "xc", "bc"))
    partial = replace(broadcast, partial=SBP.partial((0,)))
    exclusive = replace(broadcast, exclusive=SBP.exclusive((0,)))
    choices = ((split, broadcast, plain, broadcast, partial, exclusive), (plain, broadcast, other),
               (plain,), (broadcast, split, plain, other))
    expected = []
    for candidate in product(*choices):
        tensors = [candidate[i] for i in tensor_indices]
        distributed = [value for value in tensors if isinstance(value, DistributedType)]
        if not distributed or (
            len(distributed) == len(tensors)
            and all(value.placement == distributed[0].placement and value.partial is None
                    and value.exclusive is None
                    and all(isinstance(policy, SBPBroadCast) for policy in value.axis_policies)
                    for value in distributed)
        ):
            expected.append(candidate)
    assert tuple(_dsa_broadcast_choices(choices, tensor_indices)) == tuple(expected)


def test_empty_choice_sets_preserve_product_semantics():
    assert tuple(_dsa_broadcast_choices((), ())) == ((),)
    assert tuple(_dsa_broadcast_choices(((),), (0,))) == ()
