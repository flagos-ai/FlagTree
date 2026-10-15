# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

"""A fused packed-QKV weight is packed by its own bank field program.

``_fused_weight_type`` declares the fused buffer as ``(owners, local_n,
sum_of_local_k)``: dimension 0 selects an owner and the rest is that owner's
tile, holding its Q, K and V segments back to back.  The parameter-bank field
program already produces exactly those bytes -- it carries the concat, the
local-shard split and the lane packing -- so the packer has to replay it.

Concatenating the sources along the logical N axis instead yields the same
multiset of elements in a different order.  A permutation-invariant check
cannot see the difference, but every consumer reads the wrong weights, so
these tests pin the behaviour rather than the element set.
"""

from __future__ import annotations

import pytest
import torch

from triton.flagmega.artifacts.rdata import _source_tensor
from triton.flagmega.errors import ArtifactError


class _Checkpoint:
    def __init__(self, tensors):
        self._tensors = tensors

    def load_tensor(self, key, device="cpu"):
        return self._tensors[key]


class _Buffer:
    def __init__(self, keys, owners, local_n, local_k, *, group=None, row=0):
        self.id = "fused.test.buffer"
        self.weight_key = keys[0]
        self.fused_weight_keys = tuple(keys)
        # Mirrors _fused_weight_type: (owners, local_n, sum_of_local_k).
        self.shape = (owners, local_n, local_k)
        self.rdata_group = group
        self.source_node = group
        self.group_index = row


def test_single_source_buffer_loads_its_checkpoint_tensor_directly():
    value = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    buffer = _Buffer(("q",), 2, 2, 3)
    buffer.fused_weight_keys = ()

    loaded = _source_tensor(_Checkpoint({"q": value}), buffer, None)

    assert torch.equal(loaded, value)


def test_fused_buffer_is_not_packed_by_concatenating_its_sources():
    # The defect this guards: joining q/k/v on the logical N axis produces the
    # right elements in the wrong order, which no permutation-invariant check
    # rejects.  Packing must refuse rather than emit those bytes.
    tensors = {
        name: torch.zeros((rows, 3), dtype=torch.float32)
        for name, rows in (("q", 8), ("k", 4), ("v", 4))
    }
    buffer = _Buffer(("q", "k", "v"), 4, 2, 4)

    with pytest.raises(ArtifactError) as error:
        _source_tensor(_Checkpoint(tensors), buffer, None)

    message = str(error.value)
    assert "field program" in message or "parameter-bank" in message


def test_fused_buffer_without_a_resolvable_bank_field_is_rejected():
    # A fused buffer naming a group that no bank declares cannot be packed;
    # failing loudly keeps a silent wrong-layout image from being written.
    class _Module:
        nodes = ()
        node_map: dict = {}

    tensors = {
        name: torch.zeros((rows, 3), dtype=torch.float32)
        for name, rows in (("q", 8), ("k", 4), ("v", 4))
    }
    buffer = _Buffer(("q", "k", "v"), 4, 2, 4, group="absent.bank.field")

    with pytest.raises(ArtifactError) as error:
        _source_tensor(_Checkpoint(tensors), buffer, _Module())

    assert "parameter bank" in str(error.value)
