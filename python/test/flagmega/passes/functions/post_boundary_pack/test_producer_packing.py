# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (DictWeightResolver, TorchEvaluator, PagedAttentionStateConfig,
                                       create_paged_attention_state)
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


class Producers(fm.Module):

    def __init__(self, lanes=(2, 4), scalar_users=False, negative_axes=False):
        super().__init__(dialect="ntt", stage="packed", entry="main")
        self.lanes, self.scalar_users = lanes, scalar_users
        self.negative_axes = negative_axes

    def forward(self):
        ids = self.input("ids", fm.tensor_type("int64", (1, )), id="ids")
        state = self.input("state",
                           PagedAttentionStateConfig(1, 2, 32, block_size=4, num_blocks=2).ref_type, id="state")
        weight = self.weight("weight", fm.tensor_type("bfloat16", (17, 32)), source="weights", key="weight",
                             id="weight")
        embedded = fm.F.nn.embedding(ids, weight, padding_idx=0, name="embedding")
        pair = fm.F.nn.rotary_embedding(embedded, state, head_dim=32, theta=10000., name="rotary")
        outputs = [
            fm.F.tensors.pack(embedded, (8, ), axes=(-1 if self.negative_axes else 1, ), name="packed_embedding")
        ]
        for i in range(2):
            table = fm.F.tensors.get_item(pair, i, name=f"table_{i}")
            outputs.append(
                fm.F.tensors.pack(table, self.lanes, axes=(-1 if self.negative_axes else 2, ) * len(self.lanes),
                                  name=f"packed_table_{i}"))
            if self.scalar_users:
                outputs.append(table)
        if self.scalar_users:
            outputs.extend((embedded, pair))
        self.function("main", (ids, state), tuple(outputs))


@pytest.mark.parametrize("lanes", [(8, ), (2, 4), (4, 2)])
@pytest.mark.parametrize("scalar_users", [False, True])
@pytest.mark.parametrize("negative_axes", [False, True])
def test_packing_reaches_embedding_weight_and_single_stateful_table_producer(lanes, scalar_users, negative_axes):
    original = Producers(lanes, scalar_users, negative_axes).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    embedding, = (n for n in rewritten.nodes if n.op == "nn.embedding")
    rotary, = (n for n in rewritten.nodes if n.op == "nn.rotary_embedding")
    assert embedding.type.dtype == fm.vector_type("bfloat16", (8, ))
    weight = rewritten.node_map[embedding.inputs[1]]
    assert weight.op == "tensors.pack"
    assert rewritten.node_map[weight.inputs[0]].op == "builtin.weight"
    assert rotary.type.fields[0].dtype == fm.vector_type("float32", lanes)
    assert rotary.type.fields[1] == rotary.type.fields[0]
    assert rotary.effect == original.node_map["rotary"].effect
    assert rotary.inputs[0] == embedding.id
    assert [n.id for n in rewritten.nodes if n.op == "tensors.pack"] == [weight.id]
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 2, 32, block_size=4, num_blocks=2))
    state.seq_lens[0] = 3
    evaluator = TorchEvaluator(DictWeightResolver({"weight": torch.randn(17, 32).bfloat16()}))
    for token in (0, 7):
        inputs = {"ids": torch.tensor([token]), "state": state}
        torch.testing.assert_close(evaluator.run(rewritten, inputs), evaluator.run(original, inputs), rtol=0, atol=0)
    again = post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target())
    assert again.semantic_hash == rewritten.semantic_hash
