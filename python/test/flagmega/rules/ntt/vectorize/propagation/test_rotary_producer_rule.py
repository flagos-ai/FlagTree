# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (DictWeightResolver, TorchEvaluator, PagedAttentionStateConfig,
                                       create_paged_attention_state)
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.vectorize.propagation.rotary_embedding import rotary_embedding_producer_rule


class StatefulTables(fm.Module):

    def forward(self):
        ref = self.input("reference", fm.tensor_type("bfloat16", (1, 32)))
        slots = self.input("slots", fm.tensor_type("bfloat16", (1, 2, 32)))
        state = self.input("state", PagedAttentionStateConfig(1, 2, 32).ref_type)
        layer = self.input("layer", fm.tensor_type("int32", ()))
        advance = self.input("advance", fm.tensor_type("bool", ()))
        before = fm.F.nn.rotary_embedding(ref, state, head_dim=32, theta=10000., name="before")
        updated = fm.F.nn.update_paged_attention_kv_cache(slots, state, layer, advance, cache_kind="value",
                                                          layout=("seq", "head", "dim"), name="update")
        after = fm.F.nn.rotary_embedding(ref, updated, head_dim=32, theta=10000., name="after")
        outputs = []
        # A single packed projection must not discard the other tuple field.
        for pair in (before, after):
            field = fm.F.tensors.get_item(pair, 0)
            outputs.extend((fm.F.tensors.pack(field, (2, 4), axes=(-1, -1)), pair))
        self.function("main", (ref, slots, state, layer, advance), tuple(outputs))


def test_tuple_packing_replaces_each_state_read_in_place_across_sequence_update():
    module = StatefulTables(dialect="ntt", stage="packed", entry="main").build()
    rewritten = DataflowRewriter((rotary_embedding_producer_rule(), )).rewrite(module)
    assert [(n.op, n.effect) for n in rewritten.nodes if not n.effect.is_pure] == [(n.op, n.effect)
                                                                                   for n in module.nodes
                                                                                   if not n.effect.is_pure]
    reads = [n for n in rewritten.nodes if n.op == "nn.rotary_embedding"]
    assert len(reads) == 2 and all(n.attrs["output_lanes"] == (2, 4) for n in reads)
    evaluator = TorchEvaluator(DictWeightResolver({}))

    def run(graph):
        state = create_paged_attention_state(PagedAttentionStateConfig(1, 2, 32))
        result = evaluator.run(
            graph, {
                "reference": torch.zeros(1, 32).bfloat16(), "slots": torch.ones(1, 2, 32).bfloat16(), "state": state,
                "layer": torch.tensor(0, dtype=torch.int32), "advance": torch.tensor(True)
            })
        assert state.sequence_length == 1
        return result

    expected, actual = run(module), run(rewritten)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual[0], actual[2])
