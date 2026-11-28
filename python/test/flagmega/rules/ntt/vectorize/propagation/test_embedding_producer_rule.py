# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.vectorize.propagation.embedding import embedding_producer_rule


def graph(indices_shape, old_lanes, axes, distributed=False):

    class Graph(fm.Module):

        def forward(self):
            it = fm.tensor_type("int32", indices_shape)
            wt = fm.tensor_type(fm.vector_type("bfloat16", old_lanes) if old_lanes else "bfloat16", (17, 32))
            if distributed:
                mesh = fm.Placement((2, 2), "yx", "bb")
                it = fm.DistributedType(it, (fm.SBP.broadcast(), ) * len(indices_shape), mesh)
                wt = fm.DistributedType(wt, (fm.SBP.broadcast(), fm.SBP.split_contiguous((1, ))), mesh)
            indices = self.input("indices", it, id="indices")
            weight = self.input("weight", wt, id="weight")
            value = fm.F.nn.embedding(indices, weight, padding_idx=-1, name="embedding")
            packed = fm.F.tensors.pack(value, (2, 4), axes=axes, name="packed")
            self.function("main", (indices, weight), (value, packed))

    return Graph(dialect="ntt", stage="packed", entry="main").build()


@pytest.mark.parametrize("shape", [(), (4, ), (2, 4)])
@pytest.mark.parametrize("old_lanes", [(), (2, 2)])
@pytest.mark.parametrize("distributed", [False, True])
def test_feature_pack_is_projected_to_weight_with_shared_scalar_users(shape, old_lanes, distributed):
    original = graph(shape, old_lanes, (-1, -1), distributed)
    actual = DataflowRewriter((embedding_producer_rule(), )).rewrite(original)
    producers = [n for n in actual.nodes if n.op == "nn.embedding"]
    assert len(producers) == 1
    assert producers[0].type == original.node_map["packed"].type
    packed_weight = actual.node_map[producers[0].inputs[1]]
    assert packed_weight.inputs == ("weight", )
    assert packed_weight.attrs["axes"] == (1, 1)
    inputs = {
        "indices": torch.randint(0, 17, shape, dtype=torch.int32), "weight": torch.randn(17, 32, *old_lanes).bfloat16()
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(actual, inputs), evaluator.run(original, inputs), rtol=0, atol=0)


def test_token_axis_pack_is_not_projected_into_vocabulary():
    module = graph((8, ), (), (0, 0))
    assert embedding_producer_rule().apply(module.node_map["embedding"], module) is None
