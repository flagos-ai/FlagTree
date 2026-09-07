# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.embedding import Embedding, embedding


torch = pytest.importorskip("torch")


class _EmbeddingModule(fm.Module):
    def __init__(self, *, padding_idx=None) -> None:
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.padding_idx = padding_idx

    def forward(self) -> None:
        indices = self.input("input_ids", fm.tensor_type("int32", [2]), id="input_ids")
        weight = self.weight(
            "embed_tokens.weight",
            fm.tensor_type("bfloat16", [5, 4]),
            source="embedding.safetensors",
            key="model.embed_tokens.weight",
            id="embedding_weight",
        )
        output = fm.F.nn.embedding(
            indices,
            weight,
            padding_idx=self.padding_idx,
            name="token_embedding",
        )
        self.function("main", [indices], [output])


def test_embedding_static_functional_api_infers_indices_shape_plus_hidden_axis():
    module = _EmbeddingModule(padding_idx=0).build()
    node = module.node_map["token_embedding"]

    assert node.op == "nn.embedding"
    assert node.attrs == {"padding_idx": 0}
    assert node.type == fm.tensor_type("bfloat16", [2, 4])


def test_embedding_evaluator_gathers_rows_and_zeroes_padding_tokens():
    weight = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    indices = torch.tensor([3, 0, 1], dtype=torch.int32)

    actual = embedding(indices, weight, padding_idx=0)

    torch.testing.assert_close(actual[0], weight[3])
    assert torch.equal(actual[1], torch.zeros(4, dtype=torch.bfloat16))
    torch.testing.assert_close(actual[2], weight[1])


def test_embedding_rejects_padding_index_outside_static_vocabulary():
    with pytest.raises(IRSchemaError, match="outside vocabulary"):
        _EmbeddingModule(padding_idx=5).build()


def test_embedding_infers_feature_sharded_distributed_result():
    placement = fm.Placement((8, 16), "yx", "bb")
    feature_policy = fm.SBP.split_contiguous((0,))
    indices_type = fm.DistributedType(
        fm.tensor_type("int32", [1]),
        (fm.SBP.broadcast(),),
        placement,
    )
    weight_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [151936, 2048]),
        (fm.SBP.broadcast(), feature_policy),
        placement,
    )
    inputs = (
        fm.Node("indices", "builtin.var", (), indices_type, attrs={"name": "indices"}),
        fm.Node("weight", "builtin.weight", (), weight_type, attrs={"name": "weight"}),
    )

    assert Embedding.infer_type(inputs, {"padding_idx": None}) == fm.DistributedType(
        fm.tensor_type("bfloat16", [1, 2048]),
        (fm.SBP.broadcast(), feature_policy),
        placement,
    )


def test_embedding_rejects_distributed_vocabulary_or_indices_split():
    placement = fm.Placement((8, 16), "yx", "bb")
    indices_tensor = fm.tensor_type("int32", [8])
    weight_tensor = fm.tensor_type("bfloat16", [151936, 2048])
    split_indices = fm.DistributedType(
        indices_tensor,
        (fm.SBP.split_contiguous((0,)),),
        placement,
    )
    split_vocabulary = fm.DistributedType(
        weight_tensor,
        (fm.SBP.split_contiguous((1,)), fm.SBP.broadcast()),
        placement,
    )
    broadcast_indices = fm.DistributedType(
        indices_tensor,
        (fm.SBP.broadcast(),),
        placement,
    )
    broadcast_weight = fm.DistributedType(
        weight_tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    def inputs(indices_type, weight_type):
        return (
            fm.Node("indices", "builtin.var", (), indices_type, attrs={"name": "indices"}),
            fm.Node("weight", "builtin.weight", (), weight_type, attrs={"name": "weight"}),
        )

    with pytest.raises(IRSchemaError, match="indices must be broadcast"):
        Embedding.infer_type(inputs(split_indices, broadcast_weight), {"padding_idx": None})
    with pytest.raises(IRSchemaError, match="vocabulary axis must be broadcast"):
        Embedding.infer_type(inputs(broadcast_indices, split_vocabulary), {"padding_idx": None})
