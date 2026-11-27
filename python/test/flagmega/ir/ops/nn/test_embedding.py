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


def _native_embedding_module(encoding="IQ4_XS", *, output_data_type=None, lanes=(), distributed=False):
    size = {"IQ4_XS": 136, "Q6_K": 210}[encoding]

    class Graph(fm.Module):
        def forward(self):
            it = fm.tensor_type("int32", (3,))
            wt = fm.tensor_type("uint8", (3, 2 * size))
            if distributed:
                placement = fm.Placement((1,), "x", "c")
                it = fm.DistributedType(it, (fm.SBP.broadcast(),), placement)
                wt = fm.DistributedType(wt, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
            ids = self.input("ids", it, id="ids")
            weight = self.input("weight", wt, id="weight")
            output = fm.F.nn.embedding(ids, weight, weight_encoding=encoding, output_data_type=output_data_type,
                                       output_lanes=lanes, name="embedding")
            self.function("main", (ids, weight), (output,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q6_K"])
@pytest.mark.parametrize("lanes", [(), (2, 4)])
def test_native_embedding_f32_retains_values_before_bf16_rounding(encoding, lanes):
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
    from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import packed_rows
    from python.test.flagmega.ir.ops.math.test_q6_k_decode import q6_rows

    raw, expected_rows = q6_rows() if encoding == "Q6_K" else packed_rows(encoding, rows=3)
    ids = torch.tensor([2, 0, 1], dtype=torch.int32)
    module = _native_embedding_module(encoding, output_data_type="float32", lanes=lanes)
    output, = TorchEvaluator(DictWeightResolver({})).run(module, {"ids": ids, "weight": torch.from_numpy(raw)})
    expected = torch.from_numpy(expected_rows[[2, 0, 1]])
    assert output.dtype == torch.float32
    assert torch.equal(output.reshape(3, 512).view(torch.int32), expected.view(torch.int32))
    assert not torch.equal(expected, expected.bfloat16().float())
    legacy = embedding(ids, torch.from_numpy(raw), weight_encoding=encoding)
    assert torch.equal(legacy.view(torch.int16), expected.bfloat16().view(torch.int16))


def test_native_embedding_output_dtype_roundtrip_and_legacy_omission(tmp_path):
    import os
    import subprocess
    import sys

    module = _native_embedding_module(output_data_type="float32", distributed=True)
    node = module.node_map["embedding"]
    assert node.type.tensor.dtype == fm.DType.FLOAT32
    assert node.attrs["output_data_type"] == "float32"
    path = fm.emit_module(module, tmp_path / "embedding_f32.py")
    assert fm.load_module(path) == module
    code = "from triton.flagmega import ir as fm; import sys; m=fm.load_module(sys.argv[1]); assert m.semantic_hash == sys.argv[2]"
    subprocess.run([sys.executable, "-c", code, str(path), module.semantic_hash], env=os.environ.copy(), check=True)
    implicit = Embedding.normalize_attrs({"weight_encoding": "IQ4_XS"})
    explicit = Embedding.normalize_attrs({"weight_encoding": "IQ4_XS", "output_data_type": None})
    assert implicit == explicit == {"padding_idx": None, "weight_encoding": "IQ4_XS"}
    assert _native_embedding_module().node_map["embedding"].type.dtype == fm.DType.BFLOAT16


@pytest.mark.parametrize("dtype", ["int32", "float16", "unknown", False])
def test_embedding_rejects_unsupported_output_dtype(dtype):
    with pytest.raises(IRSchemaError, match="output_data_type"):
        _native_embedding_module(output_data_type=dtype)
    with pytest.raises(IRSchemaError, match="output_data_type"):
        embedding(torch.zeros(1, dtype=torch.int32), torch.zeros((1, 136), dtype=torch.uint8),
                  weight_encoding="IQ4_XS", output_data_type=dtype)


def test_embedding_explicit_output_dtype_requires_native_storage():
    with pytest.raises(IRSchemaError, match="native"):
        Embedding.normalize_attrs({"output_data_type": "float32"})


def test_native_embedding_new_contracts_reject_unsupported_gpu_codegen():
    from triton.flagmega.codegen.cpp import describe_cpp_package
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.errors import CodegenError
    from triton.flagmega.options import CompileOptions

    for encoding in ("IQ4_XS", "Q6_K"):
        module = _native_embedding_module(encoding, output_data_type="float32")
        compiler = Compiler(CompileOptions(target="nvidia-sm120", bufferize_opt_level="fast"))
        with pytest.raises(CodegenError, match="Native embedding"):
            describe_cpp_package(compiler.compile(module).module)
