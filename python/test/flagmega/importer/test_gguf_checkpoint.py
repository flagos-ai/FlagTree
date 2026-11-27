# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GGUF aliases, native storage, and safetensors coexistence contracts."""

import io
from types import SimpleNamespace

import numpy as np
import pytest

from triton.flagmega.artifacts.rdata import _write_rdata_range
from triton.flagmega.errors import ArtifactError, ImporterError
from triton.flagmega.importer import GGUFCheckpoint, TensorByteRange, checkpoint_from_path
from triton.flagmega.importer.qwen3_5 import Qwen35Layer0Importer
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.artifacts import pack_rdata
from triton.flagmega.targets import Sm120Capability

gguf = pytest.importorskip("gguf")
torch = pytest.importorskip("torch")


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "qwen-iq4.gguf"
    writer = gguf.GGUFWriter(path, arch="qwen35")
    for key, value in (("embedding_length", 256), ("feed_forward_length", 512), ("block_count", 1),
                       ("ssm.inner_size", 256), ("ssm.group_count", 2), ("ssm.time_step_rank", 2), ("ssm.conv_kernel",
                                                                                                    4)):
        writer.add_uint32("qwen35." + key, value)
    rng = np.random.default_rng(42)

    def packed(name, rows, k, encoding):
        size = 136 if encoding == gguf.GGMLQuantizationType.IQ4_XS else 144
        values = rng.integers(0, 256, (rows, k // 256 * size), dtype=np.uint8)
        writer.add_tensor(name, values, raw_dtype=encoding)

    iq4 = gguf.GGMLQuantizationType.IQ4_XS
    q4 = gguf.GGMLQuantizationType.Q4_K
    packed("token_embd.weight", 256, 256, iq4)
    packed("blk.0.attn_qkv.weight", 768, 256, iq4)
    packed("blk.0.attn_gate.weight", 256, 256, q4)
    packed("blk.0.ssm_beta.weight", 2, 256, iq4)
    packed("blk.0.ssm_alpha.weight", 2, 256, q4)
    packed("blk.0.ssm_out.weight", 256, 256, iq4)
    packed("blk.0.ffn_gate.weight", 512, 256, q4)
    packed("blk.0.ffn_up.weight", 512, 256, q4)
    packed("blk.0.ffn_down.weight", 256, 512, iq4)
    for name, shape, value in (("attn_norm.weight", (256, ), 1.25), ("post_attention_norm.weight", (256, ),
                                                                     1.5), ("ssm_conv1d.weight", (768, 4), 0.25),
                               ("ssm_a", (2, ), -2.0), ("ssm_dt.bias", (2, ), 0.1), ("ssm_norm.weight", (128, ), 0.5)):
        writer.add_tensor("blk.0." + name, np.full(shape, value, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return GGUFCheckpoint(path)


def test_gguf_preserves_packed_storage_and_scalar_semantics(checkpoint):
    prefix = "model.language_model.layers.0.linear_attn."
    assert isinstance(checkpoint_from_path(checkpoint.path), GGUFCheckpoint)
    key = prefix + "in_proj_qkv.weight"
    assert checkpoint.tensor_info(key).shape == (768, 256)
    from triton.flagmega.ir import DType
    assert checkpoint.tensor_info(key).dtype == DType.UINT8
    assert checkpoint.quantization_of(key) == "IQ4_XS"
    span = checkpoint.tensor_byte_range(key)
    assert span.nbytes == 768 * 136
    assert span.quant_type == "IQ4_XS"
    with pytest.raises(ImporterError, match="native quantized"):
        checkpoint.load_tensor(key)
    assert checkpoint.load_tensor(prefix + "conv1d.weight").shape == (768, 1, 4)
    torch.testing.assert_close(checkpoint.load_tensor(prefix + "A_log"),
                               torch.full((2, ), np.log(2), dtype=torch.float32))
    assert torch.all(checkpoint.load_tensor("blk.0.ssm_a") == -2)
    assert torch.all(checkpoint.load_tensor("model.language_model.layers.0.input_layernorm.weight") == 0.25)
    assert torch.all(checkpoint.load_tensor("blk.0.attn_norm.weight") == 1.25)


def test_gguf_compile_copies_quantized_ranges_without_materializing(checkpoint, tmp_path):
    imported = Qwen35Layer0Importer(checkpoint).import_module()
    assert imported.node_map["gated_delta_net"].attrs["value_head_order"] == "tiled"
    module = Compiler(CompileOptions(target="nvidia-sm120")).compile(imported).module
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    copied = 0
    with checkpoint.path.open("rb") as original, (assets / "rdata.bin").open("rb") as packed:
        for recipe in module.constant_recipes:
            weights = [node for node in recipe.nodes if node.op == "builtin.weight"]
            if len(weights) != 1 or checkpoint.quantization_of(weights[0].attrs["key"]) is None:
                continue
            span = checkpoint.tensor_byte_range(weights[0].attrs["key"])
            entry = next(value for value in index["entries"] if value["key"] == recipe.outputs[0])
            original.seek(span.offset)
            packed.seek(entry["offset"])
            assert packed.read(span.nbytes) == original.read(span.nbytes)
            assert entry["nbytes"] == span.nbytes
            copied += 1
    assert copied == 9
    assert Sm120Capability.from_data(Sm120Capability().to_data()) == Sm120Capability()


def test_plain_checkpoint_byte_range_still_requires_exact_allocation(tmp_path):
    backing = tmp_path / "raw.bin"
    backing.write_bytes(b"abc")
    buffer = SimpleNamespace(weight_key="weight", nbytes=4)
    with pytest.raises(ArtifactError, match="IR requires 4"):
        _write_rdata_range(io.BytesIO(), buffer, TensorByteRange(backing, 0, 3))


def test_dense_gguf_imports_all_layers_with_exact_packed_storage(checkpoint, tmp_path):
    from math import prod
    from triton.flagmega.importer.model import import_model
    from triton.flagmega.ir import DType

    path = tmp_path / 'hybrid-iq4.gguf'
    writer = gguf.GGUFWriter(path, arch='qwen35')
    metadata = {
        'embedding_length': 256, 'feed_forward_length': 512, 'block_count': 4, 'ssm.inner_size': 256, 'ssm.group_count':
        2, 'ssm.time_step_rank': 2, 'ssm.conv_kernel': 4, 'attention.head_count': 4, 'attention.head_count_kv': 2,
        'attention.key_length': 64, 'rope.dimension_count': 16
    }
    for key, value in metadata.items():
        writer.add_uint32('qwen35.' + key, value)
    for name, tensor in checkpoint._raw.items():
        names = [name.replace('blk.0.', f'blk.{i}.') for i in range(3)] if name.startswith('blk.0.') else [name]
        for target in names:
            writer.add_tensor(target, tensor.data, raw_dtype=tensor.tensor_type)
    writer.add_tensor('output.weight', checkpoint._raw['token_embd.weight'].data,
                      raw_dtype=gguf.GGMLQuantizationType.IQ4_XS)
    writer.add_tensor('output_norm.weight', np.ones((256, ), dtype=np.float32))
    for part in ('attn_norm', 'post_attention_norm'):
        writer.add_tensor(f'blk.3.{part}.weight', np.ones((256, ), dtype=np.float32))
    for part in ('gate', 'up', 'down'):
        raw = checkpoint._raw[f'blk.0.ffn_{part}.weight']
        writer.add_tensor(f'blk.3.ffn_{part}.weight', raw.data, raw_dtype=raw.tensor_type)
    for part, n in (('q', 512), ('k', 128), ('v', 128), ('output', 256)):
        writer.add_tensor(f'blk.3.attn_{part}.weight', np.zeros((n, 136), dtype=np.uint8),
                          raw_dtype=gguf.GGMLQuantizationType.IQ4_XS)
    for part in ('q', 'k'):
        writer.add_tensor(f'blk.3.attn_{part}_norm.weight', np.ones((64, ), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    source = GGUFCheckpoint(path)
    imported = import_model(source)
    assert imported.metadata['num_hidden_layers'] == 4
    assert imported.metadata['layer_types'][-1] == 'full_attention'
    assert len([n for n in imported.nodes if n.op == 'builtin.call']) == 4
    for node in imported.nodes:
        if node.op == 'builtin.weight' and source.quantization_of(node.attrs['key']) is not None:
            assert node.type.dtype == DType.UINT8
            assert prod(d.fixed_value for d in node.type.shape) == source.tensor_byte_range(node.attrs['key']).nbytes
    module = Compiler(CompileOptions(target='nvidia-sm120')).compile(imported).module
    index = pack_rdata(module, source, tmp_path / 'hybrid-assets')
    assert index['entries']
    logits = next(n for n in imported.nodes if n.id == 'logits')
    assert logits.type.dtype == DType.FLOAT32
