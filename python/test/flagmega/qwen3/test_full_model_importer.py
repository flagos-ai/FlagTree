# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega.evaluator import TorchEvaluator, create_paged_attention_state
from triton.flagmega.evaluator import CheckpointWeightResolver
from triton.flagmega.importer import Qwen3LayerImporter, Qwen3ModelImporter, import_model
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.ir import emit_module, load_module

from .helpers import full_checkpoint, full_checkpoint_wmma, full_metadata_checkpoint


def test_full_model_importer_builds_every_layer_lm_head_and_sampling():
    source = full_checkpoint(num_hidden_layers=2)
    importer = Qwen3ModelImporter(source, block_size=4, num_blocks=2)
    module = importer.import_module()
    function = module.function_map["main"]

    assert module.metadata["num_layers"] == 2
    assert module.metadata["output_boundary"] == "logits_fp32_and_greedy_token"
    assert function.parameters == ("input_ids", "paged_attention_kv_cache")
    assert function.outputs == ("logits", "next_token", "updated_state")
    decode_layer = module.function_map["decode_layer"]
    assert decode_layer.attrs == {
        "calling_convention": "device",
        "noinline": True,
        "reusable": True,
    }
    assert module.node_map["decode_layer_qkv_projection"].op == (
        "nn.qkv_parallel_linear"
    )
    assert module.node_map["decode_layer_query_rope"].op == "nn.rope"
    assert module.node_map["decode_layer_key_cache_update"].op == (
        "nn.update_paged_attention_kv_cache"
    )
    assert module.node_map["decode_layer_paged_attention"].op == (
        "nn.paged_attention"
    )
    assert module.node_map["decode_layer_attention_output"].op == "math.matmul"
    calls = tuple(node for node in module.nodes if node.op == "builtin.call")
    assert tuple(node.attrs["callee"] for node in calls) == ("decode_layer", "decode_layer")
    assert module.node_map["layer_0_id"].attrs["value"] == 0
    assert module.node_map["layer_1_id"].attrs["value"] == 1
    assert module.node_map["layer_0_advance_sequence"].attrs["value"] is False
    assert module.node_map["layer_1_advance_sequence"].attrs["value"] is True
    assert module.node_map["lm_head"].inputs[1] == "w_embed_tokens_weight"
    assert module.node_map["next_token"].op == "nn.greedy_sample"


def test_full_model_python_ir_round_trips_and_architecture_dispatches(tmp_path):
    module = import_model(full_checkpoint(num_hidden_layers=2))
    path = emit_module(module, tmp_path / "qwen3_full.py")
    source = path.read_text(encoding="utf-8")

    assert "F.nn.greedy_sample(" in source
    assert "F.builtin.call(" in source
    assert "name='layer_1_decode_layer_call'" in source
    assert "'noinline': True" in source
    assert load_module(path).semantic_hash == module.semantic_hash


def test_full_model_evaluator_updates_all_layer_cache_and_samples_argmax():
    source = full_checkpoint(num_hidden_layers=2)
    importer = Qwen3ModelImporter(source, block_size=4, num_blocks=2)
    module = importer.import_module()
    state = create_paged_attention_state(importer.state_config)

    logits, next_token, updated_state = TorchEvaluator(
        CheckpointWeightResolver(source)
    ).run(
        module,
        {
            "input_ids": torch.tensor([3], dtype=torch.int32),
            "paged_attention_kv_cache": state,
        },
    )

    assert logits.shape == (1, 32)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(next_token, logits.argmax(dim=-1).to(torch.int32))
    assert updated_state is state
    assert state.sequence_length == 1
    assert torch.count_nonzero(state.kv_caches[:, :2]).item() > 0


def test_full_model_function_dump_separates_main_and_decode_layer_and_merges(tmp_path):
    module = import_model(full_checkpoint(num_hidden_layers=2))
    dumper = DumpManager(tmp_path, DumpFlags.PASS_IR).root

    emitted = dumper.dump_module(module, "After", category=DumpFlags.PASS_IR)

    assert emitted is not None
    assert {value.function for value in emitted.functions} == {"main", "decode_layer"}
    main_source = (tmp_path / "After" / "main.py").read_text(encoding="utf-8")
    layer_source = (tmp_path / "After" / "decode_layer.py").read_text(encoding="utf-8")
    assert "F.builtin.call(" in main_source
    assert "F.nn.qwen3_paged_attention(" not in main_source
    assert "F.nn.qkv_parallel_linear(" in layer_source
    assert "F.nn.rotary_embedding(" in main_source
    assert "F.nn.rotary_embedding(" not in layer_source
    assert "F.nn.rope(" in layer_source
    assert "F.nn.update_paged_attention_kv_cache(" in layer_source
    assert "F.nn.paged_attention(" in layer_source
    assert "F.nn.qwen3_paged_attention(" not in layer_source
    assert "F.builtin.call(" not in layer_source
    assert load_module(tmp_path / "After") == module


@pytest.mark.parametrize('importer_class, prefix', [
    (Qwen3LayerImporter, 'self_attention'), (Qwen3ModelImporter, 'decode_layer'),
])
def test_qwen3_06b_attention_width_is_independent_of_hidden_width(tmp_path, importer_class, prefix):
    source = full_metadata_checkpoint(num_hidden_layers=28, hidden=1024, intermediate=3072, vocab=151936)
    importer = importer_class(source)
    module = importer.import_module()
    assert importer.config.query_size == 2048
    assert importer.config.hidden_size == 1024
    assert importer.config.kv_size == 1024
    shapes = {
        'q_weight_kn': (1024, 2048), 'k_weight_kn': (1024, 1024), 'v_weight_kn': (1024, 1024),
        'query': (1, 16, 128), 'key': (1, 8, 128), 'value': (1, 8, 128), 'attention_merged': (1, 2048),
    }
    for suffix, expected in shapes.items():
        node = module.node_map[prefix + '_' + suffix]
        assert tuple(d.fixed_value for d in node.type.shape) == expected
    output_id = 'attention_output' if importer_class is Qwen3LayerImporter else 'decode_layer_attention_output'
    output = module.node_map[output_id]
    weight = module.node_map[output.inputs[1]]
    assert tuple(d.fixed_value for d in weight.type.shape) == (1024, 2048)
    assert tuple(d.fixed_value for d in output.type.shape) == (1, 1024)
    if importer_class is Qwen3ModelImporter:
        calls = [node for node in module.nodes if node.op == 'builtin.call']
        assert len(calls) == 28
        assert all(node.attrs['callee'] == 'decode_layer' for node in calls)
    path = emit_module(module, tmp_path / 'qwen3_06b.py')
    assert load_module(path).semantic_hash == module.semantic_hash


def _reference_wide_attention_step(source, token, position, history):
    config = source.config
    linear = torch.nn.functional.linear

    def norm(value, key):
        scaled = value.float() * (value.float().square().mean(-1, keepdim=True) + config['rms_norm_eps']).rsqrt()
        return scaled.to(value.dtype) * source.load_tensor(key)

    def rope(value):
        dim = config['head_dim']
        angles = position * config['rope_theta'] ** (-torch.arange(0, dim, 2).float() / dim)
        angles = torch.cat((angles, angles))
        value = value.float()
        rotated = torch.cat((-value[..., dim // 2:], value[..., :dim // 2]), -1)
        return (value * angles.cos() + rotated * angles.sin()).bfloat16()

    hidden = source.load_tensor('model.embed_tokens.weight')[token:token + 1]
    for layer in range(config['num_hidden_layers']):
        prefix = f'model.layers.{layer}.'
        normalized = norm(hidden, prefix + 'input_layernorm.weight')
        query, key, value = [linear(normalized, source.load_tensor(prefix + 'self_attn.' + part + '_proj.weight'))
                             .reshape(-1, config['head_dim']) for part in ('q', 'k', 'v')]
        query = rope(norm(query, prefix + 'self_attn.q_norm.weight'))
        key = rope(norm(key, prefix + 'self_attn.k_norm.weight'))
        history[layer].append((key, value))
        keys, values = [torch.stack([pair[i] for pair in history[layer]]) for i in (0, 1)]
        groups = config['num_attention_heads'] // config['num_key_value_heads']
        keys = keys.repeat_interleave(groups, dim=1).transpose(0, 1)
        values = values.repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = torch.einsum('hd,hsd->hs', query.float(), keys.float()) * config['head_dim'] ** -0.5
        probabilities = scores.softmax(-1).bfloat16()
        attended = torch.einsum('hs,hsd->hd', probabilities, values).reshape(1, -1)
        hidden = hidden + linear(attended, source.load_tensor(prefix + 'self_attn.o_proj.weight'))
        normalized = norm(hidden, prefix + 'post_attention_layernorm.weight')
        gate = linear(normalized, source.load_tensor(prefix + 'mlp.gate_proj.weight'))
        up = linear(normalized, source.load_tensor(prefix + 'mlp.up_proj.weight'))
        hidden = hidden + linear(torch.nn.functional.silu(gate) * up, source.load_tensor(prefix + 'mlp.down_proj.weight'))
    hidden = norm(hidden, 'model.norm.weight')
    return hidden.float(), linear(hidden, source.load_tensor('model.embed_tokens.weight')).float()


@pytest.mark.parametrize('importer_class, layers', [(Qwen3LayerImporter, 1), (Qwen3ModelImporter, 2)])
def test_wide_attention_graph_matches_dense_reference_across_pages(tmp_path, importer_class, layers):
    source = full_checkpoint_wmma(num_hidden_layers=layers, hidden=16, intermediate=32, vocab=32,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    importer = importer_class(source, block_size=2, num_blocks=2)
    module = load_module(emit_module(importer.import_module(), tmp_path / 'wide_attention.py'))
    state = create_paged_attention_state(importer.state_config)
    evaluator = TorchEvaluator(CheckpointWeightResolver(source))
    history = [[] for _ in range(layers)]
    for position, token in enumerate((3, 7, 5)):
        hidden, logits = _reference_wide_attention_step(source, token, position, history)
        outputs = evaluator.run(module, {'input_ids': torch.tensor([token], dtype=torch.int32),
                                         'paged_attention_kv_cache': state})
        expected = hidden if importer_class is Qwen3LayerImporter else logits
        torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
        assert outputs[-1] is state
        assert state.sequence_length == position + 1
        if importer_class is Qwen3ModelImporter:
            torch.testing.assert_close(outputs[1], logits.argmax(-1).int(), rtol=0, atol=0)
        for layer in range(layers):
            keys, values = state.gather(layer_id=layer, length=position + 1)
            torch.testing.assert_close(keys, torch.stack([pair[0] for pair in history[layer]]), rtol=0, atol=0)
            torch.testing.assert_close(values, torch.stack([pair[1] for pair in history[layer]]), rtol=0, atol=0)
