# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega.evaluator import TorchEvaluator, create_paged_attention_state
from triton.flagmega.evaluator import CheckpointWeightResolver
from triton.flagmega.importer import Qwen3ModelImporter, import_model
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.ir import emit_module, load_module

from .helpers import full_checkpoint


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
