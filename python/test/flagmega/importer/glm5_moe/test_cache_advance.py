# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A decoder writes K/V at the same position and advances exactly once."""

import pytest
import torch

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer.glm5_moe import Glm5MoeImporter
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
from triton.flagmega.runtime import load

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint


@pytest.mark.parametrize("past", [0, 7, 8])
def test_mla_evaluator_writes_both_cache_fields_at_the_original_position(past):
    source = checkpoint(REDUCED)
    importer = Glm5MoeImporter(source, block_size=8, num_blocks=3)
    module = importer.import_module()
    state = create_paged_attention_state(importer.paged_config)
    state.kv_caches.fill_(.25)
    state.seq_lens.fill_(past)
    evaluator = TorchEvaluator(CheckpointWeightResolver(source))
    for step, token in enumerate((11, 17, 5)):
        position = past + step
        _, trace = evaluator.run_with_trace(module, {
            "input_ids": torch.tensor([token], dtype=torch.int32),
            "paged_attention_state": state,
        })
        assert state.sequence_length == position + 1
        assert state.slot_mapping.item() == position
        block, offset = divmod(position, 8)
        for field, name in enumerate(("key", "value")):
            expected = trace[f"decode_mla_attention_{name}_heads"].squeeze(0)
            torch.testing.assert_close(state.kv_caches[block, 0, field, offset].reshape_as(expected),
                                       expected, rtol=0, atol=0)
        next_block, next_offset = divmod(position + 1, 8)
        assert (state.kv_caches[next_block, 0, :, next_offset] == .25).all()


@pytest.mark.cuda
@pytest.mark.parametrize("graph_mode", [False, True])
def test_mla_device_advances_once_across_page_boundary(tmp_path, graph_mode):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    source = checkpoint(REDUCED)
    importer = Glm5MoeImporter(source, block_size=8, num_blocks=3)
    module = importer.import_module()
    artifact = write_artifact(Compiler().compile(module).module, tmp_path / "artifact",
                              target="nvidia-sm90", emit_executable=True, checkpoint=source)
    evaluator = TorchEvaluator(CheckpointWeightResolver(source))
    reference_state = create_paged_attention_state(importer.paged_config)
    reference_state.kv_caches.fill_(.25)
    reference_state.seq_lens.fill_(7)
    with load(artifact, device="cuda:0") as runtime:
        state = runtime.create_state()
        state.kv_caches.fill_(.25)
        state.seq_lens.fill_(7)
        ids = torch.tensor([11], dtype=torch.int32, device="cuda:0")
        logits, token = runtime.create_outputs()
        runtime.prepare(ids, state, logits=logits, next_token=token)

        def launch():
            runtime.run_into(logits, token, ids, state)

        if graph_mode:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            step = graph.replay
        else:
            step = launch
        for index, value in enumerate((11, 17, 5)):
            expected = evaluator.run(module, {"input_ids": torch.tensor([value], dtype=torch.int32),
                                             "paged_attention_state": reference_state})
            ids.fill_(value)
            step()
            torch.cuda.synchronize()
            assert state.sequence_length == reference_state.sequence_length == 8 + index
            assert state.slot_mapping.item() == 7 + index
            torch.testing.assert_close(logits.cpu(), expected[0], atol=.05, rtol=.05)
            torch.testing.assert_close(state.kv_caches.cpu(), reference_state.kv_caches, atol=.05, rtol=.05)
