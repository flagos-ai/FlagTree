# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sparse attention must tolerate empty tiles before and after visible keys."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load
from python.test.flagmega.devices import requires_nvidia_sm90


def attention_module(config, width, *, probability_rounding=None, index_traversal=None):
    class Graph(fm.Module):
        def forward(self):
            query_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 2, 2))
            query = self.input("query", query_type, id="query")
            topk = self.input("topk", fm.tensor_type("int32", (1, width)), id="topk")
            state = self.input("state", config.ref_type, id="state")
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            result = fm.F.nn.dsa_paged_attention(
                query, topk, state, layer, scale=16 ** -.5,
                layout=("seq", "head", "dim"), hidden_size=32, name="attention",
                **({} if probability_rounding is None else {"probability_rounding": probability_rounding}),
                **({} if index_traversal is None else {"index_traversal": index_traversal}))
            self.function("main", (query, topk, state), (result,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def prepare(module, directory, inputs, state, output):
    module = Compiler().compile(module).module
    artifact = write_artifact(module, directory, target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    binding = runtime.buffer_plan.function_map[module.entry]
    values = {}
    for value, buffers in binding.parameters:
        if isinstance(module.node_map[value].type, fm.RefType):
            values.update(zip(buffers, (state.kv_caches, state.query_start_loc, state.seq_lens,
                                      state.slot_mapping, state.block_table), strict=True))
        else:
            values[buffers[0]] = inputs[module.node_map[value].attrs["name"]]
    for _, buffers in binding.outputs:
        values[buffers[0]] = output
    arguments = tuple(values[str(argument["buffer"])] for argument in runtime.external_arguments)
    runtime.prepare(*arguments)
    return runtime, arguments


@requires_nvidia_sm90
@pytest.mark.parametrize("slot", [0, 64, 129])
def test_visible_key_survives_leading_and_trailing_empty_tiles(tmp_path, slot):
    config = PagedAttentionStateConfig(1, 1, 16, num_blocks=2)
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(1)
    state.slot_mapping.zero_()
    query = torch.ones((1, 2, 2, 8), dtype=torch.bfloat16, device="cuda")
    topk = torch.full((1, 130), -1, dtype=torch.int32, device="cuda")
    topk[0, slot] = 0
    expected = torch.arange(16, dtype=torch.bfloat16, device="cuda") / 8
    state.kv_caches[0, 0, 0, 0].fill_(-3)
    state.kv_caches[0, 0, 1, 0].copy_(expected.reshape(1, 2, 8))
    output = torch.empty_like(query)
    runtime, arguments = prepare(attention_module(config, 130), tmp_path / "attention",
                                 {"query": query, "topk": topk}, state, output)
    try:
        for _ in range(3):
            output.fill_(float("nan"))
            runtime.run_into(*arguments)
            torch.testing.assert_close(output.reshape(2, 16), expected.expand(2, 16), rtol=0, atol=0)
            query.mul_(-1)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@requires_nvidia_sm90
def test_indices_after_current_query_are_not_visible(tmp_path):
    config = PagedAttentionStateConfig(1, 1, 16, num_blocks=2)
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(1)
    state.slot_mapping.zero_()
    query = torch.zeros((1, 2, 2, 8), dtype=torch.bfloat16, device="cuda")
    topk = torch.tensor([[7, 0]], dtype=torch.int32, device="cuda")
    state.kv_caches[0, 0, 1, 0].fill_(1)
    state.kv_caches[0, 0, 1, 7].fill_(99)
    output = torch.empty_like(query)
    runtime, arguments = prepare(attention_module(config, 2), tmp_path / "attention",
                                 {"query": query, "topk": topk}, state, output)
    try:
        runtime.run_into(*arguments)
        torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
    finally:
        runtime.close()


@requires_nvidia_sm90
def test_probabilities_round_before_value_product(tmp_path):
    config = PagedAttentionStateConfig(1, 1, 16, num_blocks=2)
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(3)
    state.slot_mapping.fill_(2)
    query = torch.zeros((1, 2, 2, 8), dtype=torch.bfloat16, device="cuda")
    topk = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    for position, value in enumerate((1, 2, 4)):
        state.kv_caches[0, 0, 1, position].fill_(value)
    output = torch.empty_like(query)
    runtime, arguments = prepare(attention_module(config, 3), tmp_path / "attention",
                                 {"query": query, "topk": topk}, state, output)
    try:
        runtime.run_into(*arguments)
        # The semantic op casts softmax probabilities to the query dtype
        # before BF16 attention. 3 * BF16(1/3) is distinguishable at this sum.
        probability = torch.tensor(1 / 3, dtype=torch.bfloat16).float()
        expected = (probability * 7).bfloat16().item()
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
    finally:
        runtime.close()
