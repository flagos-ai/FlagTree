# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native FA3 probabilities retain their origin across context owners."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load
from python.test.flagmega.codegen.triton.kernels.paged_attention_partial.test_causal_chunk_runtime import chunk_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fixture_name", ["native_flash_attention_online_bf16.json",
                                          "native_flash_attention_online_bf16_l11.json",
                                          "native_flash_attention_short_causal_tile.json",
                                          "native_flash_attention_fma_reciprocal.json"])
def test_native_online_bf16_attention_with_empty_context_owners(tmp_path, fixture_name):
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          fixture_name).read_text())
    q, k, v, expected = (torch.tensor(fixture[n], dtype=torch.bfloat16, device="cuda")
                         for n in ("query", "key", "value", "output"))
    if q.shape[1] == 1:
        q = q.repeat(1, 2, 1)
        expected = expected.repeat(1, 2, 1)
    heads, kv_heads, length = q.shape[1], k.shape[1], k.shape[0]
    config = PagedAttentionStateConfig(3, kv_heads, 256, block_size=64, num_blocks=1)
    source = chunk_attention(config, 1, ("seq", "head", "dim"), False, True, query_heads=heads)
    source = fm.verify_module(replace(source, nodes=tuple(
        replace(n, attrs={**n.attrs, "probability_rounding": "online_bf16",
                          "softmax_block_size": 80, "softmax_traversal": "reverse"})
        if n.op == "ntt.paged_attention_partial" else n for n in source.nodes)))
    compiled = Compiler().compile(source).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90",
                                 emit_executable=True), device="cuda:0")
    state = create_paged_attention_state(config, device="cuda")
    for row in range(length):
        state.kv_caches[0, 1, 0, row].copy_(k[row].reshape(kv_heads, 32, 8))
        state.kv_caches[0, 1, 1, row].copy_(v[row].reshape(kv_heads, 32, 8))
    state.seq_lens.fill_(length)
    state.slot_mapping.fill_(length - 1)
    state.query_start_loc.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    query = q.reshape(1, heads, 32, 8)
    output = torch.full_like(query, float("nan"))
    binding = runtime.buffer_plan.function_map[compiled.entry]
    fields = (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table)
    buffers = {}
    for value, names in binding.parameters:
        if isinstance(compiled.node_map[value].type, fm.RefType):
            buffers.update(zip(names, fields, strict=True))
        else:
            buffers[names[0]] = query
    for _, names in binding.outputs:
        buffers[names[0]] = output
    arguments = tuple(buffers[str(arg["buffer"])] for arg in runtime.external_arguments)
    try:
        runtime.prepare(*arguments)
        runtime.run_into(*arguments)
        torch.testing.assert_close(output.reshape_as(expected), expected, rtol=0, atol=0)
        reports = runtime.resource_report
        reports = reports if isinstance(reports, tuple) else (reports,)
        assert all(r["spill_bytes"] == 0 for r in reports)
    finally:
        runtime.close()
