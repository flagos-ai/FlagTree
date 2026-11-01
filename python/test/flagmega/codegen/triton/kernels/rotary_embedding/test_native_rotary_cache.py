# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compare generated tables with the pinned engine's actual BF16 cache."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("noinline", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_native_rotary_cache_across_complete_request_positions(tmp_path, decode, noinline, cached):
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_rotary_cache.json").read_text())
    capacity_tokens, width = fixture["tokens"], fixture["head_dim"]
    tokens = 1 if decode else capacity_tokens
    config = PagedAttentionStateConfig(1, 2, width)

    class Rotary(fm.Module):
        def forward(self):
            reference = self.input("reference", fm.tensor_type("bfloat16", (tokens, 17)))
            state = self.input("state", config.ref_type)
            cache = (fm.F.nn.rotary_cache_table(positions=capacity_tokens,
                inverse_frequencies=tuple(fixture["inverse_frequencies"])) if cached else None)
            pair = fm.F.nn.rotary_embedding(
                reference, state, *((cache,) if cached else ()),
                head_dim=width, theta=fixture["theta"], output_dtype="bfloat16",
                output_lanes=(8,), math_mode="accurate",
                inverse_frequencies=tuple(fixture["inverse_frequencies"]),
            )
            pair_outputs = tuple(fm.F.tensors.get_item(pair, index=i) for i in range(2))
            if noinline:
                self.function("rotary", (reference, state), pair_outputs,
                              attrs={"noinline": True, "reusable": True})
                entry_reference = self.input("entry_reference", reference.type)
                entry_state = self.input("entry_state", state.type)
                result = fm.F.builtin.call(entry_reference, entry_state,
                    result_type=pair.type, callee="rotary")
                self.function("main", (entry_reference, entry_state),
                    tuple(fm.F.tensors.get_item(result, index=i) for i in range(2)))
            else:
                self.function("main", (reference, state), pair_outputs)

    module = Rotary(dialect="high_level", stage="imported", entry="main",
                    metadata={"auto_distribution": {"placement": fm.Placement((2, 2), "yx", "bb").to_data()}}).build()
    module = Compiler().compile(module).module
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90",
                                 checkpoint=MemoryCheckpoint({}, {}, {}),
                                 emit_executable=True), device="cuda:0")
    state = create_paged_attention_state(config, device="cuda")
    reference = torch.zeros((tokens, 17), dtype=torch.bfloat16, device="cuda")
    outputs = [torch.empty((tokens, 1, width), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    values = {}
    binding = runtime.buffer_plan.function_map[module.entry]
    for name, buffers in binding.parameters:
        if isinstance(module.node_map[name].type, fm.RefType):
            fields = (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table)
            values.update(zip(buffers, fields, strict=True))
        else:
            values[buffers[0]] = reference
    output_buffers = tuple(buffer for _, buffers in binding.outputs for buffer in buffers)
    values.update(zip(output_buffers, outputs, strict=True))
    arguments = tuple(values[arg["buffer"]] for arg in runtime.external_arguments)
    try:
        runtime.prepare(*arguments)
        if decode:
            packed = torch.empty((capacity_tokens, width), dtype=torch.bfloat16, device="cuda")
            for position in range(capacity_tokens):
                state.seq_lens.fill_(position)
                runtime.run_into(*arguments)
                packed[position].copy_(torch.cat((outputs[0][0, 0, :width // 2],
                                                 outputs[1][0, 0, :width // 2])))
        else:
            runtime.run_into(*arguments)
            packed = torch.cat((outputs[0][:, 0, :width // 2], outputs[1][:, 0, :width // 2]), dim=-1)
        digest = hashlib.sha256(packed.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert digest == fixture["cache_sha256"]
        reports = runtime.resource_report
        reports = reports if isinstance(reports, tuple) else (reports,)
        assert all(report["spill_bytes"] == 0 for report in reports)
    finally:
        runtime.close()
