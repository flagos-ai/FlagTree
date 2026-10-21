# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The composite adapter binds an explicit request phase on every launch."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError, RuntimeContractError
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCompositeModelModule


def graph():
    class Graph(fm.Module):
        def forward(self):
            ids = self.input("input_ids", fm.tensor_type("int32", (1,)), id="input_ids")
            phase = self.input("prefill", fm.tensor_type("bool", ()), id="prefill")
            conv = self.input("conv", fm.tensor_type("bfloat16", (3, 1, 8, 2)), id="conv")
            state = self.input("recurrent", fm.tensor_type("float32", (1, 1, 8, 8)), id="recurrent")
            paged = self.input("paged", PagedAttentionStateConfig(1, 1, 8, block_size=8, num_blocks=1).ref_type,
                               id="paged")
            scalar = fm.F.tensors.cast(phase, "float32")
            logits = fm.F.math.add(fm.F.tensors.broadcast_to(scalar, shape=(1, 8)),
                                    fm.F.tensors.broadcast_to(fm.F.tensors.cast(ids, "float32"), shape=(1, 8)),
                                    name="logits")
            token = fm.F.nn.greedy_sample(logits, name="token")
            conv_out = fm.F.math.add(conv, fm.F.tensors.broadcast_to(fm.F.tensors.cast(phase, "bfloat16"),
                                                                    shape=(3, 1, 8, 2)))
            state_out = fm.F.math.add(state, fm.F.tensors.broadcast_to(scalar, shape=(1, 1, 8, 8)))
            self.builder.metadata["composite_abi"] = {
                "schema": "flagmega.composite-abi/v1", "input_id": ids.id, "prefill_input": phase.id,
                "logits_output": logits.id, "token_output": token.id,
                "tensor_states": ({"state_id": 0, "input": (conv.id, state.id),
                                   "output": (conv_out.id, state_out.id)},),
                "paged_state": {"input": paged.id, "output": paged.id},
            }
            self.function("main", (ids, phase, conv, state, paged), (logits, token, conv_out, state_out, paged))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires CUDA")
def test_prepared_composite_adapter_switches_phase_and_restarts(tmp_path):
    compiled = Compiler().compile(graph()).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    assert isinstance(runtime, GeneratedTirCompositeModelModule)
    try:
        state = runtime.create_state()
        ids = torch.tensor([11], device="cuda", dtype=torch.int32)
        logits, token = runtime.create_outputs()
        runtime.prepare(ids, state, logits=logits, next_token=token, prefill=True)
        for phase in (True, False, True):
            state.kda.convolution.zero_()
            state.kda.recurrent.zero_()
            runtime.run_into(logits, token, ids, state, prefill=phase)
            torch.testing.assert_close(logits, torch.full_like(logits, 11 + int(phase)), rtol=0, atol=0)
            torch.testing.assert_close(state.kda.recurrent, torch.full_like(state.kda.recurrent, int(phase)),
                                       rtol=0, atol=0)
            torch.testing.assert_close(state.kda.convolution, torch.full_like(state.kda.convolution, int(phase)),
                                       rtol=0, atol=0)
        for bad in (1, "prefill", torch.tensor(True)):
            with pytest.raises(RuntimeContractError, match="Python bool"):
                runtime.run_into(logits, token, ids, state, prefill=bad)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("binding,message", [("missing", "unknown prefill"),
                                            ("input_ids", "rank-zero bool")])
def test_composite_phase_metadata_is_checked_before_loading_a_device(tmp_path, binding, message):
    source = graph()
    source = fm.verify_module(replace(source, metadata={**source.metadata,
        "composite_abi": {**source.metadata["composite_abi"], "prefill_input": binding}}))
    compiled = Compiler().compile(source).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    with pytest.raises(ArtifactError, match=message):
        load(artifact)
