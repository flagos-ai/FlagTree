# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prepared composite launches retain structural checks without host state reads."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import EvaluationError, RuntimeContractError
from triton.flagmega.ir.ops.nn._dsa_indexer_state import DsaIndexerStateConfig, create_dsa_indexer_state
from triton.flagmega.runtime import load

from .test_composite_request_phase import graph


def _with_dsa(module):
    config = DsaIndexerStateConfig((0,), 8, 4, 2)
    dsa = fm.Node("dsa", "builtin.var", (), config.ref_type, attrs={"name": "dsa"})
    function, = module.functions
    return fm.verify_module(replace(module, nodes=(*module.nodes, dsa), functions=(replace(function,
        parameters=(*function.parameters, dsa.id), outputs=(*function.outputs, dsa.id)),),
        metadata={**module.metadata, "dsa_full_indexer_layers": (0,), "dsa_index_output_width": 2,
            "composite_abi": {**module.metadata["composite_abi"],
                              "dsa_state": {"input": dsa.id, "output": dsa.id}}}))


def test_dsa_static_validation_does_not_inspect_mutable_values(monkeypatch):
    state = create_dsa_indexer_state(DsaIndexerStateConfig((0,), 8, 4, 2))
    original = torch.Tensor.item

    def no_host_read(value, *args):
        if value.untyped_storage().data_ptr() == state.seq_lens.untyped_storage().data_ptr():
            raise AssertionError("Unexpected synchronous state read")
        return original(value, *args)

    monkeypatch.setattr(torch.Tensor, "item", no_host_read)
    state.validate(dynamic_values=False)
    state.seq_lens = state.seq_lens.to(torch.int64)
    with pytest.raises(EvaluationError, match="seq_lens"):
        state.validate(dynamic_values=False)


def test_dsa_default_validation_still_rejects_invalid_capacity():
    state = create_dsa_indexer_state(DsaIndexerStateConfig((0,), 8, 4, 2))
    state.seq_lens.fill_(9)
    with pytest.raises(EvaluationError, match="capacity"):
        state.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_dsa", (False, True))
def test_prepared_composite_captures_and_replays_both_phases(tmp_path, with_dsa):
    module = graph()
    if with_dsa:
        module = _with_dsa(module)
    module = Compiler().compile(module).module
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    try:
        state = runtime.create_state()
        ids = torch.tensor([11], device="cuda", dtype=torch.int32)
        logits, token = runtime.create_outputs()
        runtime.prepare(ids, state, logits=logits, next_token=token, prefill=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        with torch.cuda.stream(stream):
            for phase in (True, False):
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    runtime.run_into(logits, token, ids, state, prefill=phase)
                graphs[phase] = captured
            state.kda.convolution.zero_(); state.kda.recurrent.zero_()
            expected = 0
            for phase in (True, False, True):
                graphs[phase].replay()
                torch.cuda.synchronize()
                expected += int(phase)
                torch.testing.assert_close(logits, torch.full_like(logits, 11 + int(phase)), rtol=0, atol=0)
                torch.testing.assert_close(state.kda.convolution, torch.full_like(state.kda.convolution, expected),
                                           rtol=0, atol=0)
                torch.testing.assert_close(state.kda.recurrent, torch.full_like(state.kda.recurrent, expected),
                                           rtol=0, atol=0)
        # Reprepare remains a content-validation boundary after external edits.
        state.paged.seq_lens.fill_(state.paged.config.max_sequence_length + 1)
        with pytest.raises(EvaluationError, match="capacity"):
            runtime.prepare(ids, state, logits=logits, next_token=token)
        state.paged.seq_lens.zero_()
        if with_dsa:
            state.dsa.seq_lens.fill_(state.dsa.config.max_sequence_length + 1)
            with pytest.raises(RuntimeContractError, match="capacity"):
                runtime.prepare(ids, state, logits=logits, next_token=token)
            state.dsa.seq_lens = state.dsa.seq_lens.to(torch.int64)
            with pytest.raises(RuntimeContractError, match="seq_lens"):
                runtime.run_into(logits, token, ids, state)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
