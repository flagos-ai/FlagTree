# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rotary tables advance the position across every row of a query chunk."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load


def _run_rotary(tmp_path, tokens, past, lanes, output_dtype="float32"):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    placement = fm.Placement((2, 2), "yx", "bb")
    config = PagedAttentionStateConfig(1, 2, 32)

    class Rotary(fm.Module):

        def forward(self):
            reference = self.input("reference", fm.tensor_type("bfloat16", (tokens, 17)))
            state = self.input("state", config.ref_type)
            pair = fm.F.nn.rotary_embedding(reference, state, head_dim=32, theta=10000., attention_scaling=1.2,
                                            output_lanes=lanes, output_dtype=output_dtype)
            self.function("main", (reference, state), (pair, ))

    module = Rotary(dialect="distributed", stage="frozen_constants", entry="main",
                    metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    fm.emit_module(module, tmp_path / "input.py")
    module = Compiler().compile(fm.load_module(tmp_path / "input.py")).module
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(past)
    reference = torch.zeros((tokens, 17), dtype=torch.bfloat16, device="cuda")
    outputs = [torch.full((tokens, 1, 32), float("nan"), device="cuda", dtype=getattr(torch, output_dtype)) for _ in range(2)]
    buffers = {}
    binding = runtime.buffer_plan.function_map[module.entry]
    for value, names in binding.parameters:
        if isinstance(module.node_map[value].type, fm.RefType):
            buffers.update(
                zip(names,
                    (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table),
                    strict=True))
        else:
            for name in names:
                buffers[name] = reference
    for _, names in binding.outputs:
        buffers.update(zip(names, outputs, strict=True))
    arguments = [buffers[str(arg["buffer"])] for arg in runtime.external_arguments]
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    return [value.cpu() for value in outputs]


@pytest.mark.parametrize("tokens,past", ((1, 3), (3, 0), (9, 13)))
@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_rotary_chunk_positions(tmp_path, tokens, past, lanes):
    outputs = _run_rotary(tmp_path, tokens, past, lanes)
    torch = pytest.importorskip("torch")
    angles = torch.outer(torch.arange(past, past + tokens).float(), 10000.**(-torch.arange(0, 32, 2).float() / 32))
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    for actual, expected in zip(outputs, (angles.cos() * 1.2, angles.sin() * 1.2)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("lanes", [(8,), (2, 4), (4, 2)])
@pytest.mark.parametrize("tokens,past", [(1, 3), (9, 13)])
def test_vector_output_is_bitwise_equal_to_scalar_kernel(tmp_path, lanes, tokens, past):
    torch = pytest.importorskip("torch")
    scalar = _run_rotary(tmp_path / "scalar", tokens, past, ())
    vector = _run_rotary(tmp_path / "vector", tokens, past, lanes)
    for actual, expected in zip(vector, scalar):
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("lanes", [(), (8,), (2, 8)])
@pytest.mark.parametrize("tokens,past", [(1, 17), (3, 81)])
def test_rotary_bf16_store_equals_explicit_fp32_table_cast(tmp_path, lanes, tokens, past):
    torch = pytest.importorskip("torch")
    wide = _run_rotary(tmp_path / "wide", tokens, past, lanes)
    narrow = _run_rotary(tmp_path / "narrow", tokens, past, lanes, output_dtype="bfloat16")
    for actual, expected in zip(narrow, wide):
        assert torch.equal(actual.view(torch.int16), expected.bfloat16().view(torch.int16))
