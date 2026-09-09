# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from python.test.flagmega.passes.tir.bufferize.test_ref_slice import state_slice_graph


@pytest.mark.parametrize("index", [None, 0, 1, 2])
@pytest.mark.parametrize("reusable", [False, True])
def test_reference_subspan_updates_only_selected_layer_across_invocations(tmp_path, index, reusable):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    compiled = Compiler().compile(state_slice_graph(index, reusable=reusable)).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_gdn_state(GatedDeltaNetStateConfig(3, 1, 2, 4, 4, 4, 16), device="cuda:0")
    state.convolution.copy_(torch.arange(state.convolution.numel(), device="cuda").reshape_as(state.convolution))
    state.recurrent.fill_(17)
    expected_state = state.convolution.clone()
    # Exactly representable products isolate addressing/aliasing from numerical
    # approximations in activation implementations.
    weight = torch.zeros((16, 1, 4), dtype=torch.bfloat16, device="cuda")
    weight[:, :, -1] = 1
    qkv = torch.arange(16, dtype=torch.bfloat16, device="cuda").reshape(1, 16)
    output = torch.empty_like(qkv)

    def arguments(layer):
        prefix = "entry_" if reusable else ""
        state_id = prefix + "state"
        values = {prefix + "qkv": qkv, prefix + "weight": weight, prefix + "layer": layer}
        state_buffers = dict(runtime.buffer_plan.function_map["main"].parameters)[state_id]
        state_fields = dict(zip(state_buffers, (state.convolution, state.recurrent)))
        return tuple(output if spec["role"] == "result" else (
            state_fields[spec["buffer"]] if spec["value"] == state_id else values[spec["value"]])
                     for spec in runtime.external_arguments)

    sequence = (1, 2, 0, 1) if index is None else (index, ) * 4
    GeneratedTirCallGraphModule.prepare(runtime, *arguments(sequence[0]))
    for step, layer in enumerate(sequence):
        qkv.add_(1)
        expected_state[layer, :, :-1, :] = expected_state[layer, :, 1:, :].clone()
        expected_state[layer, :, -1, :] = qkv.reshape(2, 8)
        expected_output = torch.nn.functional.silu(qkv)
        if reusable:
            expected_state[layer, :, :-1, :] = expected_state[layer, :, 1:, :].clone()
            expected_state[layer, :, -1, :] = expected_output.reshape(2, 8)
            expected_output = torch.nn.functional.silu(expected_output)
        GeneratedTirCallGraphModule.run_into(runtime, *arguments(layer))
        torch.cuda.synchronize()
        torch.testing.assert_close(state.convolution, expected_state, rtol=0, atol=0,
                                   msg=f"wrong state after step {step}, layer {layer}")
        torch.testing.assert_close(state.recurrent, torch.full_like(state.recurrent, 17), rtol=0, atol=0)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    assert runtime.prepare_count == 1
