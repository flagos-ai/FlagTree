# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""DSA logical layer ids and sparse selections must match the state contract."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn._dsa_indexer_state import DsaIndexerState, DsaIndexerStateConfig
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.devices import requires_nvidia_sm90


def indexer_module(config, layer, *, state_layer_ids=None, placement=None, reusable=False,
                   index_heads=2, index_pool=2, index_topk=4):
    def tensor(shape):
        value = fm.tensor_type("bfloat16", shape)
        return (fm.DistributedType(value, tuple(fm.SBP.broadcast() for _ in shape), placement)
                if placement is not None else value)

    class Graph(fm.Module):
        def forward(self):
            dim = config.index_head_dim
            parameters = [self.input(name, tensor(shape), id=name)
                          for name, shape in (("q", (1, index_heads * dim)), ("key", (1, dim)),
                                              ("gate", (1, dim)), ("weights", (1, index_heads)))]
            state = self.input("state", config.ref_type, id="state")
            layer_id = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), layer)
            advance = fm.F.builtin.scalar_const(fm.tensor_type("bool", ()), True)
            norm = self.input("norm", tensor((dim,)), id="norm")
            bias = self.input("bias", tensor((dim,)), id="bias")
            ape = self.input("ape", tensor((index_pool, dim)), id="ape")
            result = fm.F.nn.dsa_indexer(
                *parameters, state, layer_id, advance, norm, bias, ape,
                index_topk=index_topk, index_head_dim=dim, index_n_heads=index_heads, index_kpool=index_pool,
                always_select_tail=True, epsilon=1e-6,
                state_layer_ids=config.layer_ids if state_layer_ids is None else state_layer_ids,
                name="indexer")
            arguments = (*parameters, state, norm, bias, ape)
            if reusable:
                self.function("worker", arguments, (result,), attrs={
                    "calling_convention": "device", "reusable": True,
                })
                arguments = tuple(self.input(value.attrs["name"], value.type, id=f"entry_{value.id}")
                                  for value in arguments)
                result = fm.F.builtin.call(*arguments, callee="worker", result_type=result.type)
            self.function("main", arguments, (result,))

    return Graph(dialect="distributed" if placement is not None else "high_level",
                 stage="frozen_constants" if placement is not None else "imported", entry="main",
                 metadata=({"auto_distribution": {"placement": placement.to_data()}}
                           if placement is not None else {})).build()


def prepare(module, directory, inputs, state, output, *, target=None):
    compiler = Compiler()
    if target is not None:
        compiler.target = target
    compiled = compiler.compile(module).module
    artifact = write_artifact(compiled, directory, target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    binding = runtime.buffer_plan.function_map[compiled.entry]
    values = {}
    for value, buffers in binding.parameters:
        if isinstance(compiled.node_map[value].type, fm.RefType):
            values.update(zip(buffers, (state.packed_states, state.seq_lens), strict=True))
        else:
            values[buffers[0]] = inputs[compiled.node_map[value].attrs["name"]]
    for _, buffers in binding.outputs:
        values[buffers[0]] = output
    arguments = tuple(values[str(argument["buffer"])] for argument in runtime.external_arguments)
    runtime.prepare(*arguments)
    return runtime, arguments


@requires_nvidia_sm90
def test_noncontiguous_logical_layers_and_first_token_tail(tmp_path):
    config = DsaIndexerStateConfig((3, 7), 8, 4, 5)
    # Valid tensor views retain larger backing storage as a write canary. The
    # old kernel's logical-id-as-offset mistake stays inside that allocation.
    backing = torch.full((8, 8, 9), 17, dtype=torch.bfloat16, device="cuda")
    state = DsaIndexerState(backing[:2], torch.zeros(1, dtype=torch.int32, device="cuda"), config)
    state.packed_states.zero_()
    inputs = {"q": torch.ones((1, 8), dtype=torch.bfloat16, device="cuda"),
              "key": torch.tensor([[1, 2, 4, 8]], dtype=torch.bfloat16, device="cuda"),
              "gate": torch.zeros((1, 4), dtype=torch.bfloat16, device="cuda"),
              "weights": torch.ones((1, 2), dtype=torch.bfloat16, device="cuda"),
              "norm": torch.ones(4, dtype=torch.bfloat16, device="cuda"),
              "bias": torch.zeros(4, dtype=torch.bfloat16, device="cuda"),
              "ape": torch.zeros((2, 4), dtype=torch.bfloat16, device="cuda")}
    module = indexer_module(config, 3)
    reference_state = state.clone()
    expected, = TorchEvaluator(DictWeightResolver({})).run(module, {**inputs, "state": reference_state})
    output = torch.empty((1, 5), dtype=torch.int32, device="cuda")
    runtime, arguments = prepare(module, tmp_path / "indexer", inputs, state, output)
    try:
        runtime.run_into(*arguments)
        torch.testing.assert_close(backing[2:], torch.full_like(backing[2:], 17), rtol=0, atol=0)
        torch.testing.assert_close(state.packed_states, reference_state.packed_states, rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert state.sequence_length == 1
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@requires_nvidia_sm90
@pytest.mark.parametrize("layer", [3, 7])
@pytest.mark.parametrize("cooperative", [False, True])
@pytest.mark.parametrize("reusable", [False, True])
def test_eight_updates_select_distinct_pools_and_keep_tail(tmp_path, layer, cooperative, reusable):
    config = DsaIndexerStateConfig((3, 7), 8, 4, 5)
    state = DsaIndexerState(torch.zeros((2, 8, 9), dtype=torch.bfloat16, device="cuda"),
                            torch.zeros(1, dtype=torch.int32, device="cuda"), config)
    reference_state = state.clone()
    inputs = {
        "q": torch.tensor([[1, 0, 0, -1] * 2], dtype=torch.bfloat16, device="cuda"),
        "key": torch.empty((1, 4), dtype=torch.bfloat16, device="cuda"),
        "gate": torch.zeros((1, 4), dtype=torch.bfloat16, device="cuda"),
        "weights": torch.ones((1, 2), dtype=torch.bfloat16, device="cuda"),
        "norm": torch.ones(4, dtype=torch.bfloat16, device="cuda"),
        "bias": torch.zeros(4, dtype=torch.bfloat16, device="cuda"),
        "ape": torch.zeros((2, 4), dtype=torch.bfloat16, device="cuda"),
    }
    module = indexer_module(config, layer, reusable=reusable,
                            placement=fm.Placement((8, 16), "yx", "bb") if cooperative else None)
    output = torch.empty((1, 5), dtype=torch.int32, device="cuda")
    runtime, arguments = prepare(module, tmp_path / "indexer", inputs, state, output)
    try:
        for step in range(8):
            inputs["key"].copy_(torch.tensor([[1, step + 2, -step - 1, -2]],
                                             dtype=torch.bfloat16, device="cuda"))
            expected, = TorchEvaluator(DictWeightResolver({})).run(module, {**inputs, "state": reference_state})
            output.fill_(-99)
            runtime.run_into(*arguments)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(state.packed_states, reference_state.packed_states, rtol=0, atol=0)
            selected = output[output >= 0]
            assert selected.unique().numel() == selected.numel()
            assert state.sequence_length == step + 1
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@requires_nvidia_sm90
def test_full_capacity_visibility_and_tail_in_reusable_worker(tmp_path):
    config = DsaIndexerStateConfig((3, 7), 4096, 128, 2051)
    packed = torch.zeros((2, 4096, 257), dtype=torch.bfloat16, device="cuda")
    # The first visible token is in a different warp-sized reduction segment.
    # Keep a partial pool so count and first jointly determine the output tail.
    packed[1, 3071:3091, :128] = torch.arange(20, device="cuda")[:, None]
    packed[1, 3071:3091, 256] = 1
    state = DsaIndexerState(packed, torch.tensor([3091], dtype=torch.int32, device="cuda"), config)
    reference_state = state.clone()
    inputs = {"q": torch.ones((1, 4096), dtype=torch.bfloat16, device="cuda"),
              "key": torch.ones((1, 128), dtype=torch.bfloat16, device="cuda"),
              "gate": torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda"),
              "weights": torch.ones((1, 32), dtype=torch.bfloat16, device="cuda"),
              "norm": torch.ones(128, dtype=torch.bfloat16, device="cuda"),
              "bias": torch.zeros(128, dtype=torch.bfloat16, device="cuda"),
              "ape": torch.zeros((4, 128), dtype=torch.bfloat16, device="cuda")}
    module = indexer_module(config, 7, reusable=True, index_heads=32, index_pool=4, index_topk=2048)
    expected, = TorchEvaluator(DictWeightResolver({})).run(module, {**inputs, "state": reference_state})
    base = NvidiaSm90Target().triton_implementation_model
    model = replace(base, implementations=tuple(
        replace(implementation, parameters={**implementation.parameters, "compute_num_warps": 8})
        if implementation.family == "dsa_indexer" else implementation for implementation in base.implementations))
    output = torch.empty((1, 2051), dtype=torch.int32, device="cuda")
    runtime, arguments = prepare(module, tmp_path / "indexer", inputs, state, output,
                                 target=NvidiaSm90Target(triton_implementation_model=model))
    try:
        runtime.run_into(*arguments)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(state.packed_states, reference_state.packed_states, rtol=0, atol=0)
        assert state.sequence_length == 3092
        assert output[0, 2048].item() == 3091
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
