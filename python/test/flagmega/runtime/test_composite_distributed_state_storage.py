# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Persistent tensors use the physical layout of their declared external ABI."""

import os
import subprocess
from dataclasses import replace
from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.ops.nn._kda_state import create_kda_state
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule, GeneratedTirCompositeModelModule


def distributed_state_graph(*, view_results=False):
    mesh = fm.Placement((2, 2, 4), "cyx", "cbb")
    broad = fm.SBP.broadcast()
    head = fm.SBP.split_contiguous((0, 2), 1)
    convolution = fm.DistributedType(fm.tensor_type("bfloat16", (3, 8, 128, 3)),
                                     (broad, head, broad, broad), mesh)
    recurrent = fm.DistributedType(fm.tensor_type("float32", (1, 8, 128, 128)),
                                  (broad, head, broad, broad), mesh)

    class Graph(fm.Module):
        def forward(self):
            ids = self.input("input_ids", fm.tensor_type("int32", (1,)), id="input_ids")
            conv = self.input("conv", convolution, id="conv")
            state = self.input("recurrent", recurrent, id="recurrent")
            paged = self.input("paged", PagedAttentionStateConfig(1, 1, 8, block_size=8, num_blocks=1).ref_type,
                               id="paged")
            logits = fm.F.tensors.broadcast_to(fm.F.tensors.cast(ids, "float32"), shape=(1, 8), name="logits")
            token = fm.F.nn.greedy_sample(logits, name="token")
            conv_out = fm.F.math.add(conv, conv, name="conv_update")
            state_out = fm.F.math.add(state, state, name="recurrent_update")
            if view_results:
                conv_out = fm.F.tensors.reshape(conv_out, shape=(3, 8, 128, 3), name="conv_result")
                state_out = fm.F.tensors.reshape(state_out, shape=(1, 8, 128, 128), name="recurrent_result")
            self.builder.metadata["composite_abi"] = {
                "schema": "flagmega.composite-abi/v1", "input_id": ids.id,
                "logits_output": logits.id, "token_output": token.id,
                "tensor_states": ({"state_id": 0, "input": (conv.id, state.id),
                                   "output": (conv_out.id, state_out.id)},),
                "paged_state": {"input": paged.id, "output": paged.id},
            }
            self.function("main", (ids, conv, state, paged), (logits, token, conv_out, state_out, paged))

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build(), mesh


def _compile_and_load(tmp_path, *, view_results=False):
    source, mesh = distributed_state_graph(view_results=view_results)
    module = Compiler(CompileOptions(placement=mesh)).compile(source).module
    path = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(path)
    assert isinstance(runtime, GeneratedTirCompositeModelModule)
    return runtime


@pytest.mark.parametrize("view_results", [False, True])
def test_entry_state_storage_preserves_declared_chip_owner_layout(tmp_path, view_results):
    runtime = _compile_and_load(tmp_path, view_results=view_results)
    try:
        function = runtime.buffer_plan.function_map[runtime.ir_module.entry]
        bindings = dict((*function.parameters, *function.outputs))
        pair, = runtime.ir_module.metadata["composite_abi"]["tensor_states"]
        for input_id, output_id in zip(pair["input"], pair["output"], strict=True):
            before, after = (runtime.buffer_plan.buffer_map[bindings[value][0]] for value in (input_id, output_id))
            assert before.distributed_type == after.distributed_type
            assert before.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_PER_CHIP
            assert after.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_PER_CHIP
            assert before.physical_access_span.nbytes == after.physical_access_span.nbytes
    finally:
        runtime.close()


def test_composite_state_config_uses_external_physical_layout(tmp_path):
    runtime = _compile_and_load(tmp_path)
    try:
        # Owner-major components are required by the real compact-per-chip
        # ABI. The logical global head count remains eight in the typed IR.
        assert runtime._kda_config.convolution_shape == (4, 3, 1, 128, 3)
        assert runtime._kda_config.recurrent_shape == (4, 1, 1, 128, 128)
        state = create_kda_state(runtime._kda_config, device="cpu")
        assert tuple(state.convolution.shape) == (1, 4, 3, 1, 128, 3)
        assert tuple(state.recurrent.shape) == (1, 4, 1, 1, 128, 128)
        state.validate()
        assert prod(runtime._kda_config.recurrent_shape) == 4 * 128 * 128
    finally:
        runtime.close()


@pytest.mark.parametrize("chip", [0, 1])
def test_state_owner_binding_uses_the_placement_chip_count(tmp_path, monkeypatch, chip):
    runtime = _compile_and_load(tmp_path)
    try:
        # Exercise the real ownership binding without loading device buffers.
        monkeypatch.setattr(GeneratedTirCallGraphModule, "load", lambda self, device: self)
        monkeypatch.setenv("RANK", str(chip))
        runtime.load(f"cuda:{chip}")
        assert runtime._kda_config.chip_index == chip
    finally:
        runtime.close()


@pytest.fixture(scope="module")
def owned_state_device_group():
    import torch
    import torch.distributed as dist

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    context = torch.empty(1, device=device)
    uuid = "GPU-" + str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    owners = {int(row.split(",")[1]) for row in active.splitlines() if row.split(",")[0].strip() == uuid}
    # nvidia-smi reports host PIDs, which need not match this container's PID
    # namespace. Our live CUDA allocation must be the only device context.
    assert len(owners) == 1, f"{device} ({uuid}) has competing CUDA host PIDs {owners}"
    del context
    dist.init_process_group("gloo")
    yield device
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("WORLD_SIZE") != "2", reason="requires torchrun on two idle SM90 GPUs")
@pytest.mark.parametrize("view_results", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_owned_state_carry_matches_every_global_head(tmp_path, view_results, captured, owned_state_device_group):
    import torch
    import torch.distributed as dist

    device, rank = owned_state_device_group, dist.get_rank()
    runtime = _compile_and_load(tmp_path, view_results=view_results)
    graph = None
    try:
        runtime.load(device)
        state = runtime.create_state()
        assert state.kda.config.chip_index == rank
        assert state.kda.config.logical_types == runtime._kda_config.logical_types
        assert runtime.ir_module.node_map["recurrent"].type.tensor.shape[1].fixed_value == 8
        guards = []
        for value in (state.kda, state.next_kda):
            for field in ("convolution", "recurrent"):
                old = getattr(value, field)
                guard_elements = 256 // old.element_size()
                storage = torch.full((old.numel() + 2 * guard_elements,), -123., dtype=old.dtype, device=device)
                setattr(value, field, storage[guard_elements:-guard_elements].reshape(old.shape))
                guards.extend((storage[:guard_elements], storage[-guard_elements:]))
        ids = torch.full((1,), 11, dtype=torch.int32, device=device)
        logits, token = runtime.create_outputs()
        runtime.prepare(ids, state, logits=logits, next_token=token)
        assert runtime.resource_report["spill_bytes"] == 0
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        if captured:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph, stream=stream):
                runtime.run_into(logits, token, ids, state)
        torch.cuda.current_stream().wait_stream(stream)
        for mutation in range(3):
            conv = ((torch.arange(3 * 8 * 128 * 3) * 5 + mutation * 7) % 31 - 15).reshape(3, 8, 128, 3).float().div(128).bfloat16()
            recurrent = torch.arange(8 * 128 * 128).reshape(1, 8, 128, 128).float().div(65536).add(mutation / 16)
            if mutation == 0:
                conv.zero_(); recurrent.zero_()
            expected_conv = conv[:, rank * 4:(rank + 1) * 4].permute(1, 0, 2, 3).unsqueeze(2).contiguous()
            expected_recurrent = recurrent[:, rank * 4:(rank + 1) * 4].permute(1, 0, 2, 3).unsqueeze(2).contiguous()
            state.kda.convolution.copy_(expected_conv.unsqueeze(0))
            state.kda.recurrent.copy_(expected_recurrent.unsqueeze(0))
            for step in range(3):
                state.next_kda.convolution.fill_(float("nan"))
                state.next_kda.recurrent.fill_(float("nan"))
                dist.barrier()
                if graph is None:
                    runtime.run_into(logits, token, ids, state)
                else:
                    graph.replay()
                torch.cuda.synchronize()
                expected_conv = expected_conv + expected_conv
                expected_recurrent = expected_recurrent + expected_recurrent
                assert torch.equal(state.kda.convolution[0].cpu().view(torch.uint8), expected_conv.view(torch.uint8))
                assert torch.equal(state.kda.recurrent[0].cpu().view(torch.uint8), expected_recurrent.view(torch.uint8))
                assert all(bool((guard == -123).all()) for guard in guards)
                # Explicit diagnostic assembly stays outside the carried ABI
                # and replay. Check all global heads, not one rank's prefix.
                pieces = [None, None]
                dist.all_gather_object(pieces, (state.kda.convolution[0].cpu(), state.kda.recurrent[0].cpu()))
                whole_conv = torch.cat([piece[0].squeeze(2).permute(1, 0, 2, 3) for piece in pieces], dim=1)
                whole_recurrent = torch.cat([piece[1].squeeze(2).permute(1, 0, 2, 3) for piece in pieces], dim=1)
                assert torch.equal(whole_conv.view(torch.uint8), (conv * 2 ** (step + 1)).view(torch.uint8))
                assert torch.equal(whole_recurrent.view(torch.uint8), (recurrent * 2 ** (step + 1)).view(torch.uint8))
        original = state.kda.config
        state.kda.config = replace(original, chip_index=1 - rank)
        with pytest.raises(RuntimeContractError, match="KDA state"):
            runtime.run_into(logits, token, ids, state)
        state.kda.config = original
    finally:
        graph = None
        runtime.close()
