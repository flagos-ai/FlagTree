# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exercise fused gating with real unequal partial states and gate-side views."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load


def graph(config, lanes, exported, block_cyclic=False):
    mesh = fm.Placement((2, 2), "yx", "bb")
    dtype = "bfloat16" if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
    tensor = fm.tensor_type(dtype, (1, 4, 32 // lanes))
    policies = (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 2), fm.SBP.broadcast())
    query_type = fm.DistributedType(tensor, policies, mesh)
    dim_split = (fm.SBP.split_block_cyclic((0,), 1) if block_cyclic else
                 fm.SBP.split_contiguous((0,), 32 // lanes // 2))
    gated_type = fm.DistributedType(tensor, (*policies[:2], dim_split), mesh)

    class Graph(fm.Module):
        def forward(self):
            query = self.input("query", tensor, id="query")
            gate = self.input("gate", tensor, id="gate")
            state = self.input("state", config.ref_type, id="state")
            q = fm.F.distributed.boxing(query, query_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            partial = fm.F.ntt.paged_attention_partial(q, state, layer, scale=32 ** -.5,
                layout=("seq", "head", "dim"), hidden_size=128, split_hierarchy_axis=0, split_count=2)
            combined = fm.F.ntt.paged_attention_combine(
                *(fm.F.tensors.get_item(partial, index) for index in range(3)),
                layout=("seq", "head", "dim"), hidden_size=128, output_data_type=dtype,
                output_type=query_type, split_hierarchy_axis=0, split_count=2, name="combined")
            value = fm.F.distributed.sharded_view(combined, gated_type)
            gate_view = fm.F.distributed.boxing(gate, gated_type)
            sigmoid = (fm.F.math.sigmoid(gate_view) if lanes == 1 else
                       fm.F.math.vectorized_unary(gate_view, unary_op="sigmoid"))
            metadata = {"selected_vectorization": "vectorization.last_axis",
                        "selected_vector_axes": (2,), "selected_vector_lanes": (lanes,)} if lanes != 1 else {}
            gated = (fm.F.math.mul(value, sigmoid) if lanes == 1 else
                     fm.F.math.vectorized_binary(value, sigmoid, binary_op="mul", metadata=metadata))
            outputs = [fm.F.distributed.boxing(gated, tensor)]
            if exported:
                outputs.append(fm.F.distributed.boxing(combined, tensor))
            self.function("main", (query, gate, state), (fm.F.builtin.tuple(*outputs),))

    return Graph(dialect="high_level", stage="distributed", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()


def prepare(module, directory, query, gate, state, outputs):
    compiled = Compiler().compile(module).module
    gated_kernels = [function.dispatch.microkernel for function in compiled.kernel_definitions
                     if function.dispatch.semantic_op == "ntt.paged_attention_gated_combine"]
    assert all(kernel.parameters["elements_per_program"] == 32 for kernel in gated_kernels)
    artifact = write_artifact(compiled, directory, target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    binding = runtime.buffer_plan.function_map[compiled.entry]
    buffers = {}
    for value, names in binding.parameters:
        if isinstance(compiled.node_map[value].type, fm.RefType):
            buffers.update(zip(names, (state.kv_caches, state.query_start_loc, state.seq_lens,
                                       state.slot_mapping, state.block_table), strict=True))
        else:
            assert len(names) == 1
            buffers[names[0]] = {"query": query, "gate": gate}[compiled.node_map[value].attrs["name"]]
    output_names = [name for _, names in binding.outputs for name in names]
    buffers.update(zip(output_names, outputs, strict=True))
    arguments = [buffers[str(argument["buffer"])] for argument in runtime.external_arguments]
    runtime.prepare(*arguments)
    return runtime, arguments, artifact


@pytest.mark.parametrize("lanes", [1, 8])
@pytest.mark.parametrize("block_cyclic", [False, True])
def test_fused_and_unfused_device_paths_match_with_resharded_output(tmp_path, lanes, block_cyclic):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    torch.manual_seed(39)
    config = PagedAttentionStateConfig(1, 2, 32, num_blocks=4)
    state = create_paged_attention_state(config, device="cuda")
    state.kv_caches.copy_(torch.randn_like(state.kv_caches))
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32))
    shape = (1, 4, 32) if lanes == 1 else (1, 4, 32 // lanes, lanes)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate = torch.linspace(-6, 6, 128, device="cuda").to(torch.bfloat16).reshape(shape)
    fused_out, unfused_out, attention_out = (torch.empty_like(query) for _ in range(3))
    fused, fused_args, artifact = prepare(graph(config, lanes, False, block_cyclic), tmp_path / "fused", query, gate, state, (fused_out,))
    unfused, unfused_args, _ = prepare(graph(config, lanes, True, block_cyclic), tmp_path / "unfused", query, gate, state, (unfused_out, attention_out))
    source = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: paged_attention_gated_combine/decode" in source
    assert "# flagmega-kernel: elementwise/mul" not in source
    original_cache = state.kv_caches.clone()
    for length in (1, 33, 129, 255, 256, 300):
        state.seq_lens.fill_(length)
        state.slot_mapping.fill_(length - 1)
        fused_out.fill_(float("nan"))
        fused.run_into(*fused_args)
        unfused.run_into(*unfused_args)
        torch.testing.assert_close(fused_out, unfused_out, rtol=0, atol=0)
        torch.testing.assert_close(fused_out, (attention_out * gate.float().sigmoid().bfloat16()).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(state.kv_caches, original_cache, rtol=0, atol=0)
