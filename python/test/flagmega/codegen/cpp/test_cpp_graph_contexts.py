# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Captured cooperative launches retain independent runtime state and counters."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import create_paged_attention_state
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from python.test.flagmega.codegen.cpp.test_cpp_rope_cache import cache_module, sm120_torch


def test_two_cpp_runtime_graphs_replay_independent_cache_state(tmp_path):
    torch = sm120_torch()
    config, source = cache_module(True, True)
    compiled = Compiler(CompileOptions(target='nvidia-sm120')).compile(source).module
    artifact = write_artifact(compiled, tmp_path / 'artifact', target='nvidia-sm120',
                              emit_executable=True, codegen_backend='cpp')
    contexts = []
    try:
        for index, start in enumerate((0, 16)):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                runtime = load(artifact, device='cuda:0')
                state = create_paged_attention_state(config, device='cuda')
                state.kv_caches.fill_(-9)
                state.seq_lens.fill_(start)
                state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], device='cuda', dtype=torch.int32))
                semantic = (torch.arange(384, device='cuda').reshape(3, 4, 32) + index * 512).bfloat16()
                slots = semantic.reshape(3, 4, 4, 8)
                binding = runtime.buffer_plan.function_map[compiled.entry]
                fields = (state.kv_caches, state.query_start_loc, state.seq_lens,
                          state.slot_mapping, state.block_table)
                buffers = {}
                for value, names in binding.parameters:
                    if isinstance(compiled.node_map[value].type, fm.RefType):
                        buffers.update(zip(names, fields, strict=True))
                    else:
                        buffers[names[0]] = slots
                for _, names in binding.outputs:
                    buffers.update(zip(names, fields, strict=True))
                arguments = [buffers[str(arg['buffer'])] for arg in runtime.external_arguments]
                runtime.prepare(*arguments)
                # Load/prepare the CUDA function before capture, then reset
                # state on its bound stream outside the measured/replay body.
                runtime.run_into(*arguments)
                state.kv_caches.fill_(-9)
                state.seq_lens.fill_(start)
                state.slot_mapping.zero_()
                state.query_start_loc.zero_()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    runtime.run_into(*arguments)
                contexts.append((runtime, stream, graph, state, semantic, start))
        # The graphs share the library but retain different state/counter
        # storage. Enqueue alternating replays without host synchronization.
        for _ in range(8):
            for _, stream, graph, _, _, _ in contexts:
                with torch.cuda.stream(stream):
                    graph.replay()
        for runtime, stream, _, state, semantic, start in contexts:
            stream.synchronize()
            expected = torch.full_like(state.kv_caches, -9)
            for position in range(start, start + 24, 3):
                for row in range(3):
                    page = (2, 0, 3, 1)[(position + row) // 16]
                    expected[page, 1, 1, (position + row) % 16].copy_(semantic[row].reshape(4, 4, 8))
            torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
            assert state.seq_lens.item() == start + 24
            assert state.slot_mapping.item() == start + 21
            assert state.query_start_loc.tolist() == [0, 3]
            assert runtime.resource_report['cooperative_grid']
            assert runtime.resource_report['resident_blocks_per_sm'] > 0
    finally:
        for runtime, stream, _, _, _, _ in contexts:
            stream.synchronize()
            runtime.close()
