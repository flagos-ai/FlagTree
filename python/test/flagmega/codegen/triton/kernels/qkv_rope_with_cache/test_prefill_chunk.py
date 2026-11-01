# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused prefill writes every logical token, including a page crossing."""

import pytest
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import create_paged_attention_state
from triton.flagmega.runtime import load
from triton.flagmega import ir as fm
from python.test.flagmega.codegen.triton.kernels.qkv_rope_with_cache.test_qkv_owner_local_runtime import _graph


@pytest.mark.parametrize("independent_v", [False, True])
def test_fused_prefill_preserves_all_chunk_slots(tmp_path, independent_v):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    module, config = _graph(8, False, 64, False, False, independent_v, tokens=3)
    compiled = Compiler().compile(module).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0")
    try:
        values = {"q": torch.ones((3, 5, 16, 8), dtype=torch.bfloat16, device="cuda"),
                  "k": torch.ones((3, 3, 16, 8), dtype=torch.bfloat16, device="cuda"),
                  "v": (torch.arange(3 * 3 * 128, device="cuda").reshape(3, 3, 16, 8) / 8).bfloat16(),
                  "scale": torch.ones(128, device="cuda", dtype=torch.bfloat16),
                  "bias": torch.zeros(128, device="cuda", dtype=torch.bfloat16)}
        values["cos"] = torch.tensor([1., 0., -1.], device="cuda").reshape(3, 1, 1).expand(3, 1, 64).contiguous()
        values["sin"] = torch.tensor([0., 1., 0.], device="cuda").reshape(3, 1, 1).expand(3, 1, 64).contiguous()
        state = create_paged_attention_state(config, device="cuda")
        state.block_table.copy_(torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32))
        output = torch.empty_like(values["q"])
        bound = {}
        for identity, buffers in runtime.buffer_plan.function_map[compiled.entry].parameters:
            node = compiled.node_map[identity]
            if isinstance(node.type, fm.RefType):
                bound.update((buffer, getattr(state, field)) for buffer, (field, _) in zip(buffers, node.type.fields, strict=True))
            else:
                bound[buffers[0]] = values[node.attrs["name"]]
        for spec in runtime.external_arguments:
            if spec["role"] == "result":
                bound[spec["buffer"]] = output
        args = tuple(bound[spec["buffer"]] for spec in runtime.external_arguments)
        runtime.prepare(*args)
        key = torch.ones((3, 3, 128), device="cuda", dtype=torch.bfloat16)
        key[1, :, :32] = -1
        key[2, :, :64] = -1
        query = key[:, :1].expand(3, 5, 128).reshape_as(output)
        for base in (3, 6):
            state.seq_lens.fill_(base)
            state.kv_caches.fill_(-9)
            values["v"].add_(1)
            expected = state.kv_caches.clone()
            for token in range(3):
                position = base + token
                page = int(state.block_table[0, position // config.block_size])
                expected[page, 0, 0, position % config.block_size].copy_(key[token].reshape(3, 16, 8))
                expected[page, 0, 1, position % config.block_size].copy_(values["v"][token])
            runtime.run_into(*args)
            torch.cuda.synchronize()
            torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
            torch.testing.assert_close(output, query, rtol=0, atol=0)
            assert state.seq_lens.item() == base + 3
            assert state.slot_mapping.item() == base
    finally:
        runtime.close()
