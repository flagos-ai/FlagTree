# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Runtime-uniform sequence advance remains safe inside a reusable worker."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load


def test_reusable_chunk_worker_advances_after_all_owner_reads(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    placement = fm.Placement((2, 2), "yx", "bb")
    config = PagedAttentionStateConfig(3, 4, 32, block_size=16, num_blocks=4)
    slots_type = fm.tensor_type("bfloat16", (3, 4, 32))
    split = fm.DistributedType(slots_type, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()),
                               placement)
    scalar = fm.tensor_type("int32", ())
    boolean = fm.tensor_type("bool", ())

    class Graph(fm.Module):

        def forward(self):
            slots = self.input("slots", slots_type)
            state = self.input("state", config.ref_type)
            layer = self.input("layer", scalar)
            advance = self.input("advance", boolean)
            local = fm.F.distributed.force_boxing(slots, split)
            updated = fm.F.nn.update_paged_attention_kv_cache(local, state, layer, advance, cache_kind="value",
                                                              layout=("seq", "head", "dim"))
            self.function("worker", (slots, state, layer, advance), (updated, ),
                          attrs={"noinline": True, "reusable": True})
            values = self.input("entry_slots", slots_type, id="entry_slots")
            cache = self.input("entry_state", config.ref_type, id="entry_state")
            first_layer = self.input("first_layer", scalar, id="first_layer")
            second_layer = self.input("second_layer", scalar, id="second_layer")
            increment = self.input("entry_advance", boolean, id="entry_advance")
            no_advance = fm.F.builtin.scalar_const(boolean, False)
            first = fm.F.builtin.call(values, cache, first_layer, no_advance, callee="worker",
                                      result_type=config.ref_type, effect=fm.effect("read_write", config.ref_type.name))
            second = fm.F.builtin.call(values, first, second_layer, increment, callee="worker",
                                       result_type=config.ref_type,
                                       effect=fm.effect("read_write", config.ref_type.name))
            self.function("main", (values, cache, first_layer, second_layer, increment), (fm.F.builtin.tuple(second), ))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    fm.emit_module(module, tmp_path / "input.py")
    module = Compiler().compile(fm.load_module(tmp_path / "input.py")).module
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert source.count("def _flagmega_function_worker__consumer(") == 1
    assert source.count("    _flagmega_function_worker__consumer(") == 2
    runtime = load(artifact, device="cuda:0")
    state = create_paged_attention_state(config, device="cuda")
    state.kv_caches.fill_(-9)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda"))
    slots = torch.arange(384, device="cuda").reshape(3, 4, 32).bfloat16()
    binding = runtime.buffer_plan.function_map[module.entry]
    names = dict(binding.parameters)["entry_state"]
    fields = dict(
        zip(names, (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table),
            strict=True))

    def arguments(first, second, advance):
        inputs = {"entry_slots": slots, "first_layer": first, "second_layer": second, "entry_advance": advance}
        return tuple(fields[arg["buffer"]] if arg["buffer"] in fields else inputs[arg["value"]]
                     for arg in runtime.external_arguments)

    runtime.prepare(*arguments(0, 1, False))
    expected = state.kv_caches.clone()
    for base, first, second, advance in ((13, 0, 1, False), (13, 1, 2, True), (31, 2, 0, True), (34, 0, 2, False)):
        state.seq_lens.fill_(base)
        slots.add_(1)
        runtime.run_into(*arguments(first, second, advance))
        for layer in (first, second):
            for row in range(3):
                page = (2, 0, 3, 1)[(base + row) // 16]
                expected[page, layer, 1, (base + row) % 16].copy_(slots[row].reshape(4, 4, 8))
        torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
        assert state.sequence_length == base + (3 if advance else 0)
        assert state.slot_mapping.item() == base
    assert runtime.prepare_count == 1
