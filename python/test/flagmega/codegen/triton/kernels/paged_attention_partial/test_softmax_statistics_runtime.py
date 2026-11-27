# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Multiple nonempty context shards must publish maxima in the common log base."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load


def attention_module(config):
    class Attention(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="distributed", entry="main",
                             metadata={"auto_distribution": {"placement":
                                       fm.Placement((8, 16), "yx", "bb").to_data()}})

        def forward(self):
            query_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 16, 16))
            query = self.input("query", fm.DistributedType(
                query_type, (fm.SBP.broadcast(), fm.SBP.split(
                    fm.SplitStage((1,), fm.ContiguousSplit(1))), fm.SBP.broadcast()),
                fm.Placement((8, 16), "yx", "bb")))
            state = self.input("state", config.ref_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            partial = fm.F.ntt.paged_attention_partial(
                query, state, layer, scale=128 ** -.5, layout=("seq", "head", "dim"),
                hidden_size=2048, split_hierarchy_axis=0, split_count=8)
            value = fm.F.ntt.paged_attention_combine(
                fm.F.tensors.get_item(partial, 0), fm.F.tensors.get_item(partial, 1),
                fm.F.tensors.get_item(partial, 2), layout=("seq", "head", "dim"),
                hidden_size=2048, output_data_type=query_type.dtype, output_type=query.type,
                split_hierarchy_axis=0, split_count=8)
            result = fm.F.builtin.tuple(fm.F.distributed.boxing(value, query_type))
            self.function("main", (query, state), (result,))

    return Attention().build()


@pytest.mark.parametrize("length", [33, 129, 250, 300])
def test_mma_attention_merges_unequal_nonempty_shard_maxima(tmp_path, length):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    config = PagedAttentionStateConfig(1, 8, 128, num_blocks=4)
    module = Compiler().compile(attention_module(config)).module
    path = write_artifact(module, tmp_path / "attention", target="nvidia-sm90", emit_executable=True)
    source = (path / "generated_kernels.py").read_text()
    assert "paged_attention_partial/mma_tma_smem_pipeline" in source
    runtime = load(path, device="cuda:0")
    query = torch.ones((1, 16, 16, 8), device="cuda", dtype=torch.bfloat16)
    state = create_paged_attention_state(config, device="cuda")
    # Nonconsecutive pages also verify the descriptor/page-table boundary.
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32))
    keys = torch.full((length, 8, 128), .125, device="cuda", dtype=torch.bfloat16)
    keys[128:] *= -1
    values = torch.ones_like(keys)
    values[128:] *= -1
    for position in range(length):
        logical, offset = divmod(position, config.block_size)
        page = (2, 0, 3, 1)[logical]
        state.kv_caches[page, 0, 0, offset].copy_(keys[position].reshape(8, 16, 8))
        state.kv_caches[page, 0, 1, offset].copy_(values[position].reshape(8, 16, 8))
    state.seq_lens.fill_(length)
    state.slot_mapping.fill_(length - 1)
    output = torch.empty_like(query)
    binding = runtime.buffer_plan.function_map[module.entry]
    buffers = {}
    for value, names in binding.parameters:
        if isinstance(module.node_map[value].type, fm.RefType):
            buffers.update(zip(names, (state.kv_caches, state.query_start_loc, state.seq_lens,
                                       state.slot_mapping, state.block_table), strict=True))
        else:
            assert len(names) == 1
            buffers[names[0]] = query
    for value, names in binding.outputs:
        assert len(names) == 1
        buffers[names[0]] = output
    arguments = [buffers[str(argument["buffer"])] for argument in runtime.external_arguments]
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    scores = torch.einsum("qhd,khd->hqk", query.reshape(1, 16, 128).float(),
                          keys.repeat_interleave(2, dim=1).float()) * 128 ** -.5
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1),
                            values.repeat_interleave(2, dim=1).float()).to(torch.bfloat16)
    torch.testing.assert_close(output.reshape_as(expected), expected, rtol=.01, atol=.005)
