"""Warp-local QK reduction at tile/page boundaries across all context owners."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def attention_runtime(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper CUDA required")
    sys.path.insert(0, str(ROOT))
    from agent_optimizations.decode import create_target, install
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
    from triton.flagmega.runtime import load
    install()
    config = PagedAttentionStateConfig(1, 2, 256, num_blocks=16)
    placement = fm.Placement((8, 16), "yx", "bb")

    class Attention(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="distributed", entry="main",
                             metadata={"auto_distribution": {"placement": placement.to_data()}})

        def forward(self):
            query_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 16, 32))
            query = self.input("query", fm.DistributedType(query_type,
                (fm.SBP.broadcast(), fm.SBP.split(fm.SplitStage((1,), fm.ContiguousSplit(1))),
                 fm.SBP.broadcast()), placement))
            state = self.input("state", config.ref_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            partial = fm.F.ntt.paged_attention_partial(query, state, layer, scale=.0625,
                layout=("seq", "head", "dim"), hidden_size=4096, split_hierarchy_axis=0, split_count=8)
            value = fm.F.ntt.paged_attention_combine(*(fm.F.tensors.get_item(partial, i) for i in range(3)),
                layout=("seq", "head", "dim"), hidden_size=4096, output_data_type=query_type.dtype,
                output_type=query.type, split_hierarchy_axis=0, split_count=8)
            self.function("main", (query, state), (fm.F.builtin.tuple(fm.F.distributed.boxing(value, query_type)),))

    compiler = Compiler()
    compiler.target = create_target()
    module = compiler.compile(Attention().build()).module
    path = write_artifact(module, tmp_path_factory.mktemp("adaptive-attention") / "artifact",
                          target="nvidia-sm90", emit_executable=True)
    source = (path / "generated_kernels.py").read_text()
    assert "_flagmega_paged_attention_partial_reduction_layout" in source
    runtime = load(path, device="cuda:0")
    torch.manual_seed(572)
    query = torch.randn((1, 16, 32, 8), device="cuda", dtype=torch.bfloat16) * .5
    state = create_paged_attention_state(config, device="cuda")
    permutation = torch.randperm(16, device="cuda", dtype=torch.int32)
    state.block_table.copy_(permutation.reshape_as(state.block_table))
    key = torch.randn((4096, 2, 256), device="cuda", dtype=torch.bfloat16) * .5
    value = torch.randn_like(key)
    for page in range(16):
        state.kv_caches[permutation[page].long(), 0, 0].copy_(key[page * 256:(page + 1) * 256].reshape(256, 2, 32, 8))
        state.kv_caches[permutation[page].long(), 0, 1].copy_(value[page * 256:(page + 1) * 256].reshape(256, 2, 32, 8))
    output = torch.empty_like(query)
    buffers = {}
    binding = runtime.buffer_plan.function_map[module.entry]
    for node, names in binding.parameters:
        if isinstance(module.node_map[node].type, fm.RefType):
            buffers.update(zip(names, (state.kv_caches, state.query_start_loc, state.seq_lens,
                                       state.slot_mapping, state.block_table), strict=True))
        else:
            buffers[names[0]] = query
    for _, names in binding.outputs:
        buffers[names[0]] = output
    arguments = tuple(buffers[str(a["buffer"])] for a in runtime.external_arguments)
    runtime.prepare(*arguments)
    reports = runtime.resource_report
    reports = reports if isinstance(reports, tuple) else (reports,)
    assert all(row["spill_bytes"] == 0 for row in reports)
    yield torch, runtime, arguments, query, state, key, value, output
    runtime.close()
    sys.path.remove(str(ROOT))


@pytest.mark.parametrize("length", [31, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 3071])
def test_warp_local_attention_matches_fp32_softmax_at_context_boundaries(attention_runtime, length):
    torch, runtime, arguments, query, state, key, value, output = attention_runtime
    state.seq_lens.fill_(length)
    state.slot_mapping.fill_(length - 1)
    output.fill_(float("nan"))
    runtime.run_into(*arguments)
    scores = torch.einsum("qhd,thd->hqt", query.reshape(1, 16, 256).float(),
                          key[:length].repeat_interleave(8, 1).float()) * .0625
    expected = torch.einsum("hqt,thd->qhd", scores.softmax(-1),
                            value[:length].repeat_interleave(8, 1).float()).to(torch.bfloat16)
    torch.testing.assert_close(output.reshape_as(expected), expected, rtol=.005, atol=.001)
