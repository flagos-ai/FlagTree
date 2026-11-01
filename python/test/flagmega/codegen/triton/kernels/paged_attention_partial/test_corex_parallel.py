# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The parallel COREX template must preserve the portable attention ABI."""
import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def _call():
    return dict(
        family="paged_attention_partial", variant="decode_t16", symbol="parallel",
        signature="q, kv, slot, table, stats, acc", local_capacity=4,
        head_dim=128, token_tile=16, num_query_heads=16, num_kv_heads=8,
        context_shard="shard_y", head="attention_state_offsets + shard_x * 4",
        active="True", query_token="attention_state_offsets // 4", num_layers=2,
        block_size=256, scale=128 ** -.5, context_shards=4,
        query_dim_stride=8, query_lanes=8, accumulator_dim_stride=1, num_blocks=16,
        query="q + (attention_state_offsets + shard_x * 4) * 128",
        kv_cache="kv", slot_mapping="slot", block_table="table", layer_id="1",
        partial_max="stats + shard_index * 4 + attention_state_offsets",
        partial_sum="stats + 64 + shard_index * 4 + attention_state_offsets",
        partial_accumulator="acc + (shard_index * 4 + attention_state_offsets) * 128",
        semantic_attrs={"layout": ("seq", "head", "dim")},
        inputs=[{"formal": "q", "buffers": [{"abi": {
            "logical_shape": (1, 16, 16), "scalar_dtype": "bfloat16"}}]}],
        outputs=[{"formal": "result_0", "buffers": [{"abi": {
            "local_capacity_shape": (1, 4, 1)}}]}],
    )


def _render(call, warps=32):
    return TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("paged_attention_partial", "decode_t16", "corex", "bi_v150"),
        dict(render_calls=[call], num_warps=warps, distributed_entry=True,
             mesh_hierarchy=[4, 4], mesh_rank=2, mesh_levels="bb",
             mesh_axis_names=["block_y", "block_x"]),
    ).source


def test_parallel_template_requires_one_uniform_query():
    call = _call()
    call["inputs"][0]["buffers"][0]["abi"]["logical_shape"] = (2, 16, 16)
    call["outputs"][0]["buffers"][0]["abi"]["local_capacity_shape"] = (2, 2, 1)
    source = _render(call)
    assert "_flagmega_paged_attention_partial_parallel" not in source
    assert "for attention_state_start" in source


@pytest.mark.parametrize("warps", [4, 8])
def test_other_launch_contracts_use_portable_template(warps):
    assert "_flagmega_paged_attention_partial_parallel" not in _render(_call(), warps)


@pytest.mark.parametrize("warps", [16, 32])
def test_parallel_template_preserves_physical_launch_contract(warps):
    source = _render(_call(), warps)
    assert "_flagmega_paged_attention_partial_parallel" in source
    assert f"[{warps // 16}, 4, 4, 1]" in source


@pytest.fixture(scope="module", params=[16, 32])
def attention_kernels(tmp_path_factory, request):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    registry = TritonTemplateRegistry()
    portable = registry.render("kernels/paged_attention_partial/decode.py.jinja",
                               {"paged_attention_partial_variant": "decode_t16"})
    source = "import triton\nimport triton.language as tl\nimport triton.experimental.tle.language as tle\n"
    source += "FLAGMEGA_GRID_MESH = tl.constexpr(tle.device_mesh({'block': [('block_y', 4), ('block_x', 4)]}))\n"
    source += portable + _render(_call(), request.param) + '''
@triton.jit
def original(q, kv, slot, table, stats, acc):
    owner = tl.program_id(0)
    for local_head in tl.range(4):
        head = owner % 4 * 4 + local_head
        _flagmega_paged_attention_partial(q + head * 128, kv, slot, table, 1,
            stats + owner * 4 + local_head, stats + 64 + owner * 4 + local_head,
            acc + (owner * 4 + local_head) * 128, owner // 4, head, 0,
            2, 16, 8, 128, 256, 128 ** -.5, 16, 4, 8, 8, 1, 16)
'''
    directory = tmp_path_factory.mktemp("corex-parallel-attention")
    path = directory / "attention.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("corex_parallel_attention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module.original, 32), (module.parallel, request.param)


@pytest.mark.parametrize("context", [1, 17, 127, 255, 256, 257, 319, 320, 321, 511, 702, 1024, 2048])
@pytest.mark.parametrize("unaligned", [False, True])
def test_parallel_attention_exact_paged_states(attention_kernels, context, unaligned):
    torch = pytest.importorskip("torch")
    torch.manual_seed(41)
    q = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    storage = torch.randn(16 * 2 * 2 * 256 * 8 * 128 + int(unaligned),
                          device="cuda", dtype=torch.bfloat16)
    kv = storage[int(unaligned):].view(16, 2, 2, 256, 8, 128)
    table = torch.tensor([3, 5, 7, 9, 11, 13, -1, 99, 2, 4, 6, 8, 10, 12, 14, 0],
                         device="cuda", dtype=torch.int32)
    slot = torch.tensor([context - 1], device="cuda", dtype=torch.int64)
    results = []
    for kernel, warps in attention_kernels:
        stats = torch.full((128,), float("nan"), device="cuda")
        accumulator = torch.full((64, 128), float("nan"), device="cuda")
        kernel[(16,)](q, kv, slot, table, stats, accumulator,
                      num_warps=warps, enable_fp_fusion=False)
        torch.cuda.synchronize()
        results.append((stats, accumulator))
    for original, parallel in zip(*results):
        assert torch.equal(original, parallel)
