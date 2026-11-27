# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1100's decode_t16 partial loads clamped tiles and the next tile's block ids early."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

_GFX1100 = KernelTemplateSpec("paged_attention_partial", "decode_t16", "amd", "gfx1100")
_AMD = KernelTemplateSpec("paged_attention_partial", "decode_t16", "amd")
_FLAGS = ("paged_attention_partial_unmasked_tiles", "paged_attention_partial_prefetch_block_ids")


def _render(spec, **context):
    return TritonTemplateRegistry().render_kernel(spec, {"render_calls": [], **context}).source


def _loop(source):
    return source[source.index("for token_start in"):source.index("running_max = tl.max(row_max")]


def test_gfx1100_resolves_its_own_template():
    assert TritonTemplateRegistry().resolve(_GFX1100) == (
        "kernels/paged_attention_partial/platforms/amd/gfx1100/decode_t16.py.jinja"
    )
    assert TritonTemplateRegistry().resolve(
        KernelTemplateSpec("paged_attention_partial", "decode_t16", "amd", "gfx1201")
    ) == "kernels/paged_attention_partial/platforms/amd/decode_t16.py.jinja"


def test_gfx1100_tile_loads_are_unmasked_and_block_ids_load_one_tile_ahead():
    source = _render(_GFX1100)
    compile(source, "gfx1100_decode_t16.py", "exec")
    prologue = source[:source.index("for token_start in")]
    loop = _loop(source)
    assert "next_physical_block = tl.load(" in prologue
    assert "mask=" not in loop
    assert "physical_block = next_physical_block" in loop
    # The next tile's block ids are requested after this tile's K/V, before any math on them.
    assert loop.index("value_value = tl.load(") < loop.index("next_physical_block = tl.load(")
    assert loop.index("next_physical_block = tl.load(") < loop.index("score = tl.sum(")


@pytest.mark.parametrize(
    "spec",
    [_AMD, KernelTemplateSpec("paged_attention_partial", "decode_t16"),
     KernelTemplateSpec("paged_attention_partial", "decode_t32")],
)
def test_other_templates_keep_masked_loads(spec):
    source = _render(spec)
    assert "next_physical_block" not in source and "address_token" not in source
    assert "mask=valid[:, None]" in source


_WRAPPER = '''
@triton.jit
def run(query, kv_cache, slot_mapping, block_table, partial_max, partial_sum, partial_accumulator,
        token_tile: tl.constexpr):
    shard = tl.program_id(0)
    _flagmega_paged_attention_partial(
        query, kv_cache, slot_mapping, block_table, 0,
        partial_max + shard, partial_sum + shard, partial_accumulator + shard * 128,
        shard, 0, 0,
        1, 2, 1, 128, 256, 0.08838834764831845, token_tile, 4, 1, 1, 1, 8,
    )
'''


def _module(tmp_path, source, name):
    path = tmp_path / f"{name}.py"
    path.write_text("import triton\nimport triton.language as tl\n" + source + _WRAPPER)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def _run(tmp_path, source, name, query, kv_cache, slot_mapping, block_table):
    import torch

    maxima, sums = torch.empty(4, device="cuda"), torch.empty(4, device="cuda")
    accumulators = torch.empty(4 * 128, device="cuda")
    _module(tmp_path, source, name).run[(4,)](
        query, kv_cache, slot_mapping, block_table, maxima, sums, accumulators, token_tile=16)
    torch.cuda.synchronize()
    return maxima, sums, accumulators


@pytest.mark.parametrize("context", [1, 3, 17, 255, 256, 257, 512, 700, 2000, 2048])
def test_gfx1100_partial_is_bitwise_equal_to_amd_per_row(tmp_path, context):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    torch.manual_seed(context)
    query = torch.randn(128, device="cuda").bfloat16()
    kv_cache = (3 * torch.randn(8 * 2 * 256 * 128, device="cuda")).bfloat16()
    slot_mapping = torch.tensor([context - 1], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4], device="cuda", dtype=torch.int32)
    args = (query, kv_cache, slot_mapping, block_table)
    expected = _run(tmp_path, _render(_AMD), "amd_t16", *args)
    sources = {"gfx1100_t16": _render(_GFX1100)}
    for flag in _FLAGS:
        sources[flag] = _render(_AMD, **{flag: True})
    cache = kv_cache.view(8, 2, 256, 128)[block_table.long()].double()
    keys = cache[:, 0].reshape(-1, 128)[:context]
    values = cache[:, 1].reshape(-1, 128)[:context]
    reference = torch.softmax(keys @ query.double() * 128 ** -.5, 0) @ values
    for name, source in sources.items():
        maxima, sums, accumulators = _run(tmp_path, source, name, *args)
        for got, want in zip((maxima, sums, accumulators), expected):
            assert torch.equal(got, want), name
        weights = torch.exp(maxima - maxima.max()).double()
        combined = (weights[:, None] * accumulators.view(4, 128).double()).sum(0) / (
            weights * sums.double()).sum()
        torch.testing.assert_close(combined, reference, rtol=1e-4, atol=1e-4, msg=name)
