# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The AMD decode_t16 partial keeps per-token-row softmax state in the loop."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

_GENERIC = KernelTemplateSpec("paged_attention_partial", "decode_t16")
_AMD = {
    arch: KernelTemplateSpec("paged_attention_partial", "decode_t16", "amd", arch)
    for arch in ("gfx1100", "gfx1201")
}


def _render(spec, **context):
    return TritonTemplateRegistry().render_kernel(spec, {"render_calls": [], **context}).source


def _loop(source):
    return source[source.index("for token_start in"):source.index("partial_index = 0")]


@pytest.mark.parametrize("arch", sorted(_AMD))
def test_amd_decode_t16_renders_per_row_state(arch):
    assert TritonTemplateRegistry().resolve(_AMD[arch]) == (
        "kernels/paged_attention_partial/platforms/amd/decode_t16.py.jinja"
    )
    amd = _render(_AMD[arch])
    compile(amd, "amd_decode_t16.py", "exec")
    loop = _loop(amd)
    # No reduction over the token axis inside the loop; the rows merge once after it.
    body = loop[:loop.index("running_max = tl.max(row_max")]
    assert "tl.max(" not in body and "axis=0" not in body
    assert body.index("value_value = tl.load(") < body.index("score = tl.sum(")


def test_generic_decode_t16_keeps_shared_running_state():
    generic = _render(_GENERIC)
    assert "row_max" not in generic
    assert "tile_max = tl.max(score, axis=0)" in generic


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


@pytest.mark.parametrize("context", [1, 3, 17, 255, 257, 700, 2000])
def test_per_row_partial_matches_dense_reference(tmp_path, context):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    torch.manual_seed(context)
    query = torch.randn(128, device="cuda").bfloat16()
    kv_cache = (3 * torch.randn(8 * 2 * 256 * 128, device="cuda")).bfloat16()
    slot_mapping = torch.tensor([context - 1], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4], device="cuda", dtype=torch.int32)
    sources = {
        "amd_t16": _render(_AMD["gfx1201"]),
        # The per-row switch without the AMD V-load placement.
        "per_row_value_after_score": _render(_GENERIC, paged_attention_partial_per_row_softmax=True),
    }
    cache = kv_cache.view(8, 2, 256, 128)[block_table.long()].double()
    keys = cache[:, 0].reshape(-1, 128)[:context]
    values = cache[:, 1].reshape(-1, 128)[:context]
    reference = torch.softmax(keys @ query.double() * 128 ** -.5, 0) @ values
    for name, source in sources.items():
        maxima, sums = torch.empty(4, device="cuda"), torch.empty(4, device="cuda")
        accumulators = torch.empty(4 * 128, device="cuda")
        _module(tmp_path, source, name).run[(4,)](
            query, kv_cache, slot_mapping, block_table, maxima, sums, accumulators, token_tile=16)
        torch.cuda.synchronize()
        # Shards with no tokens report the empty state.
        empty = sums == 0
        assert torch.all(maxima[empty] == -float("inf"))
        assert torch.all(accumulators.view(4, 128)[empty] == 0)
        weights = torch.exp(maxima - maxima.max()).double()
        combined = (weights[:, None] * accumulators.view(4, 128).double()).sum(0) / (
            weights * sums.double()).sum()
        torch.testing.assert_close(combined, reference, rtol=1e-4, atol=1e-4, msg=name)
