# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The AMD decode_t32 partial issues the V tile load next to K, unchanged math."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

_GENERIC = KernelTemplateSpec("paged_attention_partial", "decode_t32")
_AMD = KernelTemplateSpec("paged_attention_partial", "decode_t32", "amd", "gfx1201")


def _render(spec):
    return TritonTemplateRegistry().render_kernel(spec, {"render_calls": []}).source


def test_amd_issues_value_before_the_score_math():
    assert TritonTemplateRegistry().resolve(_AMD) == (
        "kernels/paged_attention_partial/platforms/amd/decode_t32.py.jinja"
    )
    amd, generic = _render(_AMD), _render(_GENERIC)
    compile(amd, "amd_decode_t32.py", "exec")
    assert amd.index("value_value = tl.load(") < amd.index("score = tl.sum(")
    assert generic.index("value_value = tl.load(") > generic.index("probability = tl.where(")
    # Only the position of the V load differs.
    assert sorted(amd.splitlines()) == sorted(generic.splitlines())


_WRAPPER = '''
@triton.jit
def run(query, kv_cache, slot_mapping, block_table, partial_max, partial_sum, partial_accumulator):
    shard = tl.program_id(0)
    _flagmega_paged_attention_partial(
        query, kv_cache, slot_mapping, block_table, 0,
        partial_max + shard, partial_sum + shard, partial_accumulator + shard * 128,
        shard, 0, 0,
        1, 2, 1, 128, 256, 0.08838834764831845, 32, 4, 1, 1, 1, 4,
    )
'''


def _module(tmp_path, spec, name):
    path = tmp_path / f"{name}.py"
    path.write_text("import triton\nimport triton.language as tl\n" + _render(spec) + _WRAPPER)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def test_amd_partial_matches_generic_bitwise(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    torch.manual_seed(5)
    context = 700
    query = torch.randn(128, device="cuda").bfloat16()
    kv_cache = torch.randn(4 * 2 * 256 * 128, device="cuda").bfloat16()
    slot_mapping = torch.tensor([context - 1], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    results = []
    for name, spec in (("generic_partial", _GENERIC), ("amd_partial", _AMD)):
        outputs = (torch.empty(4, device="cuda"), torch.empty(4, device="cuda"),
                   torch.empty(4 * 128, device="cuda"))
        _module(tmp_path, spec, name).run[(4,)](query, kv_cache, slot_mapping, block_table, *outputs)
        results.append(outputs)
    torch.cuda.synchronize()
    for generic, amd in zip(*results):
        assert torch.equal(generic, amd)
    # Combine the four shards and compare with a dense reference.
    cache = kv_cache.view(4, 2, 256, 128)[block_table.long()].float()
    keys = cache[:, 0].reshape(-1, 128)[:context]
    values = cache[:, 1].reshape(-1, 128)[:context]
    reference = torch.softmax(keys @ query.float() * 128 ** -.5, 0) @ values
    maxima, sums, accumulators = results[1]
    weights = torch.exp(maxima - maxima.max())
    combined = (weights[:, None] * accumulators.view(4, 128)).sum(0) / (weights * sums).sum()
    torch.testing.assert_close(combined, reference, rtol=1e-4, atol=1e-4)
