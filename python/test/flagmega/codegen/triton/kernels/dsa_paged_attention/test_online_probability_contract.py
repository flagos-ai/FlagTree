# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Unnormalized BF16 probabilities retain the FP32 online normalizer."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import (
    DictWeightResolver,
    PagedAttentionStateConfig,
    TorchEvaluator,
    create_paged_attention_state,
)

from .test_decode_runtime import attention_module, prepare


def inputs(device="cpu", slots=(0, 1, 2)):
    config = PagedAttentionStateConfig(1, 1, 16, num_blocks=2)
    state = create_paged_attention_state(config, device=device)
    state.seq_lens.fill_(3)
    state.slot_mapping.fill_(2)
    query = torch.zeros((1, 2, 2, 8), dtype=torch.bfloat16, device=device)
    topk = torch.full((1, max(slots) + 1), -1, dtype=torch.int32, device=device)
    for logical, slot in enumerate(slots):
        topk[0, slot] = logical
    for position, value in enumerate((1, 2, 4)):
        state.kv_caches[0, 0, 1, position].fill_(value)
    return config, state, query, topk


def test_explicit_online_contract_rounds_before_normalization_and_resumes(tmp_path):
    config, state, query, topk = inputs()
    module = attention_module(config, 3, probability_rounding="online_bf16")
    restored = fm.load_module(fm.emit_module(module, tmp_path / "online.py"))
    assert restored.semantic_hash == module.semantic_hash
    (result,) = TorchEvaluator(DictWeightResolver({})).run(restored, {"query": query, "topk": topk, "state": state})
    expected = torch.tensor(7 / 3, dtype=torch.bfloat16)
    assert expected != (torch.tensor(1 / 3, dtype=torch.bfloat16).float() * 7).bfloat16()
    torch.testing.assert_close(result, torch.full_like(result, expected.item()), rtol=0, atol=0)


def test_default_probability_contract_keeps_its_canonical_hash():
    config, _, _, _ = inputs()
    assert (
        attention_module(config, 3).semantic_hash
        == attention_module(config, 3, probability_rounding="normalized").semantic_hash
    )


def test_unknown_probability_contract_is_rejected():
    config, _, _, _ = inputs()
    with pytest.raises(IRSchemaError, match="probability_rounding"):
        attention_module(config, 3, probability_rounding="unknown")


def test_default_index_traversal_keeps_its_canonical_hash():
    config, _, _, _ = inputs()
    assert (
        attention_module(config, 3).semantic_hash == attention_module(config, 3, index_traversal="slots").semantic_hash
    )


@pytest.mark.parametrize("value", ["unknown", 1, True, None, []])
def test_invalid_index_traversal_is_rejected(value):
    from triton.flagmega.ir.ops.nn.dsa_paged_attention import DsaPagedAttention

    config, _, _, _ = inputs()
    attrs = next(node.attrs for node in attention_module(config, 3).nodes if node.op == "nn.dsa_paged_attention")
    with pytest.raises(IRSchemaError, match="index_traversal"):
        DsaPagedAttention.normalize_attrs({**attrs, "index_traversal": value})


def test_compact_traversal_cpu_semantics_and_typed_reload(tmp_path):
    config, state, _, padded = inputs(slots=(0, 65, 129))
    generator = torch.Generator().manual_seed(0)
    query = torch.randn((1, 2, 2, 8), generator=generator).bfloat16()
    keys = torch.randn((3, 1, 2, 8), generator=generator).bfloat16()
    values = torch.randn((3, 1, 2, 8), generator=generator).bfloat16()
    state.kv_caches[0, 0, 0, :3].copy_(keys)
    state.kv_caches[0, 0, 1, :3].copy_(values)
    compact = attention_module(config, 130, probability_rounding="online_bf16", index_traversal="compact")
    restored = fm.load_module(fm.emit_module(compact, tmp_path / "compact.py"))
    assert restored.semantic_hash == compact.semantic_hash
    slots = attention_module(config, 130, probability_rounding="online_bf16")
    assert compact.semantic_hash != slots.semantic_hash
    evaluator = TorchEvaluator(DictWeightResolver({}))
    dense = torch.full_like(padded, -1)
    dense[0, :3] = torch.arange(3)

    def run(module, topk):
        return evaluator.run(module, {"query": query, "topk": topk, "state": state})[0]

    expected = run(slots, dense)
    torch.testing.assert_close(run(restored, padded), expected, rtol=0, atol=0)
    assert (run(slots, padded) != expected).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slots", [(0, 1, 2), (64, 65, 66), (0, 65, 129)])
def test_device_online_contract_handles_empty_and_multiple_tiles(tmp_path, slots):
    config, state, query, topk = inputs("cuda", slots)
    output = torch.empty_like(query)
    module = attention_module(config, topk.shape[1], probability_rounding="online_bf16")
    runtime, arguments = prepare(module, tmp_path / "attention", {"query": query, "topk": topk}, state, output)
    try:
        runtime.run_into(*arguments)
        torch.testing.assert_close(
            output, torch.full_like(output, torch.tensor(7 / 3, dtype=torch.bfloat16).item()), rtol=0, atol=0
        )
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
