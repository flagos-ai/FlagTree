# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native softmax exponent rounding must survive the strict module contract."""

from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load

from .test_mma_online_schedule import _arguments, _compile, _graph


@triton.jit
def _native_two_token_reference(Query, Key0, Key1, Value0, Value1, Out):
    head = tl.program_id(0)
    dims = tl.arange(0, 512)
    # Only feature zero is nonzero. Both scores and the BF16 PV products
    # are exact FP32 products, independently of the matrix schedule.
    q = tl.load(Query + head * 512).to(tl.float32)
    score0 = q * tl.load(Key0).to(tl.float32)
    score1 = q * tl.load(Key1).to(tl.float32)
    maximum = tl.maximum(score0, score1)
    scale = 1.4426950408889634
    p0 = tl.exp2(score0 * scale - maximum * scale)
    p1 = tl.exp2(score1 * scale - maximum * scale)
    inverse = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[p0 + p1],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    p0 = p0.to(tl.bfloat16).to(tl.float32)
    p1 = p1.to(tl.bfloat16).to(tl.float32)
    value = p0 * tl.load(Value0 + dims).to(tl.float32) + p1 * tl.load(Value1 + dims).to(tl.float32)
    tl.store(Out + head * 512 + dims, (value * inverse).to(tl.bfloat16))


@pytest.mark.parametrize("traversal", ["slots", "compact"])
@pytest.mark.parametrize("floating_point_contract", [None, "strict"])
def test_native_two_token_probabilities_preserve_rounding(tmp_path, traversal, floating_point_contract):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 512, num_blocks=2)
    source = _graph(config, 2, heads=16, index_traversal=traversal)
    if floating_point_contract is not None:
        source = replace(source, metadata={**source.metadata, "floating_point_contract": floating_point_contract})
    module, artifact = _compile(source, tmp_path, "bf16_mma")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(2)
    state.slot_mapping.fill_(1)
    generator = torch.Generator().manual_seed(0)
    q_first = (torch.randn((16,), generator=generator) * 4).bfloat16().cuda()
    keys = torch.randn((2,), generator=generator).bfloat16().cuda()
    # Close scores with a large common offset expose the exponent's
    # fused multiply-subtract rounding at the BF16 output boundary.
    keys[0], keys[1] = 256, 258
    values = torch.randn((2, 512), generator=generator).bfloat16().cuda()
    query = torch.zeros((1, 16, 64, 8), dtype=torch.bfloat16, device="cuda")
    query.reshape(16, 512)[:, 0].copy_(q_first / 16)
    state.kv_caches.zero_()
    state.kv_caches[0, 0, 0, 0].reshape(-1)[0] = keys[0]
    state.kv_caches[0, 0, 0, 1].reshape(-1)[0] = keys[1]
    state.kv_caches[0, 0, 1, 0].copy_(values[0].reshape(1, 64, 8))
    state.kv_caches[0, 0, 1, 1].copy_(values[1].reshape(1, 64, 8))
    topk = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    actual, expected = torch.empty_like(query), torch.empty_like(query)
    reference = _native_two_token_reference[(16,)](
        query, state.kv_caches[0, 0, 0, 0], state.kv_caches[0, 0, 0, 1],
        state.kv_caches[0, 0, 1, 0], state.kv_caches[0, 0, 1, 1], expected,
        num_warps=8, enable_fp_fusion=True,
    )
    assert reference.metadata.enable_fp_fusion
    runtime = load(artifact, device="cuda:0")
    try:
        arguments = _arguments(runtime, module, state, query, topk, actual)
        runtime.prepare(*arguments)
        assert all(
            prepared.compiled_kernel.metadata.enable_fp_fusion == (floating_point_contract != "strict")
            for prepared in runtime._prepared
        )
        runtime.run_into(*arguments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
