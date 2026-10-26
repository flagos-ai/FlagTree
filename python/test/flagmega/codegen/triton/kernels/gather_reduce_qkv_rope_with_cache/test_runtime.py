# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute compact partial-owner QKV fusion, including its rounding and state effects."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import create_paged_attention_state
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from python.test.flagmega.passes.test_fuse_qkv_statistics import qkv_statistics_graph
from python.test.flagmega.devices import is_nvidia_sm90


@pytest.mark.parametrize("head_dim,rotary_dim,use_mean,heads_tuple", [
    (64, None, False, (4, 2, 2)), (64, 48, True, (4, 2, 2)), (256, 64, False, (4, 2, 2)),
    (64, 48, True, (5, 3, 3)),
])
def test_compact_partial_qkv_query_cache_and_publication(tmp_path, head_dim, rotary_dim, use_mean, heads_tuple):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA required")
    source, config = qkv_statistics_graph(head_dim=head_dim, rotary_dim=rotary_dim, use_mean=use_mean,
                                         heads_tuple=heads_tuple)
    compiled = Compiler().compile(source).module
    assert len([k for k in compiled.kernel_definitions
                if k.dispatch.semantic_op == "ntt.gather_reduce_qkv_rope_with_cache"]) == 1
    assert not [k for k in compiled.kernel_definitions if k.dispatch.semantic_op in {"nn.norm_stats", "distributed.boxing"}]
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    prototype = load(artifact)
    runtime = GeneratedTirCallGraphModule(prototype.artifact, prototype.manifest,
                                         prototype.ir_module, prototype.kernel).load("cuda:0")
    for pool in runtime._pool_values.values():
        pool.fill_(0xA5)
    generator = torch.Generator(device="cuda").manual_seed(763)
    dense, compact = [], []
    for heads in heads_tuple:
        parts = torch.randn((2, heads, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
        dense.append(parts.float().sum(0).bfloat16())
        storage = torch.full((4, (heads + 1) // 2, head_dim), float("nan"), device="cuda", dtype=torch.bfloat16)
        for member in range(2):
            for head_owner in range(2):
                owned = parts[member, head_owner::2]
                storage[member * 2 + head_owner, :len(owned)].copy_(owned)
        compact.append(storage)
    values = {
        name: torch.randn(shape, generator=generator, device="cuda").to(dtype)
        for name, shape, dtype in (
            ("scale", (head_dim,), torch.bfloat16), ("bias", (head_dim,), torch.bfloat16),
            ("cos", (1, 1, rotary_dim or head_dim), torch.float32),
            ("sin", (1, 1, rotary_dim or head_dim), torch.float32),
        )
    }
    state = create_paged_attention_state(config, device="cuda")
    state.block_table.copy_(torch.tensor([[1, 0]], device="cuda", dtype=torch.int32))
    query = torch.empty((1, heads_tuple[0], head_dim), device="cuda", dtype=torch.bfloat16)
    bound = {}
    for node_id, buffers in runtime.buffer_plan.function_map[compiled.entry].parameters:
        node = compiled.node_map[node_id]
        if node_id == "partials":
            bound.update(zip(buffers, compact, strict=True))
        elif isinstance(node.type, fm.RefType):
            bound.update((buffer, getattr(state, field)) for buffer, (field, _) in zip(buffers, node.type.fields, strict=True))
        else:
            bound[buffers[0]] = values[node.attrs["name"]]
    for argument in runtime.external_arguments:
        if argument["role"] == "result":
            bound[argument["buffer"]] = query
    arguments = [bound[arg["buffer"]] for arg in runtime.external_arguments]
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    torch.cuda.synchronize()

    def expected(value):
        value = value.float().unsqueeze(0)
        mean = value.mean(-1, keepdim=True) if use_mean else 0.0
        inverse = torch.rsqrt(torch.clamp(value.square().mean(-1, keepdim=True) - mean * mean, min=0) + 1e-6)
        normalized = ((value - mean) * inverse * values["scale"] + values["bias"]).bfloat16().float()
        extent = rotary_dim or head_dim
        prefix = normalized[..., :extent]
        partner = torch.cat((-prefix[..., extent // 2:], prefix[..., :extent // 2]), -1)
        result = prefix * values["cos"] + partner * values["sin"]
        return torch.cat((result, normalized[..., extent:]), -1).bfloat16()

    torch.testing.assert_close(query, expected(dense[0]), atol=0, rtol=0)
    expected_cache = torch.zeros_like(state.kv_caches)
    expected_cache[1, 0, 0, 0] = expected(dense[1]).reshape(heads_tuple[1], head_dim // 8, 8)
    expected_cache[1, 0, 1, 0] = dense[2].reshape(heads_tuple[1], head_dim // 8, 8)
    torch.testing.assert_close(state.kv_caches, expected_cache, atol=0, rtol=0)
    assert state.sequence_length == 1
    report = runtime.resource_report
    assert report["spill_bytes"] == 0
