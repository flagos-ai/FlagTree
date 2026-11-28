# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute compact partial-owner QKV fusion, including its rounding and state effects."""

import ast

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import create_paged_attention_state
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from python.test.flagmega.passes.test_fuse_qkv_statistics import qkv_statistics_graph


@pytest.mark.parametrize("head_dim", [64, 256])
def test_partial_cache_coordinates_are_bound_in_emitted_writer(tmp_path, head_dim):
    source, _ = qkv_statistics_graph(head_dim=head_dim)
    compiled = Compiler().compile(source).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    tree = ast.parse((artifact / "generated_kernels.py").read_text())
    writers = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.endswith("__write")]
    assert writers
    for writer in writers:
        coordinates = [
            node for node in ast.walk(writer)
            if isinstance(node, ast.Name) and node.id.startswith(("qkv_k_cache_", "qkv_v_cache_"))
        ]
        used = {node.id for node in coordinates if isinstance(node.ctx, ast.Load)}
        defined = {node.id for node in coordinates if isinstance(node.ctx, ast.Store)}
        assert used, writer.name
        assert not used - defined, f"{writer.name}: unbound cache coordinates {used - defined}"


@pytest.mark.parametrize("head_dim", [64, 256])
def test_fp32_partial_storage_rounds_materialized_bf16_before_all_consumers(tmp_path, head_dim):
    source, _ = qkv_statistics_graph(head_dim=head_dim, partial_dtype="float32")
    compiled = Compiler().compile(source).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    text = (artifact / "generated_kernels.py").read_text()
    tree = ast.parse(text)
    sums = [
        node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.startswith(('qkv_q', 'qkv_k',
                                                                   'qkv_v')) and target.id.endswith('_value')
            for target in node.targets) and 'tl.sum(' in ast.unparse(node.value)
    ]
    assert len(sums) >= 3
    for node in sums:
        assert ').to(tl.bfloat16).to(tl.float32)' in ast.unparse(node.value)
    # External pointers inherit their actual typed argument ABI, so they need
    # no textual pointer cast. Rounding belongs to the complete sum above.
    assert all(field.tensor.dtype.elem_type == fm.DType.FLOAT32 for field in compiled.node_map['partials'].type.fields)


@pytest.mark.parametrize("head_dim,rotary_dim,use_mean,heads_tuple", [
    (64, None, False, (4, 2, 2)),
    (64, 48, True, (4, 2, 2)),
    (256, 64, False, (4, 2, 2)),
    (64, 48, True, (5, 3, 3)),
])
def test_compact_partial_qkv_query_cache_and_publication(tmp_path, head_dim, rotary_dim, use_mean, heads_tuple,
                                                         partial_dtype="bfloat16"):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    source, config = qkv_statistics_graph(head_dim=head_dim, rotary_dim=rotary_dim, use_mean=use_mean,
                                          heads_tuple=heads_tuple, partial_dtype=partial_dtype)
    compiled = Compiler().compile(source).module
    assert len([
        k for k in compiled.kernel_definitions if k.dispatch.semantic_op == "ntt.gather_reduce_qkv_rope_with_cache"
    ]) == 1
    assert not [
        k for k in compiled.kernel_definitions if k.dispatch.semantic_op in {"nn.norm_stats", "distributed.boxing"}
    ]
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    prototype = load(artifact)
    runtime = GeneratedTirCallGraphModule(prototype.artifact, prototype.manifest, prototype.ir_module,
                                          prototype.kernel).load("cuda:0")
    for pool in runtime._pool_values.values():
        pool.fill_(0xA5)
    generator = torch.Generator(device="cuda").manual_seed(763)
    dense, compact = [], []
    for heads in heads_tuple:
        parts = torch.randn((2, heads, head_dim), generator=generator, device="cuda",
                            dtype=getattr(torch, partial_dtype))
        dense.append(parts.float().sum(0).bfloat16())
        storage = torch.full((4, (heads + 1) // 2, head_dim), float("nan"), device="cuda",
                             dtype=getattr(torch, partial_dtype))
        for member in range(2):
            for head_owner in range(2):
                owned = parts[member, head_owner::2]
                storage[member * 2 + head_owner, :len(owned)].copy_(owned)
        compact.append(storage)
    values = {
        name: torch.randn(shape, generator=generator, device="cuda").to(dtype)
        for name, shape, dtype in (
            ("scale", (head_dim, ), torch.bfloat16),
            ("bias", (head_dim, ), torch.bfloat16),
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
            bound.update(
                (buffer, getattr(state, field)) for buffer, (field, _) in zip(buffers, node.type.fields, strict=True))
        else:
            bound[buffers[0]] = values[node.attrs["name"]]
    for argument in runtime.external_arguments:
        if argument["role"] == "result":
            bound[argument["buffer"]] = query
    arguments = [bound[arg["buffer"]] for arg in runtime.external_arguments]
    runtime.prepare(*arguments)

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

    for position in (0, 3, 4, 7):
        state.seq_lens.fill_(position)
        state.kv_caches.fill_(-9)
        query.fill_(float("nan"))
        expected_cache = state.kv_caches.clone()
        page = int(state.block_table[0, position // config.block_size])
        offset = position % config.block_size
        expected_cache[page, 0, 0, offset] = expected(dense[1]).reshape(heads_tuple[1], head_dim // 8, 8)
        expected_cache[page, 0, 1, offset] = dense[2].reshape(heads_tuple[1], head_dim // 8, 8)
        runtime.run_into(*arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(query, expected(dense[0]), atol=0, rtol=0)
        torch.testing.assert_close(state.kv_caches, expected_cache, atol=0, rtol=0)
        assert state.sequence_length == position + 1
        assert state.slot_mapping.item() == position
        assert state.query_start_loc.tolist() == [0, 1]
    report = runtime.resource_report
    assert report["spill_bytes"] == 0


def test_fp32_partial_sum_rounding_on_device(tmp_path):
    test_compact_partial_qkv_query_cache_and_publication(tmp_path, 64, None, False, (4, 2, 2), partial_dtype="float32")
