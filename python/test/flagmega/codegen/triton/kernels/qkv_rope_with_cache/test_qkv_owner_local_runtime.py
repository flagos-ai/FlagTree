# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit stats collectives followed by owner-local QKV/RoPE."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.ir.ops.nn._rotary_distribution import has_remote_rotary_pairs
from triton.flagmega.runtime import load


def _graph(lanes, use_mean, rotary, swap, external_stats, independent_v):
    mesh = fm.Placement((4, 2) if swap else (2, 4), "xy" if swap else "yx", "bb")
    head_axis, dim_axis = (1, 0) if swap else (0, 1)
    dtype = fm.DType.BFLOAT16 if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
    broad = fm.SBP.broadcast()
    policies = (broad, fm.SBP.split_block_cyclic((head_axis,), 1),
                fm.SBP.split_block_cyclic((dim_axis,), 8 // lanes))
    config = PagedAttentionStateConfig(1, 3, 128, block_size=4, num_blocks=3, lanes=8)

    class Graph(fm.Module):
        def forward(self):
            values, inputs = [], []
            for role, heads in (("q", 5), ("k", 3), ("v", 3)):
                tensor = fm.tensor_type(dtype, (1, heads, 128 // lanes))
                source = self.input(role, tensor, id=role)
                inputs.append(source)
                role_policies = ((broad, fm.SBP.split_block_cyclic((dim_axis,), 1),
                                  fm.SBP.split_block_cyclic((head_axis,), 8 // lanes))
                                 if role == "v" and independent_v else policies)
                distributed = fm.DistributedType(tensor, role_policies, mesh)
                assert not has_remote_rotary_pairs(distributed, rotary)
                values.append(fm.F.distributed.boxing(source, distributed))
            scale = self.input("scale", fm.tensor_type("bfloat16", (128,)))
            bias = self.input("bias", scale.type)
            cosine = self.input("cos", fm.tensor_type("float32", (1, 1, rotary or 128)))
            sine = self.input("sin", cosine.type)
            state = self.input("state", config.ref_type)
            inputs.extend((scale, bias, cosine, sine, state))
            parameters = tuple(fm.F.distributed.boxing(value, fm.DistributedType(value.type, (broad,) * value.type.rank, mesh))
                               for value in (scale, bias, cosine, sine))
            stats = []
            for role, value, heads in zip(("q", "k"), values[:2], (5, 3)):
                if external_stats:
                    tensor = fm.tensor_type("float32", (2 if use_mean else 1, 1, heads, 1))
                    source = self.input(f"{role}_stats", tensor, id=f"{role}_stats")
                    inputs.append(source)
                    destination = fm.DistributedType(tensor, (broad, broad, policies[1], broad), mesh)
                else:
                    source = fm.F.nn.norm_stats(value, axis=-1, use_mean=use_mean, name=f"{role}_stats")
                    assert source.type.partial == fm.SBP.partial((dim_axis,), fm.ReduceOp.SUM)
                    destination = replace(source.type, partial=None)
                stats.append(fm.F.distributed.boxing(source, destination))
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            advance = fm.F.builtin.scalar_const(fm.tensor_type("bool", ()), True)
            scale, bias, cosine, sine = parameters
            result = fm.F.nn.qkv_rope_with_cache(
                fm.F.builtin.tuple(*values), scale, scale, bias, bias, cosine, sine, state, layer, advance, *stats,
                q_axis=-1, q_epsilon=1e-6, q_use_mean=use_mean,
                k_axis=-1, k_epsilon=1e-6, k_use_mean=use_mean,
                rotary_dim=rotary, qkv_layout=("seq", "head", "dim"), attention_layout=("seq", "head", "dim"),
                name="apply",
            )
            query = fm.F.tensors.get_item(result, 0)
            query = fm.F.distributed.boxing(query, query.type.tensor)
            self.function("main", tuple(inputs), (query, fm.F.tensors.get_item(result, 1)))

    return Graph(dialect="high_level", stage="distributed", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build(), config


@pytest.mark.parametrize("lanes,use_mean,rotary,swap", [(1, False, None, False), (8, False, 64, False),
                                                      (8, True, 64, True), (1, True, None, True)])
@pytest.mark.parametrize("external_stats", [False, True])
@pytest.mark.parametrize("independent_v", [False, True])
def test_pair_local_qkv_uses_explicit_stats_and_handles_tail_heads(
    tmp_path, lanes, use_mean, rotary, swap, external_stats, independent_v,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    source, config = _graph(lanes, use_mean, rotary, swap, external_stats, independent_v)
    compiled = Compiler().compile(source).module
    kernels = [kernel for kernel in compiled.kernel_definitions if kernel.dispatch.semantic_op == "nn.qkv_rope_with_cache"]
    assert len(kernels) == 1
    qkv_type = kernels[0].parameters[0].type
    assert all(not has_remote_rotary_pairs(field, rotary) for field in qkv_type.fields[:2])
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator(device="cuda").manual_seed(713)
    values = {role: torch.randn((1, heads, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
              for role, heads in (("q", 5), ("k", 3), ("v", 3))}
    values.update({name: torch.randn(shape, generator=generator, device="cuda").to(dtype)
                   for name, shape, dtype in (("scale", (128,), torch.bfloat16), ("bias", (128,), torch.bfloat16),
                       ("cos", (1, 1, rotary or 128), torch.float32), ("sin", (1, 1, rotary or 128), torch.float32))})
    for role in ("q", "k"):
        value = values[role].float()
        mean_sum = value.sum(-1, keepdim=True)
        square_sum = value.square().sum(-1, keepdim=True)
        if external_stats:
            mean_sum = mean_sum * .25
            square_sum = square_sum * 2
        values[f"{role}_stats"] = torch.stack((mean_sum, square_sum) if use_mean else (square_sum,))
    state = create_paged_attention_state(config, device="cuda")
    state.block_table.copy_(torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32))
    output = torch.empty_like(values["q"])
    function = runtime.buffer_plan.function_map[compiled.entry]
    bound = {}
    for node_id, buffers in function.parameters:
        node = compiled.node_map[node_id]
        if isinstance(node.type, fm.RefType):
            bound.update((buffer, getattr(state, field)) for buffer, (field, _) in zip(buffers, node.type.fields, strict=True))
        else:
            bound[buffers[0]] = values[node.attrs["name"]]
    for argument in runtime.external_arguments:
        if argument["role"] == "result":
            bound[argument["buffer"]] = output
    arguments = [bound[argument["buffer"]] for argument in runtime.external_arguments]
    runtime.prepare(*arguments)

    def expected(role):
        stats = values[f"{role}_stats"]
        mean = stats[0] / 128 if use_mean else 0.0
        variance = stats[-1] / 128 - mean * mean
        normalized = ((values[role].float() - mean) * torch.rsqrt(variance.clamp_min(0) + 1e-6)
                      * values["scale"].float() + values["bias"].float()).bfloat16().float()
        extent = rotary or 128
        prefix = normalized[..., :extent]
        partner = torch.cat((-prefix[..., extent // 2:], prefix[..., :extent // 2]), -1)
        return torch.cat((prefix * values["cos"] + partner * values["sin"], normalized[..., extent:]), -1).bfloat16()

    for position in (0, 3, 4, 7, 8):
        state.seq_lens.fill_(position)
        state.kv_caches.fill_(float("nan"))
        output.fill_(float("nan"))
        runtime.run_into(*arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected("q"), rtol=0, atol=0)
        page = int(state.block_table[0, position // config.block_size])
        key = state.kv_caches[page, 0, 0, position % config.block_size].reshape(1, 3, 128)
        value = state.kv_caches[page, 0, 1, position % config.block_size].reshape(1, 3, 128)
        torch.testing.assert_close(key, expected("k"), rtol=0, atol=0)
        torch.testing.assert_close(value, values["v"], rtol=0, atol=0)
        assert state.seq_lens.item() == position + 1
        assert state.slot_mapping.item() == position
