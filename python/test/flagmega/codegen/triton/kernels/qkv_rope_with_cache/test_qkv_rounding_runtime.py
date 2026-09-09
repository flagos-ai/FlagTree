# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One fused QKV op must preserve NormApply and RoPE tensor boundaries."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import create_paged_attention_state
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


def _module(round_before_scale, rotary_dim=None, head_dim=64):
    config = PagedAttentionStateConfig(1, 1, head_dim, block_size=4, num_blocks=2, lanes=8)

    class Graph(fm.Module):
        def forward(self):
            q = self.input("q", fm.tensor_type("bfloat16", (1, 2, head_dim)))
            k = self.input("k", fm.tensor_type("bfloat16", (1, 1, head_dim)))
            v = self.input("v", k.type)
            scale = self.input("scale", fm.tensor_type("bfloat16", (head_dim,)))
            bias = self.input("bias", scale.type)
            cos = self.input("cos", fm.tensor_type("float32", (1, 1, rotary_dim or head_dim)))
            sin = self.input("sin", cos.type)
            state = self.input("state", config.ref_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0, name="layer")
            advance = fm.F.builtin.scalar_const(fm.tensor_type("bool", ()), False, name="advance")
            result = fm.F.nn.qkv_rope_with_cache(
                fm.F.builtin.tuple(q, k, v), scale, scale, bias, bias,
                cos, sin, state, layer, advance,
                q_axis=-1, q_epsilon=1e-6, q_use_mean=False,
                k_axis=-1, k_epsilon=1e-6, k_use_mean=False,
                q_round_before_scale=round_before_scale,
                k_round_before_scale=round_before_scale,
                rotary_dim=rotary_dim,
                qkv_layout=("seq", "head", "dim"), attention_layout=("seq", "head", "dim"),
                name="qkv",
            )
            query = fm.F.tensors.get_item(result, 0, name="query")
            updated = fm.F.tensors.get_item(result, 1, name="updated")
            self.function("main", (q, k, v, scale, bias, cos, sin, state), (query, updated))

    return Graph(dialect="high_level", stage="imported", entry="main").build(), config


@pytest.mark.parametrize("round_before_scale", [False, True])
@pytest.mark.parametrize("head_dim,rotary_dim", [(64, None), (64, 16), (64, 48), (256, 64)])
@pytest.mark.parametrize("cosine,sine", [(1.0, 0.0), (0.75, 0.625)])
def test_fused_qkv_preserves_bf16_norm_and_rotary_results(tmp_path, round_before_scale, cosine, sine, rotary_dim, head_dim):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    source, config = _module(round_before_scale, rotary_dim, head_dim)
    extent = rotary_dim or head_dim
    compiled = Compiler().compile(source).module
    artifact = write_artifact(compiled, tmp_path / "qkv", target="nvidia-sm90", emit_executable=True)
    prototype = load(artifact)
    runtime = GeneratedTirCallGraphModule(prototype.artifact, prototype.manifest,
                                         prototype.ir_module, prototype.kernel).load("cuda:0")
    generator = torch.Generator().manual_seed(712)
    values = {
        name: torch.randn(shape, generator=generator).to(device="cuda", dtype=torch.bfloat16)
        for name, shape in (("q", (1, 2, head_dim)), ("k", (1, 1, head_dim)), ("v", (1, 1, head_dim)), ("scale", (head_dim,)))
    }
    values["bias"] = torch.zeros_like(values["scale"])
    values["cos"] = torch.full((1, 1, extent), cosine, device="cuda")
    values["sin"] = torch.full((1, 1, extent), sine, device="cuda")
    state = create_paged_attention_state(config, device="cuda:0")
    state.kv_caches.zero_()
    output = torch.empty((1, 2, head_dim), dtype=torch.bfloat16, device="cuda")
    function = runtime.buffer_plan.function_map[compiled.entry]
    bound = {}
    for name, buffers in function.parameters:
        node = compiled.node_map[name]
        if isinstance(node.type, fm.RefType):
            bound.update((buffer, getattr(state, field)) for buffer, (field, _) in zip(buffers, node.type.fields, strict=True))
        else:
            assert len(buffers) == 1
            bound[buffers[0]] = values[node.attrs["name"]]
    for argument in runtime.external_arguments:
        if argument["role"] == "result":
            bound[argument["buffer"]] = output
    arguments = [bound[argument["buffer"]] for argument in runtime.external_arguments]
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    torch.cuda.synchronize()
    for kind in ("ttir", "llir", "ptx"):
        (tmp_path / f"kernel.{kind}").write_text(runtime._prepared.compiled_kernel.asm[kind])

    def expected(value):
        unit = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        if round_before_scale:
            unit = unit.bfloat16().float()
        normalized = (unit * values["scale"].float()).bfloat16().float()
        prefix = normalized[..., :extent]
        rotated = torch.cat((-prefix[..., extent // 2:], prefix[..., :extent // 2]), dim=-1)
        result = prefix * values["cos"].float() + rotated * values["sin"].float()
        return torch.cat((result, normalized[..., extent:]), dim=-1).bfloat16()

    torch.testing.assert_close(output, expected(values["q"]), rtol=0, atol=0)
    key = state.kv_caches[0, 0, 0, 0].reshape(1, 1, head_dim)
    torch.testing.assert_close(key, expected(values["k"]), rtol=0, atol=0)
