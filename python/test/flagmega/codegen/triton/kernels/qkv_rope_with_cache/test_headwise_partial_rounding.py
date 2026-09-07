# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute the real headwise template with two physical partial owners."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def render_kernel(round_before_scale, wide=False):
    def partial(storage, offsets):
        return {"member_width": 2, "member_count": 2, "storage": storage,
                "owner": f"qkv_{storage}_partial_member", "owner_stride": 64,
                "offset": offsets, "scalar_dtype": "tl.bfloat16"}

    def head(kind):
        offsets = f"qkv_{kind}_element_offsets"
        return {"compute_active": "True", "outer_capacity": 1, "tile": 64,
                "reduction_capacity": 64, "apply_active": "True", "use_mean": False,
                "normalization_size": 64, "epsilon": 1e-6, "scale": "scale", "scale_offset": offsets,
                "bias": "bias", "bias_offset": offsets, "head_dim": 64, "dimension": offsets,
                "cosine": "cosine", "cosine_offset": offsets, "sine": "sine", "sine_offset": offsets,
                "output": "output", "output_offset": offsets, "output_active": "True",
                "cache_offset": offsets, "partial_input": partial(kind, offsets),
                "round_before_scale": round_before_scale, "input_type": "tl.bfloat16",
                "intermediate_type": "tl.float32" if wide else "tl.bfloat16"}

    call = {"q": head("q"), "k": head("k"), "kv_cache": "cache", "v": {
        "compute_active": "True", "tile": 64, "capacity": 64, "active": "True",
        "partial_input": partial("v", "qkv_v_offsets"), "cache_offset": "64 + qkv_v_offsets"}}
    template = TritonTemplateRegistry().environment.from_string(
        'import triton\nimport triton.language as tl\nfrom triton.language.extra.cuda import libdevice\n'
        '{% import "kernels/gather_reduce_qkv_rope_with_cache/_headwise_partial.py.jinja" as headwise %}\n'
        '@triton.jit\ndef kernel(q, k, v, scale, bias, cosine, sine, output, cache):\n{{ headwise.run(call) }}\n')
    return template.render(call=call)


@pytest.mark.parametrize("round_before_scale", [False, True])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("trig_dtype", ["float32", "bfloat16"])
def test_headwise_partial_matches_unfused_bf16_boundaries(tmp_path, round_before_scale, wide, trig_dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    path = tmp_path / "partial_qkv.py"
    path.write_text(render_kernel(round_before_scale, wide))
    spec = importlib.util.spec_from_file_location("partial_qkv", path)
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    generator = torch.Generator(device="cuda").manual_seed(712)
    q, k, v = [torch.randn((2, 64), generator=generator, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    scale, bias = [torch.randn(64, generator=generator, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    cosine, sine = [torch.randn(64, generator=generator, device="cuda").to(getattr(torch, trig_dtype)) for _ in range(2)]
    output = torch.empty_like(scale)
    cache = torch.empty(128, dtype=torch.bfloat16, device="cuda")
    generated.kernel[(1,)](q, k, v, scale, bias, cosine, sine, output, cache)

    def expected(partials):
        value = partials.float().sum(0).bfloat16().float()
        unit = value * torch.rsqrt(value.square().mean() + 1e-6)
        if round_before_scale and not wide:
            unit = unit.bfloat16().float()
        normalized = unit * scale.float() + bias.float()
        if not wide:
            normalized = normalized.bfloat16()
        rotated = torch.cat((-normalized[32:], normalized[:32]))
        tables = (cosine, sine) if wide else (cosine.bfloat16(), sine.bfloat16())
        return (normalized * tables[0] + rotated * tables[1]).bfloat16()

    torch.testing.assert_close(output, expected(q), rtol=0, atol=0)
    torch.testing.assert_close(cache[:64], expected(k), rtol=0, atol=0)
    torch.testing.assert_close(cache[64:], v.float().sum(0).bfloat16(), rtol=0, atol=0)
