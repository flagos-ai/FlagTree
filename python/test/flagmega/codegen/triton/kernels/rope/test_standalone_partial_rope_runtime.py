# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The standalone template shares the call-graph RoPE numerical contract."""

import importlib.util
from pathlib import Path

import pytest
from jinja2 import Template

import triton.flagmega


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("head,rotary", [(24, None), (24, 16), (7, 2)])
def test_standalone_rope_rotates_only_its_prefix(tmp_path, dtype, head, rotary):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    template = Path(triton.flagmega.__file__).parent / "codegen/triton/kernels/rope/decode.py.jinja"
    source = ("import triton\nimport triton.language as tl\nfrom triton.language.extra.cuda import libdevice\n" +
              Template(template.read_text()).render())
    path = tmp_path / "standalone.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("partial_rope_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extent = head if rotary is None else rotary
    generator = torch.Generator().manual_seed(39)
    value = torch.randn((2, 3, head), generator=generator).to(device="cuda", dtype=getattr(torch, dtype))
    cos = torch.randn((2, 1, extent), generator=generator).cuda()
    sin = torch.randn((2, 1, extent), generator=generator).cuda()
    prefix = value[..., :extent].float()
    partner = torch.cat((-prefix[..., extent // 2:], prefix[..., :extent // 2]), dim=-1)
    expected = torch.cat(((prefix * cos.float() + partner * sin.float()).to(value.dtype), value[..., extent:]), dim=-1)
    output = torch.empty_like(value)
    module._flagmega_rope[(triton.cdiv(value.numel(), 64), )](value, cos, sin, output, value.numel(), 2, 3, head,
                                                              3 * head, head, 1, 64, rotary_dim=rotary)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
