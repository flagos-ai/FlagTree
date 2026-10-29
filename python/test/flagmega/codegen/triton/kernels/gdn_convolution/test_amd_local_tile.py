# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1201 GDN convolution tile: every history slot is loaded before the
state shift, with the generic arithmetic order, so it matches the generic tile
bitwise; gfx1100 and other platforms keep the generic tile."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

torch = pytest.importorskip("torch")

CONTEXT = {"mesh_size": 1}
AMD = KernelTemplateSpec("gdn_convolution", "persistent", "amd", "gfx1201")
GFX1100 = KernelTemplateSpec("gdn_convolution", "persistent", "amd", "gfx1100")
GENERIC = KernelTemplateSpec("gdn_convolution", "persistent", "nvidia", "sm120")


def test_amd_template_loads_history_before_the_shift_and_generic_is_unchanged():
    registry = TritonTemplateRegistry()
    amd = registry.render_kernel(AMD, CONTEXT).source
    generic = registry.render_kernel(GENERIC, CONTEXT).source
    assert amd.count("def _flagmega_gdn_convolution_local_tile(") == 1
    assert "def _flagmega_amd_gdn_convolution_accumulate(" in amd
    assert "_flagmega_amd_gdn_convolution_accumulate" not in generic
    assert "for history_index in tl.static_range(0, HISTORY):" in generic
    assert "for history_index in tl.static_range(0, HISTORY):" not in amd
    assert registry.render_kernel(GFX1100, CONTEXT).source == generic


def test_nontemporal_recurrent_state_store_is_gfx1201_only():
    registry = TritonTemplateRegistry()
    store = 'cache_modifier=".cs"'
    gfx1201 = registry.render_kernel(KernelTemplateSpec("gdn_recurrent", "persistent", "amd", "gfx1201"), CONTEXT)
    gfx1100 = registry.render_kernel(KernelTemplateSpec("gdn_recurrent", "persistent", "amd", "gfx1100"), CONTEXT)
    assert gfx1201.source.count(store) == 1
    assert store not in gfx1100.source


def _gfx1201():
    return torch.cuda.is_available() and bool(torch.version.hip) and \
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1201"


KERNEL = '''
@triton.jit
def conv_kernel(qkv, destination, conv_state, conv_weight, CHANNELS: tl.constexpr, HISTORY: tl.constexpr,
                ROUND_PRODUCTS: tl.constexpr, ROUND_BEFORE_ACTIVATION: tl.constexpr, CURRENT_FIRST: tl.constexpr):
    channels = tl.program_id(0) * 128 + tl.arange(0, 128)
    _flagmega_gdn_convolution_local_tile(
        qkv, destination, conv_state, conv_weight, channels, channels, channels * (HISTORY + 1), channels,
        channels < CHANNELS, HISTORY, 1, ROUND_PRODUCTS, ROUND_BEFORE_ACTIVATION, CURRENT_FIRST)
'''


def _module(tmp_path, spec, name):
    source = "import triton\nimport triton.language as tl\n"
    source += TritonTemplateRegistry().render_kernel(spec, CONTEXT).source
    if "def _flagmega_silu(" not in source:
        source += "\n@triton.jit\ndef _flagmega_silu(x):\n    return x * tl.sigmoid(x)\n"
    source += KERNEL
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("current_first", (True, False))
@pytest.mark.parametrize("round_products", (True, False))
@pytest.mark.parametrize("history", (1, 3, 4))
def test_amd_convolution_tile_matches_the_generic_tile_bitwise(tmp_path, history, round_products, current_first):
    if not _gfx1201():
        pytest.skip("requires a gfx1201 GPU")
    channels = 200  # a masked tail in the second tile
    generator = torch.Generator().manual_seed(history * 4 + round_products * 2 + current_first)
    qkv = torch.randn(channels, generator=generator).bfloat16().cuda()
    weight = torch.randn(channels * (history + 1), generator=generator).bfloat16().cuda()
    state = torch.randn((channels + 7) // 8 * 8 * history, generator=generator).bfloat16().cuda()
    results = []
    for spec, name in ((GENERIC, "generic_conv"), (AMD, "amd_conv")):
        conv_state = state.clone()
        output = torch.full((channels, ), float("nan"), device="cuda").bfloat16()
        _module(tmp_path, spec, name).conv_kernel[(2, )](qkv, output, conv_state, weight, channels, history,
                                                          round_products, True, current_first, num_warps=4)
        torch.cuda.synchronize()
        results.append((output.cpu(), conv_state.cpu()))
    (generic_output, generic_state), (amd_output, amd_state) = results
    assert torch.isfinite(generic_output.float()).all()
    assert torch.equal(generic_output.view(torch.int16), amd_output.view(torch.int16))
    assert torch.equal(generic_state.view(torch.int16), amd_state.view(torch.int16))
