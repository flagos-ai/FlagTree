# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A shared helper must use its invocation's dimensions, not the first call's."""

import importlib.util
import os
import ast

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _shared_template_context


@pytest.mark.parametrize("first", ["persistent", "state_smem_pipeline"])
@pytest.mark.parametrize("reverse_render", [False, True])
def test_mixed_variants_emit_one_shared_helper_set(first, reverse_render):
    variants = ("persistent", "state_smem_pipeline")
    other = next(value for value in variants if value != first)
    context = _shared_template_context([
        {"family": "gdn_recurrent", "variant": first, "head_block": 8, "query_scale": .5},
        {"family": "gdn_recurrent", "variant": other, "head_block": 256, "query_scale": .0625},
    ])
    context["render_calls"] = ()
    registry = TritonTemplateRegistry()
    order = tuple(reversed(variants)) if reverse_render else variants
    source = "\n".join(registry.render_kernel(
        KernelTemplateSpec("gdn_recurrent", variant, "nvidia", "sm90"), context).source for variant in order)
    definitions = [node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    assert len(definitions) == len(set(definitions))
    assert definitions.count("_flagmega_gdn_recurrent_local_core_tile") == 1
    assert definitions.count("_flagmega_gdn_recurrent_local_norm_tile") == 1
    assert definitions.count("_flagmega_gdn_projection_pair") == 1


def _helpers(directory):
    source = TritonTemplateRegistry(os.environ.get("FLAGMEGA_GDN_TEMPLATE_ROOT")).render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm90"),
        {"head_block": 8, "query_scale_repr": "0.5"}).source
    wrappers = '''
@triton.jit
def _flagmega_silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def norm_kernel(z, result, weight, scratch, V: tl.constexpr):
    row = tl.arange(0, triton.next_power_of_2(V))
    _flagmega_gdn_recurrent_local_norm_tile(z, result, weight, scratch,
        row, row, row, row < V, V, 1e-6)

@triton.jit
def core_kernel(qkv, state, scratch, K: tl.constexpr):
    row = tl.arange(0, 8)
    ones = tl.full((8,), 1., tl.float32)
    _flagmega_gdn_recurrent_local_core_tile(qkv, ones, ones, state, scratch,
        row, row, row < 8, 1, 1, K, 8, 1, triton.next_power_of_2(K),
        False, 1e-12, False, False)
'''
    path = directory / "gdn_dimension_helpers.py"
    path.write_text("import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n" + source + wrappers)
    spec = importlib.util.spec_from_file_location("gdn_dimension_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value_dim", (8, 16, 256))
def test_norm_helper_uses_its_own_value_dimension(tmp_path, value_dim):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = _helpers(tmp_path)
    z = torch.ones(value_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones_like(z)
    scratch = torch.arange(1, value_dim + 1, device="cuda", dtype=torch.float32)
    output = torch.empty_like(z)
    module.norm_kernel[(1,)](z, output, weight, scratch, value_dim, num_warps=8)
    expected = (scratch * torch.rsqrt(scratch.square().mean() + 1e-6) * torch.nn.functional.silu(z.float())).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("key_dim", (4, 16, 64))
def test_core_helper_uses_its_own_query_scale(tmp_path, key_dim):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = _helpers(tmp_path)
    qkv = torch.ones(2 * key_dim + 8, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros((8, key_dim), device="cuda")
    scratch = torch.empty(8, device="cuda")
    module.core_kernel[(1,)](qkv, state, scratch, key_dim, num_warps=8)
    torch.testing.assert_close(state, torch.full_like(state, key_dim ** -.5), rtol=0, atol=0)
    torch.testing.assert_close(scratch, torch.full_like(scratch, key_dim ** -.5), rtol=0, atol=0)
