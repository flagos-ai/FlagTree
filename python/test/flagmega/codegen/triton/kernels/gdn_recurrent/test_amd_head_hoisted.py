# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""AMD RDNA GDN recurrent body: head-only quantities (both gates, normalized
q/k rows) are computed once per value head, per-value quantities are [V, 1]
columns of the [V, K] tile, and the AMD target defaults to 16-value tiles."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
from triton.flagmega.targets.amd_gfx1201 import AmdGfx1201Target
from triton.flagmega.targets.portable_triton_implementations import portable_triton_implementation_model

torch = pytest.importorskip("torch")

GDN = "tir.gdn_recurrent.persistent"
CONTEXT = {"mesh_size": 1}


def _gdn_parameters(target):
    model = target.triton_implementation_model
    return next(impl.parameters for impl in model.implementations if impl.id == GDN)


def test_amd_target_defaults_gdn_to_16_value_tiles_but_keeps_a_supplied_model():
    assert _gdn_parameters(AmdGfx1201Target())["tile_state"] == (128, 16)
    supplied = AmdGfx1201Target(triton_implementation_model=portable_triton_implementation_model())
    assert _gdn_parameters(supplied)["tile_state"] == (128, 4)
    portable = next(i for i in portable_triton_implementation_model().implementations if i.id == GDN)
    assert portable.parameters["tile_state"] == (128, 4)


def test_amd_template_hoists_head_quantities_and_generic_is_unchanged():
    registry = TritonTemplateRegistry()
    amd = registry.render_kernel(KernelTemplateSpec("gdn_recurrent", "persistent", "amd", "gfx1201"), CONTEXT).source
    generic = registry.render_kernel(KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm120"),
                                     CONTEXT).source
    for helper in ("_flagmega_amd_gdn_gate_rows", "_flagmega_amd_gdn_head_qk_columns",
                   "_flagmega_amd_gdn_core_hoisted", "_flagmega_amd_gdn_norm_columns"):
        assert f"def {helper}(" in amd and helper not in generic
    assert "def _flagmega_gdn_recurrent_local_core_tile(" in amd  # the generic body remains available


def _amd_gpu():
    return torch.cuda.is_available() and bool(torch.version.hip) and \
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] in ("gfx1100", "gfx1201")


KERNELS = '''
@triton.jit
def gates_kernel(source, b_weight, a_weight, output, K: tl.constexpr, B: tl.constexpr, A: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b_value, a_value = _flagmega_amd_gdn_gate_rows(source, b_weight, a_weight, row, K, B, A)
    tl.store(output + row * 4, b_value)
    tl.store(output + row * 4 + 1, a_value)
    tl.store(output + row * 4 + 2, _flagmega_dense_rows(source, b_weight, row, row >= 0, K, 128, False, B))
    tl.store(output + row * 4 + 3, _flagmega_dense_rows(source, a_weight, row, row >= 0, K, 128, False, A))


@triton.jit
def core_kernel(qkv, state_generic, state_amd, core_generic, core_amd, beta, decay,
                V: tl.constexpr, KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, K: tl.constexpr,
                TILED: tl.constexpr):
    start = tl.program_id(0) * V
    head = start // K
    values = start + tl.arange(0, V)
    active = values < VALUE_HEADS * K
    _flagmega_gdn_recurrent_local_core_tile(
        qkv, beta, decay, state_generic, core_generic, values, values, active,
        VALUE_HEADS, KEY_HEADS, K, K, VALUE_HEADS // KEY_HEADS, K, False, 1e-12, False, False,
        TILED_VALUE_HEADS=TILED)
    columns = start + tl.arange(0, V)[:, None]
    mask = columns < VALUE_HEADS * K
    key, query = _flagmega_amd_gdn_head_qk_columns(
        qkv, tl.zeros_like(columns) + head * K, tl.zeros_like(mask) | True,
        KEY_HEADS, K, K, VALUE_HEADS // KEY_HEADS, K, False, 1e-12, False, TILED)
    _flagmega_amd_gdn_core_hoisted(qkv, beta, decay, state_amd, core_amd, columns, mask, key, query,
                                   KEY_HEADS, K, K, False)


@triton.jit
def norm_kernel(z, core, norm_weight, out_generic, out_amd, V: tl.constexpr, K: tl.constexpr):
    start = tl.program_id(0) * V
    values = start + tl.arange(0, V)
    _flagmega_gdn_recurrent_local_norm_tile(z, out_generic, norm_weight, core, values, values, values,
                                            values >= 0, K, 1e-6)
    columns = start + tl.arange(0, V)[:, None]
    _flagmega_amd_gdn_norm_columns(z, out_amd, norm_weight, core, columns, columns, columns, columns >= 0, K, 1e-6)
'''


def _module(tmp_path):
    source = "import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n"
    source += TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "amd", "gfx1201"), CONTEXT).source
    source += "\n@triton.jit\ndef _flagmega_silu(x):\n    return x * tl.sigmoid(x)\n" if "def _flagmega_silu(" not in source else ""
    source += KERNELS
    path = tmp_path / "amd_gdn_head_hoisted.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("encodings", (("IQ4_XS", "IQ4_XS"), ("Q4_K", "Q4_K")))
def test_amd_gate_rows_are_bitwise_the_single_row_dots(tmp_path, encodings):
    if not _amd_gpu():
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    rows, k = 6, 5120
    b_packed, _ = _packed_fixture(encodings[0], rows=rows, k=k)
    a_packed, _ = _packed_fixture(encodings[1], rows=rows, k=k)
    a_packed = a_packed[::-1].copy()  # distinct weights for the two gates
    source = torch.randn(k, generator=torch.Generator().manual_seed(5)).bfloat16().cuda()
    output = torch.full((rows, 4), float("nan"), device="cuda")
    _module(tmp_path).gates_kernel[(rows, )](source, torch.from_numpy(b_packed).cuda(),
                                             torch.from_numpy(a_packed).cuda(), output, k, *encodings, num_warps=4)
    torch.cuda.synchronize()
    output = output.cpu()
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 0], output[:, 2]) and torch.equal(output[:, 1], output[:, 3])


@pytest.mark.parametrize("tiled", (True, False))
@pytest.mark.parametrize("tile", (4, 16))
def test_amd_core_and_norm_columns_match_the_generic_tiles(tmp_path, tiled, tile):
    if not _amd_gpu():
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    key_heads, value_heads, k = 4, 12, 128
    generator = torch.Generator().manual_seed(11)
    qkv = torch.randn(2 * key_heads * k + value_heads * k, generator=generator).bfloat16().cuda()
    state = torch.randn(value_heads * k * k, generator=generator).cuda()
    state_generic, state_amd = state.clone(), state.clone()
    core_generic = torch.zeros(value_heads * k, device="cuda")
    core_amd = torch.zeros(value_heads * k, device="cuda")
    module = _module(tmp_path)
    module.core_kernel[(value_heads * k // tile, )](qkv, state_generic, state_amd, core_generic, core_amd, 0.37,
                                                    0.91, tile, key_heads, value_heads, k, tiled, num_warps=4)
    torch.cuda.synchronize()
    assert torch.isfinite(state_amd).all() and torch.isfinite(core_amd).all()
    # Same operations per row; only the reduction trees may differ.
    torch.testing.assert_close(state_amd, state_generic, rtol=0, atol=2e-6)
    torch.testing.assert_close(core_amd, core_generic, rtol=0, atol=2e-6)

    z = torch.randn(value_heads * k, generator=generator).bfloat16().cuda()
    norm_weight = torch.randn(k, generator=generator).bfloat16().cuda()
    out_generic = torch.empty(value_heads * k, dtype=torch.bfloat16, device="cuda")
    out_amd = torch.empty_like(out_generic)
    module.norm_kernel[(value_heads * k // tile, )](z, core_generic, norm_weight, out_generic, out_amd, tile, k,
                                                    num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(out_amd, out_generic)


@pytest.mark.parametrize("target", ("amd-gfx1100", "amd-gfx1201"))
@pytest.mark.parametrize("gate_mode,norm_mode", (("accurate", "fast"), ("fast", "accurate")))
def test_amd_gdn_call_follows_the_requested_math_modes(tmp_path, target, gate_mode, norm_mode):
    # The AMD body calls the shared gate helper and its own norm tile; both must
    # receive the call's math modes, as the generic body passes them.
    from python.test.flagmega.gdn.recurrent_helpers import recurrent_case, recurrent_module
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions

    from triton.flagmega import ir as fm

    # Native IQ4_XS gate rows over K = 256 select the AMD hoisted body.
    _, attrs, types, _ = recurrent_case(dimension=16, gate_math_mode=gate_mode, norm_math_mode=norm_mode,
                                        b_weight_encoding="IQ4_XS", a_weight_encoding="IQ4_XS")
    types.update(projection_input=fm.tensor_type("bfloat16", (3, 256)), b_weight=fm.tensor_type("uint8", (4, 136)),
                 a_weight=fm.tensor_type("uint8", (4, 136)))
    lowered = Compiler(CompileOptions(target=target)).compile(recurrent_module(types, attrs)).module
    source = (write_artifact(lowered, tmp_path / "artifact", target=target, emit_executable=True) /
              "generated_kernels.py").read_text()
    compile(source, "generated_kernels.py", "exec")
    assert "    _flagmega_amd_gdn_norm_columns(" in source  # the hoisted body is called
    assert f'GATE_MATH_MODE="{gate_mode}"' in source
    assert f"ACCURATE_SILU={norm_mode == 'accurate'}" in source
