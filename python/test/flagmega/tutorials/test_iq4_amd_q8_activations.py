# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The Qwen3.8 IQ4 RDNA tutorial's q8_1 activations: with its templates
installed, AMD IQ4_XS GEMVs quantize the activation once per CTA to q8_1_rne
in LDS and dot it with v_dot4; Q4_K, NVIDIA and uninstalled sources keep the
default kernels."""

import importlib.util
import re
from math import ceil
from types import SimpleNamespace
from pathlib import Path

import pytest

from triton.flagmega.codegen.triton import tir_package

torch = pytest.importorskip("torch")

TUTORIAL = (Path(__file__).resolve().parents[3] / "tutorials" / "flagmega" / "03-qwen3.8-27b-iq4" / "amd-rdna")
IQ4_XS_ROW_BYTES_PER_256 = 136


def _gemv():
    from python.test.flagmega.codegen.triton.kernels.block_fp8 import test_amd_native_gguf_gemv
    return test_amd_native_gguf_gemv


def _agent_optimizations():
    spec = importlib.util.spec_from_file_location("rdna_iq4_agent_optimizations",
                                                  TUTORIAL / "agent_optimizations" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(tir_package, "TritonTemplateRegistry", tir_package.TritonTemplateRegistry)
    package = _agent_optimizations()
    package.install(q8_activations=True)
    return package


def _source(tmp_path, name, build, encoding, target, k=5120):
    gemv = _gemv()
    module = gemv._projection(encoding, 64, k) if build == "projection" else gemv._glu(encoding, 64, k)
    return (gemv._artifact(tmp_path, name, module, target) / "generated_kernels.py").read_text()


@pytest.mark.parametrize("build", ("projection", "glu"))
@pytest.mark.parametrize("target", ("amd-gfx1201", "amd-gfx1100"))
def test_installed_iq4_xs_gemvs_dot_q8_activations(tmp_path, installed, target, build):
    source = _source(tmp_path, "q8", build, "IQ4_XS", target)
    # K=5120: 160 groups split into a 128-group main buffer and a 32-group tail.
    assert "q8_main_groups: tl.constexpr = 128" in source
    assert "q8_words = tle.gpu.alloc([1024], tl.int32" in source
    assert "q8_tail_words = tle.gpu.alloc([256], tl.int32" in source
    assert re.search(r"tl\.float16, nv_mma_shared_layout=False\)\n    _flagmega_amd_q8_quantize\(.*5120,\s*128, 32, q8_main_groups\)", source)
    assert "_flagmega_amd_iq4_xs_q8_rows(" in source and "v_dot4_i32_iu8" in source
    assert "q8_words, q8_scales, q8_tail_words, q8_tail_scales, q8_main_groups" in source


def test_installed_down_projection_quantizes_in_two_chunks(tmp_path, installed):
    source = _source(tmp_path, "down", "projection", "IQ4_XS", "amd-gfx1201", k=17408)
    assert "q8_main_groups: tl.constexpr = 512" in source
    assert re.search(r"_flagmega_amd_q8_quantize\(.*17408,\s*256, 32, q8_main_groups\)", source)


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_installed_q4_k_keeps_the_float_activation_path(tmp_path, installed, build):
    source = _source(tmp_path, "q4k", build, "Q4_K", "amd-gfx1201")
    assert "q8_words = tle.gpu.alloc" not in source
    assert not re.search(r"^\s+_flagmega_amd_q8_quantize\(", source, re.MULTILINE)
    assert "_flagmega_amd_native_rows(" in source


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_uninstalled_amd_sources_have_no_q8(tmp_path, build):
    source = _source(tmp_path, "amd", build, "IQ4_XS", "amd-gfx1201")
    assert "q8_" not in source and "v_dot4" not in source
    assert "_flagmega_amd_native_rows(" in source


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_install_leaves_nvidia_sources_unchanged(tmp_path, monkeypatch, build):
    before = _source(tmp_path, "before", build, "IQ4_XS", "nvidia-sm120")
    monkeypatch.setattr(tir_package, "TritonTemplateRegistry", tir_package.TritonTemplateRegistry)
    _agent_optimizations().install(q8_activations=True)
    assert _source(tmp_path, "after", build, "IQ4_XS", "nvidia-sm120") == before



def _wide_source(tmp_path, name, build, target, rows=4096):
    # 4096 rows over the 128-block mesh: 32 rows, i.e. two 16-row tiles, per block.
    gemv = _gemv()
    module = gemv._projection("IQ4_XS", rows, 5120) if build == "projection" else gemv._glu("IQ4_XS", rows, 5120)
    return (gemv._artifact(tmp_path, name, module, target) / "generated_kernels.py").read_text()


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_installed_gfx1201_native_gemvs_start_at_a_block_rotated_tile(tmp_path, installed, monkeypatch, build):
    # The gate+up tile queue replaces the rotation; this checks the static fallback.
    monkeypatch.setenv("FLAGMEGA_AMD_TILE_QUEUE", "0")
    source = _wide_source(tmp_path, "rotated", build, "amd-gfx1201")
    assert "for local_n_tile in tl.range(0, 2):" in source
    assert "local_n_start = (((local_n_tile + shard_index) % 2) * 16).to(tl.int32)" in source
    assert "for local_n_start in tl.range(0, 32, 16):" not in source


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_uninstalled_native_gemvs_keep_the_ordered_tile_loop(tmp_path, build):
    source = _wide_source(tmp_path, "ordered", build, "amd-gfx1201")
    assert "for local_n_start in tl.range(0, 32, 16):" in source
    assert "local_n_tile" not in source


@pytest.mark.parametrize(("build", "rows"), (("projection", 8), ("glu", 16)))
def test_installed_gfx1100_native_gemvs_keep_the_ordered_tile_loop(tmp_path, installed, monkeypatch, build, rows):
    monkeypatch.setenv("FLAGMEGA_AMD_TILE_QUEUE", "0")
    source = _wide_source(tmp_path, "gfx1100", build, "amd-gfx1100")
    assert f"for local_n_start in tl.range(0, 32, {rows}):" in source
    assert "local_n_tile" not in source


def _artifact_codegen(tmp_path, name, build, target, rows=4096):
    import json
    gemv = _gemv()
    module = gemv._projection("IQ4_XS", rows, 5120) if build == "projection" else gemv._glu("IQ4_XS", rows, 5120)
    artifact = gemv._artifact(tmp_path, name, module, target)
    return (artifact / "generated_kernels.py").read_text(), json.loads((artifact / "artifact.json").read_text())


def _codegen_keys(value):
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _codegen_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _codegen_keys(item)}
    return set()


@pytest.mark.parametrize("target", ("amd-gfx1201", "amd-gfx1100", "nvidia-sm120"))
def test_default_registry_requests_no_counter_slots(tmp_path, target):
    source, artifact = _artifact_codegen(tmp_path, "plain", "glu", target)
    assert "_flagmega_counters" not in source and "atomic_add" not in source
    assert not {"counter_slots", "counter_slot_argument"} & _codegen_keys(artifact)


def test_installed_gfx1201_gate_up_takes_tiles_from_the_counter_queue(tmp_path, installed):
    source, artifact = _artifact_codegen(tmp_path, "queue", "glu", "amd-gfx1201")
    # 4096 rows = 512 8-row tiles over 128 blocks: 4 even shares + 7 spare claims.
    assert "_flagmega_counters" in artifact_signature(artifact)
    assert "queue_counter = _flagmega_counters + 0" in source
    assert "for queue_step in tl.range(0, 11):" in source
    # 128 * 10 claims per invocation; the last claimer returns the counter to zero.
    assert "queue_claim = tl.atomic_add(queue_counter, 1, sem=\"relaxed\")" in source
    assert "if queue_claim == 1279:\n                tl.atomic_xchg(queue_counter, 0, sem=\"relaxed\")" in source
    assert "queue_next = 128 + queue_claim" in source
    assert "weight_rows = (0 + queue_rows * 2720)" in source
    assert "destination_offsets = (0 + queue_rows * 1)" in source
    assert "n_mask, K, Q8_MAIN_GROUPS, 8)" in source and "local_n_offsets" not in source.split("queue_counter =")[1].split("@triton.jit")[0]
    # The counter is threaded into the GLU device function from its caller.
    assert re.search(r"_glu\([^)]*_flagmega_counters\)", source)


def artifact_signature(artifact):
    return [name for value in _string_lists(artifact) for name in value if name == "_flagmega_counters"]


def _string_lists(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _string_lists(item)
    elif isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            yield value
        for item in value:
            yield from _string_lists(item)


def test_installed_gfx1100_gate_up_takes_tiles_from_the_counter_queue(tmp_path, installed):
    source, _ = _artifact_codegen(tmp_path, "queue1100", "glu", "amd-gfx1100")
    assert "queue_counter = _flagmega_counters + 0" in source and "queue_next = 128 + queue_claim" in source


@pytest.mark.parametrize("target", ("nvidia-sm120",))
def test_installed_tile_queue_stays_off_other_targets(tmp_path, installed, target):
    source, artifact = _artifact_codegen(tmp_path, "other", "glu", target)
    assert "_flagmega_counters" not in source
    assert not {"counter_slots", "counter_slot_argument"} & _codegen_keys(artifact)


def test_disabled_tile_queue_restores_the_rotated_loop(tmp_path, installed, monkeypatch):
    monkeypatch.setenv("FLAGMEGA_AMD_TILE_QUEUE", "0")
    source, artifact = _artifact_codegen(tmp_path, "off", "glu", "amd-gfx1201")
    assert "_flagmega_counters" not in source
    assert "local_n_start = (((local_n_tile + shard_index) % 2) * 16).to(tl.int32)" in source


def test_installed_queue_skips_projections(tmp_path, installed):
    # No GLU call uses the counter, so none is allocated or threaded.
    projection, artifact = _artifact_codegen(tmp_path, "projection", "projection", "amd-gfx1201")
    assert "_flagmega_counters" not in projection and "for local_n_tile in tl.range(0, 2):" in projection
    assert not {"counter_slots", "counter_slot_argument"} & _codegen_keys(artifact)


def test_installed_queue_covers_cyclic_owners(tmp_path, installed):
    # 4100 rows over 128 blocks render owner rows as two 32-row cycles; the
    # queue still covers each global row once.
    source, _ = _artifact_codegen(tmp_path, "cyclic", "glu", "amd-gfx1201", rows=4100)
    assert "queue_next = tl.full((), 513, tl.int64)" in source
    assert "n_mask = queue_rows < 4100" in source


def _glu_call(active_n, rows, capacity=136):
    return {"active_n": active_n, "local_n_capacity": capacity,
            "weight_row_offset": f"({rows}) * (5120 // 256) * 136",
            "destination_offset": f"((0)) * 17408 + ({rows}) * 1"}


CONTIGUOUS = "((local_n_offsets) + ((shard_x) * 136) + ((shard_y) * 2176))"
# gfx1100 8x24: 64-row cycles of 12288 rows, the second cycle 5120 rows long.
CYCLIC = ("(((shard_x) * 64) + ((shard_y) * 1536) + ((((local_n_offsets)) // (64)) * 12288)"
          " + (((local_n_offsets)) % (64)))")
CYCLIC_ACTIVE = "(tl.minimum(64, tl.maximum(0, (((shard_y) * -1536) + ((shard_x) * -64) + 5120))) + 64)"


@pytest.mark.parametrize(("call", "mesh"), (
    (_glu_call("136", CONTIGUOUS), (8, 16)),
    (_glu_call(CYCLIC_ACTIVE, CYCLIC, capacity=128), (8, 24)),
))
def test_queue_rows_cover_the_output_once(installed, call, mesh):
    assert installed._row_progression(call, *mesh) == (17408, "(0 + queue_rows * 1)", "(0 + queue_rows * 2720)")


@pytest.mark.parametrize("call", (
    # Every block computes the same rows.
    _glu_call("136", "(local_n_offsets)"),
    # One row left out after every block: not one progression.
    _glu_call("136", "((local_n_offsets) + ((shard_x) * 137) + ((shard_y) * 2192))"),
    # Weight rows not affine in the output row.
    {**_glu_call("136", CONTIGUOUS), "weight_row_offset": f"({CONTIGUOUS}) % 1000"},
    # A symbol the planner cannot evaluate.
    _glu_call("active_rows", CONTIGUOUS),
))
def test_queue_declines_other_row_layouts(installed, call):
    assert installed._row_progression(call, 8, 16) is None


def _amd_gpu():
    return torch.cuda.is_available() and bool(torch.version.hip) and \
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] in ("gfx1100", "gfx1201")


def _q8_kernel(tmp_path, package, k):
    """The installed AMD block_fp8 helpers and the real activation prologue for K, writing fp32 rows."""
    registry = package.RdnaIQ4Registry()
    prologue = registry.render("kernels/_amd_q8_prologue.py.jinja", {"call": SimpleNamespace(k=k, value="source")})
    source = "import triton\nimport triton.language as tl\nimport triton.experimental.tle.language as tle\n"
    source += registry.render("kernels/block_fp8/platforms/amd/simt.py.jinja", {"tn": 16, "mesh_size": 1})
    source += f'''
@triton.jit
def q8_rows_kernel(source, weight, output, N: tl.constexpr, K: tl.constexpr, ROW_BYTES: tl.constexpr):
{prologue}
    for row0 in tl.range(0, N, 16):
        row = row0 + tl.arange(0, 16)
        mask = row < N
        _flagmega_iq4_xs_linear_local_tile(source, weight, output, row.to(tl.int64) * ROW_BYTES, row, mask, K,
                                           q8_words, q8_scales, q8_tail_words, q8_tail_scales, q8_main_groups)
'''
    path = tmp_path / f"q8_rows_{k}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"q8_rows_{k}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _q8_1_reference(value, dequantized, round_half_even=True):
    """fp64 dot of q8_1_rne activations: per 32 values int8 = rne(fl32(x * fl32(127 / amax))), scale =
    fp16(amax / 127)."""
    x = value.float().reshape(-1, 32)
    maximum = x.abs().amax(1)
    # Tensor / tensor: torch evaluates a scalar numerator or divisor through a reciprocal, which is
    # not the correctly rounded quotient.
    limit = torch.full_like(maximum, 127.0)
    inverse = torch.where(maximum > 0, limit / maximum, torch.zeros_like(maximum))
    scaled = x * inverse[:, None]
    q = torch.round(scaled) if round_half_even else torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)
    scale = (maximum / limit).half().double()
    weight = torch.from_numpy(dequantized).double().reshape(dequantized.shape[0], -1, 32)
    return ((weight * q.double()[None]).sum(2) * scale[None]).sum(1)


def _run_q8_rows(tmp_path, package, value, packed, n, k):
    module = _q8_kernel(tmp_path, package, k)
    output = torch.full((n + 8, ), float("nan"), dtype=torch.float32, device="cuda")
    module.q8_rows_kernel[(1, )](value.cuda(), torch.from_numpy(packed).cuda(), output, n, k,
                                 k // 256 * IQ4_XS_ROW_BYTES_PER_256, num_warps=4)
    torch.cuda.synchronize()
    return output.double().cpu()


# K=5120 is the model's hidden size (128 main + 32 tail groups), K=17408 the
# down projection (two 256-group chunks + 32 tail groups), K=2048 a power of
# two without tail and K=768 a 16 + 8 split whose tail step masks 24 lanes.
# 40 rows end in a 16-row tile whose second half is empty, 44 in a partial one.
@pytest.mark.parametrize("k", (5120, 17408, 2048, 768))
@pytest.mark.parametrize("n", (40, 44))
def test_q8_gemv_matches_the_q8_1_rne_reference(tmp_path, installed, k, n):
    if not _amd_gpu():
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    packed, dequantized = _packed_fixture("IQ4_XS", rows=n, k=k)
    value = torch.randn(k, generator=torch.Generator().manual_seed(11)).bfloat16()
    expected = _q8_1_reference(value, dequantized)
    actual = _run_q8_rows(tmp_path, installed, value, packed, n, k)
    torch.testing.assert_close(actual[:n], expected, rtol=1e-5, atol=1e-6 * float(expected.abs().max()))
    assert torch.isnan(actual[n:]).all(), "rows past the tile end must not be stored"
    exact = value.double() @ torch.from_numpy(dequantized).double().T
    assert float((actual[:n] - exact).abs().max()) < 0.02 * float(exact.abs().max()), "q8 stays near the exact dot"


def test_q8_activations_round_half_to_even(tmp_path, installed):
    if not _amd_gpu():
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    n, k = 40, 5120
    # amax = 127 makes the scale exactly 1, so every other value j + 0.5 is a tie.
    generator = torch.Generator().manual_seed(13)
    ties = torch.randint(0, 127, (k // 32, 31), generator=generator).float() + 0.5
    signs = torch.where(torch.rand(k // 32, 31, generator=generator) < 0.5, -1.0, 1.0)
    value = torch.cat((torch.full((k // 32, 1), 127.0), ties * signs), dim=1).flatten().bfloat16()
    assert torch.equal(value.float().reshape(-1, 32)[:, 1:].abs() % 1, torch.full((k // 32, 31), 0.5))
    packed, dequantized = _packed_fixture("IQ4_XS", rows=n, k=k)
    even, away = _q8_1_reference(value, dequantized), _q8_1_reference(value, dequantized, round_half_even=False)
    tolerance = 1e-6 * float(even.abs().max())
    assert not torch.allclose(away, even, rtol=1e-5, atol=tolerance), "the fixture must separate the rounding modes"
    actual = _run_q8_rows(tmp_path, installed, value, packed, n, k)
    torch.testing.assert_close(actual[:n], even, rtol=1e-5, atol=tolerance)


def test_q8_quantize_ties_at_half_the_maximum(tmp_path, installed):
    # x = amax / 2 puts x * fl32(127 / amax) within an ulp of 63.5: the byte flips unless the
    # quotient is correctly rounded and the product rounds before the round-to-nearest-even.
    if not _amd_gpu():
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    n, k = 40, 17408
    groups = torch.randn(k // 32, 32, generator=torch.Generator().manual_seed(17)).bfloat16().float()
    groups[:, 1] = groups.abs().amax(1) / 2
    value = groups.flatten().bfloat16()
    packed, dequantized = _packed_fixture("IQ4_XS", rows=n, k=k)
    expected = _q8_1_reference(value, dequantized)
    actual = _run_q8_rows(tmp_path, installed, value, packed, n, k)
    torch.testing.assert_close(actual[:n], expected, rtol=1e-5, atol=1e-6 * float(expected.abs().max()))


def test_prologue_split_is_a_power_of_two_main_buffer_and_tail(installed):
    registry = installed.RdnaIQ4Registry()
    for k, main, tail, chunk in ((5120, 128, 32, 128), (6144, 128, 64, 128), (17408, 512, 32, 256),
                                 (2048, 64, 1, 64), (768, 16, 8, 16)):
        prologue = registry.render("kernels/_amd_q8_prologue.py.jinja", {"call": SimpleNamespace(k=k, value="x")})
        assert f"q8_main_groups: tl.constexpr = {main}\n" in prologue
        assert f"q8_tail_scales = tle.gpu.alloc([{tail}], tl.float16" in prologue
        assert re.search(rf"{k},\s*{chunk}, {tail}, q8_main_groups\)", prologue), prologue
        assert ceil(k / 32) - main <= tail


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_gfx1201_prefetches_first_weight_rows_before_quantizing(tmp_path, installed, build):
    source = _source(tmp_path, "pf", build, "IQ4_XS", "amd-gfx1201")
    calls = re.findall(r"^    _flagmega_amd_q8_prefetch\(", source, re.MULTILINE)
    assert len(calls) == (2 if build == "glu" else 1)
    assert "s_prefetch_data" in source
    # The prefetch is issued before the activation is quantized.
    assert source.index("    _flagmega_amd_q8_prefetch(") < source.index("    _flagmega_amd_q8_quantize(")


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_gfx1100_and_disabled_prefetch_render_no_prefetch(tmp_path, installed, monkeypatch, build):
    assert "_flagmega_amd_q8_prefetch" not in _source(tmp_path, "nopf", build, "IQ4_XS", "amd-gfx1100")
    monkeypatch.setenv("FLAGMEGA_AMD_Q8_PREFETCH_KB", "0")
    assert "_flagmega_amd_q8_prefetch" not in _source(tmp_path, "off", build, "IQ4_XS", "amd-gfx1201")


@pytest.mark.parametrize("build", ("projection", "glu"))
def test_gfx1100_pins_the_q8_source_pointer_to_sgprs(tmp_path, installed, build):
    def quantize(source):
        start = source.index("def _flagmega_amd_q8_quantize(")
        return source[start:source.index("\n@triton.jit", start)]

    assert "source = _flagmega_amd_uniform_parameter(source)" in quantize(
        _source(tmp_path, "pin", build, "IQ4_XS", "amd-gfx1100"))
    assert "uniform_parameter" not in quantize(_source(tmp_path, "r9700", build, "IQ4_XS", "amd-gfx1201"))


def test_gfx1100_projections_dot_8_row_tiles(tmp_path, installed):
    source = _wide_source(tmp_path, "rows8", "projection", "amd-gfx1100")
    assert "for local_n_start in tl.range(0, 32, 8):" in source
    assert "local_n_offsets = local_n_start + tl.arange(0, 8)" in source
    assert "n_mask, K, Q8_MAIN_GROUPS, 8)" in source
    r9700 = _wide_source(tmp_path, "r9700", "projection", "amd-gfx1201")
    assert "local_n_start + tl.arange(0, 16)" in r9700 and "Q8_MAIN_GROUPS, 16)" in r9700
