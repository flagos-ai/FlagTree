# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""AMD RDNA native GGUF GEMVs use the layout-aware lookup templates; the
generic (and NVIDIA) native GGUF kernels are unchanged."""

from math import ceil, prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions

torch = pytest.importorskip("torch")

BLOCK_BYTES = {"IQ4_XS": 136, "Q4_K": 144}
GENERIC_CODEBOOK = "tl.where(code == 0, -127.0"


def _packed(builder, name, encoding, n, k):
    return builder.input(name, fm.tensor_type("uint8", (n, k // 256 * BLOCK_BYTES[encoding])), id=name)


def _scale(builder, name, n, k):
    return builder.input(name, fm.tensor_type("float32", (ceil(n / 128), ceil(k / 128))), id=name)


def _projection(encoding, n, k, lanes=()):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, k)), id="value")
            weight, scale = _packed(self, "weight", encoding, n, k), _scale(self, "weight_scale", n, k)
            output = fm.F.math.block_scaled_matmul(value, weight, scale, weight_block_n=128, weight_block_k=128,
                                                   weight_encoding=encoding, output_lanes=lanes, name="projection")
            self.function("main", (value, weight, scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _glu(encoding, n, k):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, k)), id="value")
            gate, up = _packed(self, "gate", encoding, n, k), _packed(self, "up", encoding, n, k)
            gate_scale, up_scale = _scale(self, "gate_scale", n, k), _scale(self, "up_scale", n, k)
            output = fm.F.nn.matmul_glu(value, gate, up, gate_scale, up_scale, activation="silu",
                                        weight_block_n=128, weight_block_k=128, gate_weight_encoding=encoding,
                                        up_weight_encoding=encoding, name="glu")
            self.function("main", (value, gate, up, gate_scale, up_scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _artifact(tmp_path, name, module, target):
    lowered = Compiler(CompileOptions(target=target)).compile(module).module
    return write_artifact(lowered, tmp_path / name, target=target, emit_executable=True)


@pytest.mark.parametrize("kernel", ("block_fp8", "matmul_glu"))
@pytest.mark.parametrize("architecture", ("gfx1100", "gfx1201"))
def test_amd_targets_resolve_the_amd_native_gguf_template(kernel, architecture):
    registry = TritonTemplateRegistry()
    assert registry.resolve(KernelTemplateSpec(kernel, "simt", "amd", architecture)) == \
        f"kernels/{kernel}/platforms/amd/simt.py.jinja"
    assert registry.resolve(KernelTemplateSpec(kernel, "simt", "nvidia", "sm120")) == f"kernels/{kernel}/simt.py.jinja"


@pytest.mark.parametrize("encoding", ("IQ4_XS", "Q4_K"))
@pytest.mark.parametrize("build", ("projection", "glu"))
@pytest.mark.parametrize("target", ("amd-gfx1201", "amd-gfx1100"))
def test_amd_native_gguf_source_uses_the_lookup_kernel(tmp_path, target, build, encoding):
    module = _projection(encoding, 64, 512) if build == "projection" else _glu(encoding, 64, 512)
    source = (_artifact(tmp_path, "amd", module, target) / "generated_kernels.py").read_text()
    family = "block_fp8" if build == "projection" else "matmul_glu"
    assert f"# flagmega-kernel: {family}/simt platform=amd" in source
    assert "_flagmega_amd_native_rows(" in source and "v_perm_b32" in source
    assert GENERIC_CODEBOOK not in source


@pytest.mark.parametrize("build", ("projection", "glu"))
@pytest.mark.parametrize("target", ("nvidia-sm120", "nvidia-sm90"))
def test_nvidia_native_gguf_source_keeps_the_generic_kernel(tmp_path, target, build):
    module = _projection("IQ4_XS", 64, 512) if build == "projection" else _glu("IQ4_XS", 64, 512)
    source = (_artifact(tmp_path, "nvidia", module, target) / "generated_kernels.py").read_text()
    family = "block_fp8" if build == "projection" else "matmul_glu"
    assert f"# flagmega-kernel: {family}/simt platform=generic" in source
    assert GENERIC_CODEBOOK in source
    assert "_flagmega_amd_" not in source and "v_perm_b32" not in source


def _amd_target():
    if not torch.cuda.is_available() or not torch.version.hip:
        return None
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return {"gfx1100": "amd-gfx1100", "gfx1201": "amd-gfx1201"}.get(arch)


def _run(tmp_path, name, module, target, values, shape, dtype):
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    runtime = load(_artifact(tmp_path, name, module, target), device="cuda:0")
    output = torch.full(shape, float("nan"), dtype=dtype, device="cuda")
    arguments = tuple(output if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    return output.float().cpu()


def _assert_bf16_close(actual, expected):
    """Within one bf16 rounding of an fp64 reference (plus fp32 summation slack)."""
    expected = expected.reshape(actual.shape)
    assert torch.isfinite(actual).all(), "every output scalar must be written"
    tolerance = expected.abs() * 2.0**-7 + expected.abs().max() * 1e-4
    assert ((actual.double() - expected).abs() <= tolerance).all(), \
        float(((actual.double() - expected).abs() - tolerance).max())


# 2560 rows over the 8x16 mesh give each owner 20 rows: one full 16-row tile
# and a ragged 4-row tile. K=2048 runs the unmasked group loop, K=768 the
# masked tail (24 IQ4_XS groups / 12 Q4_K pairs per row).
@pytest.mark.parametrize("encoding", ("IQ4_XS", "Q4_K"))
@pytest.mark.parametrize("k", (2048, 768))
@pytest.mark.parametrize("lanes", ((), (4, )))
def test_amd_native_gguf_projection_matches_fp64(tmp_path, encoding, k, lanes):
    target = _amd_target()
    if target is None:
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    n = 2560
    packed, dequantized = _packed_fixture(encoding, rows=n, k=k)
    value = torch.randn(1, k, generator=torch.Generator().manual_seed(7)).bfloat16()
    values = {"value": value.cuda(), "weight": torch.from_numpy(packed).cuda(),
              "weight_scale": torch.ones(ceil(n / 128), ceil(k / 128), device="cuda")}
    expected = value.double() @ torch.from_numpy(dequantized).double().T
    shape = (1, n // prod(lanes), *lanes) if lanes else (1, n)
    actual = _run(tmp_path, "projection", _projection(encoding, n, k, lanes), target, values, shape, torch.bfloat16)
    _assert_bf16_close(actual, expected)


@pytest.fixture
def amd_helpers(tmp_path):
    """The AMD block_fp8 and matmul_glu helpers rendered for 16-row tiles."""
    import importlib.util

    registry = TritonTemplateRegistry()
    source = "import triton\nimport triton.language as tl\n"
    for kernel in ("block_fp8", "matmul_glu"):
        source += registry.render(f"kernels/{kernel}/platforms/amd/simt.py.jinja", {"tn": 16, "mesh_size": 1})
    source += '''
@triton.jit
def rows_kernel(source, weight, output, N: tl.constexpr, K: tl.constexpr, ROW_BYTES: tl.constexpr,
                ENCODING: tl.constexpr, FAMILY: tl.constexpr):
    for row0 in tl.range(0, N, 16):
        row = row0 + tl.arange(0, 16)
        mask = row < N
        offsets = row.to(tl.int64) * ROW_BYTES
        if FAMILY == "linear":
            if ENCODING == "IQ4_XS":
                _flagmega_iq4_xs_linear_local_tile(source, weight, output, offsets, row, mask, K)
            else:
                _flagmega_q4_k_linear_local_tile(source, weight, output, offsets, row, mask, K)
        else:
            if ENCODING == "IQ4_XS":
                result = _flagmega_iq4_xs_dot(source, weight, offsets, mask, K)
            else:
                result = _flagmega_q4_k_dot(source, weight, offsets, mask, K)
            tl.store(output + row, result, mask=mask)
'''
    path = tmp_path / "amd_native_gguf.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("amd_native_gguf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 40 rows end in a tile whose second 8-row half is empty (skipped), 44 rows in
# one whose second half is partly active; K=2048 runs the unmasked group loop,
# K=768 the masked tail.
@pytest.mark.parametrize("encoding", ("IQ4_XS", "Q4_K"))
@pytest.mark.parametrize("k", (2048, 768))
@pytest.mark.parametrize("family", ("linear", "glu"))
@pytest.mark.parametrize("n", (40, 44))
def test_amd_native_gguf_helpers_match_fp64(amd_helpers, encoding, k, family, n):
    if _amd_target() is None:
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    packed, dequantized = _packed_fixture(encoding, rows=n, k=k)
    value = torch.randn(1, k, generator=torch.Generator().manual_seed(5)).bfloat16()
    output = torch.full((n + 8, ), float("nan"), dtype=torch.float32, device="cuda")
    amd_helpers.rows_kernel[(1, )](value.cuda(), torch.from_numpy(packed).cuda(), output, n, k,
                                   k // 256 * BLOCK_BYTES[encoding], encoding, family, num_warps=4)
    expected = (value.double() @ torch.from_numpy(dequantized).double().T)[0]
    actual = output.double().cpu()
    torch.testing.assert_close(actual[:n], expected, rtol=1e-5, atol=1e-6 * float(expected.abs().max()))
    assert torch.isnan(actual[n:]).all(), "rows past the tile end must not be stored"
