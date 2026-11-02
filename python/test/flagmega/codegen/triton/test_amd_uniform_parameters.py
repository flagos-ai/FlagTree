# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1100 packages pin noinline device-function pointer parameters to SGPRs;
other targets render unchanged."""

import textwrap
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.amd_uniform_parameters import (
    HELPER,
    add_uniform_parameter_prologue,
    amd_uniform_parameters,
    skipped_symbols,
)
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions

TEMPLATE = (Path(__file__).parents[4] / "triton" / "flagmega" / "codegen" / "triton" / "entrypoints" /
            "_amd_uniform_parameters.py.jinja")


def test_only_gfx1100_pins_parameters():
    assert amd_uniform_parameters("amd", "gfx1100")
    assert not amd_uniform_parameters("amd", "gfx1201")
    assert not amd_uniform_parameters("nvidia", "sm90")


def test_prologue_covers_noinline_runtime_parameters_only():
    source = textwrap.dedent('''\
        @triton.jit(noinline=True)

        def device(source, destination, n, BLOCK: tl.constexpr, scale=2):
            """Doc."""
            tl.store(destination, tl.load(source))


        @triton.jit
        def inline_helper(pointer):
            return tl.load(pointer)


        @triton.jit(noinline=True)
        def no_parameters():
            pass
        ''')
    patched = add_uniform_parameter_prologue(source)
    assert patched.split("\n")[3:8] == [
        '    """Doc."""',
        f"    source = {HELPER}(source)",
        f"    destination = {HELPER}(destination)",
        f"    n = {HELPER}(n)",
        "    tl.store(destination, tl.load(source))",
    ]
    assert f"{HELPER}(BLOCK)" not in patched and f"{HELPER}(scale)" not in patched
    assert f"{HELPER}(pointer)" not in patched
    assert patched.count(HELPER) == 3


def test_prologue_skips_the_gemv_calls():
    source = textwrap.dedent('''\
        @triton.jit(noinline=True)
        def _call_0_projection(source, weight):
            pass


        @triton.jit(noinline=True)
        def _call_0_projection__finalize(source):
            pass


        @triton.jit(noinline=True)
        def _call_1_norm(source):
            pass
        ''')
    skip = skipped_symbols(({"family": "block_fp8", "symbol": "_call_0_projection"},
                            {"family": "norm_apply", "symbol": "_call_1_norm"}))
    patched = add_uniform_parameter_prologue(source, skip)
    assert patched.count(f" = {HELPER}(") == 1 and f"    source = {HELPER}(source)\n" in patched.split("_call_1_norm")[1]


def _projection_module():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 512)), id="value")
            weight = self.input("weight", fm.tensor_type("uint8", (64, 2 * 136)), id="weight")
            scale = self.input("weight_scale", fm.tensor_type("float32", (1, 4)), id="weight_scale")
            output = fm.F.math.block_scaled_matmul(value, weight, scale, weight_block_n=128, weight_block_k=128,
                                                   weight_encoding="IQ4_XS", name="projection")
            # A non-GEMV call after the projection: its pointer parameters must be pinned.
            output = fm.F.math.silu(output, name="activated")
            self.function("main", (value, weight, scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _generated(tmp_path, target):
    lowered = Compiler(CompileOptions(target=target)).compile(_projection_module()).module
    artifact = write_artifact(lowered, tmp_path / target, target=target, emit_executable=True)
    return (artifact / "generated_kernels.py").read_text()


def _body(source, symbol_part):
    start = source.index("def ", source.index(symbol_part) - 60)
    return source[start:source.index("\n@triton.jit", start)]


def test_gfx1100_package_pins_device_function_pointers(tmp_path):
    source = _generated(tmp_path, "amd-gfx1100")
    assert f"def {HELPER}(value):" in source
    # Without the alignment fact the pinned pointer would change the chosen layouts.
    assert "tl.multiple_of(pinned, 16)" in source
    assert f" = {HELPER}(" in _body(source, "_call_1_activated")
    assert f" = {HELPER}(" not in _body(source, "_call_0_projection")


@pytest.mark.parametrize("target", ("amd-gfx1201", "nvidia-sm90"))
def test_other_targets_render_without_the_prologue(tmp_path, target):
    assert HELPER not in _generated(tmp_path, target)


KERNELS = '''
@triton.jit(noinline=True)
def device(source, destination, n, scale):
    offsets = tl.arange(0, 128)
    mask = offsets < n
    tl.store(destination + offsets, tl.load(source + offsets, mask=mask) * scale, mask=mask)


@triton.jit
def kernel(source, destination, n):
    device(source, destination, n, 1)
'''


def test_helper_keeps_values_on_an_amd_gpu(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires an AMD GPU")
    import importlib.util

    # Triton reads jit sources from files: render the helper plus a pinned
    # device function (pointer, int and constexpr parameters) into a module.
    path = tmp_path / "amd_uniform_parameters_kernels.py"
    kernels = add_uniform_parameter_prologue(KERNELS)
    assert kernels.count(f" = {HELPER}(") == 4
    path.write_text("import triton\nimport triton.language as tl\n\n" + TEMPLATE.read_text() + kernels)
    spec = importlib.util.spec_from_file_location("amd_uniform_parameters_kernels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = torch.arange(128, device="cuda", dtype=torch.float32)
    destination = torch.full_like(source, -1.0)
    module.kernel[(1, )](source, destination, 100)
    torch.cuda.synchronize()
    assert torch.equal(destination[:100], source[:100])
    assert torch.equal(destination[100:], torch.full((28, ), -1.0, device="cuda"))
