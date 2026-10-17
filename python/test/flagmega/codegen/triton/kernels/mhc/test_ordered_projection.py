# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The ordered coefficient/RMS contract survives codegen and packing."""

import importlib.util
from pathlib import Path

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load

from python.test.flagmega.codegen.triton.kernels.mhc.test_mhc_runtime import _artifact, _run, _skip_without_sm90
from python.test.flagmega.ir.ops.nn.test_mhc_ordered_projection import module, scalar_reference


@pytest.mark.parametrize("hc,hidden,splits,warps", [(1, 128, 1, 4), (2, 768, 2, 4),
                                                 (3, 512, 2, 4), (4, 4096, 8, 8)])
def test_device_projection_matches_independent_scalar_fmaf(tmp_path, hc, hidden, splits, warps):
    _skip_without_sm90(torch)
    import triton.experimental.tle.language as tle
    import triton.flagmega.codegen.triton as codegen

    template = Path(codegen.__file__).parent / "kernels/mhc_pre/_stream_projection.py.jinja"
    source = """import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.language.extra.cuda.libdevice as libdevice
""" + template.read_text() + """
@triton.jit
def kernel(X, W, Y, S, HC: tl.constexpr, H: tl.constexpr, SPLITS: tl.constexpr,
           DL: tl.constexpr, SL: tl.constexpr, ML: tl.constexpr):
    result, square = _flagmega_mhc_stream_projection(
        X, W, HC, H, SPLITS, H, 1, 1, HC * (HC + 2), DL, SL, ML)
    rows = tl.arange(0, triton.next_power_of_2(HC * (HC + 2)))
    tl.store(Y + rows, result, rows < HC * (HC + 2))
    tl.store(S, square)
"""
    path = tmp_path / "ordered.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"mhc_ordered_{hc}_{warps}", path)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    generator = torch.Generator().manual_seed(179)
    streams = torch.randn((1, hc, hidden), generator=generator)
    # A column-major projection exercises the physical stride contract.
    weights = torch.randn((hc * hidden, hc * (hc + 2)), generator=generator).T
    expected_mix, expected_square = scalar_reference(streams, weights, splits)
    x, weight = streams.cuda(), weights.cuda()
    actual_mix = torch.empty_like(expected_mix, device="cuda")
    actual_square = torch.empty_like(expected_square, device="cuda")
    n_warps = min(warps // 4, 2)
    dot = tle.gpu.BlockEncoding([1, 1, 1], [1, 1, 32], [n_warps, warps // (4 * n_warps), 4], [2, 1, 0])
    compiled = implementation.kernel[(1,)](
        x, weight, actual_mix, actual_square, hc, hidden, splits,
        dot, tle.gpu.SlicedEncoding(0, dot), tle.gpu.BlockEncoding([1], [32], [warps], [0]), num_warps=warps)
    torch.testing.assert_close(actual_mix.cpu(), expected_mix, atol=0, rtol=0)
    torch.testing.assert_close(actual_square.cpu(), expected_square, atol=0, rtol=0)
    assert not torch.equal(expected_mix, streams.reshape(1, -1) @ weights.T)
    assert compiled.n_spills == 0


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_ordered_contract_reaches_normal_kernel_and_packed_results(tmp_path, lanes):
    _skip_without_sm90(torch)
    graph = module(tokens=2, output_lanes=lanes, collapse_reduction="sequential_fma")
    # Resume through emitted Python, as a standalone compilation does.
    graph = fm.load_module(fm.emit_module(graph, tmp_path / "input.py"))
    generator = torch.Generator().manual_seed(194)
    host = {"streams": torch.randn((2, 3, 512), generator=generator),
            "projection": torch.randn((15, 1536), generator=generator) * .02,
            "base": torch.randn((15,), generator=generator), "scale": torch.ones(3)}
    expected = TorchEvaluator(DictWeightResolver({})).run(graph, host)
    artifact = _artifact(tmp_path, "artifact", graph)
    source = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: mhc_pre/stream_fma" in source
    assert "# flagmega-kernel: mhc_pre/local" not in source
    assert "# flagmega-kernel: pack/" not in source
    runtime = load(artifact, device="cuda:0")
    try:
        names = runtime.ir_module.function_map[runtime.ir_module.entry].outputs
        outputs = {name: torch.empty_like(value, device="cuda") for name, value in zip(names, expected, strict=True)}
        _run(runtime, {name: value.cuda() for name, value in host.items()}, outputs, torch)
        for name, reference in zip(names, expected, strict=True):
            torch.testing.assert_close(outputs[name].cpu(), reference, atol=2e-6, rtol=2e-6)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


def test_ordered_projection_attributes_survive_packing_and_selection():
    compiled = Compiler().compile(module(output_lanes=(8,))).module
    mhc = [definition.dispatch for definition in compiled.kernel_definitions
           if definition.dispatch.semantic_op == "nn.mhc_pre"]
    assert len(mhc) == 1
    assert mhc[0].microkernel.variant == "stream_fma"
    assert mhc[0].semantic_attrs["projection_reduction"] == "stream_fma_128"
    assert mhc[0].semantic_attrs["projection_splits"] == 2
