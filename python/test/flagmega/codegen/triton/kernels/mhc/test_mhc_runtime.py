# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The mHC stages must reach a generated kernel and match the reference.

Before this kernel family existed, ``selected_tir_variants`` refused the whole
model with "No reviewed Triton TIR candidate was proposed for op 'nn.mhc_pre'";
the op definition and the AutoDistributed provider alone are not enough to make
an operation compilable.  These tests pin the emitted structure -- one pass over
the streams for all three coefficient groups, the Sinkhorn sweeps in registers
-- and check the arithmetic against the same vLLM transcription the op-level
suite uses, so a drift in either the kernel or the evaluator fails here.
"""

import pathlib

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.ir.ops.nn.test_mhc import (
    COEFFICIENTS,
    HC,
    HC_EPS,
    HIDDEN,
    POST_MULT,
    RMS_EPS,
    SINKHORN,
    TOKENS,
    _operands,
    _reference_post,
    _reference_pre,
)
from python.test.flagmega.devices import is_nvidia_sm90

PRE_INPUTS = ("streams", "projection", "base", "scale")
POST_INPUTS = ("value", "residual", "post", "comb")


def _pre_module(stream_dtype="float32", iterations=SINKHORN, *, output_lanes=(),
                pack_output=(), hidden=HIDDEN):
    """Build a pre-stage graph; ``stream_dtype`` sets the collapse precision.

    The default is FP32 on purpose.  With bf16 streams the collapsed output has
    an 8-bit mantissa, which rounds away a wrong-epsilon coefficient error --
    the same vacuity that let the op-level suite pass while ``pre_mix`` added
    the RMS epsilon instead of the hyper-connection one.
    """
    types = {"streams": fm.tensor_type(stream_dtype, (TOKENS, HC, hidden)),
             "projection": fm.tensor_type("float32", (COEFFICIENTS, HC * hidden)),
             "base": fm.tensor_type("float32", (COEFFICIENTS, )),
             "scale": fm.tensor_type("float32", (3, ))}

    class Graph(fm.Module):
        def forward(self):
            operands = tuple(self.input(name, types[name], id=name) for name in PRE_INPUTS)
            stage = fm.F.nn.mhc_pre(*operands, epsilon=RMS_EPS, sinkhorn_iterations=iterations,
                                    sinkhorn_epsilon=HC_EPS, post_multiplier=POST_MULT,
                                    output_lanes=output_lanes,
                                    name="pre")
            post, comb, collapsed = fm.F.tensors.get_items(stage, 0, 1, 2, name_prefix="pre")
            if pack_output:
                collapsed = fm.F.tensors.pack(collapsed, pack_output, axes=(-1,) * len(pack_output),
                                              name="packed_collapsed")
            self.function("main", operands, (post, comb, collapsed))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _post_module():
    types = {"value": fm.tensor_type("bfloat16", (TOKENS, HIDDEN)),
             "residual": fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN)),
             "post": fm.tensor_type("float32", (TOKENS, HC)),
             "comb": fm.tensor_type("float32", (TOKENS, HC, HC))}

    class Graph(fm.Module):
        def forward(self):
            operands = tuple(self.input(name, types[name], id=name) for name in POST_INPUTS)
            self.function("main", operands, (fm.F.nn.mhc_post(*operands, name="mixed"), ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _artifact(tmp_path: pathlib.Path, name: str, module):
    return write_artifact(Compiler().compile(module).module, tmp_path / name,
                          target="nvidia-sm90", emit_executable=True)


def _skip_without_sm90(torch):
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")


def _run(runtime, values, outputs, torch):
    arguments = tuple(outputs[spec["value"]] if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()


def test_the_pre_kernel_walks_the_streams_once_for_all_three_groups(tmp_path):
    source = (_artifact(tmp_path, "mhc-pre", _pre_module())
              / "generated_kernels.py").read_text("utf-8")

    assert "# flagmega-kernel: mhc_pre/local platform=generic" in source
    body = source.split("# flagmega-kernel: mhc_pre/local")[1].split("@triton.jit")[1]
    # One matvec loop feeding three group accumulators, not three loops.
    assert body.count("_fm_square_sum +=") == 1
    for group in ("_fm_pre_mix +=", "_fm_post_mix +=", "_fm_comb_mix +="):
        assert body.count(group) == 1, f"{group} must accumulate in the single stream pass"
    # The Sinkhorn sweeps stay in registers: no store inside the iteration.
    iteration = body.split("for _fm_iteration")[1].split("tl.store")[0]
    assert iteration.count("tl.sum(_fm_comb, 1)") == 1
    assert iteration.count("tl.sum(_fm_comb, 0)") == 1
    assert "tl.load" not in iteration
    # The pre coefficients take the hyper-connection epsilon, never the RMS one.
    assert f"+ {HC_EPS!r}" in body
    assert body.count("tl.store(") == 3


def test_the_post_kernel_hoists_the_mixing_matrix_out_of_the_hidden_loop(tmp_path):
    source = (_artifact(tmp_path, "mhc-post", _post_module())
              / "generated_kernels.py").read_text("utf-8")

    assert "# flagmega-kernel: mhc_post/local platform=generic" in source
    body = source.split("# flagmega-kernel: mhc_post/local")[1].split("@triton.jit")[1]
    tile_loop = body.split("for _fm_start")[1]
    assert "_fm_comb = tl.load(" not in tile_loop, "comb must be read once per token"
    assert "_fm_post = tl.load(" not in tile_loop, "post must be read once per token"
    assert tile_loop.count("tl.store(") == 1


def test_the_pre_kernel_matches_the_serving_runtime_reference(tmp_path):
    torch = pytest.importorskip("torch")
    _skip_without_sm90(torch)
    runtime = load(_artifact(tmp_path, "mhc-pre", _pre_module()), device="cuda:0")
    operands = _operands()
    streams = operands["streams"].float()
    values = {name: (streams if name == "streams" else operands[name]).cuda()
              for name in PRE_INPUTS}
    outputs = {"pre_0": torch.empty((TOKENS, HC), dtype=torch.float32, device="cuda"),
               "pre_1": torch.empty((TOKENS, HC, HC), dtype=torch.float32, device="cuda"),
               "pre_2": torch.empty((TOKENS, HIDDEN), dtype=torch.float32, device="cuda")}
    _run(runtime, values, outputs, torch)

    post, comb, collapsed = _reference_pre(streams, operands["projection"],
                                           operands["scale"], operands["base"])
    # The kernel keeps FP32 registers where the reference keeps FP32 tensors, so
    # only the reduction order differs. The collapse tolerance is tight enough to
    # catch a wrong epsilon in pre_mix, which perturbs it by ~4e-5.
    torch.testing.assert_close(outputs["pre_0"].cpu(), post, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs["pre_1"].cpu(), comb, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs["pre_2"].cpu(), collapsed, rtol=1e-6, atol=1e-6)


def test_the_post_kernel_matches_the_serving_runtime_reference(tmp_path):
    torch = pytest.importorskip("torch")
    _skip_without_sm90(torch)
    runtime = load(_artifact(tmp_path, "mhc-post", _post_module()), device="cuda:0")
    operands = _operands()
    post, comb, _ = _reference_pre(operands["streams"], operands["projection"],
                                   operands["scale"], operands["base"])
    host = {"value": operands["value"], "residual": operands["streams"],
            "post": post, "comb": comb}
    values = {name: host[name].cuda() for name in POST_INPUTS}
    outputs = {"mixed": torch.empty((TOKENS, HC, HIDDEN), dtype=torch.bfloat16, device="cuda")}
    _run(runtime, values, outputs, torch)

    expected = _reference_post(operands["value"], operands["streams"], post.unsqueeze(-1), comb)
    torch.testing.assert_close(outputs["mixed"].cpu(), expected, rtol=0, atol=8e-3)


def test_the_pre_kernel_sweeps_sinkhorn_in_the_reference_order(tmp_path):
    """Compare before the sweeps converge, where order and direction still show.

    At vLLM's 20 iterations the mixing matrix is doubly stochastic to FP32, so
    dropping a sweep or swapping the two directions changes nothing measurable.
    Two iterations leave the intermediate visible, which is what makes the
    kernel's sweep structure -- source-only first, then sink/source -- testable.
    """
    torch = pytest.importorskip("torch")
    _skip_without_sm90(torch)
    module = _pre_module(iterations=2)
    runtime = load(_artifact(tmp_path, "mhc-pre-2", module), device="cuda:0")
    operands = _operands()
    streams = operands["streams"].float()
    values = {name: (streams if name == "streams" else operands[name]).cuda()
              for name in PRE_INPUTS}
    outputs = {"pre_0": torch.empty((TOKENS, HC), dtype=torch.float32, device="cuda"),
               "pre_1": torch.empty((TOKENS, HC, HC), dtype=torch.float32, device="cuda"),
               "pre_2": torch.empty((TOKENS, HIDDEN), dtype=torch.float32, device="cuda")}
    _run(runtime, values, outputs, torch)

    _, comb, _ = _reference_pre(streams, operands["projection"], operands["scale"],
                                operands["base"], iterations=2)
    # Anti-vacuity: at two iterations the matrix must not already be converged,
    # or this test would pass for any sweep structure.
    assert (comb.sum(dim=-1) - 1.).abs().max() > 1e-4
    torch.testing.assert_close(outputs["pre_1"].cpu(), comb, rtol=1e-6, atol=1e-6)


def test_the_reference_comb_is_doubly_stochastic(tmp_path):
    """Guard against a vacuous pass: a constant matrix would satisfy any mix."""
    torch = pytest.importorskip("torch")
    operands = _operands()
    _, comb, _ = _reference_pre(operands["streams"], operands["projection"],
                                operands["scale"], operands["base"])
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones(TOKENS, HC), rtol=1e-4, atol=1e-4)
    assert comb.std() > 1e-3, "a near-constant mixing matrix would not test the sweeps"


@pytest.mark.parametrize("lanes", [(8,), (2, 4), (1,)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_packed_pre_kernel_writes_the_same_scalars_without_a_pack_kernel(tmp_path, lanes, dtype):
    from math import prod

    torch = pytest.importorskip("torch")
    _skip_without_sm90(torch)
    # Nine packed groups exercise the masked tail of a power-of-two tile.
    hidden = 72
    generator = torch.Generator().manual_seed(19)
    host = {"streams": torch.randn(TOKENS, HC, hidden, generator=generator).to(getattr(torch, dtype)),
            "projection": torch.randn(COEFFICIENTS, HC * hidden, generator=generator),
            "base": torch.randn(COEFFICIENTS, generator=generator),
            "scale": torch.randn(3, generator=generator)}
    values = {name: value.cuda() for name, value in host.items()}
    results = []
    for name, kwargs in (("scalar", {}), ("packed", {"pack_output": lanes}),
                         ("direct", {"output_lanes": lanes})):
        artifact = _artifact(tmp_path, name, _pre_module(dtype, hidden=hidden, **kwargs))
        source = (artifact / "generated_kernels.py").read_text()
        assert source.count("# flagmega-kernel: mhc_pre/local") == 1
        assert "# flagmega-kernel: pack/" not in source
        assert "# flagmega-kernel: tensor_transform/" not in source
        runtime = load(artifact, device="cuda:0")
        packed = name != "scalar"
        shape = (TOKENS, hidden // prod(lanes), *lanes) if packed else (TOKENS, hidden)
        result_buffers = (torch.empty((TOKENS, HC), dtype=torch.float32, device="cuda"),
                          torch.empty((TOKENS, HC, HC), dtype=torch.float32, device="cuda"),
                          torch.empty(shape, dtype=getattr(torch, dtype), device="cuda"))
        result_specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
        assert len(result_specs) == len(result_buffers)
        outputs = {spec["value"]: buffer for spec, buffer in zip(result_specs, result_buffers)}
        _run(runtime, values, outputs, torch)
        results.append((result_buffers[0].cpu(), result_buffers[1].cpu(),
                        result_buffers[2].cpu().reshape(TOKENS, hidden)))
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
    torch.testing.assert_close(results[2], results[0], rtol=0, atol=0)
    reference = _reference_pre(host["streams"], host["projection"], host["scale"], host["base"])
    for index in range(2):
        torch.testing.assert_close(results[2][index], reference[index], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(results[2][2], reference[2], rtol=1e-6, atol=1e-6 if dtype == "float32" else 8e-3)
