# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module

CASES = [
    ("pad", {"pad_end": (0, 3), "pad_value": -2.5}),
    ("slice_to_shape", {"shape": (6, 4)}),
    ("slice", {"starts": (1, ), "ends": (7, ), "axes": (1, ), "steps": (2, )}),
    ("slice", {"starts": (None, ), "ends": (None, ), "axes": (1, ), "steps": (-1, )}),
]


def graph(op, attrs, *, vector=False, split=False):
    dtype = fm.vector_type("bfloat16", (2, 4)) if vector else "float32"
    value_type = fm.tensor_type(dtype, (6, 7))
    if split:
        value_type = fm.DistributedType(value_type, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()),
                                        fm.Placement((2, 2), "xy", "bb"))
    metadata = {"auto_distribution": {"placement": value_type.placement.to_data()}} if split else {}
    return replace(primitive_module(fm.get_definition(f"tensors.{op}"), (value_type, ), **attrs),
                   stage="frozen_constants", metadata=metadata)


@pytest.mark.parametrize("op,attrs", CASES)
def test_tensor_transform_selects_real_generic_template(tmp_path, op, attrs):
    compiled = Compiler().compile(graph(op, attrs)).module
    render_triton_package(compiled, tmp_path)
    source = (tmp_path / "generated_kernels.py").read_text()
    family = "pad" if op == "pad" else "slice"
    assert f"# flagmega-kernel: {family}/local platform=generic" in source


@pytest.mark.parametrize("op,attrs", CASES)
@pytest.mark.parametrize("vector,split", [(False, False), (True, False), (True, True)])
def test_tensor_transform_executes_exact_values_with_tail_owners_and_lanes(tmp_path, op, attrs, vector, split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = graph(op, attrs, vector=vector, split=split)
    values = torch.arange(6 * 7 * (8 if vector else 1), device="cuda").reshape((6, 7, 2, 4) if vector else (6, 7))
    values = values.to(torch.bfloat16 if vector else torch.float32)
    expected = TorchEvaluator(DictWeightResolver({})).run(module, {"value": values})[0]
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    runtime.prepare(values)
    actual = runtime.run(values)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
