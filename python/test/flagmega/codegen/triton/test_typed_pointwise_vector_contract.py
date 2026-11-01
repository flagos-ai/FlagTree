# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical vector lanes do not require an earlier AutoVectorize decision."""

from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.vectorization import vectorization_contract
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRSchemaError

from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def typed_module(op, shape):
    attrs = {"binary_op": "add"} if op == "math.vectorized_binary" else (
        {"unary_op": "sigmoid"} if op == "math.vectorized_unary" else {})
    kind = fm.tensor_type(fm.vector_type("float32", (8,)), shape)
    definition = fm.get_definition(op)
    return primitive_module(definition, (kind,) * len(definition.input_parameters), **attrs)


@pytest.mark.parametrize("op", ["math.add", "math.mul", "math.silu",
                                "math.vectorized_binary", "math.vectorized_unary"])
@pytest.mark.parametrize("shape", [(), (2, 7)])
def test_typed_pointwise_contract_survives_normal_selection_and_codegen(tmp_path, op, shape):
    source = typed_module(op, shape)
    contract = vectorization_contract(source.node_map["output"])
    assert contract["kind"] == "typed_elements"
    assert contract["axes"] == () and contract["lanes"] == (8,)
    module = Compiler().compile(source).module
    restored = fm.load_module(fm.emit_module(module, tmp_path / "selected.py"))
    assert restored.semantic_hash == module.semantic_hash
    render_triton_package(restored, tmp_path / "codegen")
    assert "# flagmega-kernel: pack/" not in (tmp_path / "codegen/generated_kernels.py").read_text()


@pytest.mark.parametrize("op", ["math.div", "math.exp", "math.sigmoid"])
@pytest.mark.parametrize("shape", [(), (2, 7)])
def test_scalar_only_pointwise_schemas_remain_explicit(op, shape):
    with pytest.raises(IRSchemaError, match="scalar|BF16/FP32"):
        typed_module(op, shape)


@pytest.mark.parametrize("op", ["math.add", "math.vectorized_binary", "math.vectorized_unary"])
@pytest.mark.parametrize("shape", [(), (2, 7)])
def test_typed_pointwise_gpu_matches_independent_lane_reference(tmp_path, op, shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from triton.flagmega.runtime import load

    source = typed_module(op, shape)
    path = write_artifact(Compiler().compile(source).module, tmp_path / "artifact",
                          target="nvidia-sm90", emit_executable=True)
    runtime = load(path, device="cuda:0")
    try:
        value = (torch.arange(prod((*shape, 8)), device="cuda") % 31 - 15).float().reshape((*shape, 8)) / 8
        arguments = (value,) if op == "math.vectorized_unary" else (value, value + 1 / 8)
        runtime.prepare(*arguments)
        actual = runtime.run(*arguments)
        torch.cuda.synchronize()
        expected = torch.sigmoid(value) if op == "math.vectorized_unary" else arguments[0] + arguments[1]
        torch.testing.assert_close(actual, expected, rtol=1e-6 if op == "math.vectorized_unary" else 0,
                                   atol=1e-6 if op == "math.vectorized_unary" else 0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
