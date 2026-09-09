# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


def _module(op: str) -> fm.IRModule:

    class Graph(fm.Module):

        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            input_type = fm.tensor_type("bfloat16", (3, 17))
            lhs = self.input("lhs", input_type, id="lhs")
            if op in {"math.add", "math.mul"}:
                rhs = self.input("rhs", input_type, id="rhs")
                output = getattr(fm.F.math, op.rsplit(".", 1)[-1])(lhs, rhs, name="output")
                inputs = (lhs, rhs)
            elif op == "math.silu":
                output = fm.F.math.silu(lhs, name="output")
                inputs = (lhs, )
            else:
                output = fm.F.tensors.cast(lhs, fm.DType.FLOAT32, name="output")
                inputs = (lhs, )
            self.function("main", inputs, (output, ))

    return Graph().build()


@pytest.mark.parametrize("op,variant,lowered_op", [
    ("math.add", "add", "math.vectorized_binary"),
    ("math.mul", "mul", "math.vectorized_binary"),
    ("math.silu", "silu", "math.vectorized_unary"),
    ("tensors.cast", "cast", "tensors.cast"),
])
def test_package_resolves_selected_elementwise_family_variant_template(
    tmp_path,
    op,
    variant,
    lowered_op,
):
    module = Compiler().compile(_module(op)).module
    package = render_triton_package(module, tmp_path / variant)
    source = (tmp_path / variant / "generated_kernels.py").read_text("utf-8")

    assert package["kind"] == "tir_call_graph/v1"
    call = next(value for value in package["runtime_binding"]["call_abi"]["kernel_calls"]
                if value["semantic_op"] == lowered_op)
    if lowered_op == "math.vectorized_binary":
        assert call["semantic_attrs"]["binary_op"] == variant
    elif lowered_op == "math.vectorized_unary":
        assert call["semantic_attrs"]["unary_op"] == variant
    assert call["variant"] == variant
    assert {"kernel": "elementwise", "variant": variant} in package["kernel_template_specs"]
    assert f"# flagmega-kernel: elementwise/{variant} platform=generic" in source
    assert f"_flagmega_elementwise_{variant}(" in source
    assert f"def {package['symbol']}(" in source


@pytest.mark.parametrize("op", ["math.add", "math.mul", "math.silu", "tensors.cast"])
def test_selected_elementwise_template_executes_through_artifact_runtime_on_h800(
    tmp_path,
    op,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = Compiler().compile(_module(op)).module
    artifact = write_artifact(
        module,
        tmp_path / op.replace(".", "_"),
        target="nvidia-sm90",
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    lhs = torch.randn((3, 17), dtype=torch.bfloat16, device="cuda:0")
    inputs = (lhs, )
    if op in {"math.add", "math.mul"}:
        rhs = torch.randn_like(lhs)
        inputs = (lhs, rhs)
    runtime.prepare(*inputs)
    output = runtime.run(*inputs)
    torch.cuda.synchronize()

    if op == "math.add":
        expected = lhs + inputs[1]
    elif op == "math.mul":
        expected = lhs * inputs[1]
    elif op == "math.silu":
        expected = torch.nn.functional.silu(lhs.float()).to(torch.bfloat16)
    else:
        expected = lhs.float()
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
