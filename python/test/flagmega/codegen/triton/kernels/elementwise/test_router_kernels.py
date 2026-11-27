# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def compiled(op, dtype):
    tensor = fm.tensor_type(dtype, (3, 17))
    types = (tensor, tensor) if op == "div" else (tensor, )
    return Compiler().compile(primitive_module(fm.get_definition(f"math.{op}"), types)).module


@pytest.mark.parametrize("op", ["div", "sigmoid"])
def test_router_elementwise_uses_generic_selected_template(tmp_path, op):
    module = compiled(op, "float32")
    package = render_triton_package(module, tmp_path)
    source = (tmp_path / "generated_kernels.py").read_text()
    assert {"kernel": "elementwise", "variant": op} in package["kernel_template_specs"]
    assert f"# flagmega-kernel: elementwise/{op} platform=generic" in source
    assert "Qwen" not in source
    if op == "div":
        assert "tl.div_rn(" in source


@pytest.mark.parametrize("op", ["div", "sigmoid"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_selected_router_elementwise_executes_local_shards_on_h800(tmp_path, op, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H800/SM90 CUDA required")
    artifact = write_artifact(compiled(op, dtype), tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    left = torch.linspace(-11, 11, 51, device="cuda:0").reshape(3, 17).to(getattr(torch, dtype))
    values = (left, )
    if op == "div":
        right = torch.linspace(0.01, 3, 51, device="cuda:0").reshape(3, 17).to(left.dtype)
        values = (left, right)
        expected = left / right
    else:
        expected = left.float().sigmoid().to(left.dtype)
    runtime.prepare(*values)
    actual = runtime.run(*values)
    torch.cuda.synchronize()
    # Division is round-to-nearest, while sigmoid's exp implementation may
    # differ by FP32 ulps. BF16 output must preserve the declared boundary.
    tolerance = 0 if dtype == "bfloat16" or op == "div" else 2e-7
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
