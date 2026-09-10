# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


@pytest.mark.parametrize("kind", ["pointwise", "softmax", "softmax_tiled"])
@pytest.mark.parametrize("distributed", [False, True])
def test_compiled_fusion_has_one_kernel_and_correct_values(tmp_path, kind, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    columns = 5003 if kind == "softmax_tiled" else 37
    shape = (4, columns)
    value_type = fm.tensor_type("bfloat16", shape)
    metadata = {}
    if distributed:
        placement = fm.Placement((2, 2), "xy", "bb")
        value_type = fm.DistributedType(value_type, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()), placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", value_type, id="x")
            wide = fm.F.tensors.cast(x, "float32", name="wide")
            result = fm.F.math.sigmoid(wide, name="compute") if kind == "pointwise" else fm.F.nn.softmax(
                wide, name="compute")
            result = fm.F.tensors.cast(result, "bfloat16", name="result")
            self.function("main", (x, ), (result, ))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main", metadata=metadata).build()
    compiled = Compiler().compile(module).module
    calls = [node for node in compiled.nodes if node.op == "tir.call"]
    assert len(calls) == 1
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    x = torch.randn(shape, device="cuda").bfloat16()
    output = torch.full_like(x, float("nan"))
    arguments = tuple(x if item["role"] == "parameter" else output for item in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    expected = (x.float().sigmoid() if kind == "pointwise" else x.float().softmax(-1)).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0.008, atol=1e-5)
