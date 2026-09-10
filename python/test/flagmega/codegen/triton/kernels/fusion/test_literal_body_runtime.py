# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


@pytest.mark.parametrize("distributed", [False, True])
def test_manual_binary_literal_body_stays_in_registers(tmp_path, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    shape = (4, 37)

    @fm.fusion(fm.tensor_type("float32", shape))
    def square_plus_bias(value):
        return fm.F.math.add(fm.F.math.mul(value, value), fm.F.builtin.splat_const(value.type, 0.25))

    value_type = square_plus_bias.input_type
    metadata = {}
    if distributed:
        placement = fm.Placement((2, 2), "xy", "bb")
        value_type = fm.DistributedType(value_type, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()), placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", value_type)
            result = fm.F.with_ops(fm.F.math.sigmoid, value, pre_ops={"value": square_plus_bias})
            self.function("main", (value, ), (result, ))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main", metadata=metadata).build()
    compiled = Compiler().compile(module).module
    assert len([node for node in compiled.nodes if node.op == "tir.call"]) == 1
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.linspace(-3, 3, 4 * 37, device="cuda").reshape(shape)
    output = torch.full_like(value, float("nan"))
    arguments = tuple(value if item["role"] == "parameter" else output for item in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, (value.square() + 0.25).sigmoid(), rtol=2e-6, atol=1e-7)
