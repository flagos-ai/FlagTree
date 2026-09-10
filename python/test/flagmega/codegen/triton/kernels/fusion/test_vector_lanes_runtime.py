# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.ops.ntt.vectorized_cast import cast_vector_axes
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


@pytest.mark.parametrize("axes,lanes,new_lanes", [((1, ), (8, ), (4, )), ((0, 1), (2, 4), (1, 4)),
                                                  ((1, 1), (2, 4), (1, 4)), ((-1, ), (8, ), (4, )),
                                                  ((1, ), (8, ), (2, 4))])
@pytest.mark.parametrize("distributed", [False, True])
def test_fused_cast_activation_reuses_real_vector_coordinate_mapping(tmp_path, axes, lanes, new_lanes, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    shape = (4, 8)
    value_type = fm.tensor_type(fm.vector_type("bfloat16", lanes), shape)
    metadata = {}
    if distributed:
        placement = fm.Placement((2, 2), "xy", "bb")
        value_type = fm.DistributedType(value_type, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()), placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", value_type)
            wide = fm.F.ntt.vectorized_cast(x, fm.vector_type("float32", new_lanes), axes)
            y = fm.F.math.vectorized_unary(wide, unary_op="sigmoid")
            self.function("main", (x, ), (y, ))

    module = Graph(dialect="high_level", stage="frozen_constants", entry="main", metadata=metadata).build()
    compiled = Compiler().compile(module).module
    assert len([node for node in compiled.nodes if node.op == "tir.call"]) == 1
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    x = torch.linspace(-3, 3, prod((*shape, *lanes)), device="cuda").reshape((*shape, *lanes)).bfloat16()
    input_axes, output_axes = cast_vector_axes(lanes, new_lanes, axes, len(shape))
    logical = unpack_physical(x, len(shape), lanes, input_axes)
    expected = pack_physical(logical.float().sigmoid(), len(shape), new_lanes, output_axes)
    output = torch.full_like(expected, float("nan"))
    arguments = tuple(x if item["role"] == "parameter" else output for item in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=1e-7)
