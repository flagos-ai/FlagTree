# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("lanes", [(8, ), (2, 4)])
@pytest.mark.parametrize("split", [False, True])
def test_vector_sigmoid_is_bitwise_identical_to_scalar_kernel(tmp_path, dtype, lanes, split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    typ = fm.tensor_type(fm.vector_type(dtype, lanes), (4, 16))
    metadata = {}
    if split:
        placement = fm.Placement((2, 2), "yx", "bb")
        typ = fm.DistributedType(typ, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()), placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", typ, id="value")
            scalar = fm.F.math.sigmoid(fm.F.tensors.bitcast(value, dtype), name="scalar")
            vector = fm.F.math.vectorized_unary(
                value, unary_op="sigmoid", name="vector", metadata={
                    "selected_vectorization": "test.vector_sigmoid", "selected_vector_axes": (1, ) * len(lanes),
                    "selected_vector_lanes": lanes
                })
            self.function("main", (value, ), (scalar, vector))

    module = Graph(dialect="ntt", stage="frozen_constants", entry="main", metadata=metadata).build()
    compiled = Compiler().compile(module).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    value = torch.linspace(-30, 30, 512, device="cuda").to(getattr(torch, dtype)).reshape(4, 16, *lanes)
    scalar = torch.full((4, 128), float("nan"), device="cuda", dtype=getattr(torch, dtype))
    vector = torch.full_like(value, float("nan"))
    outputs = {"scalar": scalar, "vector": vector}
    arguments = tuple(value if item["role"] == "parameter" else outputs[item["buffer"]]
                      for item in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    assert torch.equal(scalar.reshape(-1), vector.reshape(-1))
