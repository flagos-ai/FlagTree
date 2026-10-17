# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A by-value scalar can be materialized into a distributed tensor."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype,values", [("bool", (True, False)), ("int32", (17, -3)),
                                         ("float32", (1.25, -2.5))])
def test_rank_zero_tensor_load_uses_scalar_pointer_and_mask(tmp_path, dtype, values):
    placement = fm.Placement((2, 4), "yx", "bb")
    scalar = fm.tensor_type(dtype, ())
    local = fm.DistributedType(scalar, (), placement)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", scalar, id="source")
            loaded = fm.F.distributed.force_boxing(source, local, name="load_scalar")
            vector = fm.F.tensors.broadcast_to(loaded, shape=(8,))
            result = fm.F.distributed.force_boxing(vector, fm.tensor_type(dtype, (8,)))
            self.function("main", (source,), (result,))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    output = torch.empty(8, dtype={"bool": torch.bool, "int32": torch.int32, "float32": torch.float32}[dtype], device="cuda")
    try:
        def arguments(value):
            return tuple(output if spec["role"] == "result" else value for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments(values[0]))
        for value in values:
            GeneratedTirCallGraphModule.run_into(runtime, *arguments(value))
            torch.testing.assert_close(output, torch.full_like(output, value), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
