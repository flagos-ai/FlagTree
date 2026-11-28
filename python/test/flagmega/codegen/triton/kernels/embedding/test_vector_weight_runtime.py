# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("lanes", [(), (8, ), (2, 4)])
@pytest.mark.parametrize("split", [False, True])
def test_embedding_reads_every_weight_lane_and_writes_every_output_lane(tmp_path, lanes, split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    mesh = fm.Placement((2, 2), "yx", "bb")

    class Embedding(fm.Module):

        def forward(self):
            dtype = fm.vector_type("bfloat16", lanes) if lanes else "bfloat16"
            it = fm.DistributedType(fm.tensor_type("int32", (1, )), (fm.SBP.broadcast(), ), mesh)
            policy = fm.SBP.split_contiguous((1, )) if split else fm.SBP.broadcast()
            wt = fm.DistributedType(fm.tensor_type(dtype, (17, 32)), (fm.SBP.broadcast(), policy), mesh)
            indices, weight = self.input("indices", it), self.input("weight", wt)
            output = fm.F.nn.embedding(indices, weight, padding_idx=0)
            self.function("main", (indices, weight), (output, ))

    module = Embedding(dialect="distributed", stage="frozen_constants", entry="main",
                       metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    weight = torch.randn(17, 32, *lanes, device="cuda").bfloat16()
    for token in (0, 7, 16):
        indices = torch.tensor([token], dtype=torch.int32, device="cuda")
        runtime.prepare(indices, weight)
        actual = runtime.run(indices, weight)
        expected = torch.zeros_like(weight[:1]) if token == 0 else weight[token:token + 1]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
