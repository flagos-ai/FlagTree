# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from python.test.flagmega.codegen.triton.candidates.test_packed_partial_tail import projection


@pytest.mark.parametrize("k,n,output_split", [(64, 8, False), (96, 24, False), (96, 48, True)])
def test_packed_partial_projection_masks_local_output_and_reduction_tails(tmp_path, k, n, output_split):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    compiled = Compiler().compile(projection(k, n, output_split=output_split)).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    lhs = torch.full((1, k), 1 / 16, device="cuda", dtype=torch.bfloat16)
    logical_rhs = (torch.arange(k * n, device="cuda").reshape(k, n).remainder(5) - 2).bfloat16() / 16
    rhs = logical_rhs.reshape(k // 16, 2, 8, n // 8, 8).permute(0, 3, 4, 1, 2).contiguous()
    runtime.prepare(lhs, rhs)
    output = runtime.run(lhs, rhs)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, lhs @ logical_rhs, rtol=0, atol=0)
